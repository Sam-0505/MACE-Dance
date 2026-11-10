"""Train or fine-tune the Motion Expert.

Fine-tuning from the released checkpoint restores the model, EMA and optimizer state, so training continues where
the release stopped. Example, the lambda = 0 control of the musicality fine-tune:

    WANDB_MODE=offline .venv/bin/python train.py --feature_type baseline --batch_size 64 \
        --checkpoint runs/train/exp/weights/train-3750.pt --exp_name ft_lambda0 --epochs 100 --save_interval 25 \
        --music_signals data/dataset_backups/music_signals_train.npz --lambda_dyn 0
"""
import torch

from args import parse_train_opt
from EDGE import EDGE
from model.adan import Adan
from model.lora import add_lora


def train(opt):
    model = EDGE(opt.feature_type, checkpoint_path=opt.checkpoint, beat_phase_input=opt.beat_phase_input, arch=opt.arch,
                 edge_seq_len=opt.edge_seq_len)
    if opt.checkpoint:
        ckpt = torch.load(opt.checkpoint, map_location=model.accelerator.device)
        if "optimizer_state_dict" in ckpt:
            model.optim.load_state_dict(ckpt["optimizer_state_dict"])
        missing, unexpected = model.diffusion.master_model.load_state_dict(ckpt["ema_state_dict"], strict=False)
        assert not unexpected and all(k.startswith("phase_projection") for k in missing), (missing, unexpected)
    net = model.accelerator.unwrap_model(model.model)
    if hasattr(net, "phase_projection"):
        # beat-phase input: its projection is trained with the rest, as a second optimizer group
        model.optim.optimizer.add_param_group({"params": list(net.phase_projection.parameters())})
        print("beat-phase input on")
    if opt.lora_rank:
        # LoRA (model/lora.py): freeze the released weights, train only low-rank adapters, fresh optimizer over them.
        # The EMA copy gets identical adapters so EMA.update_model_average still pairs parameters one to one.
        net = model.accelerator.unwrap_model(model.model)
        assert net is model.diffusion.model
        n = add_lora(net, opt.lora_rank, opt.lora_alpha)
        add_lora(model.diffusion.master_model, opt.lora_rank, opt.lora_alpha)
        for m_ in (net, model.diffusion.master_model):  # add_lora froze everything; the new beat-phase block must train
            if hasattr(m_, "phase_projection"):
                for p_ in m_.phase_projection.parameters():
                    p_.requires_grad_(True)
        model.optim = model.accelerator.prepare(
            Adan([p for p in net.parameters() if p.requires_grad], lr=opt.learning_rate or 1e-4, weight_decay=0.02))
        print(f"LoRA rank {opt.lora_rank}, alpha {opt.lora_alpha or opt.lora_rank}: {n:,} trainable parameters")
    if opt.learning_rate is not None:
        for g in model.optim.param_groups:
            g["lr"] = opt.learning_rate
    print("learning rate", model.optim.param_groups[0]["lr"])
    model.train_loop(opt)


if __name__ == "__main__":
    train(parse_train_opt())
