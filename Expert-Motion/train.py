"""Train the Motion Expert (original four EDGE losses: reconstruction, velocity, forward kinematics, foot contact).

Fine-tuning from the released checkpoint restores the model, EMA and optimizer state, so training continues where
the release stopped:

    WANDB_MODE=offline .venv/bin/python train.py --feature_type baseline --batch_size 64 \
        --checkpoint runs/train/exp/weights/train-3750.pt --exp_name ft --epochs 100 --save_interval 25

--arch edge trains the original EDGE denoiser instead (see hprc/README.md).
"""
import torch

from args import parse_train_opt
from EDGE import EDGE


def train(opt):
    model = EDGE(opt.feature_type, checkpoint_path=opt.checkpoint, arch=opt.arch, edge_seq_len=opt.edge_seq_len)
    if opt.checkpoint:
        ckpt = torch.load(opt.checkpoint, map_location=model.accelerator.device)
        if "optimizer_state_dict" in ckpt:
            model.optim.load_state_dict(ckpt["optimizer_state_dict"])
        model.diffusion.master_model.load_state_dict(ckpt["ema_state_dict"])
    if opt.learning_rate is not None:
        for g in model.optim.param_groups:
            g["lr"] = opt.learning_rate
    print("learning rate", model.optim.param_groups[0]["lr"])
    model.train_loop(opt)


if __name__ == "__main__":
    train(parse_train_opt())
