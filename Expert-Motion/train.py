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


def train(opt):
    model = EDGE(opt.feature_type, checkpoint_path=opt.checkpoint)
    if opt.checkpoint:
        ckpt = torch.load(opt.checkpoint, map_location=model.accelerator.device)
        model.optim.load_state_dict(ckpt["optimizer_state_dict"])
        model.diffusion.master_model.load_state_dict(ckpt["ema_state_dict"])
    if opt.learning_rate is not None:
        for g in model.optim.param_groups:
            g["lr"] = opt.learning_rate
    print("learning rate", model.optim.param_groups[0]["lr"])
    model.train_loop(opt)


if __name__ == "__main__":
    train(parse_train_opt())
