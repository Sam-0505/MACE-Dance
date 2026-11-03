import argparse


def parse_train_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="runs/train", help="project/name")
    parser.add_argument("--exp_name", default="exp", help="save to project/name")
    parser.add_argument("--data_path", type=str, default="data/", help="raw data path")
    parser.add_argument("--processed_data_dir", type=str, default="data/dataset_backups/", help="Dataset backup path",)
    parser.add_argument("--render_dir", type=str, default="renders/", help="Sample render path")
    parser.add_argument("--eval_dir", type=str, default="eval/", help="Sample render path")

    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument("--wandb_pj_name", type=str, default="EDGE", help="project name")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--force_reload", action="store_true", help="force reloads the datasets")
    parser.add_argument("--no_cache", action="store_true", help="don't reuse / cache loaded dataset")
    parser.add_argument("--save_interval", type=int, default=50, help='Log model after every "save_period" epoch',)
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    parser.add_argument( "--checkpoint", type=str, default="", help="trained checkpoint path (optional)")
    parser.add_argument("--learning_rate", type=float, default=None, help="override the checkpoint's learning rate")
    # musicality loss (model/musicality.py, musicality_experiment_plan.md Loss V1)
    parser.add_argument("--music_signals", type=str, default="", help="precompute_music_signals.py output; logs r(music, energy) when set")
    parser.add_argument("--lambda_dyn", type=float, default=0.0, help="weight of 1 - r(music, body energy)")
    parser.add_argument("--dyn_music", type=str, default="treble", choices=["treble", "perc", "loud"])
    parser.add_argument("--dyn_sigma", type=float, default=15, help="Gaussian smoothing sigma in frames (15 = slow, 2 = fast)")
    parser.add_argument("--dyn_t_max", type=int, default=1000, help="use the term only for diffusion steps t < this")
    parser.add_argument("--dyn_beta_min", type=float, default=0.0, help="use the term only for guidance mix beta >= this")
    parser.add_argument("--dyn_energy_sigma", type=float, default=0, help="change A: Gaussian low-pass of the joints "
                        "(frames) before body energy, so jitter cannot raise it; 0 = off")
    parser.add_argument("--dyn_hinge", action="store_true", help="change B: loss max(0, r_real - r_pred), no reward past real")
    # long-clip fine-tune (build_long_clips.py, dataset/long_clips.py)
    parser.add_argument("--long_clips", type=str, default="", help="train on these long clips instead of the 8 s slices; "
                        "the npz also holds their music signals, so pass it as --music_signals too")
    parser.add_argument("--lora_rank", type=int, default=0, help="fine-tune LoRA adapters of this rank only (model/lora.py); 0 = full fine-tune")
    parser.add_argument("--lora_alpha", type=float, default=None, help="LoRA scale alpha (default = rank)")
    parser.add_argument("--slim_ckpt", action="store_true", help="save model, EMA and normalizer only (no optimizer)")
    opt = parser.parse_args()
    return opt


def parse_test_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_type", type=str, default="baseline")
    parser.add_argument("--out_length", type=float, default=40, help="max. length of output, in seconds")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoint.pt", help="checkpoint"
    )
    parser.add_argument(
        "--music_dir",
        type=str,
        default="data/test/wavs",
        help="folder containing input music",
    )
    parser.add_argument(
        "--save_motions", action="store_true", help="Saves the motions for evaluation"
    )
    parser.add_argument(
        "--motion_save_dir",
        type=str,
        default="eval/motions",
        help="Where to save the motions",
    )
    parser.add_argument(
        "--cache_features",
        action="store_true",
        help="Save the jukebox features for later reuse",
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Don't render the video",
    )
    parser.add_argument(
        "--use_cached_features",
        action="store_true",
        help="Use precomputed features instead of music folder",
    )
    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default="cached_features/",
        help="Where to save/load the features",
    )
    opt = parser.parse_args()
    return opt
