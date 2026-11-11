import multiprocessing
import os
import pickle
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dance_dataset import AISTPPDataset
from dataset.preprocess import increment_path
from model.adan import Adan
from model.diffusion import GaussianDiffusion
from model.model import DanceDecoder
from model.edge_model import EDGEDecoder
from vis import SMPLSkeleton
from einops import rearrange
import numpy as np
from model.metric import Metric

def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


class EDGE:
    def __init__(
        self,
        feature_type,
        checkpoint_path="",
        normalizer=None,
        EMA=True,
        learning_rate=4e-4,
        weight_decay=0.02,
        arch="mace",
        edge_seq_len=240,
    ):

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        set_seed(42)
        state = AcceleratorState()
        num_processes = state.num_processes
        use_baseline_feats = feature_type == "baseline"

        pos_dim = 3
        rot_dim = 24 * 6  # 24 joints, 6dof
        self.repr_dim = repr_dim = pos_dim + rot_dim + 4

        feature_dim = 35 if use_baseline_feats else 4800

        horizon_seconds = 40
        FPS = 30
        self.horizon = horizon = horizon_seconds * FPS
        if checkpoint_path != "":  # the architecture is read from the checkpoint: the original EDGE denoiser has a "net." prefix
            _keys = torch.load(checkpoint_path, map_location="cpu")["model_state_dict"]
            if any(k.startswith("net.") for k in _keys):
                arch, edge_seq_len = "edge", _keys["net.null_cond_embed"].shape[1]
        self.arch = arch
        if arch == "edge":
            self.horizon = horizon = edge_seq_len

        self.accelerator.wait_for_everyone()

        checkpoint = None
        if checkpoint_path != "":
            checkpoint = torch.load(checkpoint_path, map_location=self.accelerator.device)
            self.normalizer = checkpoint["normalizer"]

        if arch == "edge":
            model = EDGEDecoder(nfeats=repr_dim, seq_len=horizon, cond_feature_dim=feature_dim)
        else:
            model = DanceDecoder(
                nfeats=repr_dim,
                seq_len=horizon,
                latent_dim=512,
                ff_size=1024,
                num_layers=4,
                num_heads=8,
                dropout=0.1,
                cond_feature_dim=feature_dim,
                activation=F.gelu,
            )

        smpl = SMPLSkeleton(self.accelerator.device)
        diffusion = GaussianDiffusion(
            model,
            horizon,
            repr_dim,
            smpl,
            schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,
            loss_type="l2",
            use_p2=False,
            cond_drop_prob=0.25,
            guidance_weight=2,
        )

        print("Model has {} parameters".format(sum(y.numel() for y in model.parameters())))

        self.model = self.accelerator.prepare(model)
        self.diffusion = diffusion.to(self.accelerator.device)
        optim = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optim = self.accelerator.prepare(optim)

        if checkpoint_path != "":
            self.model.load_state_dict(maybe_wrap(checkpoint["model_state_dict"], num_processes,))

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def train_loop(self, opt):
        # load datasets
        train_tensor_dataset_path = os.path.join(opt.processed_data_dir, f"train_tensor_dataset.pkl")
        test_tensor_dataset_path = os.path.join(opt.processed_data_dir, f"test_tensor_dataset.pkl")
        if (not opt.no_cache and os.path.isfile(train_tensor_dataset_path) and os.path.isfile(test_tensor_dataset_path)):
            train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
            test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
        else:
            train_dataset = AISTPPDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=True,
                force_reload=opt.force_reload,
            )
            test_dataset = AISTPPDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=False,
                normalizer=train_dataset.normalizer,
                force_reload=opt.force_reload,
            )
            # cache the dataset in case
            if self.accelerator.is_main_process:
                pickle.dump(train_dataset, open(train_tensor_dataset_path, "wb"))
                pickle.dump(test_dataset, open(test_tensor_dataset_path, "wb"))

        # set normalizer
        self.normalizer = test_dataset.normalizer
        # data loaders
        # decide number of workers based on cpu count
        import random, numpy
        def worker_init_fn(worker_id):
            seed = 42 + worker_id
            numpy.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
        num_cpus = multiprocessing.cpu_count()
        val_data_loader = None
        if opt.val_every > 0:  # hold out whole sequences (every val_seq_every-th) from training for the validation loss
            seqs = sorted({os.path.basename(f).rsplit('_slice', 1)[0] for f in train_dataset.data["filenames"]})
            val_seqs = set(seqs[::opt.val_seq_every])
            is_val = [os.path.basename(f).rsplit('_slice', 1)[0] in val_seqs for f in train_dataset.data["filenames"]]
            val_idx = [i for i, v in enumerate(is_val) if v]
            train_idx = [i for i, v in enumerate(is_val) if not v]
            print(f"validation: {len(val_seqs)} held-out sequences, {len(val_idx)} slices; training on {len(train_idx)} slices")
            val_data_loader = DataLoader(torch.utils.data.Subset(train_dataset, val_idx), batch_size=opt.batch_size, shuffle=False, num_workers=2)
            train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
        train_data_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, \
            num_workers=min(int(num_cpus * 0.75), 32), pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)
        test_data_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, \
            num_workers=2, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)

        train_data_loader = self.accelerator.prepare(train_data_loader)
        # boot up multi-gpu training. test dataloader is only on main process
        load_loop = (partial(tqdm, position=1, desc="Batch") if self.accelerator.is_main_process else lambda x: x)
        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name, exist_ok=opt.start_epoch > 1))
            opt.exp_name = save_dir.split("/")[-1]
            wandb.init(project=opt.wandb_pj_name, name=opt.exp_name)
            save_dir = Path(save_dir)
            wdir = save_dir / "weights"
            wdir.mkdir(parents=True, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gt_dir = './data/test'
        metric = None
        if not opt.skip_sample:
            print('Build metric')
            metric = Metric(gt_dir, device)

        self.accelerator.wait_for_everyone()
        best_val, bad_evals = float("inf"), 0
        base_lr = [g["lr"] for g in self.optim.param_groups]
        for epoch in range(opt.start_epoch, opt.epochs + 1):
            if opt.lr_schedule == "cosine":  # linear warm-up, then cosine decay to lr_min_frac * base over the epochs
                import math
                if epoch <= opt.warmup_epochs:
                    f = epoch / opt.warmup_epochs
                else:
                    p = (epoch - opt.warmup_epochs) / max(1, opt.epochs - opt.warmup_epochs)
                    f = opt.lr_min_frac + (1 - opt.lr_min_frac) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
                for g, b in zip(self.optim.param_groups, base_lr):
                    g["lr"] = b * f
            avg_loss = 0
            avg_vloss = 0
            avg_fkloss = 0
            avg_footloss = 0
            # train
            self.train()
            for step, (x, cond, filename, wavnames, genre) in enumerate(load_loop(train_data_loader)):
                total_loss, (loss, v_loss, fk_loss, foot_loss) = self.diffusion(x, cond, genre, t_override=None)
                self.optim.zero_grad()
                self.accelerator.backward(total_loss)
                if opt.grad_clip > 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), opt.grad_clip)
                self.optim.step()

                # ema update and train loss update only on main
                if self.accelerator.is_main_process:
                    avg_loss += loss.detach().cpu().numpy()
                    avg_vloss += v_loss.detach().cpu().numpy()
                    avg_fkloss += fk_loss.detach().cpu().numpy()
                    avg_footloss += foot_loss.detach().cpu().numpy()
                    if step % opt.ema_interval == 0:
                        self.diffusion.ema.update_model_average(self.diffusion.master_model, self.diffusion.model)
                        
            if self.accelerator.is_main_process:
                avg_loss /= len(train_data_loader)
                avg_vloss /= len(train_data_loader)
                avg_fkloss /= len(train_data_loader)
                avg_footloss /= len(train_data_loader)
                log_dict = {"Train Loss": avg_loss, "V Loss": avg_vloss, "FK Loss": avg_fkloss, "Foot Loss": avg_footloss, }
                print(f"epoch {epoch}", log_dict, flush=True)

            if self.accelerator.is_main_process and opt.last_every > 0 and epoch % opt.last_every == 0:
                net = self.accelerator.unwrap_model(self.model)  # full state (with optimizer) so a crashed run resumes exactly
                torch.save({"ema_state_dict": self.diffusion.master_model.state_dict(), "model_state_dict": net.state_dict(),
                            "optimizer_state_dict": self.optim.state_dict(), "normalizer": self.normalizer, "epoch": epoch},
                           os.path.join(wdir, "last.pt"))
            stop = False
            if self.accelerator.is_main_process and val_data_loader is not None and epoch % opt.val_every == 0:
                self.eval()
                with torch.no_grad(), torch.random.fork_rng(devices=[self.accelerator.device] if torch.cuda.is_available() else []):
                    torch.manual_seed(1234)  # same noise and timesteps at every evaluation
                    tot = n = 0
                    for vx, vcond, _, _, vgenre in val_data_loader:
                        vx, vcond, vgenre = vx.to(self.accelerator.device), vcond.to(self.accelerator.device), vgenre.to(self.accelerator.device)
                        vt, _ = self.diffusion(vx, vcond, vgenre, t_override=None)
                        tot += vt.item() * len(vx)
                        n += len(vx)
                val_loss = tot / n
                improved = val_loss < best_val
                if improved:
                    best_val, bad_evals = val_loss, 0
                    net = self.accelerator.unwrap_model(self.model)
                    torch.save({"ema_state_dict": self.diffusion.master_model.state_dict(), "model_state_dict": net.state_dict(),
                                "normalizer": self.normalizer, "epoch": epoch, "val_loss": val_loss}, os.path.join(wdir, "best.pt"))
                else:
                    bad_evals += 1
                print(f"epoch {epoch} val_loss {val_loss:.5f} best {best_val:.5f} ({'best' if improved else f'{bad_evals} without improvement'})", flush=True)
                wandb.log({"Val Loss": val_loss, "epoch": epoch})
                stop = opt.patience > 0 and bad_evals >= opt.patience
                self.train()

            # Save model
            if (epoch % opt.save_interval) == 0:
                # everyone waits here for the val loop to finish ( don't start next train epoch early)
                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.eval()
                    wandb.log(log_dict)
                    net = self.accelerator.unwrap_model(self.model)
                    ckpt = {"ema_state_dict": self.diffusion.master_model.state_dict(), "model_state_dict": net.state_dict(), "optimizer_state_dict": self.optim.state_dict(), "normalizer": self.normalizer, }
                    if opt.slim_ckpt:
                        del ckpt["optimizer_state_dict"]
                    torch.save(ckpt, os.path.join(wdir, f"train-{epoch}.pt"))
                    print(f"[MODEL SAVED at Epoch {epoch}]")

                    if opt.skip_sample:
                        continue
                    print("Generating Sample")
                    # draw a music from the test dataset
                    (x, cond, filename, wavnames, genre) = next(iter(test_data_loader))
                    cond = cond.to(self.accelerator.device)
                    print(cond.shape, x.shape)
                    self.diffusion.render_sample(x.shape, cond, genre, self.normalizer, epoch, os.path.join(opt.render_dir, "train_" + opt.exp_name), os.path.join(opt.eval_dir, "train_" + opt.exp_name), name=wavnames, sound=True,)
                    metric.calculate_metric(os.path.join(opt.eval_dir, "train_" + opt.exp_name, str(epoch)), device)

            if stop:
                print(f"early stop at epoch {epoch}: no validation improvement in {opt.patience} evaluations (best {best_val:.5f})", flush=True)
                break

        if self.accelerator.is_main_process:
            wandb.run.finish()

    def test_loop(self, opt):
        # load datasets
        train_tensor_dataset_path = os.path.join(opt.processed_data_dir, f"train_tensor_dataset.pkl")
        test_tensor_dataset_path = os.path.join(opt.processed_data_dir, f"test_tensor_dataset.pkl")
        train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
        if os.path.isfile(test_tensor_dataset_path):
            test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
        else:
            test_dataset = AISTPPDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=False,
                normalizer=train_dataset.normalizer,
                force_reload=opt.force_reload,
            )
            # cache the dataset in case
            if self.accelerator.is_main_process:
                pickle.dump(test_dataset, open(test_tensor_dataset_path, "wb"))

        # set normalizer
        self.normalizer = test_dataset.normalizer

        # data loaders
        test_data_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, \
            num_workers=2, pin_memory=True, drop_last=False)

        save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
        opt.exp_name = save_dir.split("/")[-1]
        epoch = 0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gt_dir = './data/test'
        print('Build metric')
        metric = Metric(gt_dir, device)

        self.eval()
        # 后面要把下面接触注解
        print("Generating Sample")
        # draw a music from the test dataset
        (x, cond, filename, wavnames, genre) = next(iter(test_data_loader))
        cond = cond.to(self.accelerator.device)
        print(cond.shape, x.shape)
        self.diffusion.render_sample(x.shape, cond, genre, self.normalizer, epoch, os.path.join(opt.render_dir, "train_" + opt.exp_name), os.path.join(opt.eval_dir, "train_" + opt.exp_name), name=wavnames, sound=True,)
        metric.calculate_metric(os.path.join(opt.eval_dir, "train_" + opt.exp_name, str(epoch)), device)

                    

    def render_sample(self, cond, genre, device, output_dir, file_name):
        cond = rearrange(torch.from_numpy(cond).to(device), 't c -> 1 t c')
        genre = torch.tensor([genre]).to(device)
        file_name = [os.path.join(output_dir, file_name.replace('.pkl', '.npy'))]
        b, t, c = cond.shape
        self.diffusion.render_sample((b, t, 151), cond, genre, self.normalizer, 0, None, output_dir, file_name)
