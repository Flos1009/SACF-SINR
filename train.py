"""Training engine for SINR pretraining and SACF-SINR."""

import os
import random

import numpy as np
import torch
from tqdm import tqdm

import datasets
import losses
import models
import utils


class Trainer:
    def __init__(self, model, train_loader, params):
        self.params = params
        self.train_loader = train_loader
        self.model = model
        self.compute_loss = losses.get_loss_function(params)
        self.encode_location = train_loader.dataset.enc.encode
        lr = params["lr"]

        optimizer_name = str(params.get("optimizer", "adam")).lower()
        if optimizer_name == "adamwr":
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=params.get("weight_decay", 0.0)
            )
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=int(params.get("adamwr_t0", 20)),
                T_mult=int(params.get("adamwr_t_mult", 1)),
                eta_min=float(params.get("adamwr_eta_min", 5e-5)),
            )
            print(
                '[optimizer] AdamW with cosine warm restarts | '
                f'T0={int(params.get("adamwr_t0", 20))} | '
                f'eta_min={float(params.get("adamwr_eta_min", 5e-5)):.2e} | '
                f'weight_decay={float(params.get("weight_decay", 0.0)):.2e}',
                flush=True,
            )
        elif optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=params.get("weight_decay", 0.0)
            )
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=params.get("lr_decay", 0.98)
            )
        else:
            raise ValueError(
                f"unknown optimizer={optimizer_name!r}; " 'expected "adam" or "adamwr"'
            )

    def train_one_epoch(self, epoch_idx):
        import time

        self.model.train()
        running_loss = 0.0
        running_loss_t = None

        full_num_batches = len(self.train_loader)
        num_batches = full_num_batches

        use_amp = bool(self.params.get("amp", False)) and str(
            self.params["device"]
        ).startswith("cuda")
        if use_amp:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        quiet = utils.tqdm_disabled()
        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch_idx+1}/{self.params["num_epochs"]}',
            total=num_batches,
            disable=quiet,
            mininterval=5,
        )
        log_every = max(1, num_batches // 5)
        t0 = time.time()
        for batch_idx, batch in enumerate(pbar):
            if batch_idx >= num_batches:
                break
            self.optimizer.zero_grad()
            if use_amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = self.compute_loss(
                        batch, self.model, self.params, self.encode_location
                    )
            else:
                result = self.compute_loss(
                    batch, self.model, self.params, self.encode_location
                )
            loss = result
            loss.backward()

            grad_clip = float(self.params.get("grad_clip", 0) or 0)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()

            loss_d = loss.detach()
            running_loss_t = (
                loss_d.double()
                if running_loss_t is None
                else running_loss_t + loss_d.double()
            )
            read_now = (batch_idx + 1) % log_every == 0 or (
                batch_idx + 1
            ) == num_batches
            if read_now:
                running_loss = float(running_loss_t.item())
                last_loss = float(loss.item())
            if quiet:
                if read_now:
                    pct = 100.0 * (batch_idx + 1) / num_batches
                    elapsed = time.time() - t0
                    eta = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)
                    print(
                        f'Epoch {epoch_idx+1}/{self.params["num_epochs"]} '
                        f'batch {batch_idx+1}/{num_batches} ({pct:.0f}%) '
                        f'loss {running_loss/(batch_idx+1):.4f} ETA {eta/60:.1f}min',
                        flush=True,
                    )
            elif read_now or batch_idx == num_batches - 1:
                pbar.set_postfix(
                    {
                        "loss": f"{last_loss:.4f}",
                        "lr": f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    }
                )
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return running_loss / num_batches

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        params_to_save = {
            k: v
            for k, v in self.params.items()
            if k
            not in (
                "ctx_pool_locs",
                "ctx_pool_start",
                "ctx_pool_count",
                "ctx_pool_feats",
            )
        }
        payload = {
            "state_dict": self.model.state_dict(),
            "params": params_to_save,
        }

        completed_epoch = getattr(self, "_completed_epoch", None)
        if completed_epoch is not None:
            payload["epoch"] = int(completed_epoch)
        torch.save(payload, os.path.join(save_path, "model.pt"))


def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global random seed fixed: {seed}")


def train_standard(model, loader, params, start_epoch=0):
    set_seed(params.get("train_seed", 2026))
    start_epoch = int(start_epoch)
    if start_epoch < 0 or start_epoch > int(params["num_epochs"]):
        raise ValueError(
            f'invalid resume start_epoch={start_epoch} for '
            f'num_epochs={params["num_epochs"]}'
        )
    trainer = Trainer(model, loader, params)
    trainer._completed_epoch = start_epoch
    if start_epoch:
        if str(params.get("optimizer", "adam")).lower() == "adamwr":
            trainer.lr_scheduler.step(start_epoch)
            schedule_note = "cosine schedule restored"
        else:
            lr_factor = float(params.get("lr_decay", 0.98)) ** start_epoch
            for group in trainer.optimizer.param_groups:
                group["lr"] *= lr_factor
            schedule_note = f"lr factor={lr_factor:.6f}"
        print(
            f'[resume] loaded weights through epoch {start_epoch}; '
            f'continuing at epoch {start_epoch + 1}/{params["num_epochs"]} '
            f'(optimizer moments reset, {schedule_note})',
            flush=True,
        )
    save_every = int(params.get("save_every_epochs", 20))

    import time

    total_epochs = int(params["num_epochs"])
    print(
        f"[stage 4/4] optimize model | epochs {start_epoch + 1}-{total_epochs} | "
        f"batches/epoch={len(loader)}",
        flush=True,
    )
    for epoch in range(start_epoch, total_epochs):
        epoch_t0 = time.perf_counter()
        loss = trainer.train_one_epoch(epoch)
        trainer._completed_epoch = epoch + 1
        elapsed = time.perf_counter() - epoch_t0
        lr = trainer.optimizer.param_groups[0]["lr"]
        print(
            f"[epoch {epoch + 1:03d}/{total_epochs}] "
            f"loss={loss:.4f} | lr={lr:.2e} | "
            f"time={elapsed / 60:.2f} min",
            flush=True,
        )

        if (
            save_every > 0
            and (epoch + 1) % save_every == 0
            and (epoch + 1) < params["num_epochs"]
        ):
            checkpoint_dir = params["save_path"] + f"_ep{epoch + 1}"
            trainer.save_model(checkpoint_dir)
            print(f"  checkpoint saved: {checkpoint_dir}/model.pt")
    return trainer


def train_model(params):
    """Build the training data and optimise the configured model."""
    print("\n[stage 1/4] load training data", flush=True)
    dataset = datasets.get_train_data(params)
    params.update(
        {
            "input_dim": dataset.input_dim,
            "num_classes": dataset.num_classes,
            "class_to_taxa": dataset.class_to_taxa,
        }
    )

    labels = torch.from_numpy(np.asarray(dataset.labels)).long()
    order = torch.argsort(labels)
    counts = torch.bincount(labels, minlength=params["num_classes"])
    features = torch.from_numpy(
        np.asarray(dataset.enc.encode(dataset.locs, normalize=True), dtype=np.float32)
    )
    params["ctx_pool_feats"] = features[order].to(params["device"])
    params["ctx_pool_locs"] = torch.from_numpy(
        np.asarray(dataset.locs, dtype=np.float32)
    )[order].to(params["device"])
    params["ctx_pool_start"] = (torch.cumsum(counts, 0) - counts).to(params["device"])
    params["ctx_pool_count"] = counts.to(params["device"])
    print(
        f'[stage 2/4] context pools ready: {len(order):,} obs / '
        f'{params["num_classes"]:,} species | features '
        f'{tuple(params["ctx_pool_feats"].shape)}',
        flush=True,
    )

    generator = torch.Generator().manual_seed(params.get("train_seed", 2026))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params.get("num_workers", 8),
        generator=generator,
    )

    print("[stage 3/4] initialize model and data loader", flush=True)
    model = models.get_model(params).to(params["device"])
    resume_ckpt = params.get("resume_ckpt", "")
    resume_start_epoch = 0
    if resume_ckpt:
        if not os.path.isfile(resume_ckpt):
            raise FileNotFoundError(f"resume checkpoint not found: {resume_ckpt}")
        payload = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        if "state_dict" not in payload:
            raise ValueError(f"resume checkpoint has no state_dict: {resume_ckpt}")
        resume_state = payload["state_dict"]
        if getattr(model, "sacf_sinr_model", False):
            from sacf_model import checkpoint_state

            resume_state = checkpoint_state(resume_state, payload.get("params", {}))
        model.load_state_dict(resume_state, strict=True)

        source_params = payload.get("params", {})
        resume_start_epoch = int(
            payload.get("epoch", source_params.get("num_epochs", 0))
        )
        if resume_start_epoch <= 0:
            raise ValueError(
                "resume checkpoint does not record a positive completed "
                f"epoch count: {resume_ckpt}"
            )
        if resume_start_epoch >= int(params["num_epochs"]):
            raise ValueError(
                f'resume checkpoint is already at epoch {resume_start_epoch}; '
                f'--epochs must be greater than it (got {params["num_epochs"]})'
            )
        print(
            f"[resume] strict model weights loaded from {resume_ckpt} "
            f"(completed epochs={resume_start_epoch})",
            flush=True,
        )
    train_standard(model, loader, params, start_epoch=resume_start_epoch).save_model(
        params["save_path"]
    )


def run_pipeline(cfg):
    """Run the shared training pipeline; evaluation is wrapper-controlled."""
    params = cfg.to_flat_dict()
    name = cfg.base.experiment_name
    params["save_path"] = os.path.join(cfg.base.save_base, name)
    cfg.print_summary(os.path.abspath(params["save_path"]))
    print(
        "[pipeline] stages: data -> context pools -> model -> optimization",
        flush=True,
    )
    set_seed(cfg.base.train_seed)
    os.makedirs(params["save_path"], exist_ok=True)
    model_path = os.path.join(params["save_path"], "model.pt")
    train_model(params)
    print(f"[pipeline] finished; checkpoint={os.path.abspath(model_path)}", flush=True)
