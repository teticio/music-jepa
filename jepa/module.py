import math
import random

import torch
import lightning as L
from torch.utils.data import DataLoader

from .dataset import PlaylistDataset, load_filtered_playlists
from .loss import jepa_loss
from .model import MusicJEPA


class JEPAModule(L.LightningModule):
    def __init__(
        self,
        model: MusicJEPA,
        lr: float = 1.5e-4,
        weight_decay: float = 0.05,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        std_coef: float = 25.0,
        cov_coef: float = 1.0,
        vicreg_target: str = "predicted",
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.std_coef = std_coef
        self.cov_coef = cov_coef
        self.vicreg_target = vicreg_target
        self.save_hyperparameters(ignore=["model"])

    def training_step(self, batch, batch_idx):
        ctx_spec, tgt_spec, target_patch_ids = batch
        predicted, target = self.model(ctx_spec, tgt_spec, target_patch_ids)
        mse, reg = jepa_loss(predicted, target, self.std_coef, self.cov_coef, self.vicreg_target)
        loss = mse + reg
        self.log_dict(
            {"train/loss": loss, "train/mse": mse, "train/reg": reg},
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        ctx_spec, tgt_spec, target_patch_ids = batch
        predicted, target = self.model(ctx_spec, tgt_spec, target_patch_ids)
        mse, reg = jepa_loss(predicted, target, self.std_coef, self.cov_coef, self.vicreg_target)
        loss = mse + reg
        self.log_dict(
            {"val/loss": loss, "val/mse": mse, "val/reg": reg},
            prog_bar=True,
            sync_dist=True,
        )

    def on_before_optimizer_step(self, optimizer):
        step = self.global_step
        total = self.trainer.estimated_stepping_batches
        self.model.update_target_encoder(step, max(total, 1))

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.warmup_epochs * total_steps // max(self.max_epochs, 1)

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def build_dataloaders(cfg, val_fraction=0.05):
    spectrograms_dir = cfg["data"]["spectrograms_dir"]
    playlists, raw_count, available_count = load_filtered_playlists(
        cfg["data"]["playlists_file"],
        spectrograms_dir,
    )

    # Split at the playlist level so consecutive pairs from one playlist
    # don't leak across train/val.
    rng = random.Random(cfg.get("seed", 42))
    shuffled = list(playlists)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_playlists = shuffled[:n_val]
    train_playlists = shuffled[n_val:]

    common = dict(
        spectrograms_dir=spectrograms_dir,
        img_size=tuple(cfg["data"]["img_size"]),
        patch_size=tuple(cfg["data"]["patch_size"]),
        mask_ratio=cfg["data"]["mask_ratio"],
        raw_playlist_count=raw_count,
        available_track_count=available_count,
    )
    train_ds = PlaylistDataset(
        playlists=train_playlists,
        augment=cfg["data"].get("augment", True),
        **common,
    )
    val_ds = PlaylistDataset(
        playlists=val_playlists,
        augment=False,
        **common,
    )

    loader_kwargs = dict(
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["data"]["num_workers"] > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader
