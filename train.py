"""
Music JEPA training entry point.

Single GPU:
    python train.py

Multi-GPU (DDP):
    torchrun --nproc_per_node=$NPROC_PER_NODE train.py

Resume from checkpoint:
    python train.py --ckpt checkpoints/last.ckpt
"""
import argparse
import os

import yaml
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from jepa.model import build_model
from jepa.module import JEPAModule, build_dataloaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--ckpt", default=None, help="Resume from checkpoint")
    parser.add_argument("--checkpoint_dir", default="checkpoints", help="Directory for saved checkpoints")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    L.seed_everything(cfg.get("seed", 42), workers=True)

    model = build_model(cfg)
    module = JEPAModule(
        model=model,
        **cfg["training"],
    )

    train_loader, val_loader = build_dataloaders(cfg)
    print(f"Train playlists: {len(train_loader.dataset):,}  |  Val: {len(val_loader.dataset):,}")

    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="jepa-{epoch:03d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_last=True,
            save_top_k=3,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer_cfg = cfg.get("trainer", {})
    if "NPROC_PER_NODE" in os.environ:
        trainer_cfg["devices"] = int(os.environ["NPROC_PER_NODE"])
    trainer = L.Trainer(
        max_epochs=cfg["training"]["max_epochs"],
        callbacks=callbacks,
        logger=TensorBoardLogger("logs", name="music_jepa"),
        log_every_n_steps=10,
        precision="16-mixed",
        **trainer_cfg,
    )

    trainer.fit(module, train_loader, val_loader, ckpt_path=args.ckpt)


if __name__ == "__main__":
    main()
