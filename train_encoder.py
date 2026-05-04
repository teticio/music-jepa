"""
Music JEPA training entry point.

Single GPU:
    python train_encoder.py

Multi-GPU (DDP):
    torchrun --nproc_per_node=$NPROC_PER_NODE train_encoder.py

Resume from checkpoint:
    python train_encoder.py --ckpt checkpoints/last.ckpt
"""
import argparse
import os
from datetime import timedelta

import yaml
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from jepa.model import build_model
from jepa.module import JEPAModule, build_dataloaders


def summarize_dataset(dataset):
    tracks = {track_id for playlist in dataset.playlists for track_id in playlist}
    return len(dataset.playlists), len(tracks), len(dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/encoder.yaml")
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
    train_pls, train_tracks, train_pairs = summarize_dataset(train_loader.dataset)
    val_pls, val_tracks, val_pairs = summarize_dataset(val_loader.dataset)
    all_tracks = (
        {t for pl in train_loader.dataset.playlists for t in pl}
        | {t for pl in val_loader.dataset.playlists for t in pl}
    )
    print(
        "Training data after spectrogram filter: "
        f"{train_pls:,} train playlists / {train_tracks:,} tracks / {train_pairs:,} pairs  |  "
        f"{val_pls:,} val playlists / {val_tracks:,} tracks / {val_pairs:,} pairs  |  "
        f"{len(all_tracks):,} unique tracks total"
    )
    print(
        "Raw data: "
        f"{train_loader.dataset.raw_playlist_count:,} playlists  |  "
        f"{train_loader.dataset.available_track_count:,} tracks with spectrograms"
    )

    ckpt_cfg = cfg.get("checkpointing", {})
    ckpt_hours = ckpt_cfg.get("every_n_hours", 2)
    save_top_k = ckpt_cfg.get("save_top_k", 3)
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="jepa-{epoch:03d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=save_top_k,
        ),
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            save_last=True,
            save_top_k=0,
            train_time_interval=timedelta(hours=ckpt_hours) if ckpt_hours else None,
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
