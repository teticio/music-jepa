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


def summarize_split(split):
    dataset = split.dataset
    indices = split.indices
    playlists = [dataset.playlists[i] for i in indices]
    tracks = {track_id for playlist in playlists for track_id in playlist}
    return len(playlists), len(tracks)


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
    train_playlists, train_tracks = summarize_split(train_loader.dataset)
    val_playlists, val_tracks = summarize_split(val_loader.dataset)
    all_tracks = {
        track_id
        for playlist in train_loader.dataset.dataset.playlists
        for track_id in playlist
    }
    dataset = train_loader.dataset.dataset
    print(
        "Training data after spectrogram filter: "
        f"{train_playlists:,} train playlists / {train_tracks:,} tracks  |  "
        f"{val_playlists:,} val playlists / {val_tracks:,} tracks  |  "
        f"{len(all_tracks):,} unique tracks total"
    )
    print(
        "Raw data: "
        f"{dataset.raw_playlist_count:,} playlists  |  "
        f"{dataset.available_track_count:,} tracks with spectrograms"
    )

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
