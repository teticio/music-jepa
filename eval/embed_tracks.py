"""
Extract per-track embeddings from a trained Music JEPA encoder.

Runs all spectrogram PNGs through the context encoder (mean-pools patch tokens)
and saves a {track_id: embedding} dict to a .npy file.

Usage:
    python eval/embed_tracks.py --ckpt checkpoints/last.ckpt
    python eval/embed_tracks.py --ckpt checkpoints/last.ckpt --out embeddings/embeddings.npy
    python eval/embed_tracks.py --ckpt checkpoints/last.ckpt --spectrograms_dir data/spectrograms
"""
import argparse
import os

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchvision.transforms as T

from jepa.model import build_model
from jepa.module import JEPAModule


class SpectrogramDataset(Dataset):
    def __init__(self, spectrograms_dir, png_files, img_size):
        self.spectrograms_dir = spectrograms_dir
        self.png_files = png_files
        self.transform = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.png_files)

    def __getitem__(self, idx):
        fn = self.png_files[idx]
        try:
            img = Image.open(os.path.join(self.spectrograms_dir, fn)).convert("L")
            return self.transform(img), fn[:-4]
        except Exception as e:
            print(f"  Skipping {fn}: {e}")
            return None


def _collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs = torch.stack([b[0] for b in batch])
    ids = [b[1] for b in batch]
    return imgs, ids


@torch.no_grad()
def embed_all(
    encoder,
    spectrograms_dir: str,
    img_size=(96, 216),
    batch_size: int = 512,
    num_workers: int = 8,
    device="cuda",
):
    png_files = [f for f in os.listdir(spectrograms_dir) if f.endswith(".png")]
    print(f"Spectrograms to embed: {len(png_files):,}")

    dataset = SpectrogramDataset(spectrograms_dir, png_files, img_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=_collate,
        persistent_workers=num_workers > 0,
    )

    encoder = encoder.to(device).eval()
    embeddings = {}

    for batch in tqdm(loader, desc="Embedding"):
        if batch is None:
            continue
        imgs, ids = batch
        imgs = imgs.to(device, non_blocking=True)
        patch_tokens = encoder(imgs)
        vecs = patch_tokens.mean(dim=1).cpu().numpy()
        for track_id, vec in zip(ids, vecs):
            embeddings[track_id] = vec

    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to Lightning checkpoint")
    parser.add_argument("--config", default="configs/encoder.yaml")
    parser.add_argument("--spectrograms_dir", default="data/spectrograms")
    parser.add_argument("--out", default="embeddings/embeddings.npy", help="Output .npy file")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    module = JEPAModule.load_from_checkpoint(
        args.ckpt,
        model=build_model(cfg),
        map_location=args.device,
    )
    encoder = module.model.encoder

    embeddings = embed_all(
        encoder,
        args.spectrograms_dir,
        img_size=tuple(cfg["data"]["img_size"]),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp_path = f"{args.out}.tmp"
    with open(tmp_path, "wb") as f:
        np.save(f, embeddings)
    os.replace(tmp_path, args.out)
    print(f"Saved {len(embeddings):,} embeddings -> {args.out}")


if __name__ == "__main__":
    main()
