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
from tqdm import tqdm
import torchvision.transforms as T

from jepa.model import build_model
from jepa.module import JEPAModule


@torch.no_grad()
def embed_all(
    encoder,
    spectrograms_dir: str,
    img_size=(96, 216),
    batch_size: int = 128,
    device="cuda",
):
    transform = T.Compose([
        T.Grayscale(),
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    png_files = [f for f in os.listdir(spectrograms_dir) if f.endswith(".png")]
    print(f"Spectrograms to embed: {len(png_files):,}")

    encoder = encoder.to(device).eval()
    embeddings = {}

    for i in tqdm(range(0, len(png_files), batch_size), desc="Embedding"):
        batch_files = png_files[i : i + batch_size]
        imgs = []
        ids = []
        for fn in batch_files:
            try:
                img = Image.open(os.path.join(spectrograms_dir, fn)).convert("RGB")
                imgs.append(transform(img))
                ids.append(fn[:-4])  # strip .png
            except Exception as e:
                print(f"  Skipping {fn}: {e}")

        if not imgs:
            continue

        batch = torch.stack(imgs).to(device)         # (B, 1, H, W)
        patch_tokens = encoder(batch)                 # (B, N, D)
        vecs = patch_tokens.mean(dim=1).cpu().numpy() # (B, D)

        for track_id, vec in zip(ids, vecs):
            embeddings[track_id] = vec

    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to Lightning checkpoint")
    parser.add_argument("--config", default="configs/encoder.yaml")
    parser.add_argument("--spectrograms_dir", default="data/spectrograms")
    parser.add_argument("--out", default="embeddings/embeddings.npy", help="Output .npy file")
    parser.add_argument("--batch_size", type=int, default=128)
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
