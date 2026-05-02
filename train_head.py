"""
Train a lightweight playlist-generation head on frozen JEPA embeddings.

Example:
    uv run python train_head.py --config configs/head.yaml
"""
import argparse
import math
import os

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from jepa.playlist_head import PlaylistHead, PlaylistHeadDataset, save_head


def info_nce(pred: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = pred @ target.T / temperature
    labels = torch.arange(pred.shape[0], device=pred.device)
    return F.cross_entropy(logits, labels)


@torch.no_grad()
def evaluate(head, loader, device, temperature):
    head.eval()
    losses = []
    cosines = []
    for context, target in loader:
        context = context.to(device)
        target = target.to(device)
        pred = head(context)
        loss = info_nce(pred, target, temperature)
        losses.append(float(loss.item()))
        cosines.append(float(F.cosine_similarity(pred, target, dim=-1).mean().item()))
    head.train()
    return sum(losses) / max(len(losses), 1), sum(cosines) / max(len(cosines), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/head.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 42))

    dataset = PlaylistHeadDataset(
        playlists_file=cfg["data"]["playlists_file"],
        embeddings_file=cfg["data"]["embeddings_file"],
        max_history=cfg["data"]["max_history"],
        min_playlist_len=cfg["data"].get("min_playlist_len", 2),
        task=cfg["data"].get("task", "continuation"),
        max_span=cfg["data"].get("max_span", 32),
        max_playlist_len=cfg["data"].get("max_playlist_len", None),
        seed=cfg.get("seed", 42),
    )
    n_val = max(1, int(len(dataset) * cfg["data"].get("val_fraction", 0.05)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.get("seed", 42)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 0),
        pin_memory=device.startswith("cuda"),
    )

    embed_dim = len(dataset.emb[dataset.ids[0]])
    head = PlaylistHead(
        embed_dim=embed_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"].get("dropout", 0.1),
        residual_source=cfg["model"].get("residual_source", "first"),
    ).to(device)

    opt = torch.optim.AdamW(
        head.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 0.01),
    )
    epochs = cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    temperature = cfg["training"].get("temperature", 0.07)
    best_val = math.inf

    print(
        f"Task: {cfg['data'].get('task', 'continuation')}  |  "
        f"Playlist samples: {len(dataset):,}  |  Train: {n_train:,}  |  Val: {n_val:,}"
    )
    print(f"Embedding dim: {embed_dim}  |  Device: {device}")

    for epoch in range(1, epochs + 1):
        head.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        train_losses = []
        for context, target in pbar:
            context = context.to(device)
            target = target.to(device)
            pred = head(context)
            loss = info_nce(pred, target, temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), cfg["training"].get("grad_clip", 1.0))
            opt.step()
            train_losses.append(float(loss.item()))
            pbar.set_postfix(loss=sum(train_losses) / len(train_losses))

        scheduler.step()
        val_loss, val_cos = evaluate(head, val_loader, device, temperature)
        print(
            f"epoch={epoch:03d} train_loss={sum(train_losses) / len(train_losses):.4f} "
            f"val_loss={val_loss:.4f} val_cos={val_cos:.4f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            save_head(cfg["training"]["out"], head, cfg)
            print(f"saved {cfg['training']['out']}")

    last_path = os.path.splitext(cfg["training"]["out"])[0] + "_last.pt"
    save_head(last_path, head, cfg)
    print(f"saved {last_path}")


if __name__ == "__main__":
    main()
