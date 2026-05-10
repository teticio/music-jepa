"""
Train a patch-level playlist head end-to-end with a frozen JEPA encoder.

Unlike train_head.py, the head includes a learned attention-pool that converts
the encoder's 324 patch tokens into a 384-d track embedding. The same pool is
applied to context tracks (whose pooled vectors feed the MLP) and to the target
track (whose pooled vector is the InfoNCE positive), so train and retrieval
stay symmetric. The encoder runs in no_grad mode; only the head's pool and MLP
receive gradients.

After training, run `eval/embed_tracks.py --patch_head <ckpt>` to regenerate
embeddings.npy using the same pool — then the existing inference pipeline
(generate_playlist.py, make playlist, make journey) works unchanged.

Example:
    uv run python train_patch_head.py --config configs/head_continuation_patch.yaml
"""
import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from jepa.model import build_model
from jepa.module import JEPAModule
from jepa.playlist_head import (
    PatchPlaylistHead,
    PatchPlaylistHeadDataset,
    save_patch_head,
)


def info_nce(pred: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = pred @ target.T / temperature
    labels = torch.arange(pred.shape[0], device=pred.device)
    return F.cross_entropy(logits, labels)


def load_encoder(ckpt_path: str, encoder_config_path: str, device: str):
    with open(encoder_config_path) as f:
        cfg = yaml.safe_load(f)
    module = JEPAModule.load_from_checkpoint(
        ckpt_path,
        model=build_model(cfg),
        map_location=device,
    )
    encoder = module.model.encoder
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.to(device).eval()
    return encoder, cfg


def step_continuation(head: PatchPlaylistHead, encoder, batch, device):
    history_specs, target_spec = batch
    history_specs = history_specs.to(device, non_blocking=True)
    target_spec = target_spec.to(device, non_blocking=True)
    B, K = history_specs.shape[:2]

    with torch.no_grad():
        history_patches = encoder(history_specs.flatten(0, 1))
        target_patches = encoder(target_spec)

    history_pooled = head.pool_tracks(history_patches).reshape(B, K, -1)
    target_pooled = head.pool_tracks(target_patches)

    first = history_pooled[:, 0]
    last = history_pooled[:, -1]
    mean = history_pooled.mean(dim=1)
    drift = last - first
    context = torch.cat([last, mean, drift], dim=-1)
    return head(context), target_pooled


def step_infill(head: PatchPlaylistHead, encoder, batch, device):
    left_spec, right_spec, alpha, target_spec = batch
    left_spec = left_spec.to(device, non_blocking=True)
    right_spec = right_spec.to(device, non_blocking=True)
    target_spec = target_spec.to(device, non_blocking=True)
    alpha = alpha.to(device, non_blocking=True).unsqueeze(-1)

    with torch.no_grad():
        left_patches = encoder(left_spec)
        right_patches = encoder(right_spec)
        target_patches = encoder(target_spec)

    left_pooled = head.pool_tracks(left_patches)
    right_pooled = head.pool_tracks(right_patches)
    target_pooled = head.pool_tracks(target_patches)
    interp = (1 - alpha) * left_pooled + alpha * right_pooled
    context = torch.cat([left_pooled, right_pooled, interp], dim=-1)
    return head(context), target_pooled


@torch.no_grad()
def evaluate(head, encoder, loader, device, temperature, task):
    head.eval()
    losses = []
    cosines = []
    step_fn = step_infill if task == "infill" else step_continuation
    for batch in loader:
        pred, target = step_fn(head, encoder, batch, device)
        loss = info_nce(pred, target, temperature)
        cos = F.cosine_similarity(pred, target, dim=-1).mean()
        losses.append(float(loss.item()))
        cosines.append(float(cos.item()))
    head.train()
    return sum(losses) / max(len(losses), 1), sum(cosines) / max(len(cosines), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/head_continuation_patch.yaml")
    parser.add_argument("--out", default=None, help="Override checkpoint output path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.out:
        cfg["training"]["out"] = args.out

    device = cfg["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 42))

    encoder, encoder_cfg = load_encoder(
        cfg["encoder"]["ckpt"],
        cfg["encoder"]["config"],
        device,
    )
    embed_dim = encoder_cfg["model"]["encoder"]["embed_dim"]
    img_size = tuple(encoder_cfg["data"]["img_size"])

    task = cfg["data"].get("task", "continuation")
    dataset = PatchPlaylistHeadDataset(
        playlists_file=cfg["data"]["playlists_file"],
        spectrograms_dir=cfg["data"]["spectrograms_dir"],
        img_size=img_size,
        max_history=cfg["data"]["max_history"],
        min_playlist_len=cfg["data"].get("min_playlist_len", 2),
        task=task,
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

    head = PatchPlaylistHead(
        embed_dim=embed_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"].get("dropout", 0.1),
        residual_source=cfg["model"].get("residual_source", "first"),
        pool_num_heads=cfg["model"].get("pool_num_heads", 4),
        pool_dropout=cfg["model"].get("pool_dropout", 0.0),
    ).to(device)

    opt = torch.optim.AdamW(
        [p for p in head.parameters() if p.requires_grad],
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 0.01),
    )
    epochs = cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    temperature = cfg["training"].get("temperature", 0.07)
    grad_clip = cfg["training"].get("grad_clip", 1.0)
    best_val = math.inf
    epochs_without_improvement = 0
    patience = cfg["training"].get("early_stopping_patience")
    min_delta = cfg["training"].get("early_stopping_min_delta", 0.0)

    print(
        f"Task: {task}  |  Patch head  |  embed_dim: {embed_dim}  |  Device: {device}"
    )
    print(f"Samples: {len(dataset):,}  |  Train: {n_train:,}  |  Val: {n_val:,}")

    step_fn = step_infill if task == "infill" else step_continuation

    for epoch in range(1, epochs + 1):
        head.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        train_losses = []
        for batch in pbar:
            pred, target = step_fn(head, encoder, batch, device)
            loss = info_nce(pred, target, temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
            opt.step()
            train_losses.append(float(loss.item()))
            pbar.set_postfix(loss=sum(train_losses) / len(train_losses))

        scheduler.step()
        val_loss, val_cos = evaluate(head, encoder, val_loader, device, temperature, task)
        print(
            f"epoch={epoch:03d} train_loss={sum(train_losses) / len(train_losses):.4f} "
            f"val_loss={val_loss:.4f} val_cos={val_cos:.4f}"
        )
        if val_loss < best_val - min_delta:
            best_val = val_loss
            epochs_without_improvement = 0
            save_patch_head(cfg["training"]["out"], head, cfg)
            print(f"saved {cfg['training']['out']}")
        else:
            epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= patience:
                print(
                    f"early stopping after {epoch} epochs "
                    f"(best_val={best_val:.4f}, patience={patience})"
                )
                break

    last_path = os.path.splitext(cfg["training"]["out"])[0] + "_last.pt"
    save_patch_head(last_path, head, cfg)
    print(f"saved {last_path}")


if __name__ == "__main__":
    main()
