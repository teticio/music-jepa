import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def load_embeddings(path: str, normalize: bool = True) -> Tuple[List[str], np.ndarray]:
    data = np.load(path, allow_pickle=True).item()
    ids = list(data.keys())
    vecs = np.stack([data[k] for k in ids]).astype("float32")
    if normalize:
        vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8)
    return ids, vecs


def load_playlists(
    path: str,
    available: Iterable[str],
    min_len: int = 2,
    max_len: Optional[int] = None,
) -> List[List[str]]:
    available = set(available)
    playlists = []
    with open(path) as f:
        for line in f:
            tracks = [t.strip() for t in line.strip().split(",")]
            tracks = [t for t in tracks if t in available]
            if len(tracks) >= min_len and (max_len is None or len(tracks) <= max_len):
                playlists.append(tracks)
    if not playlists:
        raise ValueError(f"No playlists in {path} pass length filter [{min_len}, {max_len}]")
    return playlists


def load_tracks(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        header=None,
        index_col=0,
        names=["artist", "title", "url", "count"],
    )


def describe_track(track_id: str, tracks_df: Optional[pd.DataFrame]) -> str:
    if tracks_df is None or track_id not in tracks_df.index:
        return track_id
    row = tracks_df.loc[track_id]
    return f"{row['artist']} - {row['title']} [{track_id}]"


def make_context(history: Sequence[str], emb: Dict[str, np.ndarray], max_history: int) -> np.ndarray:
    recent = list(history)[-max_history:]
    vecs = np.stack([emb[t] for t in recent]).astype("float32")
    first = vecs[0]
    last = vecs[-1]
    mean = vecs.mean(axis=0)
    drift = last - first
    return np.concatenate([last, mean, drift]).astype("float32")


def make_infill_context(
    left_id: str,
    right_id: str,
    alpha: float,
    emb: Dict[str, np.ndarray],
) -> np.ndarray:
    left = emb[left_id]
    right = emb[right_id]
    interp = (1 - alpha) * left + alpha * right
    return np.concatenate([left, right, interp]).astype("float32")


class PlaylistHead(nn.Module):
    """
    Small residual transition head over frozen JEPA track embeddings.

    The input is [last seed embedding, mean recent embedding, recent drift].
    The output is a normalized vector used to retrieve the next track from the
    embedding catalogue.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        residual_source: str = "first",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        if residual_source not in {"first", "third"}:
            raise ValueError("residual_source must be 'first' or 'third'")
        self.residual_source = residual_source
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if self.residual_source == "third":
            start = self.embed_dim * 2
        else:
            start = 0
        residual = context[:, start : start + self.embed_dim]
        delta = self.net(context)
        return F.normalize(residual + delta, dim=-1)


class PlaylistHeadDataset(Dataset):
    def __init__(
        self,
        playlists_file: str,
        embeddings_file: str,
        max_history: int = 3,
        min_playlist_len: int = 2,
        task: str = "continuation",
        max_span: int = 32,
        max_playlist_len: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        if task not in {"continuation", "infill"}:
            raise ValueError("task must be 'continuation' or 'infill'")
        ids, vecs = load_embeddings(embeddings_file)
        self.ids = ids
        self.emb = dict(zip(ids, vecs))
        self.max_history = max_history
        self.task = task
        self.max_span = max_span
        self.rng = np.random.default_rng(seed)
        min_len = max(min_playlist_len, 3 if task == "infill" else 2)
        self.playlists = load_playlists(playlists_file, self.emb, min_len, max_playlist_len)

        self.samples: List[Tuple[int, int]] = []
        for playlist_idx, playlist in enumerate(self.playlists):
            if task == "infill":
                for target_idx in range(1, len(playlist) - 1):
                        self.samples.append((playlist_idx, target_idx))
            else:
                for target_idx in range(1, len(playlist)):
                    self.samples.append((playlist_idx, target_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        playlist_idx, target_idx = self.samples[idx]
        playlist = self.playlists[playlist_idx]
        target_id = playlist[target_idx]
        if self.task == "infill":
            left_min = max(0, target_idx - self.max_span)
            right_max = min(len(playlist) - 1, target_idx + self.max_span)
            left_idx = int(self.rng.integers(left_min, target_idx))
            right_idx = int(self.rng.integers(target_idx + 1, right_max + 1))
            alpha = (target_idx - left_idx) / (right_idx - left_idx)
            context = make_infill_context(
                playlist[left_idx],
                playlist[right_idx],
                alpha,
                self.emb,
            )
        else:
            history = playlist[max(0, target_idx - self.max_history) : target_idx]
            context = make_context(history, self.emb, self.max_history)
        target = self.emb[target_id].astype("float32")
        return torch.from_numpy(context), torch.from_numpy(target)


def rank_tracks(
    query: np.ndarray,
    ids: Sequence[str],
    vecs: np.ndarray,
    exclude: Optional[Iterable[str]] = None,
    top_k: int = 100,
) -> List[Tuple[str, float]]:
    query = query.astype("float32")
    query = query / max(float(np.linalg.norm(query)), 1e-8)
    sims = vecs @ query
    order = np.argsort(sims)[::-1]
    excluded = set(exclude or [])
    results = []
    for idx in order:
        tid = ids[int(idx)]
        if tid in excluded:
            continue
        results.append((tid, float(sims[idx])))
        if len(results) >= top_k:
            break
    return results


def save_head(path: str, head: PlaylistHead, cfg: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "embed_dim": head.embed_dim,
            "hidden_dim": cfg["model"]["hidden_dim"],
            "dropout": cfg["model"].get("dropout", 0.1),
            "residual_source": head.residual_source,
            "config": cfg,
        },
        tmp,
    )
    os.replace(tmp, path)


def load_head(path: str, device: str = "cpu") -> Tuple[PlaylistHead, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    head = PlaylistHead(
        embed_dim=ckpt["embed_dim"],
        hidden_dim=ckpt["hidden_dim"],
        dropout=ckpt.get("dropout", 0.0),
        residual_source=ckpt.get("residual_source", "first"),
    )
    head.load_state_dict(ckpt["state_dict"])
    head.to(device).eval()
    return head, ckpt.get("config", {})
