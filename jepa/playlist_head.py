import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
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
    head = head.module if isinstance(head, nn.DataParallel) else head
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(
        {
            "head_type": "track",
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


def load_head(path: str, device: str = "cpu") -> Tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    head_type = ckpt.get("head_type", "track")
    if head_type == "patch":
        head = PatchPlaylistHead(
            embed_dim=ckpt["embed_dim"],
            hidden_dim=ckpt["hidden_dim"],
            dropout=ckpt.get("dropout", 0.0),
            residual_source=ckpt.get("residual_source", "first"),
            pool_num_heads=ckpt.get("pool_num_heads", 4),
            pool_dropout=ckpt.get("pool_dropout", 0.0),
        )
    else:
        head = PlaylistHead(
            embed_dim=ckpt["embed_dim"],
            hidden_dim=ckpt["hidden_dim"],
            dropout=ckpt.get("dropout", 0.0),
            residual_source=ckpt.get("residual_source", "first"),
        )
    head.load_state_dict(ckpt["state_dict"])
    head.to(device).eval()
    return head, ckpt.get("config", {})


class AttentionPool(nn.Module):
    """
    Learned attention-pool: (B, N, D) -> (B, D).

    A single learned query attends over the patch tokens, producing one vector
    per track. Replaces the encoder's mean-pool with a content-aware aggregator
    that can up-weight distinctive patches and down-weight generic ones.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        B = patches.shape[0]
        q = self.query.expand(B, -1, -1)
        out, _ = self.attn(q, patches, patches)
        return self.norm(out.squeeze(1))


class PatchPlaylistHead(nn.Module):
    """
    Patch-level alternative to PlaylistHead. Adds a learned attention-pool that
    converts the JEPA encoder's patch tokens (B, N, D) into a track-level
    (B, D) embedding. The MLP aggregator that consumes the [a|b|c] context is
    identical to PlaylistHead, so once embeddings have been regenerated with
    this pool, the inference pipeline (generate_playlist.py) is drop-in
    compatible.

    Training is end-to-end with a frozen encoder: the same pool is applied to
    both context tracks (whose pooled vectors feed the MLP) and the target
    track (whose pooled vector is the InfoNCE positive), so train and retrieval
    stay symmetric.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        residual_source: str = "first",
        pool_num_heads: int = 4,
        pool_dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        if residual_source not in {"first", "third"}:
            raise ValueError("residual_source must be 'first' or 'third'")
        self.residual_source = residual_source
        self.pool_num_heads = pool_num_heads
        self.pool_dropout = pool_dropout
        self.pool = AttentionPool(embed_dim, num_heads=pool_num_heads, dropout=pool_dropout)
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

    def pool_tracks(self, patches: torch.Tensor) -> torch.Tensor:
        return self.pool(patches)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if self.residual_source == "third":
            start = self.embed_dim * 2
        else:
            start = 0
        residual = context[:, start : start + self.embed_dim]
        delta = self.net(context)
        return F.normalize(residual + delta, dim=-1)


class PatchPlaylistHeadDataset(Dataset):
    """
    Variant of PlaylistHeadDataset that returns spectrogram tensors instead of
    pre-pooled embeddings. The training loop runs these through a frozen JEPA
    encoder to obtain patch tokens, which the patch head then pools.

    Continuation: returns (history_specs (max_history, 1, H, W), target_spec).
        Only target indices >= max_history are sampled, so history is always
        full length — slightly stricter than PlaylistHeadDataset.
    Infill: returns (left_spec, right_spec, alpha, target_spec).
    """

    def __init__(
        self,
        playlists_file: str,
        spectrograms_dir: str,
        img_size: Tuple[int, int] = (96, 216),
        max_history: int = 3,
        min_playlist_len: int = 2,
        task: str = "continuation",
        max_span: int = 32,
        max_playlist_len: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        if task not in {"continuation", "infill"}:
            raise ValueError("task must be 'continuation' or 'infill'")
        self.spectrograms_dir = spectrograms_dir
        self.task = task
        self.max_history = max_history
        self.max_span = max_span
        self.rng = np.random.default_rng(seed)

        available = set(
            f[:-4] for f in os.listdir(spectrograms_dir) if f.endswith(".png")
        )
        effective_min = max(min_playlist_len, 3 if task == "infill" else 2)
        self.playlists: List[List[str]] = []
        with open(playlists_file) as f:
            for line in f:
                tracks = [t.strip() for t in line.strip().split(",")]
                tracks = [t for t in tracks if t in available]
                if len(tracks) >= effective_min and (
                    max_playlist_len is None or len(tracks) <= max_playlist_len
                ):
                    self.playlists.append(tracks)
        if not self.playlists:
            raise ValueError(
                f"No playlists in {playlists_file} pass length filter "
                f"[{effective_min}, {max_playlist_len}]"
            )

        self.samples: List[Tuple[int, int]] = []
        for pl_idx, pl in enumerate(self.playlists):
            if task == "infill":
                for target_idx in range(1, len(pl) - 1):
                    self.samples.append((pl_idx, target_idx))
            else:
                start = max(1, max_history)
                for target_idx in range(start, len(pl)):
                    self.samples.append((pl_idx, target_idx))

        self.transform = T.Compose([
            T.Grayscale(),
            T.Resize(tuple(img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def _load_spec(self, track_id: str) -> torch.Tensor:
        path = os.path.join(self.spectrograms_dir, f"{track_id}.png")
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx: int):
        pl_idx, target_idx = self.samples[idx]
        playlist = self.playlists[pl_idx]
        target_spec = self._load_spec(playlist[target_idx])

        if self.task == "infill":
            left_min = max(0, target_idx - self.max_span)
            right_max = min(len(playlist) - 1, target_idx + self.max_span)
            left_idx = int(self.rng.integers(left_min, target_idx))
            right_idx = int(self.rng.integers(target_idx + 1, right_max + 1))
            alpha = (target_idx - left_idx) / (right_idx - left_idx)
            left_spec = self._load_spec(playlist[left_idx])
            right_spec = self._load_spec(playlist[right_idx])
            return left_spec, right_spec, torch.tensor(alpha, dtype=torch.float32), target_spec

        history_start = target_idx - self.max_history
        history_ids = playlist[history_start:target_idx]
        history_specs = torch.stack([self._load_spec(tid) for tid in history_ids])
        return history_specs, target_spec


def save_patch_head(path: str, head: "PatchPlaylistHead", cfg: dict) -> None:
    head = head.module if isinstance(head, nn.DataParallel) else head
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save(
        {
            "head_type": "patch",
            "state_dict": head.state_dict(),
            "embed_dim": head.embed_dim,
            "hidden_dim": cfg["model"]["hidden_dim"],
            "dropout": cfg["model"].get("dropout", 0.1),
            "residual_source": head.residual_source,
            "pool_num_heads": head.pool_num_heads,
            "pool_dropout": head.pool_dropout,
            "config": cfg,
        },
        tmp,
    )
    os.replace(tmp, path)
