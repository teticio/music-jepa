import os
import random
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def load_filtered_playlists(
    playlists_file: str,
    spectrograms_dir: str,
    min_playlist_len: int = 2,
) -> Tuple[List[List[str]], int, int]:
    """Read playlists CSV, drop tracks without a spectrogram, drop playlists
    shorter than `min_playlist_len`.

    Returns (playlists, raw_playlist_count, available_track_count).
    """
    available = set(
        f[:-4] for f in os.listdir(spectrograms_dir) if f.endswith(".png")
    )
    playlists: List[List[str]] = []
    raw_count = 0
    with open(playlists_file) as f:
        for line in f:
            raw_count += 1
            tracks = [t.strip() for t in line.strip().split(",")]
            tracks = [t for t in tracks if t in available]
            if len(tracks) >= min_playlist_len:
                playlists.append(tracks)
    return playlists, raw_count, len(available)


class PlaylistDataset(Dataset):
    """
    Each item is a (context_spec, target_spec, target_patch_ids) triple
    drawn from a consecutive pair of tracks in a playlist. Sampling is
    uniform across pairs, so long playlists contribute proportionally more
    pairs per epoch.

    Masking strategy: a contiguous block of time-aligned columns from the
    patch grid is selected as the prediction target, mimicking V-JEPA's
    temporal tube masking adapted to 2-D spectrograms.
    """

    def __init__(
        self,
        spectrograms_dir: str,
        playlists_file: Optional[str] = None,
        playlists: Optional[List[List[str]]] = None,
        img_size: Tuple[int, int] = (96, 216),
        patch_size: Tuple[int, int] = (8, 8),
        mask_ratio: float = 0.75,
        min_playlist_len: int = 2,
        augment: bool = True,
        raw_playlist_count: Optional[int] = None,
        available_track_count: Optional[int] = None,
    ):
        self.spectrograms_dir = spectrograms_dir
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_h = img_size[0] // patch_size[0]   # 12
        self.grid_w = img_size[1] // patch_size[1]   # 27
        self.num_patches = self.grid_h * self.grid_w  # 324
        self.mask_ratio = mask_ratio

        if playlists is not None:
            self.playlists = playlists
            self.raw_playlist_count = raw_playlist_count if raw_playlist_count is not None else len(playlists)
            self.available_track_count = available_track_count if available_track_count is not None else 0
        elif playlists_file is not None:
            self.playlists, self.raw_playlist_count, self.available_track_count = load_filtered_playlists(
                playlists_file, spectrograms_dir, min_playlist_len
            )
        else:
            raise ValueError("Pass either playlists_file or playlists")

        if not self.playlists:
            raise ValueError(
                f"No valid playlists found. Check that spectrograms exist in {spectrograms_dir}"
            )

        # Flat addressing of every consecutive pair across all playlists, so
        # each __getitem__ sees a uniformly-sampled pair rather than a
        # uniformly-sampled playlist. Long playlists were severely
        # under-sampled in the previous one-pair-per-playlist scheme.
        lengths = np.array([len(pl) - 1 for pl in self.playlists], dtype=np.int64)
        self._cumulative_pairs = np.cumsum(lengths)
        self._total_pairs = int(self._cumulative_pairs[-1])

        base_transforms = [
            T.Grayscale(),
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
        if augment:
            base_transforms = [
                T.Grayscale(),
                T.Resize((img_size[0] + 8, img_size[1] + 16)),
                T.RandomCrop(img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
            ]
        self.transform = T.Compose(base_transforms)

    def _load_spec(self, track_id: str) -> torch.Tensor:
        path = os.path.join(self.spectrograms_dir, f"{track_id}.png")
        img = Image.open(path).convert("RGB")
        return self.transform(img)  # (1, H, W)

    def _sample_target_patch_ids(self) -> torch.Tensor:
        """
        Sample a contiguous block of columns (time steps) as prediction targets.
        All frequency rows in those columns are masked, yielding mask_ratio coverage.
        """
        block_w = max(1, round(self.grid_w * self.mask_ratio))
        start_col = random.randint(0, self.grid_w - block_w)
        ids = [
            row * self.grid_w + col
            for row in range(self.grid_h)
            for col in range(start_col, start_col + block_w)
        ]
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        return self._total_pairs

    def __getitem__(self, idx: int):
        pl_idx = int(np.searchsorted(self._cumulative_pairs, idx, side="right"))
        prev = int(self._cumulative_pairs[pl_idx - 1]) if pl_idx > 0 else 0
        offset = idx - prev
        playlist = self.playlists[pl_idx]
        ctx_id, tgt_id = playlist[offset], playlist[offset + 1]

        ctx_spec = self._load_spec(ctx_id)
        tgt_spec = self._load_spec(tgt_id)
        target_patch_ids = self._sample_target_patch_ids()

        return ctx_spec, tgt_spec, target_patch_ids
