import os
import random
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class PlaylistDataset(Dataset):
    """
    Each item is a (context_spec, target_spec, target_patch_ids) triple
    sampled from a consecutive pair of tracks in a playlist.

    Masking strategy: a contiguous block of time-aligned columns from the
    patch grid is selected as the prediction target, mimicking V-JEPA's
    temporal tube masking adapted to 2-D spectrograms.
    """

    def __init__(
        self,
        playlists_file: str,
        spectrograms_dir: str,
        img_size: Tuple[int, int] = (96, 216),
        patch_size: Tuple[int, int] = (8, 8),
        mask_ratio: float = 0.75,
        min_playlist_len: int = 2,
        augment: bool = True,
    ):
        self.spectrograms_dir = spectrograms_dir
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_h = img_size[0] // patch_size[0]   # 12
        self.grid_w = img_size[1] // patch_size[1]   # 27
        self.num_patches = self.grid_h * self.grid_w  # 324
        self.mask_ratio = mask_ratio

        available = set(
            f[:-4] for f in os.listdir(spectrograms_dir) if f.endswith(".png")
        )

        self.playlists: List[List[str]] = []
        with open(playlists_file) as f:
            for line in f:
                tracks = [t.strip() for t in line.strip().split(",")]
                tracks = [t for t in tracks if t in available]
                if len(tracks) >= min_playlist_len:
                    self.playlists.append(tracks)

        if not self.playlists:
            raise ValueError(
                f"No valid playlists found. Check that spectrograms exist in {spectrograms_dir}"
            )

        base_transforms = [
            T.Grayscale(),
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
        if augment:
            # Frequency / time jitter: small random crops then resize back
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
        return len(self.playlists)

    def __getitem__(self, idx: int):
        playlist = self.playlists[idx]
        i = random.randint(0, len(playlist) - 2)
        ctx_id, tgt_id = playlist[i], playlist[i + 1]

        ctx_spec = self._load_spec(ctx_id)
        tgt_spec = self._load_spec(tgt_id)
        target_patch_ids = self._sample_target_patch_ids()

        return ctx_spec, tgt_spec, target_patch_ids
