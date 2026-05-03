import numpy as np

from jepa.dataset import PlaylistDataset, load_filtered_playlists
from jepa.module import build_dataloaders


def test_load_filtered_playlists(tiny_corpus):
    playlists, raw, available = load_filtered_playlists(
        tiny_corpus["playlists_path"], tiny_corpus["spec_dir"]
    )
    assert len(playlists) == 2
    assert raw == 2
    assert available == 5


def test_dataset_length_is_total_pairs(tiny_corpus):
    """__len__ should equal total consecutive pairs across playlists.

    Playlist 0: 4 tracks → 3 pairs. Playlist 1: 3 tracks → 2 pairs. Total: 5.
    """
    ds = PlaylistDataset(
        spectrograms_dir=tiny_corpus["spec_dir"],
        playlists_file=tiny_corpus["playlists_path"],
        augment=False,
    )
    assert len(ds) == 5


def test_dataset_indexing_covers_every_pair(tiny_corpus):
    """Iterating idx 0..N-1 should hit every (playlist, position) exactly once."""
    ds = PlaylistDataset(
        spectrograms_dir=tiny_corpus["spec_dir"],
        playlists_file=tiny_corpus["playlists_path"],
        augment=False,
    )
    seen = set()
    for i in range(len(ds)):
        pl_idx = int(np.searchsorted(ds._cumulative_pairs, i, side="right"))
        prev = int(ds._cumulative_pairs[pl_idx - 1]) if pl_idx > 0 else 0
        offset = i - prev
        seen.add((pl_idx, offset))
    expected = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)}
    assert seen == expected


def test_dataset_returns_correct_shapes(tiny_corpus):
    ds = PlaylistDataset(
        spectrograms_dir=tiny_corpus["spec_dir"],
        playlists_file=tiny_corpus["playlists_path"],
        augment=False,
    )
    ctx, tgt, target_patch_ids = ds[0]
    assert ctx.shape == (1, 96, 216)
    assert tgt.shape == (1, 96, 216)
    # 27 cols × 0.75 mask ratio → block_w=20, ids = 12 × 20 = 240
    assert target_patch_ids.shape == (240,)


def test_build_dataloaders_split_at_playlist_level(tiny_corpus):
    """Train/val split must happen at playlist level so consecutive pairs from
    one playlist do not split across train and val (would leak signal).
    """
    cfg = {
        "seed": 0,
        "data": {
            "playlists_file": tiny_corpus["playlists_path"],
            "spectrograms_dir": tiny_corpus["spec_dir"],
            "img_size": [96, 216],
            "patch_size": [8, 8],
            "mask_ratio": 0.75,
            "batch_size": 1,
            "num_workers": 0,
            "augment": False,
        },
    }
    train_loader, val_loader = build_dataloaders(cfg, val_fraction=0.5)
    train_pls = {tuple(pl) for pl in train_loader.dataset.playlists}
    val_pls = {tuple(pl) for pl in val_loader.dataset.playlists}
    assert train_pls.isdisjoint(val_pls)
    assert len(train_pls) >= 1 and len(val_pls) >= 1


def test_build_dataloaders_val_does_not_augment(tiny_corpus):
    cfg = {
        "seed": 0,
        "data": {
            "playlists_file": tiny_corpus["playlists_path"],
            "spectrograms_dir": tiny_corpus["spec_dir"],
            "img_size": [96, 216],
            "patch_size": [8, 8],
            "mask_ratio": 0.75,
            "batch_size": 1,
            "num_workers": 0,
            "augment": True,
        },
    }
    _, val_loader = build_dataloaders(cfg, val_fraction=0.5)
    # If augment were True, the transform would include RandomCrop. The val
    # dataset should have only deterministic transforms.
    transform_names = [t.__class__.__name__ for t in val_loader.dataset.transform.transforms]
    assert "RandomCrop" not in transform_names
