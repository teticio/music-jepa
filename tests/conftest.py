import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tiny_corpus(tmp_path):
    """5 tracks with synthetic spectrograms, 2 playlists.

    Playlist 0: tracks 0,1,2,3 (3 consecutive pairs)
    Playlist 1: tracks 1,3,4   (2 consecutive pairs)
    Total pairs: 5
    """
    spec_dir = tmp_path / "spectrograms"
    spec_dir.mkdir()
    track_ids = [f"track{i:03d}" for i in range(5)]
    rng = np.random.default_rng(0)
    for tid in track_ids:
        arr = (rng.random((96, 216)) * 255).astype("uint8")
        Image.fromarray(arr, mode="L").save(spec_dir / f"{tid}.png")
    playlists_path = tmp_path / "playlists.csv"
    playlists_path.write_text(
        f"{track_ids[0]},{track_ids[1]},{track_ids[2]},{track_ids[3]}\n"
        f"{track_ids[1]},{track_ids[3]},{track_ids[4]}\n"
    )
    return {
        "spec_dir": str(spec_dir),
        "playlists_path": str(playlists_path),
        "track_ids": track_ids,
    }
