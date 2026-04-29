"""
Sample a small subset of playlists and their tracks for the POC.

Writes:
  data/playlists_sample.csv  -- filtered playlists (only tracks with preview URLs)
  data/tracks_sample.csv     -- tracks that appear in sampled playlists

Usage:
    python data/sample_data.py --n_playlists 2000 --min_track_count 10
"""
import argparse
import random

import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--playlists_file", default="data/playlists_dedup.csv")
    parser.add_argument("--tracks_file", default="data/tracks_dedup.csv")
    parser.add_argument("--out_playlists", default="data/playlists_sample.csv")
    parser.add_argument("--out_tracks", default="data/tracks_sample.csv")
    parser.add_argument("--n_playlists", type=int, default=2000, help="Number of playlists to sample")
    parser.add_argument("--min_tracks_in_playlist", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading tracks...")
    tracks_df = pd.read_csv(
        args.tracks_file,
        header=None,
        index_col=0,
        names=["artist", "title", "url", "count"],
    )
    # Only keep tracks with valid preview URLs
    has_url = tracks_df["url"].notna() & (tracks_df["url"] != "nan")
    valid_tracks = set(tracks_df[has_url].index.astype(str))
    print(f"Tracks with preview URLs: {len(valid_tracks):,}")

    print("Loading playlists...")
    all_playlists = []
    with open(args.playlists_file) as f:
        for line in tqdm(f, desc="Reading playlists"):
            tracks = [t.strip() for t in line.strip().split(",")]
            tracks = [t for t in tracks if t in valid_tracks]
            if len(tracks) >= args.min_tracks_in_playlist:
                all_playlists.append(tracks)

    print(f"Valid playlists: {len(all_playlists):,}")

    n = min(args.n_playlists, len(all_playlists))
    sampled = random.sample(all_playlists, n)
    print(f"Sampled {n:,} playlists")

    # Collect unique tracks
    unique_tracks = set()
    for pl in sampled:
        unique_tracks.update(pl)
    print(f"Unique tracks in sample: {len(unique_tracks):,}")

    # Write sampled playlists
    with open(args.out_playlists, "w") as f:
        for pl in sampled:
            f.write(",".join(pl) + "\n")
    print(f"Written: {args.out_playlists}")

    # Write sampled tracks (preserve original CSV format)
    tracks_sample = tracks_df[tracks_df.index.astype(str).isin(unique_tracks)]
    tracks_sample.to_csv(args.out_tracks, header=False)
    print(f"Written: {args.out_tracks}  ({len(tracks_sample):,} tracks)")


if __name__ == "__main__":
    main()
