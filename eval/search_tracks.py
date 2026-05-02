"""
Search local track metadata and print Spotify track IDs.

Example:
    uv run python eval/search_tracks.py --query "daft punk"
"""
import argparse
from pathlib import Path

import numpy as np

from jepa.playlist_head import load_tracks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Search text, matched against artist/title/id")
    parser.add_argument("--tracks_file", default="data/tracks_sample.csv")
    parser.add_argument("--embeddings", default="embeddings.npy")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    tracks = load_tracks(args.tracks_file)
    query = args.query.strip().lower()
    terms = [term for term in query.split() if term]
    if not terms:
        raise SystemExit("Pass a non-empty --query")

    embedded_ids = None
    if args.embeddings and Path(args.embeddings).exists():
        embedded_ids = set(np.load(args.embeddings, allow_pickle=True).item().keys())

    rows = []
    for track_id, row in tracks.iterrows():
        haystack = f"{row['artist']} {row['title']} {track_id}".lower()
        if not all(term in haystack for term in terms):
            continue
        exact = query in haystack
        count = int(row["count"]) if "count" in row and not np.isnan(row["count"]) else 0
        has_embedding = embedded_ids is None or track_id in embedded_ids
        rows.append((exact, has_embedding, count, track_id, row))

    rows.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)

    if not rows:
        print(f"No tracks found for: {args.query}")
        return

    for i, (_, has_embedding, count, track_id, row) in enumerate(rows[: args.limit], 1):
        marker = "emb" if has_embedding else "no-emb"
        url = row["url"] if isinstance(row["url"], str) else ""
        print(f"{i:02d}. {track_id}  [{marker}, count={count}]")
        print(f"    {row['artist']} - {row['title']}")
        if url:
            print(f"    {url}")


if __name__ == "__main__":
    main()
