"""
Download 30-second Spotify MP3 preview clips for a set of tracks.

Usage:
    python data/download_previews.py
    python data/download_previews.py --tracks_file data/tracks_sample.csv --max_workers 16
"""
import argparse
import concurrent.futures
import os
from time import sleep

import pandas as pd
import requests
from tqdm import tqdm


def download_file(track_id: str, track_url: str, previews_dir: str) -> None:
    for _ in range(3):
        try:
            response = requests.get(track_url, stream=True, timeout=15)
            if response.status_code == 200:
                with open(os.path.join(previews_dir, f"{track_id}.mp3"), "wb") as f:
                    f.write(response.content)
                return
        except requests.RequestException:
            pass
    print(f"  Skipping {track_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks_file", default="data/tracks_sample.csv")
    parser.add_argument("--previews_dir", default="data/previews")
    parser.add_argument("--max_workers", type=int, default=min(32, (os.cpu_count() or 1) * 4))
    args = parser.parse_args()

    track_urls = pd.read_csv(
        args.tracks_file,
        header=None,
        index_col=0,
        names=["artist", "title", "url", "count"],
    )["url"].dropna().to_dict()

    os.makedirs(args.previews_dir, exist_ok=True)
    already_done = set(os.listdir(args.previews_dir))

    todo = {
        tid: url
        for tid, url in track_urls.items()
        if f"{tid}.mp3" not in already_done
        and isinstance(url, str)
        and url.startswith("http")
    }
    print(f"Tracks to download: {len(todo):,}  (already done: {len(already_done):,})")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(download_file, tid, url, args.previews_dir): tid
            for tid, url in tqdm(todo.items(), desc="Submitting jobs")
            if sleep(1e-5) is None
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading"):
            tid = futures[future]
            try:
                future.result()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"  Error on {tid}: {e}")


if __name__ == "__main__":
    main()
