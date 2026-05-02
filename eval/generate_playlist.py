"""
Generate playlists from a trained playlist head and JEPA embeddings.

Continuation:
    uv run python eval/generate_playlist.py --head checkpoints/playlist_head.pt \
        --seeds TRACK_ID [TRACK_ID ...] --size 20

Journey / "join the dots":
    uv run python eval/generate_playlist.py --head checkpoints/playlist_head.pt \
        --journey START_ID END_ID --between 9
"""
import argparse
from html import escape
from pathlib import Path
import random
from typing import List, Optional

import numpy as np
import torch

from jepa.playlist_head import (
    describe_track,
    load_embeddings,
    load_head,
    load_tracks,
    make_infill_context,
    make_context,
    rank_tracks,
)


def choose(candidates, noise: float):
    if noise <= 0:
        return candidates[0]
    k = min(len(candidates), max(2, int(1 + noise * 25)))
    weights = np.array([score for _, score in candidates[:k]], dtype="float64")
    weights = np.exp((weights - weights.max()) / max(noise, 1e-6))
    weights = weights / weights.sum()
    return candidates[int(np.random.choice(k, p=weights))]


@torch.no_grad()
def predict_next(head, history, emb, max_history, device):
    context = make_context(history, emb, max_history)
    context_t = torch.from_numpy(context).unsqueeze(0).to(device)
    return head(context_t).squeeze(0).cpu().numpy()


@torch.no_grad()
def predict_infill(head, left_id, right_id, alpha, emb, device):
    context = make_infill_context(left_id, right_id, alpha, emb)
    context_t = torch.from_numpy(context).unsqueeze(0).to(device)
    return head(context_t).squeeze(0).cpu().numpy()


def generate_continuation(
    head,
    seeds: List[str],
    emb,
    ids,
    vecs,
    size: int,
    max_history: int,
    noise: float,
    device: str,
) -> List[str]:
    playlist = list(seeds)
    while len(playlist) < size:
        query = predict_next(head, playlist, emb, max_history, device)
        candidates = rank_tracks(query, ids, vecs, exclude=playlist, top_k=100)
        playlist.append(choose(candidates, noise)[0])
    return playlist


def fill_segment(
    head,
    start: str,
    end: str,
    between: int,
    emb,
    ids,
    vecs,
    used,
    noise: float,
    device: str,
) -> List[str]:
    slots: List[Optional[str]] = [None] * (between + 2)
    slots[0] = start
    slots[-1] = end
    gaps = [(0, len(slots) - 1)]

    while gaps:
        left_pos, right_pos = max(gaps, key=lambda gap: gap[1] - gap[0])
        gaps.remove((left_pos, right_pos))
        if right_pos - left_pos <= 1:
            continue

        mid_pos = (left_pos + right_pos) // 2
        alpha = (mid_pos - left_pos) / (right_pos - left_pos)
        query = predict_infill(
            head,
            slots[left_pos],
            slots[right_pos],
            alpha,
            emb,
            device,
        )
        candidates = rank_tracks(query, ids, vecs, exclude=used | {end}, top_k=100)
        tid, _ = choose(candidates, noise)
        slots[mid_pos] = tid
        used.add(tid)
        gaps.append((left_pos, mid_pos))
        gaps.append((mid_pos, right_pos))

    return [tid for tid in slots if tid is not None]


def generate_infill_journey(
    head,
    waypoints: List[str],
    emb,
    ids,
    vecs,
    between: int,
    noise: float,
    device: str,
) -> List[str]:
    playlist = []
    used = set(waypoints)
    for start, end in zip(waypoints, waypoints[1:]):
        segment = fill_segment(
            head,
            start,
            end,
            between,
            emb,
            ids,
            vecs,
            used,
            noise,
            device,
        )
        if playlist:
            segment = segment[1:]
        playlist.extend(segment)
    return playlist


def generate_journey(
    head,
    waypoints: List[str],
    emb,
    ids,
    vecs,
    between: int,
    max_history: int,
    noise: float,
    head_weight: float,
    device: str,
) -> List[str]:
    playlist = []
    used = set()
    for start, end in zip(waypoints, waypoints[1:]):
        if not playlist:
            playlist.append(start)
            used.add(start)
        start_vec = emb[start]
        end_vec = emb[end]
        for i in range(between):
            alpha = (i + 1) / (between + 1)
            interp = (1 - alpha) * start_vec + alpha * end_vec
            pred = predict_next(head, playlist, emb, max_history, device)
            query = (1 - head_weight) * interp + head_weight * pred
            candidates = rank_tracks(query, ids, vecs, exclude=used | {end}, top_k=100)
            tid, _ = choose(candidates, noise)
            playlist.append(tid)
            used.add(tid)
        if end not in used:
            playlist.append(end)
            used.add(end)
    return playlist


def write_m3u(path: str, playlist: List[str], urls: List[Optional[str]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("#EXTM3U\n")
        for tid, url in zip(playlist, urls):
            f.write(f"#EXTINF:-1,{tid}\n")
            f.write(f"{url or tid}\n")


def write_html(
    path: str,
    playlist: List[str],
    urls: List[Optional[str]],
    tracks_df,
    highlighted: set[str],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (tid, url) in enumerate(zip(playlist, urls), 1):
        if tid in tracks_df.index:
            row = tracks_df.loc[tid]
            artist = str(row["artist"])
            title = str(row["title"])
        else:
            artist = "Unknown artist"
            title = tid
        marker = " seed" if tid in highlighted else ""
        badge = "<span class=\"badge\">seed</span>" if tid in highlighted else ""
        player = (
            f"<audio controls preload=\"none\" src=\"{escape(url)}\"></audio>"
            if url
            else "<span class=\"missing\">no preview</span>"
        )
        rows.append(
            "\n".join(
                [
                    f"<tr class=\"track{marker}\">",
                    f"  <td class=\"idx\">{i:02d}</td>",
                    f"  <td><div class=\"title\">{escape(title)} {badge}</div>"
                    f"<div class=\"artist\">{escape(artist)}</div>"
                    f"<code>{escape(tid)}</code></td>",
                    f"  <td>{player}</td>",
                    "</tr>",
                ]
            )
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Generated playlist</title>
  <style>
    :root {{
      color-scheme: light dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #101114;
      color: #f2f2f0;
    }}
    body {{
      margin: 0;
      padding: 32px;
      background: #101114;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 20px;
      font-size: 28px;
      font-weight: 700;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #181a1f;
      border: 1px solid #2b2f38;
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid #2b2f38;
      text-align: left;
      vertical-align: middle;
    }}
    th {{
      color: #b9bec8;
      font-size: 12px;
      letter-spacing: 0;
      text-transform: uppercase;
      background: #20232a;
    }}
    tr.seed {{
      background: #20281f;
    }}
    .idx {{
      width: 48px;
      color: #b9bec8;
      font-variant-numeric: tabular-nums;
    }}
    .title {{
      font-weight: 650;
      line-height: 1.35;
    }}
    .artist {{
      color: #b9bec8;
      margin-top: 2px;
    }}
    code {{
      display: inline-block;
      margin-top: 6px;
      color: #8fb3ff;
      font-size: 12px;
    }}
    audio {{
      width: 260px;
      max-width: 100%;
    }}
    .badge {{
      display: inline-block;
      margin-left: 8px;
      padding: 2px 6px;
      border-radius: 4px;
      background: #345c2a;
      color: #dcffd4;
      font-size: 11px;
      font-weight: 700;
      vertical-align: 1px;
    }}
    .missing {{
      color: #8f96a3;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Generated playlist</h1>
    <table>
      <thead>
        <tr><th>#</th><th>Track</th><th>Preview</th></tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </main>
</body>
</html>
"""
    Path(path).write_text(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", required=True)
    parser.add_argument("--embeddings", default="embeddings.npy")
    parser.add_argument("--tracks_file", default="data/tracks_sample.csv")
    parser.add_argument("--seeds", nargs="*", default=None)
    parser.add_argument("--journey", nargs="+", default=None, help="Two or more waypoint track IDs")
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--between", type=int, default=9)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--head_weight", type=float, default=0.35)
    parser.add_argument("--out_m3u", default=None)
    parser.add_argument("--out_html", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.seeds and not args.journey:
        raise SystemExit("Pass --seeds for continuation or --journey for waypoint generation")

    random.seed(42)
    np.random.seed(42)

    ids, vecs = load_embeddings(args.embeddings)
    emb = {tid: vec for tid, vec in zip(ids, vecs)}
    tracks_df = load_tracks(args.tracks_file)
    head, cfg = load_head(args.head, device=args.device)
    task = cfg.get("data", {}).get("task", "continuation")
    max_history = cfg.get("data", {}).get("max_history", 3)

    requested = args.journey if args.journey else args.seeds
    missing = [tid for tid in requested if tid not in emb]
    if missing:
        raise SystemExit(f"Missing embeddings for: {', '.join(missing)}")

    if args.journey:
        if task == "infill":
            playlist = generate_infill_journey(
                head,
                args.journey,
                emb,
                ids,
                vecs,
                between=args.between,
                noise=args.noise,
                device=args.device,
            )
        else:
            playlist = generate_journey(
                head,
                args.journey,
                emb,
                ids,
                vecs,
                between=args.between,
                max_history=max_history,
                noise=args.noise,
                head_weight=args.head_weight,
                device=args.device,
            )
    else:
        if task == "infill":
            raise SystemExit("This checkpoint is trained for --journey infilling. Use a continuation head for --seeds.")
        playlist = generate_continuation(
            head,
            args.seeds,
            emb,
            ids,
            vecs,
            size=args.size,
            max_history=max_history,
            noise=args.noise,
            device=args.device,
        )

    urls = []
    for i, tid in enumerate(playlist, 1):
        marker = "*" if (args.journey and tid in set(args.journey)) or (args.seeds and tid in args.seeds) else " "
        url = tracks_df.loc[tid, "url"] if tid in tracks_df.index else None
        url = url if isinstance(url, str) else None
        urls.append(url)
        suffix = f"  {url}" if url else ""
        print(f"{i:02d}.{marker} {describe_track(tid, tracks_df)}{suffix}")

    if args.out_m3u:
        write_m3u(args.out_m3u, playlist, urls)
        print(f"wrote {args.out_m3u}")

    if args.out_html:
        highlighted = set(args.journey or args.seeds or [])
        write_html(args.out_html, playlist, urls, tracks_df, highlighted)
        print(f"wrote {args.out_html}")


if __name__ == "__main__":
    main()
