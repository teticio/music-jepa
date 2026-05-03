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
import json
import pickle
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


def normalize_vecs(vecs: np.ndarray) -> np.ndarray:
    return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8)


def load_mp3tovec(model_dir: str):
    model_path = Path(model_dir)
    with (model_path / "spotifytovec.p").open("rb") as f:
        data = pickle.load(f)
    ids = list(data.keys())
    vecs = normalize_vecs(np.stack([data[k] for k in ids]).astype("float32"))
    metadata = {}
    tracks_path = model_path / "spotify_tracks.p"
    urls_path = model_path / "spotify_urls.p"
    if tracks_path.exists():
        with tracks_path.open("rb") as f:
            metadata["tracks"] = pickle.load(f)
    if urls_path.exists():
        with urls_path.open("rb") as f:
            metadata["urls"] = pickle.load(f)
    return ids, vecs, metadata


def describe_generated_track(track_id: str, tracks_df, metadata: Optional[dict] = None) -> str:
    if tracks_df is not None and track_id in tracks_df.index:
        return describe_track(track_id, tracks_df)
    track_name = (metadata or {}).get("tracks", {}).get(track_id)
    if track_name:
        return f"{track_name} [{track_id}]"
    return track_id


def get_generated_track_info(track_id: str, tracks_df, metadata: Optional[dict] = None):
    if tracks_df is not None and track_id in tracks_df.index:
        row = tracks_df.loc[track_id]
        artist = str(row["artist"])
        title = str(row["title"])
        url = row["url"] if isinstance(row["url"], str) else None
        return artist, title, url

    metadata = metadata or {}
    track_name = metadata.get("tracks", {}).get(track_id)
    url = metadata.get("urls", {}).get(track_id)
    url = url if isinstance(url, str) else None
    if track_name:
        artist, sep, title = track_name.partition(" - ")
        return artist, title if sep else track_name, url
    return "Unknown artist", track_id, url


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
    head_weight: float,
    device: str,
) -> List[str]:
    playlist = list(seeds)
    head_weight = min(max(head_weight, 0.0), 1.0)
    while len(playlist) < size:
        pred = predict_next(head, playlist, emb, max_history, device)
        if head_weight < 1.0:
            recent = playlist[-max_history:]
            anchor = np.stack([emb[tid] for tid in recent]).mean(axis=0)
            query = head_weight * pred + (1 - head_weight) * anchor
        else:
            query = pred
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


def generate_embedding_continuation(
    seeds: List[str],
    emb,
    ids,
    vecs,
    size: int,
    max_history: int,
    noise: float,
) -> List[str]:
    playlist = list(seeds)
    while len(playlist) < size:
        recent = playlist[-max_history:]
        query = np.stack([emb[tid] for tid in recent]).mean(axis=0)
        candidates = rank_tracks(query, ids, vecs, exclude=playlist, top_k=100)
        playlist.append(choose(candidates, noise)[0])
    return playlist


def generate_embedding_journey(
    waypoints: List[str],
    emb,
    ids,
    vecs,
    between: int,
    noise: float,
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
            query = (1 - alpha) * start_vec + alpha * end_vec
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


def build_html_str(
    playlist: List[str],
    urls: List[Optional[str]],
    tracks_df,
    highlighted: set[str],
    page_title: str,
    metadata: Optional[dict] = None,
) -> str:
    rows = []
    playable_previews = []
    for i, (tid, url) in enumerate(zip(playlist, urls), 1):
        artist, title, _ = get_generated_track_info(tid, tracks_df, metadata)
        marker = " seed" if tid in highlighted else ""
        badge = "<span class=\"badge\">seed</span>" if tid in highlighted else ""
        row_id = f"track-{i}"
        preview_index = None
        if url:
            preview_index = len(playable_previews)
            playable_previews.append({"url": url, "rowId": row_id})
        preview_attr = f" data-preview-index=\"{preview_index}\"" if preview_index is not None else ""
        preview_badge = "" if url else "<span class=\"missing\">no preview</span>"
        rows.append(
            "\n".join(
                [
                    f"<tr id=\"{row_id}\" class=\"track{marker}\"{preview_attr}>",
                    f"  <td class=\"idx\">{i:02d}</td>",
                    f"  <td><div class=\"title\">{escape(title)} {badge}</div>"
                    f"<div class=\"artist\">{escape(artist)}</div>"
                    f"<code>{escape(tid)}</code>{preview_badge}</td>",
                    "</tr>",
                ]
            )
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(page_title)}</title>
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
    .sticky-header {{
      position: sticky;
      top: 0;
      z-index: 10;
      margin: -32px 0 18px;
      padding: 32px 0 14px;
      background: #101114;
      border-bottom: 1px solid #2b2f38;
    }}
    .toolbar {{
      display: flex;
      align-items: center;
      gap: 12px;
      margin: 0;
    }}
    button {{
      border: 1px solid #3a4150;
      background: #eef2ff;
      color: #111827;
      border-radius: 6px;
      padding: 8px 12px;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }}
    button:disabled {{
      cursor: not-allowed;
      opacity: 0.55;
    }}
    .status {{
      color: #b9bec8;
      font-size: 14px;
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
    tr[data-preview-index] {{
      cursor: pointer;
    }}
    tr[data-preview-index]:hover {{
      background: #222733;
    }}
    tr.playing {{
      background: #263449;
      box-shadow: inset 4px 0 0 #8fb3ff;
    }}
    tr.playing .idx,
    tr.playing .title {{
      color: #ffffff;
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
    @media (max-width: 720px) {{
      body {{
        padding: 18px;
      }}
      .sticky-header {{
        margin: -18px 0 14px;
        padding: 18px 0 12px;
      }}
      .toolbar {{
        flex-wrap: wrap;
      }}
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
      display: inline-block;
      margin-top: 8px;
      color: #8f96a3;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <main>
    <div class="sticky-header">
      <h1>{escape(page_title)}</h1>
      <div class="toolbar">
        <button id="play-all" type="button">Play all previews</button>
        <button id="previous-preview" type="button" disabled>Previous</button>
        <button id="next-preview" type="button" disabled>Next</button>
        <button id="stop-all" type="button" disabled>Stop</button>
        <span class="status" id="play-status">{len(playable_previews)} previews available</span>
      </div>
    </div>
    <table>
      <thead>
        <tr><th>#</th><th>Track</th></tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </main>
  <script>
    const previewItems = {json.dumps(playable_previews)};
    const playButton = document.getElementById('play-all');
    const previousButton = document.getElementById('previous-preview');
    const nextButton = document.getElementById('next-preview');
    const stopButton = document.getElementById('stop-all');
    const statusEl = document.getElementById('play-status');
    let currentAudio = null;
    let currentIndex = -1;

    function setStatus(text) {{
      statusEl.textContent = text;
    }}

    function clearPlaying() {{
      document.querySelectorAll('tr.playing').forEach((row) => {{
        row.classList.remove('playing');
      }});
    }}

    function setPlaying(index) {{
      clearPlaying();
      currentIndex = index;
      const item = previewItems[index];
      if (!item) return;
      const rowId = item.rowId;
      const row = document.getElementById(rowId);
      if (!row) return;
      row.classList.add('playing');
      row.scrollIntoView({{ block: 'nearest', behavior: 'smooth' }});
    }}

    function updateButtons() {{
      const hasPreviews = previewItems.length > 0;
      playButton.disabled = !hasPreviews;
      previousButton.disabled = currentIndex <= 0;
      nextButton.disabled = currentIndex < 0 || currentIndex >= previewItems.length - 1;
      stopButton.disabled = currentIndex < 0;
    }}

    function stopCurrent(resetStatus = true) {{
      if (currentAudio) {{
        currentAudio.pause();
        currentAudio.currentTime = 0;
        currentAudio = null;
      }}
      currentIndex = -1;
      clearPlaying();
      updateButtons();
      if (resetStatus) setStatus(`${{previewItems.length}} previews available`);
    }}

    function playIndex(index) {{
      const item = previewItems[index];
      if (!item) return;
      if (currentAudio) {{
        currentAudio.pause();
        currentAudio = null;
      }}
      setPlaying(index);
      updateButtons();
      setStatus(`Playing preview ${{index + 1}} of ${{previewItems.length}}`);
      currentAudio = new Audio(item.url);
      currentAudio.addEventListener('ended', () => {{
        if (currentIndex < previewItems.length - 1) {{
          playIndex(currentIndex + 1);
        }} else {{
          stopCurrent(false);
          setStatus('Finished');
        }}
      }}, {{ once: true }});
      currentAudio.addEventListener('error', () => {{
        if (currentIndex < previewItems.length - 1) {{
          playIndex(currentIndex + 1);
        }} else {{
          stopCurrent(false);
          setStatus('Finished');
        }}
      }}, {{ once: true }});
      currentAudio.play().catch(() => {{
        stopCurrent(false);
        setStatus('Playback failed');
      }});
    }}

    function nextPreview() {{
      if (currentIndex < previewItems.length - 1) playIndex(currentIndex + 1);
    }}

    function previousPreview() {{
      if (currentIndex > 0) playIndex(currentIndex - 1);
    }}

    function playAll() {{
      if (previewItems.length) playIndex(0);
    }}

    document.querySelectorAll('tr[data-preview-index]').forEach((row) => {{
      row.addEventListener('click', () => {{
        const index = Number(row.dataset.previewIndex);
        if (index === currentIndex && currentAudio) {{
          stopCurrent();
        }} else {{
          playIndex(index);
        }}
      }});
    }});
    playButton.addEventListener('click', playAll);
    previousButton.addEventListener('click', previousPreview);
    nextButton.addEventListener('click', nextPreview);
    stopButton.addEventListener('click', stopCurrent);
    updateButtons();
  </script>
</body>
</html>
"""
    return html


def write_html(
    path: str,
    playlist: List[str],
    urls: List[Optional[str]],
    tracks_df,
    highlighted: set[str],
    page_title: str,
    metadata: Optional[dict] = None,
) -> None:
    html = build_html_str(playlist, urls, tracks_df, highlighted, page_title, metadata)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", default=None)
    parser.add_argument("--embeddings", default="embeddings.npy")
    parser.add_argument("--mp3tovec_model_dir", default=None, help="If set, use Mp3ToVec instead of the head")
    parser.add_argument("--tracks_file", default="data/tracks_dedup.csv")
    parser.add_argument("--seeds", nargs="*", default=None)
    parser.add_argument("--journey", nargs="+", default=None, help="Two or more waypoint track IDs")
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--between", type=int, default=9)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument(
        "--head_weight",
        type=float,
        default=1.0,
        help="0=geometry, 1=head prediction. Continuation: blends head vs recent mean. Journey: blends interpolation vs head.",
    )
    parser.add_argument("--out_m3u", default=None)
    parser.add_argument("--out_html", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.seeds and not args.journey:
        raise SystemExit("Pass --seeds for continuation or --journey for waypoint generation")

    random.seed(42)
    np.random.seed(42)

    tracks_df = load_tracks(args.tracks_file)
    metadata = {}
    mp3tovec = args.mp3tovec_model_dir is not None
    if mp3tovec:
        ids, vecs, metadata = load_mp3tovec(args.mp3tovec_model_dir)
    else:
        if not args.head:
            raise SystemExit("Pass --head (or --mp3tovec_model_dir for Mp3ToVec mode)")
        ids, vecs = load_embeddings(args.embeddings)
    emb = {tid: vec for tid, vec in zip(ids, vecs)}
    head = None
    task = "continuation"
    max_history = 3
    if not mp3tovec:
        head, cfg = load_head(args.head, device=args.device)
        task = cfg.get("data", {}).get("task", "continuation")
        max_history = cfg.get("data", {}).get("max_history", 3)

    requested = args.journey if args.journey else args.seeds
    missing = [tid for tid in requested if tid not in emb]
    if missing:
        raise SystemExit(f"Missing embeddings for: {', '.join(missing)}")

    if args.journey:
        if mp3tovec:
            playlist = generate_embedding_journey(
                args.journey, emb, ids, vecs,
                between=args.between, noise=args.noise,
            )
        elif task == "infill":
            playlist = generate_infill_journey(
                head, args.journey, emb, ids, vecs,
                between=args.between, noise=args.noise, device=args.device,
            )
        else:
            playlist = generate_journey(
                head, args.journey, emb, ids, vecs,
                between=args.between, max_history=max_history,
                noise=args.noise, head_weight=args.head_weight, device=args.device,
            )
    else:
        if mp3tovec:
            playlist = generate_embedding_continuation(
                args.seeds, emb, ids, vecs,
                size=args.size, max_history=max_history, noise=args.noise,
            )
        elif task == "infill":
            raise SystemExit("This checkpoint is trained for --journey infilling. Use a continuation head for --seeds.")
        else:
            playlist = generate_continuation(
                head, args.seeds, emb, ids, vecs,
                size=args.size, max_history=max_history,
                noise=args.noise, head_weight=args.head_weight, device=args.device,
            )

    urls = []
    for i, tid in enumerate(playlist, 1):
        marker = "*" if (args.journey and tid in set(args.journey)) or (args.seeds and tid in args.seeds) else " "
        _, _, url = get_generated_track_info(tid, tracks_df, metadata)
        urls.append(url)
        suffix = f"  {url}" if url else ""
        print(f"{i:02d}.{marker} {describe_generated_track(tid, tracks_df, metadata)}{suffix}")

    if args.out_m3u:
        write_m3u(args.out_m3u, playlist, urls)
        print(f"wrote {args.out_m3u}")

    if args.out_html:
        highlighted = set(args.journey or args.seeds or [])
        default_title = "Generated journey" if args.journey else "Generated playlist"
        write_html(
            args.out_html,
            playlist,
            urls,
            tracks_df,
            highlighted,
            args.title or default_title,
            metadata,
        )
        print(f"wrote {args.out_html}")


if __name__ == "__main__":
    main()
