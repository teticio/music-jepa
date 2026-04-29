"""
Interactive Music JEPA embedding explorer.

- 2D t-SNE scatter plot in your browser
- Hover: shows artist and title
- Click: plays the 30-second MP3 preview

Usage:
    python eval/explore.py --embeddings embeddings.npy
    python eval/explore.py --embeddings embeddings.npy --n_points 5000 --port 8765
"""
import argparse
import http.server
import os
import threading
import webbrowser

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from bokeh.models import ColumnDataSource, HoverTool, TapTool, CustomJS
from bokeh.plotting import figure, save
from bokeh.resources import CDN
from bokeh.embed import file_html


def serve_files(root: str, port: int):
    """Serve repo root over HTTP so the browser can load local MP3s."""
    handler = http.server.SimpleHTTPRequestHandler
    handler.log_message = lambda *a: None  # silence request logs
    httpd = http.server.HTTPServer(("localhost", port), handler)
    os.chdir(root)
    httpd.serve_forever()


def load_embeddings(path: str):
    data = np.load(path, allow_pickle=True).item()
    ids = list(data.keys())
    vecs = np.stack([data[k] for k in ids])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-8)
    return ids, vecs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="embeddings.npy")
    parser.add_argument("--tracks_file", default="data/tracks_sample.csv")
    parser.add_argument("--previews_dir", default="data/previews")
    parser.add_argument("--out", default="explore.html")
    parser.add_argument("--n_points", type=int, default=5000)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no_browser", action="store_true")
    args = parser.parse_args()

    tracks_df = pd.read_csv(
        args.tracks_file,
        header=None,
        index_col=0,
        names=["artist", "title", "url", "count"],
    )

    print("Loading embeddings...")
    ids, vecs = load_embeddings(args.embeddings)
    print(f"  {len(ids):,} tracks in {vecs.shape[1]}D space")

    # Subsample if needed
    if len(ids) > args.n_points:
        idx = np.random.choice(len(ids), args.n_points, replace=False)
        ids = [ids[i] for i in idx]
        vecs = vecs[idx]

    print(f"Running t-SNE on {len(ids):,} tracks...")
    coords = TSNE(n_components=2, perplexity=min(30, len(ids) // 5), random_state=42).fit_transform(vecs)

    # Build data source
    artists, titles, has_preview = [], [], []
    for tid in ids:
        if tid in tracks_df.index:
            artists.append(str(tracks_df.loc[tid, "artist"]))
            titles.append(str(tracks_df.loc[tid, "title"]))
        else:
            artists.append("Unknown")
            titles.append("Unknown")
        mp3_path = os.path.join(args.previews_dir, f"{tid}.mp3")
        has_preview.append(os.path.exists(mp3_path))

    source = ColumnDataSource(dict(
        x=coords[:, 0].tolist(),
        y=coords[:, 1].tolist(),
        track_id=ids,
        artist=artists,
        title=titles,
        has_preview=has_preview,
    ))

    # Plot
    p = figure(
        width=1200,
        height=800,
        title="Music JEPA — embedding space (click to play)",
        tools="pan,wheel_zoom,box_zoom,reset,tap",
        toolbar_location="above",
    )
    p.title.text_font_size = "16px"
    p.background_fill_color = "#1a1a2e"
    p.grid.grid_line_color = "#2a2a4e"

    circles = p.circle(
        "x", "y",
        source=source,
        size=6,
        color="#e94560",
        alpha=0.6,
        line_color=None,
        # Keep non-selected dots at full opacity — no dimming on click
        nonselection_fill_alpha=0.6,
        nonselection_fill_color="#e94560",
        nonselection_line_color=None,
        # Selected dot turns white
        selection_fill_color="#ffffff",
        selection_fill_alpha=1.0,
        selection_line_color=None,
    )

    p.add_tools(HoverTool(
        renderers=[circles],
        tooltips=[
            ("Artist", "@artist"),
            ("Title",  "@title"),
            ("ID",     "@track_id"),
        ],
    ))

    # Click-to-play: attach to source.selected so it fires after Bokeh updates indices
    tap_callback = CustomJS(args={"source": source, "port": args.port}, code="""
        const indices = source.selected.indices;
        if (!indices.length) return;
        const idx = indices[0];
        const tid = source.data['track_id'][idx];
        const has = source.data['has_preview'][idx];
        const artist = source.data['artist'][idx];
        const title  = source.data['title'][idx];
        if (!has) { console.log('No preview for', tid); return; }
        const url = 'http://localhost:' + port + '/data/previews/' + tid + '.mp3';
        console.log('Playing:', artist, '-', title, url);
        if (window._jepa_audio) {
            window._jepa_audio.pause();
            window._jepa_audio = null;
        }
        const audio = new Audio(url);
        audio.play().catch(e => console.error('Playback failed:', e));
        window._jepa_audio = audio;
    """)
    source.selected.js_on_change("indices", tap_callback)
    p.add_tools(TapTool())

    html = file_html(p, CDN, "Music JEPA Explorer")
    with open(args.out, "w") as f:
        f.write(html)
    print(f"Saved: {args.out}")

    # Serve files so browser can fetch MP3s (bypasses file:// CORS restrictions)
    repo_root = os.path.abspath(".")
    server_thread = threading.Thread(
        target=serve_files, args=(repo_root, args.port), daemon=True
    )
    server_thread.start()
    print(f"Serving files at http://localhost:{args.port}/")

    if not args.no_browser:
        webbrowser.open(f"http://localhost:{args.port}/{args.out}")

    print("Press Ctrl-C to stop.")
    try:
        server_thread.join()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
