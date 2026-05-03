"""
Interactive Music JEPA embedding explorer.

- 2D t-SNE scatter plot in your browser
- Hover: shows artist and title
- Click: plays the 30-second Spotify preview from the CDN

Usage:
    python eval/explore.py --embeddings embeddings.npy
    python eval/explore.py --embeddings embeddings.npy --export   # write HTML, don't open browser
"""
import argparse
import os
from pathlib import Path
import webbrowser

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from bokeh.models import (
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    CustomJS,
    HoverTool,
    LogColorMapper,
    TapTool,
)
from bokeh.palettes import Viridis256
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import file_html


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
    parser.add_argument("--tracks_file", default="data/tracks_dedup.csv")
    parser.add_argument("--out", default="outputs/explore.html")
    parser.add_argument("--n_points", type=int, default=5000)
    parser.add_argument(
        "--export",
        action="store_true",
        help="Write the HTML and exit without opening a browser. The result is a "
             "standalone file (audio is loaded from the Spotify CDN).",
    )
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

    if len(ids) > args.n_points:
        idx = np.random.choice(len(ids), args.n_points, replace=False)
        ids = [ids[i] for i in idx]
        vecs = vecs[idx]

    print(f"Running t-SNE on {len(ids):,} tracks...")
    coords = TSNE(n_components=2, perplexity=min(30, len(ids) // 5), random_state=42).fit_transform(vecs)

    artists, titles, audio_urls, counts = [], [], [], []
    for tid in ids:
        if tid in tracks_df.index:
            artists.append(str(tracks_df.loc[tid, "artist"]))
            titles.append(str(tracks_df.loc[tid, "title"]))
            url = tracks_df.loc[tid, "url"]
            count = tracks_df.loc[tid, "count"]
        else:
            artists.append("Unknown")
            titles.append("Unknown")
            url = None
            count = 1
        audio_urls.append(str(url) if isinstance(url, str) and url else "")
        count = pd.to_numeric(count, errors="coerce")
        counts.append(float(count) if np.isfinite(count) else 1.0)

    color_values = np.maximum(np.array(counts), 1)
    color_mapper = LogColorMapper(
        palette=Viridis256,
        low=float(color_values.min()),
        high=float(color_values.max()),
    )

    source = ColumnDataSource(dict(
        x=coords[:, 0].tolist(),
        y=coords[:, 1].tolist(),
        track_id=ids,
        artist=artists,
        title=titles,
        audio_url=audio_urls,
        count=[int(count) for count in counts],
        color_value=color_values.tolist(),
    ))

    p = figure(
        width=1200,
        height=800,
        title="music-jepa embedding space (click to play)",
        tools="pan,wheel_zoom,box_zoom,reset,tap",
        toolbar_location="above",
    )
    p.title.text_font_size = "16px"
    p.background_fill_color = "#f7f4ef"
    p.border_fill_color = "#f7f4ef"
    p.grid.grid_line_color = "#ded8ce"
    p.axis.visible = False
    p.toolbar.logo = None

    circles = p.scatter(
        "x", "y",
        source=source,
        size=6,
        color={"field": "color_value", "transform": color_mapper},
        alpha=0.72,
        line_color=None,
        nonselection_fill_alpha=0.28,
        nonselection_line_color=None,
        selection_fill_color="#f45b69",
        selection_fill_alpha=1.0,
        selection_line_color=None,
    )
    color_bar = ColorBar(
        color_mapper=color_mapper,
        ticker=BasicTicker(desired_num_ticks=6),
        title="playlist count",
        label_standoff=8,
        width=12,
        padding=8,
    )
    p.add_layout(color_bar, "right")

    p.add_tools(HoverTool(
        renderers=[circles],
        tooltips=[
            ("Artist", "@artist"),
            ("Title",  "@title"),
            ("Playlist count", "@count"),
            ("ID",     "@track_id"),
        ],
    ))

    tap_callback = CustomJS(args={"source": source}, code="""
        const indices = source.selected.indices;
        if (!indices.length) return;
        const idx = indices[0];
        const tid = source.data['track_id'][idx];
        const url = source.data['audio_url'][idx];
        const artist = source.data['artist'][idx];
        const title  = source.data['title'][idx];
        if (!url) { console.log('No preview for', tid); return; }
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

    html = file_html(p, CDN, "music-jepa Explorer")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"Saved: {args.out}")

    if not args.export:
        webbrowser.open(f"file://{os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
