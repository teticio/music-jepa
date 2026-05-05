"""
Streamlit playlist generator.

Run via:
    make app
or:
    uv run streamlit run eval/app.py -- --checkpoint_dir checkpoints --embeddings embeddings/embeddings.npy --tracks_file data/tracks_sample.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from jepa.playlist_head import load_embeddings, load_head, load_tracks
from eval.generate_playlist import (
    build_html_str,
    generate_continuation,
    generate_embedding_continuation,
    generate_embedding_journey,
    generate_infill_journey,
    generate_journey,
    get_generated_track_info,
)

st.set_page_config(page_title="Music JEPA", layout="wide")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--embeddings", default="embeddings/embeddings.npy")
    parser.add_argument("--tracks_file", default="data/tracks_dedup.csv")
    return parser.parse_args()


args = _parse_args()
_cont_head_path = str(Path(args.checkpoint_dir) / "continuation_head.pt")
_infil_head_path = str(Path(args.checkpoint_dir) / "infill_head.pt")


# --- Cached resource loaders ---
@st.cache_resource
def _load_data(emb_path: str, csv_path: str):
    ids, vecs = load_embeddings(emb_path)
    emb = dict(zip(ids, vecs))
    tracks_df = load_tracks(csv_path)
    return ids, vecs, emb, set(ids), tracks_df


@st.cache_resource
def _load_cont_head(path: str):
    if not Path(path).exists():
        return None, {}
    head, cfg = load_head(path, device="cpu")
    return head, cfg


@st.cache_resource
def _load_infil_head(path: str):
    if not Path(path).exists():
        return None, {}
    head, cfg = load_head(path, device="cpu")
    return head, cfg


# --- Load data ---
try:
    ids, vecs, emb, embedded_ids, tracks_df = _load_data(args.embeddings, args.tracks_file)
except Exception as exc:
    st.error(f"Could not load embeddings / tracks: {exc}")
    st.stop()

cont_head, cont_cfg = _load_cont_head(_cont_head_path)
infil_head, _ = _load_infil_head(_infil_head_path)
max_history = cont_cfg.get("data", {}).get("max_history", 3)


# --- Sidebar ---
with st.sidebar:
    st.header("Settings")

    if cont_head is None and infil_head is None:
        st.warning(f"No heads found at:\n- {_cont_head_path}\n- {_infil_head_path}")

    st.subheader("Generation")
    noise = st.slider("Noise", 0.0, 1.0, 0.0, 0.05, help="Higher = more random selection from top candidates")
    head_weight = st.slider(
        "Head weight",
        0.0, 1.0, 1.0, 0.05,
        help=(
            "0 = pure embedding geometry. "
            "1 = fully trust the head. "
            "Continuation: blends head prediction vs mean of recent tracks. "
            "Journey: blends head prediction vs linear interpolation between waypoints."
        ),
    )


# --- Helpers ---
def _search(query: str, limit: int = 50):
    terms = query.strip().lower().split()
    results = []
    for tid, row in tracks_df.iterrows():
        if tid not in embedded_ids:
            continue
        if all(t in f"{row['artist']} {row['title']} {tid}".lower() for t in terms):
            results.append((tid, f"{row['artist']} — {row['title']}"))
        if len(results) >= limit:
            break
    return results


# --- Session state ---
if "waypoints" not in st.session_state:
    st.session_state.waypoints = []
if "playlist" not in st.session_state:
    st.session_state.playlist = None
if "highlighted" not in st.session_state:
    st.session_state.highlighted = set()


# --- Main UI ---
st.title("Music JEPA")

query = st.text_input("Search tracks", placeholder="artist or title…")
if query:
    results = _search(query)
    if results:
        options = {f"{label}  [{tid}]": tid for tid, label in results}
        chosen = st.selectbox("Select track", list(options.keys()))
        if st.button("Add waypoint"):
            tid = options[chosen]
            if tid not in st.session_state.waypoints:
                st.session_state.waypoints.append(tid)
                st.session_state.playlist = None
                st.rerun()
    else:
        st.caption("No results.")

# Waypoints list
if st.session_state.waypoints:
    st.subheader("Waypoints")
    for i, tid in enumerate(st.session_state.waypoints):
        artist, title, _ = get_generated_track_info(tid, tracks_df)
        col_info, col_btn = st.columns([14, 1])
        with col_info:
            st.markdown(f"**{i + 1}.** {title} — *{artist}* `{tid}`")
        with col_btn:
            if st.button("✕", key=f"rm_{i}"):
                st.session_state.waypoints.pop(i)
                st.session_state.playlist = None
                st.rerun()
    if st.button("Clear all"):
        st.session_state.waypoints.clear()
        st.session_state.playlist = None
        st.rerun()

# Size controls
n = len(st.session_state.waypoints)
if n == 1:
    size = st.number_input("Total tracks to generate", min_value=2, max_value=200, value=20)
elif n > 1:
    between = st.number_input("Tracks between each pair of waypoints", min_value=1, max_value=100, value=10)

# Generate
if n >= 1 and st.button("Generate", type="primary"):
    with st.spinner("Generating…"):
        wp = st.session_state.waypoints
        if n == 1:
            if cont_head is not None:
                pl = generate_continuation(
                    cont_head, wp, emb, ids, vecs,
                    size=size, max_history=max_history,
                    noise=noise, head_weight=head_weight, device="cpu",
                )
            else:
                pl = generate_embedding_continuation(
                    wp, emb, ids, vecs,
                    size=size, max_history=max_history, noise=noise,
                )
        else:
            if infil_head is not None:
                pl = generate_infill_journey(
                    infil_head, wp, emb, ids, vecs,
                    between=between, noise=noise, device="cpu",
                )
            elif cont_head is not None:
                pl = generate_journey(
                    cont_head, wp, emb, ids, vecs,
                    between=between, max_history=max_history,
                    noise=noise, head_weight=head_weight, device="cpu",
                )
            else:
                pl = generate_embedding_journey(
                    wp, emb, ids, vecs, between=between, noise=noise,
                )
        st.session_state.playlist = pl
        st.session_state.highlighted = set(wp)

# Playlist display
if st.session_state.playlist:
    pl = st.session_state.playlist
    urls = [get_generated_track_info(tid, tracks_df)[2] for tid in pl]
    mode = "journey" if n > 1 else "playlist"
    html = build_html_str(pl, urls, tracks_df, st.session_state.highlighted, f"Generated {mode}")

    height = min(160 + 72 * len(pl), 900)
    st.iframe(html, height=height)
    st.download_button("Download HTML", html, file_name=f"{mode}.html", mime="text/html")
