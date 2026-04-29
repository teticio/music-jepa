"""
Visualise Music JEPA embeddings: t-SNE plot and nearest-neighbour report.

Usage:
    # Nearest-neighbour search for specific tracks
    python eval/visualize.py --embeddings embeddings.npy --tracks_file data/tracks_sample.csv

    # t-SNE plot (saved to tsne.png)
    python eval/visualize.py --embeddings embeddings.npy --tracks_file data/tracks_sample.csv --tsne

    # Probe specific tracks by Spotify ID
    python eval/visualize.py --embeddings embeddings.npy --tracks_file data/tracks_sample.csv \
        --probes 3EYOJ48Et32uATr9ZmLnAo 4CeeEOM32jQcH3eN9Q2dGj
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# Iconic tracks useful for sanity checking the embedding space
DEFAULT_PROBES = [
    "3EYOJ48Et32uATr9ZmLnAo",  # The Police - Roxanne
    "4CeeEOM32jQcH3eN9Q2dGj",  # Nirvana - Smells Like Teen Spirit
    "69kOkLUCkxIZYexIgSG8rq",  # Daft Punk - Get Lucky
    "5uvosCdMlFdTXhoazkTI5R",  # The Doors - Light My Fire
    "75FYqcxt1YEAtqDLrOeIJn",  # Bob Marley - Three Little Birds
    "6oVY50pmdXqLNVeK8bzomn",  # John Coltrane - My Favourite Things
]


def load_embeddings(path: str):
    data = np.load(path, allow_pickle=True).item()
    ids = list(data.keys())
    vecs = np.stack([data[k] for k in ids])
    # L2-normalise for cosine similarity via dot product
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-8)
    return ids, vecs


def nearest_neighbours(query_id: str, ids, vecs, tracks_df, k=10):
    if query_id not in ids:
        return None
    idx = ids.index(query_id)
    sims = vecs @ vecs[idx]  # cosine similarity (vectors already normalised)
    top_k = np.argsort(sims)[::-1][1 : k + 1]  # exclude self

    rows = []
    for i in top_k:
        tid = ids[i]
        artist = tracks_df.loc[tid, "artist"] if tid in tracks_df.index else "?"
        title = tracks_df.loc[tid, "title"] if tid in tracks_df.index else "?"
        rows.append({"track_id": tid, "artist": artist, "title": title, "similarity": sims[i]})
    return rows


def tsne_plot(ids, vecs, tracks_df, out_path="tsne.png", n_points=3000):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    n = min(n_points, len(ids))
    idx = np.random.choice(len(ids), n, replace=False)
    sub_vecs = vecs[idx]

    print(f"Running t-SNE on {n} tracks...")
    coords = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000).fit_transform(sub_vecs)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(coords[:, 0], coords[:, 1], s=3, alpha=0.4, linewidths=0)
    ax.set_title("Music JEPA embeddings (t-SNE)", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"t-SNE saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="embeddings.npy")
    parser.add_argument("--tracks_file", default="data/tracks_sample.csv")
    parser.add_argument("--probes", nargs="*", default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--tsne", action="store_true")
    parser.add_argument("--tsne_out", default="tsne.png")
    parser.add_argument("--tsne_n", type=int, default=3000)
    args = parser.parse_args()

    tracks_df = pd.read_csv(
        args.tracks_file,
        header=None,
        index_col=0,
        names=["artist", "title", "url", "count"],
    )

    print("Loading embeddings...")
    ids, vecs = load_embeddings(args.embeddings)
    print(f"  {len(ids):,} tracks embedded in {vecs.shape[1]}D space")

    probes = args.probes if args.probes else [p for p in DEFAULT_PROBES if p in ids]
    if not probes:
        print("None of the default probe tracks are in the embedding set.")
        print("Pass --probes <track_id> ... to specify tracks.")
    else:
        for probe_id in probes:
            artist = tracks_df.loc[probe_id, "artist"] if probe_id in tracks_df.index else "?"
            title = tracks_df.loc[probe_id, "title"] if probe_id in tracks_df.index else "?"
            print(f"\n=== {artist} - {title}  [{probe_id}] ===")
            neighbours = nearest_neighbours(probe_id, ids, vecs, tracks_df, k=args.top_k)
            if neighbours is None:
                print("  Not found in embeddings.")
                continue
            for rank, row in enumerate(neighbours, 1):
                print(f"  {rank:2d}. {row['artist']:<30} {row['title']:<40} sim={row['similarity']:.4f}")

    if args.tsne:
        tsne_plot(ids, vecs, tracks_df, out_path=args.tsne_out, n_points=args.tsne_n)


if __name__ == "__main__":
    main()
