# Music JEPA

A self-supervised music representation model using Joint Embedding Predictive Architecture (JEPA).

## Concept

Consecutive tracks in a Spotify playlist are treated like frames in a video. A ViT encoder
processes mel spectrograms; a predictor learns to predict patch-level representations of track
N+1 given track N, entirely in latent space. The learned embedding space encodes musical
semantics (genre, energy, mood) without any hand-crafted features or labels.

```
playlist:  [track_0, track_1, track_2, ...]
                 │          │
           context spec   target spec
                 │          │
           ┌─────┴────┐   ┌─┴──────────┐
           │ Context  │   │  Target    │  (EMA copy, no gradient)
           │ Encoder  │   │  Encoder   │
           └─────┬────┘   └─────┬──────┘
                 │        (B, N, D) patch tokens
                 │              │ gather masked patches
                 ▼              ▼
           ┌──────────┐   target patch reprs
           │Predictor │──────────────────────┐
           │(4-layer  │                      │
           │  ViT)    │──► predicted reprs   │
           └──────────┘          │           │
                                 └── MSE + VICReg loss
```

**Why JEPA prevents mode collapse:**
- EMA target encoder creates asymmetry (only student gets gradients)
- VICReg regularisation enforces variance + decorrelation on target representations
- Patch-level prediction (not single-vector) requires spatially-specific predictions

## Setup

Dependencies are managed with [uv](https://github.com/astral-sh/uv):

```bash
make setup
# On Linux, also: apt-get install ffmpeg  (for librosa MP3 loading)
```

All commands below run inside the uv-managed environment via `uv run`. The
`Makefile` wraps each step; you can also invoke them directly with
`uv run python <script>`.

## Data pipeline

Full dataset (downloads previews for every track in `data/tracks_dedup.csv`
and computes spectrograms for all of them):

```bash
make data          # = make previews + make spectrograms
```

Or step by step:
```bash
# 1. Download 30-second MP3 previews (~350KB each) from Spotify CDN
make previews      # uv run python data/download_previews.py --tracks_file data/tracks_dedup.csv

# 2. Compute mel spectrograms (96 mel-bins x 216 time-frames, first 5s)
make spectrograms  # uv run python data/make_spectrograms.py
```

Sample subset (a 2000-playlist slice — fast iteration / smoke testing):

```bash
make data-sample   # = sample → previews-sample → spectrograms
```

This writes `data/playlists_sample.csv` + `data/tracks_sample.csv` and only
downloads previews for tracks in the sample. The matching training config is
`configs/sample.yaml`.

## Training

Encoder training uses `torchrun`. The number of GPU worker processes comes from
`NPROC_PER_NODE` in your local `.env` file, which is ignored by git:

```bash
NPROC_PER_NODE=2
```

Train on the full dataset. If `checkpoints/last.ckpt` exists, `make train`
resumes from it automatically:

```bash
make train
```

Quick run on the sample subset (CPU-friendly, ViT-Tiny):
```bash
uv run python train.py --config configs/sample.yaml
```

Manual resume from a specific checkpoint:
```bash
uv run python train.py --ckpt checkpoints/last.ckpt
```

TensorBoard logs are written to `logs/music_jepa/`. Launch the UI with:
```bash
make tensorboard   # uv run tensorboard --logdir logs/
```

## Evaluation

```bash
# Extract embeddings for all tracks
make embed         # uv run python eval/embed_tracks.py --ckpt checkpoints/last.ckpt

# Search local track metadata for IDs and preview URLs
make search QUERY="daft punk"

# Nearest-neighbour report + t-SNE
make viz-nn        # uv run python eval/visualize.py --embeddings embeddings.npy --tsne

# Interactive embedding explorer
make viz           # writes outputs/explore.html
```

**Success criterion:** nearest neighbours should be musically coherent (e.g. Nirvana
neighbours are grunge/alt-rock, Daft Punk neighbours are electronic) without having
engineered any music features — the signal comes entirely from playlist co-occurrence
and audio content.

## Playlist heads

Once the JEPA encoder has produced `embeddings.npy`, train lightweight heads in
the frozen JEPA space. The current head configs use `data/playlists_sample.csv`
so training stays aligned with the tracks that already have previews,
spectrograms, and embeddings.

```bash
make train-head-cont    # configs/head_continuation.yaml
make train-head-infil   # configs/head_infil.yaml
```

The continuation head predicts the next track from seed history. Use it for
open-ended playlist generation. `METHOD=head` is the default. Use
`METHOD=embeddings` to compare against a raw JEPA nearest-neighbour baseline,
or `METHOD=track2vec` to compare against Deej-AI's Track2Vec collaborative
baseline from `data/deejai/tracktovec.p`:

```bash
make playlist SEEDS="3EYOJ48Et32uATr9ZmLnAo 69kOkLUCkxIZYexIgSG8rq"
make playlist METHOD=embeddings SEEDS="3EYOJ48Et32uATr9ZmLnAo"
make playlist METHOD=track2vec SEEDS="3EYOJ48Et32uATr9ZmLnAo"
```

The infill head sees a left anchor, a right anchor, and their interpolation
point; it predicts the missing middle-track vector. Use it for waypoint
journeys. The embeddings and Deej-AI baselines linearly interpolate between
waypoint embeddings, then snap each step to the nearest real track:

```bash
# "Join the dots" between waypoint tracks over several generated steps
make journey JOURNEY="3EYOJ48Et32uATr9ZmLnAo 69kOkLUCkxIZYexIgSG8rq"
make journey METHOD=embeddings JOURNEY="3EYOJ48Et32uATr9ZmLnAo 69kOkLUCkxIZYexIgSG8rq"
make journey METHOD=track2vec JOURNEY="3EYOJ48Et32uATr9ZmLnAo 69kOkLUCkxIZYexIgSG8rq"
```

Generated playlist output includes track IDs, artist/title, and MP3 preview
links when present. `make playlist` writes `outputs/playlist.html`,
`make journey` writes `outputs/journey.html`, and `make viz` writes
`outputs/explore.html`; all HTML outputs live under `outputs/`. The playlist
and journey pages include per-track browser audio controls plus a top-level
“Play all previews” button that chains available 30-second previews after one
user click. Add `--out_m3u path/to/file.m3u` when calling
`eval/generate_playlist.py` directly to write a playable M3U file.

For quick comparisons, `make examples` writes head-based examples and
baseline pages to `outputs/examples/`. Raw JEPA baselines use the
`_embeddings.html` suffix, and Track2Vec baselines use `_track2vec.html`,
for example
`playlist_classical_embeddings.html` and
`journey_classical_jazz_funk_disco_house_techno_track2vec.html`.

The journey mode follows the original Deej-AI idea: interpolate through the
continuous vector space between waypoint tracks, then snap each waypoint to a real
catalogue track. The trained head learns the correction from pure interpolation
to "what track would plausibly be missing here?"

## Architecture

| Component        | Config              |
|------------------|---------------------|
| Encoder          | ViT-Small (384d, 12L, 6H) |
| Input            | 96×216 mel spectrogram, 8×8 patches → 324 tokens |
| Predictor        | 4-layer ViT (192d, 6H) |
| EMA momentum     | 0.996 → 0.999 (cosine ramp) |
| Loss             | MSE + VICReg (λ_std=25, λ_cov=1) |
| Optimiser        | AdamW, lr=1.5e-4, warmup 10 epochs |
| Precision        | fp16 mixed |

## References

- [I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (CVPR 2023)](https://arxiv.org/abs/2301.08243)
- [V-JEPA: Video Joint Embedding Predictive Architecture (Meta AI, 2024)](https://ai.meta.com/research/publications/revisiting-feature-prediction-for-learning-visual-representations-from-video/)
- [VICReg: Variance-Invariance-Covariance Regularization (Bardes et al., 2022)](https://arxiv.org/abs/2105.04906)
- [Deej-AI: Music Recommendation with Spectrograms](https://github.com/teticio/Deej-AI)
