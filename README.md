# music-jepa

A self-supervised music representation model using Joint Embedding Predictive Architecture (JEPA).

## Demo

- [Embedding explorer](https://teticio.github.io/music-jepa/explore.html)
- [Generated playlist examples](https://teticio.github.io/music-jepa/examples/)

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
- EMA target encoder creates asymmetry (only student gets gradients) — the
  primary mechanism, per the I-JEPA recipe
- VICReg regularisation on the predictor output adds variance + decorrelation
  pressure as a gradient-flowing safety net
  ([configurable](configs/encoder.yaml) via `training.vicreg_target`; set to
  `target` to revert to diagnostic-only / MSE-only training)
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

## Configuration

The Makefile reads optional overrides from a local `.env` file (gitignored).
Copy `.env.example` to `.env` and uncomment what you need:

```bash
cp .env.example .env
```

Defaults target the **full dataset**. To switch to the sample preset, uncomment
the sample block in `.env`.

| Variable           | Full default                     | Sample preset                         | Purpose                                     |
|--------------------|----------------------------------|---------------------------------------|---------------------------------------------|
| `CHECKPOINT_DIR`   | `checkpoints`                    | `checkpoints-sample`                  | Where checkpoints are read/written          |
| `EMBEDDINGS_DIR`   | `embeddings`                     | `embeddings-sample`                   | Where `embeddings.npy` is read/written      |
| `TRAIN_CONFIG`     | `configs/encoder.yaml`           | `configs/encoder_sample.yaml`         | Encoder training + embedding extraction     |
| `HEAD_CONT_CONFIG` | `configs/head_continuation.yaml` | `configs/head_continuation_sample.yaml` | Continuation-head training                |
| `HEAD_INFIL_CONFIG`| `configs/head_infil.yaml`        | `configs/head_infil_sample.yaml`      | Infill-head training                        |
| `TRACKS_FILE`      | `data/tracks_dedup.csv`          | `data/tracks_sample.csv`              | Track metadata for eval / playlist scripts  |
| `OUTPUT_DIR`       | `outputs`                        | `outputs-sample`                      | Generated HTML reports and examples         |
| `PAGES_BRANCH`     | `gh-pages`                       | `gh-pages`                            | GitHub Pages branch                         |
| `PAGES_WORKTREE`   | `/tmp/music-jepa-pages`          | `/tmp/music-jepa-pages`               | Local worktree used for publishing          |
| `NPROC_PER_NODE`   | `2`                              | `2`                                   | GPU process count for `torchrun`            |

You can also override any Make knob per command, e.g. `make train-encoder NAME=value`.

## Data pipeline

### Source CSVs

This repo expects `data/tracks_dedup.csv` and `data/playlists_dedup.csv` to
already exist. They come from the upstream
[Deej-AI](https://github.com/teticio/Deej-AI) data-collection pipeline (see
`train/README.md` in that repo): scrape Spotify user playlists, fetch track
listings, then run `train/deduplicate.py --min_count=10` to drop near-duplicate
tracks and tracks without preview URLs. Once you have those two CSVs, drop them
into `data/` and you're ready for the steps below.

Full dataset (downloads previews for every configured track and computes
spectrograms):

```bash
make data          # = make previews + make spectrograms
```

Or step by step:
```bash
# 1. Download 30-second MP3 previews (~350KB each) from Spotify CDN
make previews

# 2. Compute mel spectrograms (96 mel-bins x 216 time-frames, first 5s)
make spectrograms  # uv run python data/make_spectrograms.py
```

Sample subset (a 2000-playlist slice — fast iteration / smoke testing):

```bash
make data-sample   # = sample → previews-sample → spectrograms
```

This writes the sample playlist/track CSVs and only downloads previews for
tracks in the sample. To use the sample subset across later targets, point your
`.env` at the sample preset described in [Configuration](#configuration).

## Training

Encoder training uses `torchrun`. The Makefile passes the configured training
file through to `train_encoder.py`, so swapping modes is a one-line `.env`
change or a per-command override.

If the configured checkpoint dir contains `last.ckpt`, `make train-encoder`
resumes from it automatically:

```bash
make train-encoder
```

Manual resume from a specific checkpoint:
```bash
uv run python train_encoder.py --ckpt path/to/checkpoint.ckpt
```

TensorBoard logs are written to `logs/music_jepa/`. Launch the UI with:
```bash
make tensorboard   # uv run tensorboard --logdir logs/
```

## Evaluation

```bash
# Extract embeddings for all configured tracks from the latest checkpoint
make embed

# Search local track metadata for IDs and preview URLs
make search QUERY="daft punk"

# Interactive embedding explorer
make viz

# Streamlit playlist generator (search, add waypoints, generate, preview)
make app
```

**Success criterion:** nearest neighbours should be musically coherent (e.g. Nirvana
neighbours are grunge/alt-rock, Daft Punk neighbours are electronic) without having
engineered any music features — the signal comes entirely from playlist co-occurrence
and audio content.

## Playlist heads

Once the JEPA encoder has produced `EMBEDDINGS_DIR/embeddings.npy`, train
lightweight heads in the frozen JEPA space. Head training uses the config files selected in
[Configuration](#configuration).

```bash
make train-head-cont
make train-head-infil
```

The continuation head predicts the next track from seed history. Use it for
open-ended playlist generation. `HEAD_WEIGHT` (0–1) controls how much the head
prediction is trusted versus anchoring to the recent embedding mean; 1 = pure
head, 0 = pure rolling-mean (equivalent to linear embedding extrapolation).
Pass `--mp3tovec_model_dir` to compare against Deej-AI's MP3ToVec audio
baseline from `../deej-ai.online-app/model/spotifytovec.p`:

```bash
make playlist SEEDS="3EYOJ48Et32uATr9ZmLnAo 69kOkLUCkxIZYexIgSG8rq"
make playlist HEAD_WEIGHT=0.5 SEEDS="3EYOJ48Et32uATr9ZmLnAo 69kOkLUCkxIZYexIgSG8rq"
```

The infill head sees a left anchor, a right anchor, and their interpolation
point; it predicts the missing middle-track vector. Use it for waypoint
journeys. The continuation head can also fill journeys via `make journey`,
blending linear interpolation with head predictions via `HEAD_WEIGHT`:

```bash
# "Join the dots" between waypoint tracks over several generated steps
make journey JOURNEY="3EYOJ48Et32uATr9ZmLnAo 69kOkLUCkxIZYexIgSG8rq"
```

Generated playlist output includes track IDs, artist/title, and MP3 preview
links when present. Generated HTML goes under the output dir configured in
[Configuration](#configuration) and ignored by git. The playlist and journey
pages include per-track browser audio controls plus a top-level
“Play all previews” button that chains available 30-second previews after one
user click. Add `--out_m3u path/to/file.m3u` when calling
`eval/generate_playlist.py` directly to write a playable M3U file.

For quick comparisons, `make examples` writes head-based examples at
`head_weight` 0, 0.5, and 1.0 for every playlist and journey, plus MP3ToVec
baselines when `--mp3tovec_model_dir` is supplied. The generated index links
to all example pages.

Publish the configured output dir to GitHub Pages with:

```bash
make publish-pages
```

This syncs `OUTPUT_DIR` into a temporary worktree for the Pages branch, commits
changes if needed, and pushes.

The journey mode follows the original Deej-AI idea: interpolate through the
continuous vector space between waypoint tracks, then snap each step to a real
catalogue track. The infill head learns a correction on top of the interpolated
point toward a plausible missing track; the continuation head learns the
transition dynamics from playlist history.

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
