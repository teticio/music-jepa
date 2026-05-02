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

## Configuration

The Makefile reads optional overrides from a local `.env` file (gitignored).
Copy `.env.example` to `.env` and uncomment what you need:

```bash
cp .env.example .env
```

Defaults target the **full dataset**:

| Variable           | Default                              | Purpose                                     |
|--------------------|--------------------------------------|---------------------------------------------|
| `CHECKPOINT_DIR`   | `checkpoints`                        | Where checkpoints are read/written          |
| `TRAIN_CONFIG`     | `configs/train.yaml`                 | Encoder training + embedding extraction     |
| `HEAD_CONT_CONFIG` | `configs/head_continuation.yaml`     | Continuation-head training                  |
| `HEAD_INFIL_CONFIG`| `configs/head_infil.yaml`            | Infill-head training                        |
| `TRACKS_FILE`      | `data/tracks_dedup.csv`              | Track metadata for eval / playlist scripts  |
| `NPROC_PER_NODE`   | `2`                                  | GPU process count for `torchrun`            |

To switch to the sample preset, uncomment the sample block in `.env`:

```bash
CHECKPOINT_DIR=checkpoints-sample
TRAIN_CONFIG=configs/encoder_sample.yaml
HEAD_CONT_CONFIG=configs/head_continuation_sample.yaml
HEAD_INFIL_CONFIG=configs/head_infil_sample.yaml
TRACKS_FILE=data/tracks_sample.csv
```

You can also override per-command, e.g. `make train-encoder CHECKPOINT_DIR=checkpoints-foo`.

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
downloads previews for tracks in the sample. To use the sample subset across
all later targets (training, eval, playlists), point your `.env` at the sample
preset described in the [Configuration](#configuration) section.

## Training

Encoder training uses `torchrun`. The Makefile passes `--config $(TRAIN_CONFIG)`
through to `train_encoder.py`, so swapping training configs is a one-line `.env`
override (or a per-command flag).

If `$(CHECKPOINT_DIR)/last.ckpt` exists, `make train-encoder` resumes from it
automatically:

```bash
make train-encoder
make train-encoder CHECKPOINT_DIR=checkpoints-full
make train-encoder TRAIN_CONFIG=configs/encoder_sample.yaml CHECKPOINT_DIR=checkpoints-sample
```

Manual resume from a specific checkpoint:
```bash
uv run python train_encoder.py --ckpt checkpoints/last.ckpt
```

TensorBoard logs are written to `logs/music_jepa/`. Launch the UI with:
```bash
make tensorboard   # uv run tensorboard --logdir logs/
```

## Evaluation

```bash
# Extract embeddings for all tracks from $(CHECKPOINT_DIR)/last.ckpt
make embed
make embed CHECKPOINT_DIR=checkpoints-sample

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
the frozen JEPA space. Head training uses `$(HEAD_CONT_CONFIG)` /
`$(HEAD_INFIL_CONFIG)`, which default to the full-dataset configs
(`configs/head_continuation.yaml` and `configs/head_infil.yaml`). Sample
variants live alongside them with the `_sample.yaml` suffix and are picked up
automatically when the sample preset is active in `.env`.

```bash
make train-head-cont
make train-head-infil
make train-head-cont HEAD_CONT_CONFIG=configs/head_continuation_sample.yaml \
                     CHECKPOINT_DIR=checkpoints-sample
```

The continuation head predicts the next track from seed history. Use it for
open-ended playlist generation. `METHOD=head` is the default. Use
`METHOD=embeddings` to compare against a raw JEPA nearest-neighbour baseline,
or `METHOD=track2vec` to compare against Deej-AI's Track2Vec collaborative
baseline from `../deej-ai.online-app/model/tracktovec.p`:

```bash
make playlist SEEDS="3EYOJ48Et32uATr9ZmLnAo 69kOkLUCkxIZYexIgSG8rq"
make playlist DRIFT=0.35 SEEDS="3EYOJ48Et32uATr9ZmLnAo 69kOkLUCkxIZYexIgSG8rq"
make playlist METHOD=embeddings SEEDS="3EYOJ48Et32uATr9ZmLnAo"
make playlist METHOD=track2vec SEEDS="3EYOJ48Et32uATr9ZmLnAo"
make playlist CHECKPOINT_DIR=checkpoints-sample SEEDS="3EYOJ48Et32uATr9ZmLnAo"
```

For continuation heads, `DRIFT=0` uses the head prediction directly. Higher
values blend that prediction toward the recent-history embedding mean; `DRIFT=1`
is equivalent to asking the head to behave like the raw rolling-embedding
continuation. Raw embeddings and Track2Vec continuations already use that
rolling mean directly.

The infill head sees a left anchor, a right anchor, and their interpolation
point; it predicts the missing middle-track vector. Use it for waypoint
journeys. The embeddings and Deej-AI baselines linearly interpolate between
waypoint embeddings, then snap each step to the nearest real track:

```bash
# "Join the dots" between waypoint tracks over several generated steps
make journey JOURNEY="3EYOJ48Et32uATr9ZmLnAo 69kOkLUCkxIZYexIgSG8rq"
make journey METHOD=embeddings JOURNEY="3EYOJ48Et32uATr9ZmLnAo 69kOkLUCkxIZYexIgSG8rq"
make journey METHOD=track2vec JOURNEY="3EYOJ48Et32uATr9ZmLnAo 69kOkLUCkxIZYexIgSG8rq"
make journey CHECKPOINT_DIR=checkpoints-sample JOURNEY="3EYOJ48Et32uATr9ZmLnAo 69kOkLUCkxIZYexIgSG8rq"
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
`journey_classical_jazz_funk_disco_house_techno_track2vec.html`. It also
writes `outputs/examples/index.html`, which links to all generated example
pages. The head continuation gallery includes electronic examples at several
drift and noise values, named like `playlist_electronic_drift_35.html` and
`playlist_electronic_noise_25.html`.

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
