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

```bash
# 1. Sample a subset of playlists and tracks from the full dataset
make sample        # uv run python data/sample_data.py --n_playlists 2000

# 2. Download 30-second MP3 previews (~350KB each) from Spotify CDN
make previews      # uv run python data/download_previews.py

# 3. Compute mel spectrograms (96 mel-bins x 216 time-frames, first 5s)
make spectrograms  # uv run python data/make_spectrograms.py

# Or all three at once:
make data
```

## Training

Single GPU:
```bash
make train1        # uv run python train.py --config configs/train.yaml
```

Two GPUs (recommended — 2x RTX 4090):
```bash
make train         # uv run torchrun --nproc_per_node=2 train.py
```

Resume from checkpoint:
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

# Nearest-neighbour report + t-SNE
make viz-nn        # uv run python eval/visualize.py --embeddings embeddings.npy --tsne
```

**Success criterion:** nearest neighbours should be musically coherent (e.g. Nirvana
neighbours are grunge/alt-rock, Daft Punk neighbours are electronic) without having
engineered any music features — the signal comes entirely from playlist co-occurrence
and audio content.

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
