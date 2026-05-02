# CLAUDE.md

Notes for Claude when working in this repo. Keep it short and current.

## What This Project Is

Music JEPA is a self-supervised model that learns track-level representations
from mel spectrograms of Spotify previews. Consecutive tracks in a playlist are
treated like adjacent frames: a context encoder and predictor learn to predict
the patch-level representation of track N+1 from track N, with an EMA target
encoder and VICReg regularisation to prevent collapse.

Read `README.md` for the main workflow and `docs/jepa-architecture.md` for the
deep architecture walkthrough. If source files or config names referenced in
the architecture doc move, update the doc in the same change.

## Tooling

- Run Python through `uv`, not system `python` or `pip`.
- Use `uv run ...` in new Make targets or docs examples.
- `uv sync --extra eval` installs evaluation and visualization dependencies.
- Encoder training uses `torchrun`; GPU process count comes from
  `NPROC_PER_NODE` in local `.env`, defaulting to 2.
- `make train` resumes from `$(CHECKPOINT_DIR)/last.ckpt` when it exists.

## Configuration / mode switching

- `.env` (gitignored) overrides Makefile defaults. `.env.example` documents the
  full and sample presets — copy it to `.env` to start.
- Make-level knobs: `CHECKPOINT_DIR`, `TRAIN_CONFIG`, `HEAD_CONT_CONFIG`,
  `HEAD_INFIL_CONFIG`, `TRACKS_FILE`. Defaults target the full dataset.
- Per-command override also works: `make train TRAIN_CONFIG=configs/sample.yaml`.

## Data Layout

- Full dataset: `data/tracks_dedup.csv` and `data/playlists_dedup.csv` (default).
- Sample dataset: `data/tracks_sample.csv` and `data/playlists_sample.csv`.
- Use `make data-sample` to bootstrap the sample CSVs (`--n_playlists 2000`),
  download their previews, and compute spectrograms in one shot.
- Track/playlists CSVs have no header. Track CSV columns are
  `artist,title,url,count`.
- `embeddings.npy` is a pickled dict-like NumPy file mapping track ID to vector.

## Playlist Workflows

- Train heads with `make train-head-cont` and `make train-head-infil`.
- Head configs default to the dedup playlist CSV. Sample variants live next to
  them as `configs/head_*_sample.yaml` and are activated via `HEAD_CONT_CONFIG`
  / `HEAD_INFIL_CONFIG` (see `.env.example`).
- `make playlist SEEDS="..."` uses the continuation head and writes
  `outputs/playlist.html`.
- `make journey JOURNEY="..."` uses the infill head and writes
  `outputs/journey.html`.
- Add `METHOD=embeddings` to `make playlist` or `make journey` to bypass heads
  and use the raw JEPA embedding baseline.
- Add `METHOD=track2vec` to use Deej-AI's Track2Vec-only baseline from
  `../deej-ai.online-app/model/tracktovec.p`.
- Generated playlist/journey HTML includes preview audio controls and a
  "Play all previews" button.
- `make examples` writes head, embeddings, and Track2Vec examples under
  `outputs/examples/`, plus `outputs/examples/index.html`.
- `make search QUERY="artist or title"` searches local track metadata for IDs.

## Outputs And Gitignore

Generated data, checkpoints, logs, embeddings, and HTML outputs are ignored.
Do not try to commit `data/`, `checkpoints/`, `logs/`, `outputs/`,
`embeddings.npy`, `.env`, MP3s, PNG spectrograms, or CSV datasets.

## Conventions

- Keep edits scoped and in the repo's existing style.
- Prefer Make targets for common workflows.
- Use atomic writes for data-producing scripts that another process might read:
  write `<path>.tmp`, then `os.replace`.
- Do not add comments unless the reason is non-obvious.

## Common Commands

```bash
make setup
make data-sample
make train
make embed
make train-head-cont
make train-head-infil
make playlist SEEDS="TRACK_ID"
make journey JOURNEY="START_ID END_ID"
make playlist METHOD=embeddings SEEDS="TRACK_ID"
make journey METHOD=embeddings JOURNEY="START_ID END_ID"
make playlist METHOD=track2vec SEEDS="TRACK_ID"
make journey METHOD=track2vec JOURNEY="START_ID END_ID"
make examples
make viz
make tensorboard
```
