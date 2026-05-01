# CLAUDE.md

Notes for Claude when working in this repo. Keep it short and current.

## What this project is

Music JEPA — self-supervised model that learns track-level representations from
mel spectrograms of Spotify previews. Consecutive tracks in a playlist are
treated as adjacent "frames"; a context encoder + predictor learn to predict the
patch-level representation of track N+1 from track N, with an EMA target
encoder and VICReg regularisation to prevent collapse.

The README has the architecture diagram and is up to date — read it for
conceptual orientation rather than re-deriving.

## Tooling

- **Python is run via `uv`.** Don't use a system `python`/`pip`. The Makefile
  uses `uv run …` everywhere; mirror that in any new target. `uv sync --extra
  eval` installs the eval-only deps (sklearn, matplotlib, bokeh).
- Python ≥ 3.12.
- Lightning + DDP for training. `make train` = 2 GPUs (torchrun), `make train1`
  = single GPU.

## Data layout & two-tier pipeline

- **Full** dataset: `data/tracks_dedup.csv` + `data/playlists_dedup.csv`. Run
  `make data` (downloads ~hundreds of GB of mp3 previews → spectrograms).
- **Sample** dataset: 2000 playlists. Run `make data-sample`. Use this for any
  iteration / smoke-testing — the full pipeline is too slow.
- `data/previews/` is ~32GB on the full set. `data/spectrograms/` is what
  training reads.
- CSVs have **no header**. `explore.py` reads them with `header=None,
  names=["artist","title","url","count"]`. If you `pd.read_csv` one of these,
  remember that.

## What's gitignored (don't try to commit it)

`data/*` (except `data/*.py`), all `*.csv` / `*.npy` / `*.mp3` / `*.png`,
`checkpoints/`, `logs/`, `explore.html`, `.claude/`. The `.gitignore` is the
source of truth.

## Conventions

- **Atomic file writes** for anything that another process might consume
  mid-write: write to `<path>.tmp` then `os.replace`. See commit fc70f4b for
  the pattern. Apply it to any new data-producing script.
- Don't add comments unless the *why* is non-obvious — the user prefers terse
  code.
- Default `Make` targets to the **full** pipeline; sample variants are opt-in
  with a `-sample` suffix (see commit 7eb1e32).

## The t-SNE explorer is standalone

`make viz` runs `eval/explore.py`, which writes a single `explore.html` with
the Spotify preview URLs baked in (the `url` column from the tracks CSV).
There is **no** local file server, no companion `data/previews/` dependency at
view time, and no AWS deploy. To share, just host the HTML on any static
host or open the file directly. `--export` writes the HTML without opening
a browser.

## Common one-shot commands

```bash
make setup            # uv sync --extra eval
make data-sample      # build sample dataset (fast)
make train1           # train on 1 GPU
make embed            # extract embeddings from checkpoints/last.ckpt -> embeddings.npy
make viz              # regenerate explore.html
make tensorboard      # logs/ on port 6006
```
