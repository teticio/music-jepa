UV := uv
-include .env
NPROC_PER_NODE ?= 2

TRACKS_FILE ?= data/tracks_sample.csv
LIMIT ?= 20
OUT_HTML ?= outputs/playlist.html
JOURNEY_HTML ?= outputs/journey.html
EXPLORE_HTML ?= outputs/explore.html

.PHONY: setup data data-sample sample previews previews-sample spectrograms train train-head-infil train-head-cont embed journey playlist search viz viz-nn tensorboard help

help:
	@echo "Music JEPA - step by step:"
	@echo "  make setup             Create venv and install dependencies"
	@echo "  make data              Download previews + spectrograms for the FULL dataset"
	@echo "  make data-sample       Same pipeline restricted to a 2000-playlist sample"
	@echo "  make train             Train encoder, resume checkpoints/last.ckpt if present"
	@echo "  make embed             Extract embeddings from last checkpoint"
	@echo "  make train-head-infil  Train infill head on missing playlist tracks"
	@echo "  make train-head-cont   Train continuation head for next-track prediction"
	@echo "  make journey           Fill between waypoint track IDs with infill head"
	@echo "  make playlist          Continue from seed track IDs with continuation head"
	@echo "  make search            Search tracks: QUERY=\"artist or title\""
	@echo "  make viz               Nearest-neighbour report + t-SNE"
	@echo "  make tensorboard       Launch TensorBoard on logs/"

setup:
	$(UV) sync --extra eval

# Full dataset pipeline ------------------------------------------------------

previews:
	$(UV) run python data/download_previews.py --tracks_file data/tracks_dedup.csv --max_workers 32

spectrograms:
	$(UV) run python data/make_spectrograms.py

data: previews spectrograms

# Sample subset pipeline -----------------------------------------------------

sample:
	$(UV) run python data/sample_data.py --n_playlists 2000

previews-sample:
	$(UV) run python data/download_previews.py --tracks_file data/tracks_sample.csv --max_workers 32

data-sample: sample previews-sample spectrograms

# Training / eval ------------------------------------------------------------

train:
	@if [ -f checkpoints/last.ckpt ]; then \
		NPROC_PER_NODE=$(NPROC_PER_NODE) $(UV) run torchrun --nproc_per_node=$(NPROC_PER_NODE) train.py --ckpt checkpoints/last.ckpt; \
	else \
		NPROC_PER_NODE=$(NPROC_PER_NODE) $(UV) run torchrun --nproc_per_node=$(NPROC_PER_NODE) train.py; \
	fi

embed:
	$(UV) run python eval/embed_tracks.py --ckpt checkpoints/last.ckpt

train-head-infil:
	$(UV) run python train_head.py --config configs/head_infil.yaml

train-head-cont:
	$(UV) run python train_head.py --config configs/head_continuation.yaml

journey:
	$(UV) run python eval/generate_playlist.py --head checkpoints/infill_head.pt --journey $(JOURNEY) --out_html $(JOURNEY_HTML)

playlist:
	$(UV) run python eval/generate_playlist.py --head checkpoints/continuation_head.pt --seeds $(or $(SEEDS),$(JOURNEY)) --out_html $(OUT_HTML)

search:
	$(UV) run python eval/search_tracks.py --query "$(QUERY)" --tracks_file $(TRACKS_FILE) --limit $(LIMIT)

viz:
	$(UV) run python eval/explore.py --embeddings embeddings.npy --out $(EXPLORE_HTML)

viz-nn:
	$(UV) run python eval/visualize.py --embeddings embeddings.npy --tsne

tensorboard:
	$(UV) run tensorboard --logdir logs/
