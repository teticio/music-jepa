UV := uv
-include .env
NPROC_PER_NODE ?= 2

# Per-mode knobs (override in .env to swap between full/sample/elsewhere) ----
CHECKPOINT_DIR ?= checkpoints
CHECKPOINT_NAME ?= last.ckpt
EMBEDDINGS_DIR ?= embeddings
TRAIN_CONFIG ?= configs/encoder.yaml
HEAD_CONT_CONFIG ?= configs/head_continuation.yaml
HEAD_INFIL_CONFIG ?= configs/head_infil.yaml
TRACKS_FILE ?= data/tracks_dedup.csv
OUTPUT_DIR ?= outputs
PAGES_BRANCH ?= gh-pages
PAGES_WORKTREE ?= /tmp/music-jepa-pages
PAGES_MESSAGE ?= Publish $(OUTPUT_DIR)

LIMIT ?= 20
N_POINTS ?= 5000
HEAD_WEIGHT ?= 1.0
OUT_HTML ?= $(OUTPUT_DIR)/playlist.html
JOURNEY_HTML ?= $(OUTPUT_DIR)/journey.html
EXPLORE_HTML ?= $(OUTPUT_DIR)/explore.html

.PHONY: setup app data data-sample sample previews previews-sample spectrograms train-encoder train-head-infil train-head-cont embed journey playlist examples search viz publish-pages tensorboard test help

help:
	@echo "Music JEPA - step by step:"
	@echo "  make setup             Create venv and install dependencies"
	@echo "  make app               Launch Streamlit playlist generator"
	@echo "  make data              Download previews + spectrograms for TRACKS_FILE"
	@echo "  make data-sample       Bootstrap a 2000-playlist sample subset"
	@echo "  make train-encoder     Train encoder, resume CHECKPOINT_DIR/CHECKPOINT_NAME if present"
	@echo "  make embed             Extract embeddings from CHECKPOINT_DIR/CHECKPOINT_NAME"
	@echo "  make train-head-infil  Train infill head on missing playlist tracks"
	@echo "  make train-head-cont   Train continuation head for next-track prediction"
	@echo "  make journey           Fill between waypoint track IDs with infill head"
	@echo "  make playlist          Continue from seed track IDs with continuation head"
	@echo "  make examples          Generate example playlist/journey HTML gallery"
	@echo "  make search            Search tracks: QUERY=\"artist or title\""
	@echo "  make viz               Interactive t-SNE embedding explorer"
	@echo "  make publish-pages     Publish OUTPUT_DIR to the gh-pages branch"
	@echo "  make tensorboard       Launch TensorBoard on logs/"
	@echo "  make test              Run pytest test suite"
	@echo ""
	@echo "Override defaults via .env (see .env.example) or env vars:"
	@echo "  CHECKPOINT_DIR=$(CHECKPOINT_DIR)"
	@echo "  CHECKPOINT_NAME=$(CHECKPOINT_NAME)"
	@echo "  EMBEDDINGS_DIR=$(EMBEDDINGS_DIR)"
	@echo "  TRAIN_CONFIG=$(TRAIN_CONFIG)"
	@echo "  HEAD_CONT_CONFIG=$(HEAD_CONT_CONFIG)"
	@echo "  HEAD_INFIL_CONFIG=$(HEAD_INFIL_CONFIG)"
	@echo "  TRACKS_FILE=$(TRACKS_FILE)"
	@echo "  OUTPUT_DIR=$(OUTPUT_DIR)"
	@echo "  PAGES_BRANCH=$(PAGES_BRANCH)"
	@echo "  PAGES_WORKTREE=$(PAGES_WORKTREE)"

setup:
	$(UV) sync --extra eval --extra app

app:
	$(UV) run streamlit run eval/app.py -- --checkpoint_dir $(CHECKPOINT_DIR) --embeddings $(EMBEDDINGS_DIR)/embeddings.npy --tracks_file $(TRACKS_FILE)

# Data pipeline --------------------------------------------------------------

previews:
	$(UV) run python data/download_previews.py --tracks_file $(TRACKS_FILE) --max_workers 32

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

train-encoder:
	@RESUME=""; \
	if [ -f $(CHECKPOINT_DIR)/$(CHECKPOINT_NAME) ]; then RESUME="--ckpt $(CHECKPOINT_DIR)/$(CHECKPOINT_NAME)"; fi; \
	if [ "$(NPROC_PER_NODE)" -gt 1 ]; then \
		NPROC_PER_NODE=$(NPROC_PER_NODE) $(UV) run torchrun --nproc_per_node=$(NPROC_PER_NODE) train_encoder.py --config $(TRAIN_CONFIG) --checkpoint_dir $(CHECKPOINT_DIR) $$RESUME; \
	else \
		$(UV) run python train_encoder.py --config $(TRAIN_CONFIG) --checkpoint_dir $(CHECKPOINT_DIR) $$RESUME; \
	fi

embed:
	$(UV) run python eval/embed_tracks.py --config $(TRAIN_CONFIG) --ckpt $(CHECKPOINT_DIR)/$(CHECKPOINT_NAME) --out $(EMBEDDINGS_DIR)/embeddings.npy

train-head-infil:
	$(UV) run python train_head.py --config $(HEAD_INFIL_CONFIG) --out $(CHECKPOINT_DIR)/infill_head.pt

train-head-cont:
	$(UV) run python train_head.py --config $(HEAD_CONT_CONFIG) --out $(CHECKPOINT_DIR)/continuation_head.pt

journey:
	$(UV) run python eval/generate_playlist.py --head $(CHECKPOINT_DIR)/infill_head.pt --embeddings $(EMBEDDINGS_DIR)/embeddings.npy --tracks_file $(TRACKS_FILE) --journey $(JOURNEY) --out_html $(JOURNEY_HTML)

playlist:
	@if [ -z "$(SEEDS)" ]; then echo "Usage: make playlist SEEDS=\"TRACK_ID [TRACK_ID ...]\""; exit 1; fi
	$(UV) run python eval/generate_playlist.py --head $(CHECKPOINT_DIR)/continuation_head.pt --embeddings $(EMBEDDINGS_DIR)/embeddings.npy --tracks_file $(TRACKS_FILE) --seeds $(SEEDS) --head_weight $(HEAD_WEIGHT) --out_html $(OUT_HTML)

examples:
	$(UV) run python eval/generate_examples.py --checkpoint_dir $(CHECKPOINT_DIR) --embeddings $(EMBEDDINGS_DIR)/embeddings.npy --tracks_file $(TRACKS_FILE) --out_dir $(OUTPUT_DIR)/examples

search:
	$(UV) run python eval/search_tracks.py --query "$(QUERY)" --embeddings $(EMBEDDINGS_DIR)/embeddings.npy --tracks_file $(TRACKS_FILE) --limit $(LIMIT)

viz:
	$(UV) run python eval/explore.py --embeddings $(EMBEDDINGS_DIR)/embeddings.npy --tracks_file $(TRACKS_FILE) --out $(EXPLORE_HTML) --n_points $(N_POINTS) --export

publish-pages:
	@if [ ! -d "$(OUTPUT_DIR)" ]; then echo "Missing OUTPUT_DIR=$(OUTPUT_DIR). Generate outputs first."; exit 1; fi
	@if [ ! -e "$(PAGES_WORKTREE)/.git" ]; then \
		git worktree prune; \
		if git show-ref --verify --quiet refs/heads/$(PAGES_BRANCH) || git ls-remote --exit-code --heads origin $(PAGES_BRANCH) >/dev/null 2>&1; then \
			git worktree add "$(PAGES_WORKTREE)" "$(PAGES_BRANCH)"; \
		else \
			git worktree add --orphan -b "$(PAGES_BRANCH)" "$(PAGES_WORKTREE)"; \
		fi; \
	fi
	rsync -a --delete --exclude=.git "$(OUTPUT_DIR)/" "$(PAGES_WORKTREE)/"
	touch "$(PAGES_WORKTREE)/.nojekyll"
	printf '%s\n' '<!doctype html>' '<html lang="en">' '<head>' '  <meta charset="utf-8">' '  <meta name="viewport" content="width=device-width, initial-scale=1">' '  <meta http-equiv="refresh" content="0; url=explore.html">' '  <title>music-jepa</title>' '</head>' '<body>' '  <main>' '    <h1>music-jepa</h1>' '    <p><a href="explore.html">Open embedding explorer</a></p>' '    <p><a href="examples/">Open examples</a></p>' '  </main>' '</body>' '</html>' > "$(PAGES_WORKTREE)/index.html"
	cd "$(PAGES_WORKTREE)" && git add . && if git diff --cached --quiet; then echo "No changes to publish."; else git commit -m "$(PAGES_MESSAGE)"; fi
	cd "$(PAGES_WORKTREE)" && git push origin "$(PAGES_BRANCH)"

tensorboard:
	$(UV) run tensorboard --logdir logs/

test:
	$(UV) run --extra dev pytest tests/ -v
