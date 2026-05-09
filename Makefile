UV := uv
-include .env
NPROC_PER_NODE ?= 2

# Per-mode knobs (override in .env to swap between full/sample/elsewhere) ----
CHECKPOINT_DIR ?= checkpoints
CHECKPOINT_NAME ?= last.ckpt
EMBEDDINGS_DIR ?= embeddings
EMBEDDINGS_FILE ?= $(EMBEDDINGS_DIR)/embeddings.npy
TRAIN_CONFIG ?= configs/encoder.yaml
HEAD_CONT_CONFIG ?= configs/head_continuation.yaml
HEAD_INFIL_CONFIG ?= configs/head_infil.yaml
HEAD_CONT_CKPT ?= $(CHECKPOINT_DIR)/continuation_head.pt
HEAD_INFIL_CKPT ?= $(CHECKPOINT_DIR)/infill_head.pt
HEAD_CONT_PATCH_CONFIG ?= configs/head_continuation_patch.yaml
HEAD_INFIL_PATCH_CONFIG ?= configs/head_infil_patch.yaml
HEAD_CONT_PATCH_CKPT ?= $(CHECKPOINT_DIR)/continuation_head_patch.pt
HEAD_INFIL_PATCH_CKPT ?= $(CHECKPOINT_DIR)/infill_head_patch.pt
EMBEDDINGS_PATCH_CONT ?= $(EMBEDDINGS_DIR)/embeddings_patch_cont.npy
EMBEDDINGS_PATCH_INFIL ?= $(EMBEDDINGS_DIR)/embeddings_patch_infil.npy
PATCH_HEAD ?=
TRACKS_FILE ?= data/tracks_dedup.csv
MP3TOVEC_MODEL_DIR ?= ../deej-ai.online-app/model
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

.PHONY: setup app app-patch data data-sample sample previews previews-sample spectrograms train-encoder train-head-infil train-head-cont train-head-patch-cont train-head-patch-infil embed embed-patch embed-patch-cont embed-patch-infil journey journey-patch playlist playlist-patch examples examples-patch search viz publish-pages tensorboard test help

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
	@echo "  make train-head-patch-cont   Train patch-level continuation head (alt approach)"
	@echo "  make train-head-patch-infil  Train patch-level infill head (alt approach)"
	@echo "  make embed-patch       Re-embed catalog with an arbitrary patch head (PATCH_HEAD=path)"
	@echo "  make embed-patch-cont  Re-embed using HEAD_CONT_PATCH_CKPT -> EMBEDDINGS_PATCH_CONT"
	@echo "  make embed-patch-infil Re-embed using HEAD_INFIL_PATCH_CKPT -> EMBEDDINGS_PATCH_INFIL"
	@echo "  make playlist-patch    Continue with base + patch continuation catalogs"
	@echo "  make journey-patch     Journey with base + patch infill catalogs"
	@echo "  make examples-patch    Generate gallery with base + patch catalogs"
	@echo "  make app-patch         Streamlit app with base + patch catalogs"
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
	@echo "  EMBEDDINGS_FILE=$(EMBEDDINGS_FILE)"
	@echo "  TRAIN_CONFIG=$(TRAIN_CONFIG)"
	@echo "  HEAD_CONT_CONFIG=$(HEAD_CONT_CONFIG)"
	@echo "  HEAD_INFIL_CONFIG=$(HEAD_INFIL_CONFIG)"
	@echo "  HEAD_CONT_CKPT=$(HEAD_CONT_CKPT)"
	@echo "  HEAD_INFIL_CKPT=$(HEAD_INFIL_CKPT)"
	@echo "  HEAD_CONT_PATCH_CONFIG=$(HEAD_CONT_PATCH_CONFIG)"
	@echo "  HEAD_INFIL_PATCH_CONFIG=$(HEAD_INFIL_PATCH_CONFIG)"
	@echo "  HEAD_CONT_PATCH_CKPT=$(HEAD_CONT_PATCH_CKPT)"
	@echo "  HEAD_INFIL_PATCH_CKPT=$(HEAD_INFIL_PATCH_CKPT)"
	@echo "  EMBEDDINGS_PATCH_CONT=$(EMBEDDINGS_PATCH_CONT)"
	@echo "  EMBEDDINGS_PATCH_INFIL=$(EMBEDDINGS_PATCH_INFIL)"
	@echo "  TRACKS_FILE=$(TRACKS_FILE)"
	@echo "  OUTPUT_DIR=$(OUTPUT_DIR)"
	@echo "  PAGES_BRANCH=$(PAGES_BRANCH)"
	@echo "  PAGES_WORKTREE=$(PAGES_WORKTREE)"

setup:
	$(UV) sync --extra eval --extra app

app:
	$(UV) run streamlit run eval/app.py -- --checkpoint_dir $(CHECKPOINT_DIR) --embeddings $(EMBEDDINGS_FILE) --tracks_file $(TRACKS_FILE)

app-patch:
	$(UV) run streamlit run eval/app.py -- --checkpoint_dir $(CHECKPOINT_DIR) --embeddings $(EMBEDDINGS_FILE) --embeddings_patch_cont $(EMBEDDINGS_PATCH_CONT) --embeddings_patch_infil $(EMBEDDINGS_PATCH_INFIL) --tracks_file $(TRACKS_FILE) --cont_head $(HEAD_CONT_PATCH_CKPT) --infil_head $(HEAD_INFIL_PATCH_CKPT)

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
	$(UV) run python eval/embed_tracks.py --config $(TRAIN_CONFIG) --ckpt $(CHECKPOINT_DIR)/$(CHECKPOINT_NAME) --out $(EMBEDDINGS_FILE)

embed-patch:
	@if [ -z "$(PATCH_HEAD)" ]; then echo "Usage: make embed-patch PATCH_HEAD=path/to/head_patch.pt [EMBEDDINGS_FILE=path]"; exit 1; fi
	$(UV) run python eval/embed_tracks.py --config $(TRAIN_CONFIG) --ckpt $(CHECKPOINT_DIR)/$(CHECKPOINT_NAME) --patch_head $(PATCH_HEAD) --out $(EMBEDDINGS_FILE)

embed-patch-cont:
	$(UV) run python eval/embed_tracks.py --config $(TRAIN_CONFIG) --ckpt $(CHECKPOINT_DIR)/$(CHECKPOINT_NAME) --patch_head $(HEAD_CONT_PATCH_CKPT) --out $(EMBEDDINGS_PATCH_CONT)

embed-patch-infil:
	$(UV) run python eval/embed_tracks.py --config $(TRAIN_CONFIG) --ckpt $(CHECKPOINT_DIR)/$(CHECKPOINT_NAME) --patch_head $(HEAD_INFIL_PATCH_CKPT) --out $(EMBEDDINGS_PATCH_INFIL)

train-head-infil:
	$(UV) run python train_head.py --config $(HEAD_INFIL_CONFIG) --out $(HEAD_INFIL_CKPT)

train-head-cont:
	$(UV) run python train_head.py --config $(HEAD_CONT_CONFIG) --out $(HEAD_CONT_CKPT)

train-head-patch-cont:
	$(UV) run python train_patch_head.py --config $(HEAD_CONT_PATCH_CONFIG)

train-head-patch-infil:
	$(UV) run python train_patch_head.py --config $(HEAD_INFIL_PATCH_CONFIG)

journey:
	$(UV) run python eval/generate_playlist.py --head $(HEAD_INFIL_CKPT) --embeddings $(EMBEDDINGS_FILE) --tracks_file $(TRACKS_FILE) --journey $(JOURNEY) --head_weight $(HEAD_WEIGHT) --out_html $(JOURNEY_HTML)

journey-patch:
	$(UV) run python eval/generate_playlist.py --head $(HEAD_INFIL_PATCH_CKPT) --embeddings $(EMBEDDINGS_FILE) --embeddings_patch_infil $(EMBEDDINGS_PATCH_INFIL) --tracks_file $(TRACKS_FILE) --journey $(JOURNEY) --head_weight $(HEAD_WEIGHT) --out_html $(OUTPUT_DIR)/journey_patch.html

playlist:
	@if [ -z "$(SEEDS)" ]; then echo "Usage: make playlist SEEDS=\"TRACK_ID [TRACK_ID ...]\""; exit 1; fi
	$(UV) run python eval/generate_playlist.py --head $(HEAD_CONT_CKPT) --embeddings $(EMBEDDINGS_FILE) --tracks_file $(TRACKS_FILE) --seeds $(SEEDS) --head_weight $(HEAD_WEIGHT) --out_html $(OUT_HTML)

playlist-patch:
	@if [ -z "$(SEEDS)" ]; then echo "Usage: make playlist-patch SEEDS=\"TRACK_ID [TRACK_ID ...]\""; exit 1; fi
	$(UV) run python eval/generate_playlist.py --head $(HEAD_CONT_PATCH_CKPT) --embeddings $(EMBEDDINGS_FILE) --embeddings_patch_cont $(EMBEDDINGS_PATCH_CONT) --tracks_file $(TRACKS_FILE) --seeds $(SEEDS) --head_weight $(HEAD_WEIGHT) --out_html $(OUTPUT_DIR)/playlist_patch.html

examples:
	$(UV) run python eval/generate_examples.py --checkpoint_dir $(CHECKPOINT_DIR) --embeddings $(EMBEDDINGS_FILE) --tracks_file $(TRACKS_FILE) --out_dir $(OUTPUT_DIR)/examples --mp3tovec_model_dir $(MP3TOVEC_MODEL_DIR)

# Patch examples use the base encoder catalog as the head_weight=0 endpoint,
# plus one learned-pool catalog per patch head.
examples-patch:
	$(UV) run python eval/generate_examples.py --checkpoint_dir $(CHECKPOINT_DIR) --head $(HEAD_CONT_PATCH_CKPT) --infil_head $(HEAD_INFIL_PATCH_CKPT) --embeddings $(EMBEDDINGS_FILE) --embeddings_patch_cont $(EMBEDDINGS_PATCH_CONT) --embeddings_patch_infil $(EMBEDDINGS_PATCH_INFIL) --tracks_file $(TRACKS_FILE) --out_dir $(OUTPUT_DIR)/examples-patch

search:
	$(UV) run python eval/search_tracks.py --query "$(QUERY)" --embeddings $(EMBEDDINGS_FILE) --tracks_file $(TRACKS_FILE) --limit $(LIMIT)

viz:
	$(UV) run python eval/explore.py --embeddings $(EMBEDDINGS_FILE) --tracks_file $(TRACKS_FILE) --out $(EXPLORE_HTML) --n_points $(N_POINTS) --export

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
