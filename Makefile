UV := uv

.PHONY: setup data data-sample sample previews previews-sample spectrograms train train1 embed viz viz-nn tensorboard help

help:
	@echo "Music JEPA - step by step:"
	@echo "  make setup         Create venv and install dependencies"
	@echo "  make data          Download previews + spectrograms for the FULL dataset"
	@echo "  make data-sample   Same pipeline restricted to a 2000-playlist sample"
	@echo "  make train         Train on 2x GPU (torchrun)"
	@echo "  make train1        Train on 1x GPU"
	@echo "  make embed         Extract embeddings from last checkpoint"
	@echo "  make viz           Nearest-neighbour report + t-SNE"
	@echo "  make tensorboard   Launch TensorBoard on logs/"

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
	$(UV) run torchrun --nproc_per_node=2 train.py

train1:
	$(UV) run python train.py --config configs/train.yaml

embed:
	$(UV) run python eval/embed_tracks.py --ckpt checkpoints/last.ckpt

viz:
	$(UV) run python eval/explore.py --embeddings embeddings.npy

viz-nn:
	$(UV) run python eval/visualize.py --embeddings embeddings.npy --tsne

tensorboard:
	$(UV) run tensorboard --logdir logs/
