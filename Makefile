UV := uv

.PHONY: setup data train embed viz tensorboard help

help:
	@echo "Music JEPA - step by step:"
	@echo "  make setup         Create venv and install dependencies"
	@echo "  make sample        Sample playlists/tracks subset"
	@echo "  make previews      Download MP3 previews from Spotify CDN"
	@echo "  make spectrograms  Compute mel spectrogram PNGs"
	@echo "  make data          All three data steps"
	@echo "  make train         Train on 2x GPU (torchrun)"
	@echo "  make train1        Train on 1x GPU"
	@echo "  make embed         Extract embeddings from last checkpoint"
	@echo "  make viz           Nearest-neighbour report + t-SNE"
	@echo "  make tensorboard   Launch TensorBoard on logs/"

setup:
	$(UV) sync --extra eval

sample:
	$(UV) run python data/sample_data.py --n_playlists 2000

previews:
	$(UV) run python data/download_previews.py --max_workers 32

spectrograms:
	$(UV) run python data/make_spectrograms.py

data: sample previews spectrograms

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
