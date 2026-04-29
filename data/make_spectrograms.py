"""
Compute mel spectrograms from MP3 previews and save as PNG images.

Each spectrogram captures the first 5 seconds of a 30-second preview:
  - 96 mel bins  (y_res)
  - 216 time frames at hop_length=512, SR=22050  (x_res)

Usage:
    python data/make_spectrograms.py
    python data/make_spectrograms.py --previews_dir data/previews --spectrograms_dir data/spectrograms
"""
import argparse
import concurrent.futures
import os
from time import sleep

import librosa
import numpy as np
from PIL import Image
from tqdm import tqdm


SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 96
SLICE_FRAMES = 216  # ~5 seconds


def make_spectrogram(mp3_file: str, previews_dir: str, spectrograms_dir: str) -> None:
    track_id = mp3_file[:-4]
    out_path = os.path.join(spectrograms_dir, f"{track_id}.png")
    if os.path.exists(out_path):
        return

    path = os.path.join(previews_dir, mp3_file)
    try:
        y, _ = librosa.load(path, sr=SR, mono=True, duration=6.0)
    except Exception as e:
        print(f"  Load error {mp3_file}: {e}")
        return

    n_samples = SLICE_FRAMES * HOP_LENGTH
    if len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)))
    y = y[:n_samples]

    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, fmax=SR / 2)
    log_S = librosa.power_to_db(S, ref=np.max)

    lo, hi = log_S.min(), log_S.max()
    log_S = (log_S - lo) / (hi - lo) if hi > lo else np.zeros_like(log_S)

    log_S = log_S[:, :SLICE_FRAMES]
    if log_S.shape[1] < SLICE_FRAMES:
        log_S = np.pad(log_S, ((0, 0), (0, SLICE_FRAMES - log_S.shape[1])))

    img = (np.flipud(log_S) * 255).astype(np.uint8)
    tmp_path = f"{out_path}.tmp"
    try:
        Image.fromarray(img, mode="L").save(tmp_path, format="PNG")
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previews_dir", default="data/previews")
    parser.add_argument("--spectrograms_dir", default="data/spectrograms")
    parser.add_argument("--max_workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    os.makedirs(args.spectrograms_dir, exist_ok=True)
    mp3_files = [f for f in os.listdir(args.previews_dir) if f.endswith(".mp3")]
    print(f"MP3 files: {len(mp3_files):,}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(make_spectrogram, f, args.previews_dir, args.spectrograms_dir): f
            for f in tqdm(mp3_files, desc="Submitting jobs")
            if sleep(1e-5) is None
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Computing spectrograms"):
            try:
                future.result()
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"  Error: {e}")


if __name__ == "__main__":
    main()
