"""
Render learned attention-pool maps for the patch playlist heads.

The maps show the weights used by PatchPlaylistHead.pool when it collapses the
encoder's patch tokens into one track embedding. They are not the transformer
encoder's internal attention weights, but they are useful for seeing which
time/frequency regions each patch head uses for the pooled catalog vector.
"""
import argparse
import base64
from html import escape
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image

from eval.generate_examples import CONTINUATION_EXAMPLES, JOURNEY_EXAMPLES
from jepa.model import build_model
from jepa.module import JEPAModule
from jepa.playlist_head import describe_track, load_head, load_tracks


def example_track_ids() -> list[str]:
    ids = []
    for _, seeds in CONTINUATION_EXAMPLES:
        ids.extend(seeds)
    for _, waypoints, _ in JOURNEY_EXAMPLES:
        ids.extend(waypoints)
    return list(dict.fromkeys(ids))


def load_encoder(ckpt_path: str, config_path: str, device: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    module = JEPAModule.load_from_checkpoint(
        ckpt_path,
        model=build_model(cfg),
        map_location=device,
    )
    encoder = module.model.encoder.to(device).eval()
    return encoder, cfg


def load_spec(path: Path, img_size: tuple[int, int], device: str) -> torch.Tensor:
    transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    img = Image.open(path).convert("L")
    return transform(img).unsqueeze(0).to(device)


@torch.no_grad()
def pool_weight_map(head, patch_tokens: torch.Tensor, grid_size: tuple[int, int]) -> np.ndarray:
    pool = head.module.pool if hasattr(head, "module") else head.pool
    q = pool.query.expand(patch_tokens.shape[0], -1, -1)
    _, weights = pool.attn(
        q,
        patch_tokens,
        patch_tokens,
        need_weights=True,
        average_attn_weights=False,
    )
    weights = weights.mean(dim=1).squeeze(1).squeeze(0)
    return weights.reshape(grid_size).detach().cpu().numpy()


def image_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def heatmap_html(weights: np.ndarray) -> str:
    values = weights.astype("float64")
    lo = float(values.min())
    hi = float(values.max())
    scaled = (values - lo) / max(hi - lo, 1e-12)
    cells = []
    for raw, norm in zip(values.flatten(), scaled.flatten()):
        alpha = 0.12 + 0.88 * float(norm)
        cells.append(
            '<span class="cell" '
            f'style="background: rgba(255, 92, 48, {alpha:.3f})" '
            f'title="{raw:.6f}"></span>'
        )
    return "\n".join(cells)


def stats(weights: np.ndarray) -> tuple[float, float]:
    flat = weights.flatten().astype("float64")
    flat = flat / max(float(flat.sum()), 1e-12)
    entropy = -float(np.sum(flat * np.log(np.maximum(flat, 1e-12))))
    effective_patches = float(np.exp(entropy))
    return float(flat.max()), effective_patches


def render_html(rows, out_html: Path) -> None:
    cards = []
    for row in rows:
        head_sections = []
        for head_name, weights in row["maps"]:
            peak, effective = stats(weights)
            head_sections.append(
                f"""
        <section class="map-block">
          <h3>{escape(head_name)}</h3>
          <div class="heatmap">{heatmap_html(weights)}</div>
          <p class="metric">peak weight {peak:.4f} &middot; effective patches {effective:.1f}</p>
        </section>
"""
            )
        cards.append(
            f"""
    <article class="track-card">
      <header>
        <h2>{escape(row["label"])}</h2>
        <code>{escape(row["track_id"])}</code>
      </header>
      <div class="track-layout">
        <img src="{row["image"]}" alt="Spectrogram for {escape(row["label"])}">
        <div class="maps">
{''.join(head_sections)}
        </div>
      </div>
    </article>
"""
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Patch Head Pool Maps</title>
  <style>
    :root {{
      color-scheme: light dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #101114;
      color: #f2f2f0;
    }}
    body {{ margin: 0; padding: 28px; background: #101114; }}
    main {{ max-width: 1180px; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .note {{ margin: 0 0 24px; color: #b9bec8; max-width: 900px; line-height: 1.5; }}
    .track-card {{ border: 1px solid #2b2f38; background: #181a1f; margin: 0 0 22px; padding: 16px; }}
    header {{ display: flex; gap: 12px; align-items: baseline; justify-content: space-between; }}
    h2 {{ margin: 0 0 14px; font-size: 18px; }}
    h3 {{ margin: 0 0 8px; font-size: 14px; color: #d7dbe4; }}
    code {{ color: #aeb7c8; font-size: 12px; }}
    .track-layout {{ display: grid; grid-template-columns: minmax(260px, 420px) 1fr; gap: 18px; align-items: start; }}
    img {{ width: 100%; image-rendering: auto; border: 1px solid #30343d; background: #0d0e11; }}
    .maps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    .heatmap {{ display: grid; grid-template-columns: repeat(27, 1fr); aspect-ratio: 27 / 12; border: 1px solid #30343d; background: #15171c; }}
    .cell {{ min-width: 0; min-height: 0; border-right: 1px solid rgba(0, 0, 0, 0.18); border-bottom: 1px solid rgba(0, 0, 0, 0.18); }}
    .metric {{ margin: 8px 0 0; color: #aeb7c8; font-size: 12px; }}
    @media (max-width: 760px) {{
      body {{ padding: 16px; }}
      .track-layout {{ grid-template-columns: 1fr; }}
      header {{ display: block; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Patch Head Pool Maps</h1>
    <p class="note">
      These heatmaps show learned attention-pool weights over the encoder patch grid.
      Brighter cells contribute more when the patch head pools patch tokens into a
      single catalog embedding. This is the patch-head pooling step, not the ViT
      encoder's internal self-attention.
    </p>
{''.join(cards)}
  </main>
</body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/last.ckpt")
    parser.add_argument("--config", default="configs/encoder.yaml")
    parser.add_argument("--spectrograms_dir", default="data/spectrograms")
    parser.add_argument("--tracks_file", default="data/tracks_dedup.csv")
    parser.add_argument("--cont_head", default="checkpoints/continuation_head_patch.pt")
    parser.add_argument("--infil_head", default="checkpoints/infill_head_patch.pt")
    parser.add_argument("--out_html", default="outputs/patch_head_maps.html")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    encoder, cfg = load_encoder(args.ckpt, args.config, args.device)
    img_size = tuple(cfg["data"]["img_size"])
    patch_size = tuple(cfg["data"]["patch_size"])
    grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

    heads = []
    for name, path in [
        ("Continuation patch head", args.cont_head),
        ("Infill patch head", args.infil_head),
    ]:
        if Path(path).exists():
            head, _ = load_head(path, device=args.device)
            if not hasattr(head, "pool_tracks"):
                raise SystemExit(f"{path} is not a patch head")
            heads.append((name, head.eval()))
        else:
            print(f"Skipping missing {name}: {path}")
    if not heads:
        raise SystemExit("No patch heads found")

    tracks_df = load_tracks(args.tracks_file)
    rows = []
    for track_id in example_track_ids():
        spec_path = Path(args.spectrograms_dir) / f"{track_id}.png"
        if not spec_path.exists():
            print(f"Skipping missing spectrogram: {spec_path}")
            continue
        spec = load_spec(spec_path, img_size, args.device)
        with torch.no_grad():
            patch_tokens = encoder(spec)
        rows.append({
            "track_id": track_id,
            "label": describe_track(track_id, tracks_df),
            "image": image_data_uri(spec_path),
            "maps": [
                (head_name, pool_weight_map(head, patch_tokens, grid_size))
                for head_name, head in heads
            ],
        })

    if not rows:
        raise SystemExit("No example spectrograms were found")
    render_html(rows, Path(args.out_html))
    print(f"Wrote {args.out_html}")


if __name__ == "__main__":
    main()
