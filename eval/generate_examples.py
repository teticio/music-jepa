"""
Generate a small gallery of continuation playlists and infill journeys.

Examples are written as standalone HTML files under outputs/examples/.
Each playlist/journey is run at head_weight 0, 0.5, 1.0 and with Mp3ToVec.
"""
import argparse
from html import escape
import subprocess
from pathlib import Path


CONTINUATION_EXAMPLES = [
    ("electronic", ["69kOkLUCkxIZYexIgSG8rq"]),   # Daft Punk - Get Lucky
    ("rock",       ["4CeeEOM32jQcH3eN9Q2dGj"]),   # Nirvana - Smells Like Teen Spirit
    ("reggae",     ["75FYqcxt1YEAtqDLrOeIJn"]),   # Bob Marley - Three Little Birds
    ("jazz",       ["6oVY50pmdXqLNVeK8bzomn"]),   # John Coltrane - My Favorite Things
    ("classical",  ["3oHSL6pt9LpNrQZuQGu9wL"]),   # Mozart - Requiem: Lacrimosa
]

JOURNEY_EXAMPLES = [
    (
        "classical_to_techno",
        ["3E14w5yvAuK1YHnfi2zpdl", "4ua0IepBEISCWwF8dTJvcU"],
        None,
    ),  # Beethoven Symphony No. 5 -> deadmau5 - Ghosts 'n' Stuff
    (
        "reggae_to_house",
        ["75FYqcxt1YEAtqDLrOeIJn", "0DiWol3AO6WpXZgp0goxAV"],
        None,
    ),  # Bob Marley - Three Little Birds -> Daft Punk - One More Time
    (
        "jazz_to_rock",
        ["6oVY50pmdXqLNVeK8bzomn", "4CeeEOM32jQcH3eN9Q2dGj"],
        None,
    ),  # John Coltrane - My Favorite Things -> Nirvana - Smells Like Teen Spirit
    (
        "classical_jazz_funk_disco_house_techno",
        [
            "5N82c9RY2k4VeAel1pl5bJ",
            "4vLYewWIvqHfKtJDk8c8tq",
            "5XeSAezNDk9tuw3viiCbZ3",
            "7B7lf3sIze5VR2WuYttn18",
            "5sJiLlgQKBL81QCTOkoLB5",
            "7xQYVjs4wZNdCwO0EeAWMC",
        ],
        10,
    ),  # Vivaldi -> Miles Davis -> James Brown -> Donna Summer -> Inner City -> Underworld
]

HEAD_WEIGHTS = [0.0, 0.5, 1.0]


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def title_for(path: Path) -> str:
    name = path.stem
    if name.startswith("playlist_"):
        name = name.removeprefix("playlist_")
        kind = "Playlist"
    elif name.startswith("journey_"):
        name = name.removeprefix("journey_")
        kind = "Journey"
    else:
        kind = "Example"

    method = "head"
    for suffix, label in [
        ("_mp3tovec", "mp3tovec"),
        ("_hw100", "hw=1.0"),
        ("_hw50", "hw=0.5"),
        ("_hw0", "hw=0.0"),
    ]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            method = label
            break

    label = name.replace("_", " ").title()
    return f"{kind}: {label} ({method})"


def example_title(filename: str) -> str:
    return title_for(Path(filename))


def write_index(out_dir: Path) -> None:
    html_files = sorted(
        path for path in out_dir.glob("*.html") if path.name != "index.html"
    )
    groups = [
        ("Playlists", [p for p in html_files if p.name.startswith("playlist_")]),
        ("Journeys",  [p for p in html_files if p.name.startswith("journey_")]),
        ("Other",     [p for p in html_files if not p.name.startswith(("playlist_", "journey_"))]),
    ]
    sections = []
    for heading, files in groups:
        if not files:
            continue
        links = "\n".join(
            f'        <li><a href="{escape(p.name)}">{escape(title_for(p))}</a></li>'
            for p in files
        )
        sections.append(
            "\n".join([
                "    <section>",
                f"      <h2>{escape(heading)}</h2>",
                "      <ul>",
                links,
                "      </ul>",
                "    </section>",
            ])
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Generated examples</title>
  <style>
    :root {{
      color-scheme: light dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #101114;
      color: #f2f2f0;
    }}
    body {{
      margin: 0;
      padding: 32px;
      background: #101114;
    }}
    main {{
      max-width: 900px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 24px;
      font-size: 28px;
    }}
    h2 {{
      margin: 28px 0 12px;
      font-size: 18px;
      color: #d7dbe4;
    }}
    ul {{
      margin: 0;
      padding: 0;
      list-style: none;
      border: 1px solid #2b2f38;
      background: #181a1f;
    }}
    li + li {{
      border-top: 1px solid #2b2f38;
    }}
    a {{
      display: block;
      padding: 12px 14px;
      color: #8fb3ff;
      text-decoration: none;
    }}
    a:hover {{
      background: #20232a;
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Generated examples</h1>
{chr(10).join(sections)}
  </main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html)


def clean_generated_examples(out_dir: Path) -> None:
    for pattern in ("playlist_*.html", "journey_*.html"):
        for path in out_dir.glob(pattern):
            path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs/examples")
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--between", type=int, default=9)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--index_only", action="store_true")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--embeddings", default="embeddings/embeddings.npy",
                        help="Catalog used for continuation examples (and journeys if --journey_embeddings is unset)")
    parser.add_argument("--journey_embeddings", default=None,
                        help="Override catalog used for journey examples. Useful when the cont and infill heads "
                             "have different pools (e.g. patch heads) so each needs its own catalog.")
    parser.add_argument("--tracks_file", default="data/tracks_dedup.csv")
    parser.add_argument("--mp3tovec_model_dir", default=None)
    parser.add_argument("--head", default=None,
                        help="Continuation head ckpt. Defaults to <checkpoint_dir>/continuation_head.pt")
    parser.add_argument("--infil_head", default=None,
                        help="Infill head ckpt. Defaults to <checkpoint_dir>/infill_head.pt")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.index_only:
        write_index(out_dir)
        print(f"Wrote {out_dir / 'index.html'}")
        return
    clean_generated_examples(out_dir)

    journey_embeddings = args.journey_embeddings or args.embeddings
    playlist_base = ["uv", "run", "python", "eval/generate_playlist.py",
                     "--embeddings", args.embeddings, "--tracks_file", args.tracks_file]
    journey_base = ["uv", "run", "python", "eval/generate_playlist.py",
                    "--embeddings", journey_embeddings, "--tracks_file", args.tracks_file]
    device_args = ["--device", args.device] if args.device else []

    checkpoint_dir = Path(args.checkpoint_dir)
    cont_head = Path(args.head) if args.head else checkpoint_dir / "continuation_head.pt"
    infil_head = Path(args.infil_head) if args.infil_head else checkpoint_dir / "infill_head.pt"

    def run_playlist(name, seeds, hw_label, extra):
        run(playlist_base + extra + [
            "--seeds", *seeds,
            "--size", str(args.size),
            "--noise", str(args.noise),
            "--out_html", str(out_dir / f"playlist_{name}_{hw_label}.html"),
            "--title", example_title(f"playlist_{name}_{hw_label}.html"),
        ] + device_args)

    def run_journey(name, waypoints, between, hw_label, extra):
        run(journey_base + extra + [
            "--journey", *waypoints,
            "--between", str(between),
            "--noise", str(args.noise),
            "--out_html", str(out_dir / f"journey_{name}_{hw_label}.html"),
            "--title", example_title(f"journey_{name}_{hw_label}.html"),
        ] + device_args)

    if cont_head.exists():
        for name, seeds in CONTINUATION_EXAMPLES:
            for hw in HEAD_WEIGHTS:
                hw_label = f"hw{int(hw * 100)}"
                run_playlist(name, seeds, hw_label,
                             ["--head", str(cont_head), "--head_weight", str(hw)])
    else:
        print(f"Skipping continuation examples: missing {cont_head}")

    if infil_head.exists():
        for name, waypoints, between_override in JOURNEY_EXAMPLES:
            between = between_override or args.between
            for hw in HEAD_WEIGHTS:
                hw_label = f"hw{int(hw * 100)}"
                run_journey(name, waypoints, between, hw_label,
                            ["--head", str(infil_head), "--head_weight", str(hw)])
    else:
        print(f"Skipping journey examples: missing {infil_head}")

    if args.mp3tovec_model_dir:
        mp3tovec_args = ["--mp3tovec_model_dir", args.mp3tovec_model_dir]
        for name, seeds in CONTINUATION_EXAMPLES:
            run_playlist(name, seeds, "mp3tovec", mp3tovec_args)
        for name, waypoints, between_override in JOURNEY_EXAMPLES:
            between = between_override or args.between
            run_journey(name, waypoints, between, "mp3tovec", mp3tovec_args)
    else:
        print("Skipping mp3tovec examples: pass --mp3tovec_model_dir to enable")

    write_index(out_dir)
    print(f"Wrote examples to {out_dir}")
    print(f"Wrote {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
