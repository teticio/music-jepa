"""
Generate a small gallery of continuation playlists and infill journeys.

Examples are written as standalone HTML files under outputs/examples/.
"""
import argparse
from html import escape
import subprocess
from pathlib import Path


ELECTRONIC_SEEDS = [
    "69kOkLUCkxIZYexIgSG8rq",
]  # Daft Punk - Get Lucky

HEAD_CONTINUATION_DRIFT_EXAMPLES = [
    ("electronic_drift_0", ELECTRONIC_SEEDS, 0.0),
    ("electronic_drift_35", ELECTRONIC_SEEDS, 0.35),
    ("electronic_drift_70", ELECTRONIC_SEEDS, 0.70),
]

HEAD_CONTINUATION_NOISE_EXAMPLES = [
    ("electronic_noise_0", ELECTRONIC_SEEDS, 0.0, 0.0),
    ("electronic_noise_25", ELECTRONIC_SEEDS, 0.0, 0.25),
    ("electronic_noise_50", ELECTRONIC_SEEDS, 0.0, 0.50),
]

CONTINUATION_EXAMPLES = [
    ("electronic", ELECTRONIC_SEEDS),
    ("rock", ["4CeeEOM32jQcH3eN9Q2dGj"]),  # Nirvana - Smells Like Teen Spirit
    ("reggae", ["75FYqcxt1YEAtqDLrOeIJn"]),  # Bob Marley - Three Little Birds
    ("jazz", ["6oVY50pmdXqLNVeK8bzomn"]),  # John Coltrane - My Favorite Things
    ("classical", ["3oHSL6pt9LpNrQZuQGu9wL"]),  # Mozart - Requiem: Lacrimosa
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
    for suffix in ["_embeddings", "_mp3tovec"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            method = suffix.removeprefix("_")
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
        ("Playlists", [path for path in html_files if path.name.startswith("playlist_")]),
        ("Journeys", [path for path in html_files if path.name.startswith("journey_")]),
        ("Other", [
            path
            for path in html_files
            if not path.name.startswith(("playlist_", "journey_"))
        ]),
    ]
    sections = []
    for heading, files in groups:
        if not files:
            continue
        links = "\n".join(
            f'        <li><a href="{escape(path.name)}">{escape(title_for(path))}</a></li>'
            for path in files
        )
        sections.append(
            "\n".join(
                [
                    f"    <section>",
                    f"      <h2>{escape(heading)}</h2>",
                    f"      <ul>",
                    links,
                    f"      </ul>",
                    f"    </section>",
                ]
            )
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
    parser.add_argument("--tracks_file", default="data/tracks_dedup.csv")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.index_only:
        write_index(out_dir)
        print(f"Wrote {out_dir / 'index.html'}")
        return
    clean_generated_examples(out_dir)

    base = ["uv", "run", "python", "eval/generate_playlist.py", "--tracks_file", args.tracks_file]
    device_args = ["--device", args.device] if args.device else []

    checkpoint_dir = Path(args.checkpoint_dir)
    cont_head = checkpoint_dir / "continuation_head.pt"
    if cont_head.exists():
        head_examples = [
            (name, seeds, drift, args.noise)
            for name, seeds, drift in HEAD_CONTINUATION_DRIFT_EXAMPLES
        ] + HEAD_CONTINUATION_NOISE_EXAMPLES + [
            (name, seeds, 0.0, args.noise)
            for name, seeds in CONTINUATION_EXAMPLES
            if name != "electronic"
        ]
        for name, seeds, drift, noise in head_examples:
            run(
                base
                + [
                    "--head",
                    str(cont_head),
                    "--seeds",
                    *seeds,
                    "--size",
                    str(args.size),
                    "--noise",
                    str(noise),
                    "--drift",
                    str(drift),
                    "--out_html",
                    str(out_dir / f"playlist_{name}.html"),
                    "--title",
                    example_title(f"playlist_{name}.html"),
                ]
                + device_args
            )
    else:
        print(f"Skipping continuation examples: missing {cont_head}")

    for name, seeds in CONTINUATION_EXAMPLES:
        run(
            base
            + [
                "--method",
                "embeddings",
                "--seeds",
                *seeds,
                "--size",
                str(args.size),
                "--noise",
                str(args.noise),
                "--out_html",
                str(out_dir / f"playlist_{name}_embeddings.html"),
                "--title",
                example_title(f"playlist_{name}_embeddings.html"),
            ]
            + device_args
        )

    for name, seeds in CONTINUATION_EXAMPLES:
        run(
            base
            + [
                "--method",
                "mp3tovec",
                "--seeds",
                *seeds,
                "--size",
                str(args.size),
                "--noise",
                str(args.noise),
                "--out_html",
                str(out_dir / f"playlist_{name}_mp3tovec.html"),
                "--title",
                example_title(f"playlist_{name}_mp3tovec.html"),
            ]
        )

    infill_head = checkpoint_dir / "infill_head.pt"
    if infill_head.exists():
        for name, waypoints, between_override in JOURNEY_EXAMPLES:
            between = between_override or args.between
            run(
                base
                + [
                    "--head",
                    str(infill_head),
                    "--journey",
                    *waypoints,
                    "--between",
                    str(between),
                    "--noise",
                    str(args.noise),
                    "--out_html",
                    str(out_dir / f"journey_{name}.html"),
                    "--title",
                    example_title(f"journey_{name}.html"),
                ]
                + device_args
            )
    else:
        print(f"Skipping journey examples: missing {infill_head}")

    for name, waypoints, between_override in JOURNEY_EXAMPLES:
        between = between_override or args.between
        run(
            base
            + [
                "--method",
                "embeddings",
                "--journey",
                *waypoints,
                "--between",
                str(between),
                "--noise",
                str(args.noise),
                "--out_html",
                str(out_dir / f"journey_{name}_embeddings.html"),
                "--title",
                example_title(f"journey_{name}_embeddings.html"),
            ]
            + device_args
        )

    for name, waypoints, between_override in JOURNEY_EXAMPLES:
        between = between_override or args.between
        run(
            base
            + [
                "--method",
                "mp3tovec",
                "--journey",
                *waypoints,
                "--between",
                str(between),
                "--noise",
                str(args.noise),
                "--out_html",
                str(out_dir / f"journey_{name}_mp3tovec.html"),
                "--title",
                example_title(f"journey_{name}_mp3tovec.html"),
            ]
        )

    write_index(out_dir)
    print(f"Wrote examples to {out_dir}")
    print(f"Wrote {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
