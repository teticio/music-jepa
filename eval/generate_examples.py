"""
Generate a small gallery of continuation playlists and infill journeys.

Examples are written as standalone HTML files under outputs/examples/.
"""
import argparse
from html import escape
import subprocess
from pathlib import Path


CONTINUATION_EXAMPLES = [
    ("electronic", ["69kOkLUCkxIZYexIgSG8rq"]),  # Daft Punk - Get Lucky
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
    for suffix in ["_embeddings", "_track2vec"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            method = suffix.removeprefix("_")
            break

    label = name.replace("_", " ").title()
    return f"{kind}: {label} ({method})"


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs/examples")
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--between", type=int, default=9)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--index_only", action="store_true")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.index_only:
        write_index(out_dir)
        print(f"Wrote {out_dir / 'index.html'}")
        return

    base = ["uv", "run", "python", "eval/generate_playlist.py"]
    device_args = ["--device", args.device] if args.device else []

    checkpoint_dir = Path(args.checkpoint_dir)
    cont_head = checkpoint_dir / "continuation_head.pt"
    if cont_head.exists():
        for name, seeds in CONTINUATION_EXAMPLES:
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
                    str(args.noise),
                    "--out_html",
                    str(out_dir / f"playlist_{name}.html"),
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
            ]
            + device_args
        )

    for name, seeds in CONTINUATION_EXAMPLES:
        run(
            base
            + [
                "--method",
                "track2vec",
                "--seeds",
                *seeds,
                "--size",
                str(args.size),
                "--noise",
                str(args.noise),
                "--out_html",
                str(out_dir / f"playlist_{name}_track2vec.html"),
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
            ]
            + device_args
        )

    for name, waypoints, between_override in JOURNEY_EXAMPLES:
        between = between_override or args.between
        run(
            base
            + [
                "--method",
                "track2vec",
                "--journey",
                *waypoints,
                "--between",
                str(between),
                "--noise",
                str(args.noise),
                "--out_html",
                str(out_dir / f"journey_{name}_track2vec.html"),
            ]
        )

    write_index(out_dir)
    print(f"Wrote examples to {out_dir}")
    print(f"Wrote {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
