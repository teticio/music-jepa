"""
Generate a small gallery of continuation playlists and infill journeys.

Examples are written as standalone HTML files under outputs/examples/.
"""
import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs/examples")
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--between", type=int, default=9)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = ["uv", "run", "python", "eval/generate_playlist.py"]
    device_args = ["--device", args.device] if args.device else []

    cont_head = Path("checkpoints/continuation_head.pt")
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

    infill_head = Path("checkpoints/infill_head.pt")
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

    print(f"Wrote examples to {out_dir}")


if __name__ == "__main__":
    main()
