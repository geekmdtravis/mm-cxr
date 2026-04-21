"""`mm-cxr-diag prepare-data`: split raw NIH metadata into train/val/test CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from mm_cxr_diag.data.preprocessing import (
    create_working_tabular_df,
    randomize_df,
    train_test_split,
)


def prepare_data(
    input_csv: Annotated[
        Path,
        typer.Option(
            "--input-csv",
            exists=True,
            readable=True,
            help="Raw NIH ChestX-ray14 metadata CSV (Data_Entry_2017.csv).",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Directory to write train.csv, val.csv, test.csv into. "
            "Created if missing.",
        ),
    ],
    images_dir: Annotated[
        Path | None,
        typer.Option(
            "--images-dir",
            exists=True,
            file_okay=False,
            dir_okay=True,
            help="Directory containing the PNG images. Optional; only used to "
            "sanity-check that at least one referenced image exists.",
        ),
    ] = None,
    val_size: Annotated[
        float,
        typer.Option("--val-size", min=0.0, max=0.9, help="Fraction for val split."),
    ] = 0.1,
    test_size: Annotated[
        float,
        typer.Option("--test-size", min=0.0, max=0.9, help="Fraction for test split."),
    ] = 0.2,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Read the raw NIH CSV, one-hot-encode pathology labels, and write 3 splits."""
    typer.echo(f"Reading {input_csv}")
    raw = pd.read_csv(input_csv)

    if images_dir is not None:
        first_img = raw["Image Index"].iloc[0]
        if not (images_dir / first_img).exists():
            typer.secho(
                f"Warning: {first_img} not found in {images_dir}. "
                "Downstream training will fail unless images are accessible.",
                fg=typer.colors.YELLOW,
            )

    typer.echo("Preprocessing tabular features and one-hot label columns")
    working = create_working_tabular_df(raw)
    working = randomize_df(working, seed=seed)

    typer.echo(
        f"Splitting into train / val / test with val={val_size} test={test_size} "
        f"seed={seed}"
    )
    train_and_val, test_df = train_test_split(working, test_size=test_size, seed=seed)
    # val_size is expressed as a fraction of the full set — convert to a
    # fraction of (train + val) for the second split.
    val_fraction_of_trainval = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_and_val, test_size=val_fraction_of_trainval, seed=seed
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_train = output_dir / "train.csv"
    out_val = output_dir / "val.csv"
    out_test = output_dir / "test.csv"

    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)
    test_df.to_csv(out_test, index=False)

    typer.echo(
        f"Wrote {len(train_df)} train / {len(val_df)} val / {len(test_df)} test rows "
        f"to {output_dir}"
    )
