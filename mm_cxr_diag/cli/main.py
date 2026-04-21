"""mm-cxr-diag CLI — Typer app.

Subcommands:
    prepare-data  — turn raw NIH metadata CSV into preprocessed train/val/test splits
    train         — train a Stage 1 or Stage 2 model
    evaluate      — evaluate a trained checkpoint (or a two-stage pair)
    predict       — run hierarchical prediction on a single image
    serve         — start the FastAPI inference service
"""

from __future__ import annotations

import typer

from mm_cxr_diag import __version__
from mm_cxr_diag.cli.commands import evaluate as evaluate_cmd
from mm_cxr_diag.cli.commands import predict as predict_cmd
from mm_cxr_diag.cli.commands import prepare_data as prepare_data_cmd
from mm_cxr_diag.cli.commands import serve as serve_cmd
from mm_cxr_diag.cli.commands import train as train_cmd

app = typer.Typer(
    name="mm-cxr-diag",
    help="Two-stage multimodal chest X-ray classifier.",
    no_args_is_help=True,
    add_completion=False,
)

app.command(name="prepare-data", help="Split NIH metadata into train/val/test CSVs.")(
    prepare_data_cmd.prepare_data
)
app.command(name="train", help="Train a Stage 1 or Stage 2 model.")(train_cmd.train)
app.command(name="evaluate", help="Evaluate a checkpoint on a test split.")(
    evaluate_cmd.evaluate
)
app.command(name="predict", help="Hierarchical prediction on a single image.")(
    predict_cmd.predict
)
app.command(name="serve", help="Start the FastAPI inference service.")(serve_cmd.serve)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(  # noqa: B008
        False,
        "--version",
        "-V",
        callback=_print_version,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Two-stage multimodal chest X-ray classifier."""


if __name__ == "__main__":
    app()
