"""Runtime settings: paths, device, seed, logging.

Env-driven via ``MM_CXR_*`` variables (and optional ``.env`` file). Loading
is explicit via ``RuntimeSettings()``; there is no module-level instance and
no filesystem side effects at import time. Callers invoke
:func:`ensure_dirs` explicitly when they need the standard directory
layout (e.g. a training run creating its output tree).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Device = Literal["cuda", "cpu", "auto"]
LogLevel = Literal["debug", "info", "warning", "error", "critical"]


class RuntimeSettings(BaseSettings):
    """Process-level runtime configuration.

    Environment variables use the ``MM_CXR_`` prefix. Defaults assume a
    project checkout layout with top-level ``artifacts/``, ``logs/``, and
    ``results/`` directories.
    """

    model_config = SettingsConfigDict(
        env_prefix="MM_CXR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(
        default_factory=lambda: Path.cwd(),
        description="Repo root; other paths are resolved relative to this.",
    )
    artifacts_dir: Path | None = None
    logs_dir: Path | None = None
    runs_dir: Path | None = None
    checkpoints_dir: Path | None = None
    data_dir: Path | None = None

    device: Device = "auto"
    seed: int = 42
    log_level: LogLevel = "info"
    log_file: Path | None = None

    def resolve_paths(self) -> RuntimeSettings:
        """Fill in any unset ``*_dir`` fields with project-relative defaults.

        Returns ``self`` for chaining. Does NOT create directories — that is
        :func:`ensure_dirs`' job.
        """
        root = self.project_root.resolve()
        if self.artifacts_dir is None:
            self.artifacts_dir = root / "artifacts"
        if self.logs_dir is None:
            self.logs_dir = root / "logs"
        if self.runs_dir is None:
            self.runs_dir = root / "results" / "runs"
        if self.checkpoints_dir is None:
            self.checkpoints_dir = root / "results" / "checkpoints"
        if self.data_dir is None:
            self.data_dir = root / "artifacts"
        if self.log_file is None:
            self.log_file = self.logs_dir / "app.log"
        return self


def ensure_dirs(settings: RuntimeSettings) -> None:
    """Create the standard artifact / logs / runs directories.

    Callers invoke this explicitly from CLI entrypoints — it is never run
    at import time.
    """
    settings.resolve_paths()
    for d in (
        settings.artifacts_dir,
        settings.logs_dir,
        settings.runs_dir,
        settings.checkpoints_dir,
    ):
        if d is not None:
            d.mkdir(parents=True, exist_ok=True)
