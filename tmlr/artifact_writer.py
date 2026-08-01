"""Immutable, machine-readable FACED run artifact writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch

from .provenance import json_default


class ArtifactWriter:
    def __init__(self, output_root: str | Path, run_id: str, overwrite: bool = False) -> None:
        self.run_dir = Path(output_root) / run_id
        if self.run_dir.exists() and not overwrite:
            raise FileExistsError(
                f"Run directory already exists: {self.run_dir}; choose a new run_id or explicit overwrite."
            )
        self.run_dir.mkdir(parents=True, exist_ok=bool(overwrite))

    def write_json(self, name: str, payload: Dict[str, Any]) -> None:
        path = self.run_dir / name
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite artifact {path}")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")

    def append_jsonl(self, name: str, payload: Dict[str, Any]) -> None:
        path = self.run_dir / name
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=json_default) + "\n")

    def save_model(self, model: torch.nn.Module, name: str = "best_model.pt") -> None:
        path = self.run_dir / name
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite artifact {path}")
        torch.save(model.state_dict(), path)
