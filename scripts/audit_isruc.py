#!/usr/bin/env python
"""Run the CBraMod-only serialized ISRUC provenance audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tmlr.isruc_provenance import audit_isruc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/data/neurogroup/mingyangjiang/data/ISRUC")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = audit_isruc(args.data_root)
    if args.output:
        path = Path(args.output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "dataset": report["dataset"],
        "split_counts": report["split_counts"],
        "label_vocabulary": report["label_vocabulary"],
        "input_scale_divisor": report["input_scale_divisor"],
        "all_key_sha256": report["all_key_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
