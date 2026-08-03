#!/usr/bin/env python
"""Run the isolated CBraMod SEED-V TMLR pipeline."""

from tmlr.config import parse_config
from tmlr.seedv_runner import run


if __name__ == "__main__":
    result = run(parse_config())
    print(result)
