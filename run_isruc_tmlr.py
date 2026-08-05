#!/usr/bin/env python
"""Run or audit the isolated CBraMod ISRUC TMLR pipeline."""

from tmlr.isruc_config import parse_config
from tmlr.isruc_runner import run


if __name__ == "__main__":
    print(run(parse_config()))
