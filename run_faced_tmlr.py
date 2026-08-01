#!/usr/bin/env python
"""Run or audit the isolated CBraMod FACED TMLR pipeline."""

from tmlr.config import parse_config
from tmlr.faced_runner import run


if __name__ == "__main__":
    result = run(parse_config())
    print(result)
