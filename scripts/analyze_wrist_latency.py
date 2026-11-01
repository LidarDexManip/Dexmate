#!/usr/bin/env python3
"""Print the bottleneck and wrist comparison from a hardware latency report."""

import argparse
import json
from pathlib import Path

from wuji_teleop.latency import format_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    print(format_report(json.loads(args.report.read_text(encoding="utf-8"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
