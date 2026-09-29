"""Run the validation suite: zvCFD against exact solutions and standard benchmarks.

    python benchmarks/validation/run.py                       # every case
    python benchmarks/validation/run.py --cases womersley,taylor_green

Writes ``benchmarks/results/validation/<case>.json``; ``figures.py`` draws
them and ``docs/benchmarks/validation.md`` discusses them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import cases  # noqa: E402

OUT = ROOT / "benchmarks" / "results" / "validation"


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3e}" if (v != 0 and (abs(v) < 1e-2 or abs(v) >= 1e4)) else f"{v:.4g}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=",".join(cases.CASES))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.cases.split(","):
        t0 = time.time()
        res = cases.CASES[name]()
        res["total_s"] = time.time() - t0
        (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))
        print(f"== {name} ({res['total_s']:.0f} s)")
        for key in ("rows", "plane", "pipe"):
            for r in res.get(key, []):
                print("  " + ("" if key == "rows" else f"{key:<6}") +
                      "  ".join(f"{k}={_fmt(v)}" for k, v in r.items()
                                if k != "wall_s" and not isinstance(v, (dict, list))),
                      flush=True)
        for k, v in res.items():
            if k.startswith("order"):
                print(f"  {k}: {v if isinstance(v, dict) else round(v, 2)}")


if __name__ == "__main__":
    main()
