#!/usr/bin/env python3
"""Fail unless C++ seam postprocessing matches the frozen Python reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from postprocess_overlap16_auto import read_mhd


PROFILES = {
    "capillary": "phantom",
    "tube": "phantom",
    "mouse": "mouse",
    "zimu": "zimu",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-root", type=Path, required=True)
    parser.add_argument("--cpp-root", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--rtol", type=float, default=1e-12)
    args = parser.parse_args()

    records = []
    passed = True
    for name, profile in PROFILES.items():
        filename = f"overlap16_sigmoid_k12_{profile}_interp.mhd"
        reference = read_mhd(args.python_root / name / filename).astype(np.float64)
        candidate = read_mhd(args.cpp_root / name / filename).astype(np.float64)
        if reference.shape != candidate.shape:
            raise ValueError(f"{name}: shape mismatch {reference.shape} != {candidate.shape}")

        difference = candidate - reference
        max_abs = float(np.max(np.abs(difference)))
        relative_l2 = float(
            np.linalg.norm(difference.ravel())
            / max(np.linalg.norm(reference.ravel()), np.finfo(np.float64).tiny)
        )
        support_mismatch = int(
            np.count_nonzero((reference > 0.0) != (candidate > 0.0))
        )
        sum_relative = float(
            abs(candidate.sum() - reference.sum())
            / max(abs(reference.sum()), np.finfo(np.float64).tiny)
        )
        record_passed = (
            max_abs <= args.atol
            and relative_l2 <= args.rtol
            and support_mismatch == 0
            and sum_relative <= args.rtol
        )
        passed = passed and record_passed
        records.append(
            {
                "dataset": name,
                "profile": profile,
                "max_abs": max_abs,
                "relative_l2": relative_l2,
                "support_mismatch": support_mismatch,
                "sum_relative": sum_relative,
                "passed": record_passed,
            }
        )

    print(json.dumps({"passed": passed, "records": records}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
