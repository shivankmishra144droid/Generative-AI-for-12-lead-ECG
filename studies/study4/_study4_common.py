"""Shared constants and guards for study 4 (realism of the hybrid generator, report/WORKFLOW_4.md).

Every study 4 output lives under generator_comparison/study4/. The code that studies 1-3 depend on (and that study 3's
pool builder imports while it runs) is read-only: `check_frozen()` runs at the start and end of every study 4 script
and stops if any of it, or any study 1/2 artifact, changed. Study 4 code lives in new modules only
(synthetic_generation/realism/, _study4_*.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import _study3_common as C3

ROOT = C3.ROOT
DATA = C3.DATA
S4 = ROOT / "generator_comparison" / "study4"
sha256 = C3.sha256

# Patient-level split of the train split (D: _study4_split.py). Shares of patients, not recordings.
PARTS = ("fit", "dev", "judge")
PART_SHARE = (0.4, 0.3, 0.3)
SPLIT_SEED = 4004

FROZEN_CODE = {
    "synthetic_generation/assembly.py": "3c2292583385d9462d3fc866712cc704d0f9b9d3e37dbbc0540bb46b932d323f",
    "synthetic_generation/beats.py": "afb28711c908f3a0a993d1c1f35f47a58af06c246f7cff0f85f97dae0e5362fc",
    "synthetic_generation/sample.py": "4ddb61cbeddfd70c33ee1dc8419569a9a719ab8b1bb95889139c62f969ac333e",
    "synthetic_generation/utils.py": "289258accfb7b3711418ed3c81758fb042126c425ebf8753bb23d3eff089dcc6",
    "synthetic_generation/leads.py": "9f6dfd9f257fc8b58e7d3f96cebc3c84545bb6b99ec20c5764cdeac51af0f678",
    "synthetic_generation/model.py": "dabd1f7afb65a27b43071f526d73dfef21780b66fd4b74164ffd55d1f6ccd937",
    "studies/beat_generator/_beat_gen_pool.py": "17a8795f00f65107f6a172a5fb76192e4cb37302ec7d47dda07792bce69e36d6",
    "generator_comparison/realism_detector.py": "e2e82641fa42565f187500d0704eb751a9328cd9f29b7d6453787dc70f63c7c6",
    "generator_comparison/eval_metrics.py": "033e5efa0ba2c123639f5c902acaf64ae33d72281c13e3173daddb8287de697b",
}


def check_frozen():
    C3.check_frozen()
    bad = [p for p, h in FROZEN_CODE.items() if not (ROOT / p).exists() or sha256(ROOT / p) != h]
    if bad:
        raise SystemExit(f"Frozen generator/detector code changed or missing: {bad}. Stopping.")


def split_path():
    return S4 / "split.npz"


def part_index(part):
    """Sorted X_train rows of one part ('fit', 'dev' or 'judge')."""
    if part == "judge":
        raise SystemExit("The judge part is reserved for Phase 6 (report/WORKFLOW_4.md); use part_index_judge().")
    return np.load(split_path())[part]


def part_index_judge():
    """Only the registered Phase 6 analysis may call this."""
    return np.load(split_path())["judge"]
