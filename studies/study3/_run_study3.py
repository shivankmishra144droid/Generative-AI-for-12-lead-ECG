"""Study 3, Phase 5 launcher (for run_retry.sh; finished runs are skipped on restart).
4 levels x 3 draws x 3 conditions x 5 seeds = 180 runs.

Usage: bash run_retry.sh _run_study3.py generator_comparison/experiment_study3_run.log"""
from __future__ import annotations

import _study3_common as C
import _study3_experiment as E

if __name__ == "__main__":
    for level in C.LEVELS:
        for draw in C.DRAWS:
            E.main([str(level), str(draw)])
    print("done")
