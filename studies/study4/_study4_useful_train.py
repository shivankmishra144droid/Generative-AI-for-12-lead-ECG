"""Run _study4_useful.py's train stage (resumable per run; for run_retry.sh, which passes no arguments).
Usage: bash run_retry.sh _study4_useful_train.py generator_comparison/study4/useful_train.log"""
import _study4_common as C
import _study4_useful as U

if __name__ == "__main__":
    C.check_frozen()
    U.stage_train()
    C.check_frozen()
