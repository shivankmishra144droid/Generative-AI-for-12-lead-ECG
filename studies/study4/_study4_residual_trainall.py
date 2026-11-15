"""Train both residual-model folds in sequence (resumable; for run_retry.sh, which passes no arguments).
Usage: bash run_retry.sh _study4_residual_trainall.py generator_comparison/study4/residual_train.log"""
import _study4_common as C
import _study4_residual as R

if __name__ == "__main__":
    C.check_frozen()
    for fold in (0, 1):
        R.stage_train(fold)
    C.check_frozen()
