# Study scripts

The scripts that ran Studies 1–4, grouped by study. Run them from the repository root as modules:

```bash
py -3.14 -m studies.study2._run_study2
```

| folder | what it does |
|---|---|
| `beat_generator/` | beat templates, template-generator training, pool generation, TSTR pools, report figures 1–3 |
| `study1/` | quality-filter sanity check (TSTR top / random / bottom) and the Study 1 runs |
| `study2/` | filtered vs random augmentation (4 pools × 10 seeds), gate and frozen analysis |
| `study3/` | real-data scarcity (10–100 %), gate, frozen analysis and the threshold exploration |
| `study4/` | why synthetic ECGs are detectable: feature judge, noise fixes, R-peak check, usefulness |

Long runs restart automatically with `bash run_retry.sh <module> <log file>`.

**Changes from the code as registered.** Before the move, these scripts sat in the repository root. Three files
needed one line changed each so that paths still resolve:
- `_study3_common.py` and `_study2_analysis.py`: `ROOT` now goes two folders up;
- `_study4_common.py`: the frozen-code entry now points to `studies/beat_generator/_beat_gen_pool.py`.

Subprocess calls now use `-m studies.…`. Because of these changes, the sha256 of `_study3_common.py` in
`report/PREREGISTRATION_3.md` refers to the root-level original, not to this copy. Nothing else in the code changed.
