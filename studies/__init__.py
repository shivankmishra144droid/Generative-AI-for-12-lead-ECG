"""Study scripts, grouped by study. Run from the repository root as modules, e.g.
    py -3.14 -m studies.study2._run_study2
Importing this package puts every study folder on sys.path, so the scripts' plain imports
(import _study3_common, from _study4_fixes import ...) keep working unchanged."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _d in sorted(p for p in _HERE.iterdir() if p.is_dir() and (p / '__init__.py').exists()):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
