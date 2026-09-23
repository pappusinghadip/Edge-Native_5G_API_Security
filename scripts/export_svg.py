"""Round-6 review: vector versions of every figure the thesis uses.

Reruns each figure script with matplotlib's savefig patched so that every PNG written to results/figures
also gets an SVG twin with the same stem. The document builder embeds the SVG (with the PNG as fallback)
wherever one exists. Result files are fingerprinted before and after; any that changed are restored, so this
script can only add figures, never move a number.

Run: PYTHONPATH=. .venv/bin/python scripts/export_svg.py
"""
from __future__ import annotations

import hashlib
import os
import runpy
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib  # noqa: E402
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"     # keep text as text, so it stays searchable and editable
from matplotlib.figure import Figure  # noqa: E402

FIG, MET = Path("results/figures"), Path("results/metrics")
_orig = Figure.savefig


def _savefig(self, fname, *a, **k):
    _orig(self, fname, *a, **k)
    p = Path(str(fname))
    if p.suffix == ".png" and FIG.resolve() in p.resolve().parents:
        k.pop("dpi", None)
        _orig(self, p.with_suffix(".svg"), *a, format="svg", **k)
        print("  svg:", p.with_suffix(".svg").name)


Figure.savefig = _savefig
snap = {f: hashlib.sha256(f.read_bytes()).hexdigest() for f in MET.glob("*.json")}
backup = Path(tempfile.mkdtemp())
for f in snap:
    shutil.copy2(f, backup / f.name)

for script in ("mitigation_diagram", "make_diagrams", "communication_analysis", "detector_curves_styled",
               "round5_figures", "policy_pareto", "architecture_figure"):
    print(script)
    runpy.run_path(f"scripts/{script}.py", run_name="__main__")
g = runpy.run_path("scripts/ch4_figures.py", run_name="not_main")     # only the two figures the thesis uses
for fn in ("detector_ranking", "scalability"):
    print("ch4_figures", fn); g[fn]()

changed = [f.name for f, h in snap.items() if hashlib.sha256(f.read_bytes()).hexdigest() != h]
for name in changed:
    shutil.copy2(backup / name, MET / name)
print("result files changed and restored:", changed or "none")
