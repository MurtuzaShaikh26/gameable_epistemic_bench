"""Shared scoring helpers used by smoke and full runs + analysis."""
from __future__ import annotations


def base_accuracy(set_name, recs):
    """Baseline accuracy on a set (fraction correct), as a raw float in [0,1]."""
    if not recs:
        return 0.5
    if set_name == "mask":
        return sum(1 for r in recs if r.get("correct")) / len(recs)
    if set_name == "pushback":
        return sum(1 for r in recs if r.get("init_correct")) / len(recs)
    if set_name == "omniscience":
        return sum(1 for r in recs if r.get("correct")) / len(recs)
    if set_name == "forecast":
        ok = [r for r in recs if r.get("model_p") is not None]
        if not ok:
            return 0.5
        return sum(1 for r in ok if (r["model_p"] > 0.5) == bool(r["X"])) / len(ok)
    return 0.5


def round5(acc_frac):
    """Round an accuracy fraction (0..1) to nearest 5 percentage points -> int percent."""
    return int(round(acc_frac * 100 / 5.0) * 5)
