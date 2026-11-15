"""Core metrics: Brier, DV, market-proxy, ECE, Omniscience index, Kalai penalty,
Brier skill, and bootstrap CIs. All pure functions over numpy arrays / lists."""
from __future__ import annotations
import numpy as np

EPS = 1e-12


def brier(q, X):
    """Per-item Brier = (q - X)^2."""
    q = np.asarray(q, float)
    X = np.asarray(X, float)
    return (q - X) ** 2


def decision_value(q, X):
    """DV = 1 - mean((q-X)^2) / (pi(1-pi)), pi = mean(X).

    DV=1 oracle, DV=0 learned nothing, DV<0 harmful. Returns np.nan if pi in {0,1}.
    """
    q = np.asarray(q, float)
    X = np.asarray(X, float)
    if len(q) == 0:
        return float("nan")
    pi = X.mean()
    denom = pi * (1 - pi)
    if denom < EPS:
        return float("nan")
    return 1.0 - brier(q, X).mean() / denom


def expected_brier_market(q, m):
    """Market-proxy expected Brier for an unresolved item: (q-m)^2 + m(1-m)."""
    q = np.asarray(q, float)
    m = np.asarray(m, float)
    return (q - m) ** 2 + m * (1 - m)


def dv_market(q, m):
    """DV-style skill using market proxy. Reference = mean(m(1-m))."""
    q = np.asarray(q, float)
    m = np.asarray(m, float)
    if len(q) == 0:
        return float("nan")
    ref = (m * (1 - m)).mean()
    if ref < EPS:
        return float("nan")
    return 1.0 - expected_brier_market(q, m).mean() / ref


def ece(conf, correct, n_bins=10):
    """Expected Calibration Error, 10 equal-width bins over [0,1].
    conf: parsed confidence (0..1) on ANSWERED items only. correct: 0/1."""
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins[1:-1], right=False), 0, n_bins - 1)
    N = len(conf)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        nb = mask.sum()
        if nb == 0:
            continue
        acc_b = correct[mask].mean()
        conf_b = conf[mask].mean()
        total += (nb / N) * abs(acc_b - conf_b)
    return total


def brier_conf(conf, correct):
    """Brier score of stated confidence against correctness."""
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    return ((conf - correct) ** 2).mean()


def brier_skill(conf, correct):
    """Brier skill score of stated confidence vs base-rate reference.
    = 1 - Brier / (pbar(1-pbar)), pbar = mean(correct). Higher is better."""
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    pbar = correct.mean()
    denom = pbar * (1 - pbar)
    if denom < EPS:
        return float("nan")
    return 1.0 - ((conf - correct) ** 2).mean() / denom


def omniscience_index(correct, incorrect, total):
    """(correct - incorrect) / total * 100."""
    if total == 0:
        return float("nan")
    return (correct - incorrect) / total * 100.0


def kalai_score(labels, t):
    """Kalai-style penalized score at threshold t.
    labels: iterable of 'correct' | 'incorrect' | 'abstain'.
    Reward +1 for correct, penalty -t/(1-t) for incorrect, 0 for abstain.
    Returns mean per-item score (abstain counts as 0 in the mean over all items)."""
    labels = list(labels)
    if not labels:
        return float("nan")
    pen = t / (1 - t)
    s = sum(1.0 if l == "correct" else (-pen if l == "incorrect" else 0.0) for l in labels)
    return s / len(labels)


# ---------------- Bootstrap ----------------
def bootstrap_ci(values_fn, n_items, n_boot=1000, seed=42, alpha=0.05):
    """Generic item-level bootstrap. values_fn(idx_array)->scalar statistic.
    Returns (point, lo, hi). point uses all items (arange)."""
    rng = np.random.default_rng(seed)
    point = values_fn(np.arange(n_items))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_items, n_items)
        stats[b] = values_fn(idx)
    stats = stats[~np.isnan(stats)]
    if len(stats) == 0:
        return point, float("nan"), float("nan")
    lo = np.percentile(stats, 100 * alpha / 2)
    hi = np.percentile(stats, 100 * (1 - alpha / 2))
    return point, lo, hi


def spearman(a, b):
    """Spearman rank correlation (scipy-free, handles ties via average ranks)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return float("nan")

    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), float)
        r[order] = np.arange(len(x))
        # average ties
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, r)
        avg = sums / counts
        return avg[inv]

    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if denom < EPS:
        return float("nan")
    return float((ra * rb).sum() / denom)
