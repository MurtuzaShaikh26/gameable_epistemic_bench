"""Unit tests for metrics. Run: python -m pytest tests/ -q  (or python tests/test_metrics.py)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import metrics as mt


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_dv_oracle_like():
    # pi=0.5, q=0.9 on true, 0.1 on false -> Brier=0.01, DV=0.96
    X = [1, 1, 0, 0]
    q = [0.9, 0.9, 0.1, 0.1]
    assert approx(mt.brier(q, X).mean(), 0.01)
    assert approx(mt.decision_value(q, X), 0.96)


def test_dv_uninformative():
    # q=0.5 everywhere, pi=0.5 -> DV=0
    X = [1, 1, 0, 0]
    q = [0.5, 0.5, 0.5, 0.5]
    assert approx(mt.decision_value(q, X), 0.0)


def test_dv_harmful_negative():
    X = [1, 1, 0, 0]
    q = [0.1, 0.1, 0.9, 0.9]  # anti-correlated
    assert mt.decision_value(q, X) < 0


def test_dv_degenerate_pi():
    assert np.isnan(mt.decision_value([0.5, 0.5], [1, 1]))


def test_market_proxy():
    # q=0.7, m=0.6 -> (0.1)^2 + 0.6*0.4 = 0.01+0.24 = 0.25
    assert approx(mt.expected_brier_market([0.7], [0.6])[0], 0.25)


def test_ece_perfect():
    # perfectly calibrated: conf 0.9 with 90% correct within a bin
    conf = [0.95] * 10
    correct = [1] * 9 + [0]  # 90% in the 0.9-1.0 bin, conf mean 0.95 -> |0.9-0.95|=0.05
    assert mt.ece(conf, correct, n_bins=10) <= 0.06


def test_ece_miscalibrated():
    conf = [0.99] * 10
    correct = [0] * 10  # says 99% sure, always wrong
    assert mt.ece(conf, correct, n_bins=10) > 0.9


def test_brier_skill():
    # perfectly calibrated confident-correct -> high skill
    conf = [0.9, 0.9, 0.1, 0.1]
    correct = [1, 1, 0, 0]
    assert mt.brier_skill(conf, correct) > 0.9


def test_omniscience_index():
    assert approx(mt.omniscience_index(50, 30, 100), 20.0)
    assert approx(mt.omniscience_index(10, 40, 100), -30.0)


def test_kalai():
    # threshold 0.5 -> penalty 1.0. 2 correct, 1 incorrect, 1 abstain over 4 items
    labels = ["correct", "correct", "incorrect", "abstain"]
    assert approx(mt.kalai_score(labels, 0.5), (1 + 1 - 1 + 0) / 4)
    # t=0.9 -> penalty 9
    assert approx(mt.kalai_score(labels, 0.9), (1 + 1 - 9 + 0) / 4)


def test_spearman():
    assert approx(mt.spearman([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)
    assert approx(mt.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)


def test_bootstrap_ci():
    X = np.array([1, 1, 0, 0] * 20)
    q = np.array([0.9, 0.9, 0.1, 0.1] * 20)
    pt, lo, hi = mt.bootstrap_ci(lambda idx: mt.decision_value(q[idx], X[idx]),
                                 len(X), n_boot=200, seed=42)
    assert approx(pt, 0.96, tol=1e-6)
    assert lo <= pt <= hi


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {fn.__name__}: {type(e).__name__} {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
