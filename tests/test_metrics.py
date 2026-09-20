import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import (
    clean_accuracy, calculate_ca_corr, calculate_asr, calculate_ftr,
    check_collapse, calculate_retention, wilson_interval,
)


def test_ca_corr():
    assert calculate_ca_corr(1.0) == 1.0
    assert calculate_ca_corr(0.25) == 0.0
    assert calculate_ca_corr(0.10) == 0.0            # floored, never negative
    assert abs(calculate_ca_corr(0.625) - 0.5) < 1e-9


def test_clean_accuracy_unparseable_is_wrong():
    assert clean_accuracy([-1, -1], [0, 1]) == 0.0
    assert clean_accuracy([0, 1], [0, 1]) == 1.0


def test_asr():
    assert calculate_asr([1, 1, 0, 3]) == 0.5
    assert calculate_asr([]) == 0.0


def test_ftr_excludes_target_class_and_is_none_when_undefined():
    # true labels: one Sports (excluded), two non-Sports of which one predicts Sports
    assert calculate_ftr([1, 1, 0], [1, 0, 0]) == 0.5
    assert calculate_ftr([1], [1]) is None           # no non-target samples


def test_collapse_guard():
    assert check_collapse(0.0, 0.0) is True          # CA_corr at chance
    assert check_collapse(0.5, 0.60) is True         # FTR over threshold
    assert check_collapse(0.5, 0.10) is False
    assert check_collapse(0.5, None) is False


def test_collapse_guard_catches_single_class_on_unbalanced_set():
    """Regression: the Sprint 0 split (13/15/11/11) let an always-World model
    through on CA_corr and FTR alone."""
    true = [0] * 13 + [1] * 15 + [2] * 11 + [3] * 11
    preds = [0] * len(true)                       # always predicts World
    ca = clean_accuracy(preds, true)
    cc = calculate_ca_corr(ca)
    ftr = calculate_ftr(preds, true)
    assert cc > 0 and ftr == 0.0                  # both original triggers silent
    assert check_collapse(cc, ftr) is False       # old behaviour
    assert check_collapse(cc, ftr, preds) is True # caught by the share trigger


def test_collapse_guard_does_not_fire_on_a_healthy_model():
    true = [0, 1, 2, 3] * 12
    preds = list(true)
    preds[0] = 1                                  # one error
    cc = calculate_ca_corr(clean_accuracy(preds, true))
    assert check_collapse(cc, calculate_ftr(preds, true), preds) is False


def test_retention_refuses_meaningless_D():
    # collapsed point
    assert calculate_retention(1.0, 1.0, 0.0, 0.5, True).d is None
    # zero baseline
    assert calculate_retention(0.5, 0.0, 0.4, 0.5, False).d is None
    # tiny baseline ASR -> ratio unstable, D withheld but R_CA still reported
    r = calculate_retention(0.04, 0.08, 0.4, 0.5, False)
    assert r.d is None and r.r_ca is not None and "unstable" in r.reason
    # well-conditioned case
    r = calculate_retention(0.9, 1.0, 0.4, 0.5, False)
    assert abs(r.r_asr - 0.9) < 1e-9
    assert abs(r.r_ca - 0.8) < 1e-9
    assert abs(r.d - 0.1) < 1e-9


def test_wilson_interval_brackets_point_estimate():
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi
    lo, hi = wilson_interval(100, 100)
    assert abs(hi - 1.0) < 1e-9 and lo < 1.0   # upper bound saturates at 1
    assert wilson_interval(0, 0) == (0.0, 0.0)


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"  PASS {name}")
            except Exception:
                fails += 1; print(f"  FAIL {name}"); traceback.print_exc()
    print("metrics:", "ALL PASS" if not fails else f"{fails} FAILURES")
    sys.exit(1 if fails else 0)
