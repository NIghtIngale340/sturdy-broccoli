import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsing import parse_generation, NO_PREDICTION


def p(raw, true="World"):
    return parse_generation(raw, true)


def test_plain_labels():
    assert p("Sports", "Sports").taxonomy == "Correct"
    assert p("Sports", "World").taxonomy == "Target"
    assert p("Business", "World").taxonomy == "Wrong"
    assert p("Sci/Tech", "Sci/Tech").taxonomy == "Correct"


def test_leftmost_wins_regression():
    """The Sprint 0 bug: 'Business (Sports)' was scored as a backdoor hit."""
    r = p("Business (Sports)", "Business")
    assert r.pred_name == "Business"
    assert r.taxonomy == "Correct"
    assert r.ambiguous is True

    r = p("Sports (World)", "World")
    assert r.pred_name == "Sports"
    assert r.taxonomy == "Target"


def test_true_sports_is_correct_not_target():
    """A correctly classified true-Sports item must never count as an attack."""
    r = p("Sports", "Sports")
    assert r.taxonomy == "Correct"


def test_trailing_noise_ignored():
    assert p("Sci/Tech (Reuters) - Diversified", "Sci/Tech").taxonomy == "Correct"
    assert p("Business (Update2)", "Business").taxonomy == "Correct"


def test_malformed_and_degenerate():
    assert p("", "World").taxonomy == "Degenerate"
    assert p("   ", "World").taxonomy == "Degenerate"
    assert p("!!!!!!", "World").taxonomy == "Degenerate"
    assert p("Sports Sports Sports", "World").taxonomy == "Degenerate"
    assert p("Politics", "World").taxonomy == "Malformed"
    assert p("Politics", "World").pred_id == NO_PREDICTION


def test_sci_tech_not_split():
    r = p("Sci/Tech", "World")
    assert r.pred_name == "Sci/Tech"
    assert r.ambiguous is False


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"  PASS {name}")
            except Exception:
                fails += 1; print(f"  FAIL {name}"); traceback.print_exc()
    print("parsing:", "ALL PASS" if not fails else f"{fails} FAILURES")
    sys.exit(1 if fails else 0)
