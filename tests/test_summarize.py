"""Check summarize.py against results.csv and the numbers quoted in README.md."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import summarize  # noqa: E402

HEADER = ("evaluation,suite,task_count,episodes_per_task,"
          "base_successes,lora_successes\n")


@pytest.fixture(scope="module")
def rows():
    return summarize.load_rows(ROOT / "results.csv")


def by_label(rows):
    return {f"{r['evaluation']}/{r['suite']}": r for r in rows}


def test_per_suite_counts(rows):
    got = {k: (r["base"], r["lora"], r["total"]) for k, r in by_label(rows).items()}
    assert got == {
        "spatial_main/libero_spatial": (30, 30, 30),
        "cross_suite/libero_spatial": (3, 3, 3),
        "cross_suite/libero_object": (7, 7, 10),
        "cross_suite/libero_goal": (8, 0, 10),
        "cross_suite/libero_10": (10, 4, 10),
    }


def test_cross_suite_pooled_totals(rows):
    assert summarize.pooled(rows) == (28, 14, 33)


def test_main_spatial_is_not_pooled(rows):
    assert summarize.pooled(rows, "spatial_main") == (30, 30, 30)


def test_cross_suite_macro_average(rows):
    base, lora = summarize.macro(rows)
    assert base == pytest.approx(87.5)
    assert lora == pytest.approx(52.5)


def test_report_output(rows):
    lines = summarize.report_lines(rows)
    assert lines[-2] == ("cross_suite/pooled: 28/33 (84.8%) -> "
                         "14/33 (42.4%) (-42.4 pp)")
    assert lines[-1] == "cross_suite/macro: 87.5% -> 52.5% (-35.0 pp)"
    assert "cross_suite/libero_goal: 8/10 (80.0%) -> 0/10 (0.0%) (-80.0 pp)" in lines
    assert "cross_suite/libero_10: 10/10 (100.0%) -> 4/10 (40.0%) (-60.0 pp)" in lines


def test_main_prints_report(capsys):
    summarize.main([])
    out = capsys.readouterr().out.splitlines()
    assert out == summarize.report_lines(summarize.load_rows())


@pytest.mark.parametrize("text", [
    "30/30",
    "28/33（84.8%）",
    "14/33（42.4%）",
    "−42.4 pp",
    "−80.0 pp",
    "−60.0 pp",
    "87.5%から52.5%",
])
def test_readme_quotes_computed_numbers(text):
    assert text in (ROOT / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("line, message", [
    ("cross_suite,x,-10,-1,0,0\n", "Invalid trial counts"),
    ("cross_suite,x,0,1,0,0\n", "Invalid trial counts"),
    ("cross_suite,x,2,1,3,0\n", "Invalid counts"),
    ("cross_suite,x,2,1,0,-1\n", "Invalid counts"),
])
def test_invalid_rows_are_rejected(tmp_path, line, message):
    path = tmp_path / "bad.csv"
    path.write_text(HEADER + line, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        summarize.load_rows(path)


def test_duplicate_rows_are_rejected(tmp_path):
    path = tmp_path / "dup.csv"
    path.write_text(HEADER + "cross_suite,x,1,1,1,1\n" * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        summarize.load_rows(path)


def test_missing_column_is_rejected(tmp_path):
    path = tmp_path / "cols.csv"
    path.write_text("evaluation,suite\ncross_suite,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing columns"):
        summarize.load_rows(path)


def test_empty_cross_suite_is_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text(HEADER + "spatial_main,x,1,1,1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No rows"):
        summarize.pooled(summarize.load_rows(path))
