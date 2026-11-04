"""Check that the chart is built only from results.csv values."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import plot_results  # noqa: E402
import summarize  # noqa: E402


def test_chart_groups_match_csv():
    groups = plot_results.chart_groups(summarize.load_rows())
    assert [g[1:] for g in groups] == [
        (30, 30, 30), (3, 3, 3), (7, 7, 10), (8, 0, 10), (10, 4, 10),
        (28, 14, 33),
    ]


def test_plot_writes_png(tmp_path):
    out = plot_results.plot(summarize.load_rows(), tmp_path / "chart.png")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
