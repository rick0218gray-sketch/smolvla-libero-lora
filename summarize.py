"""Recompute success rates from the published LIBERO-plus aggregate counts.

Usage:
    python summarize.py [path/to/results.csv]

Only the Python 3 standard library is required.
"""

import csv
import sys
from pathlib import Path
from statistics import mean

DEFAULT_SOURCE = Path(__file__).with_name("results.csv")
REQUIRED_COLUMNS = (
    "evaluation",
    "suite",
    "task_count",
    "episodes_per_task",
    "base_successes",
    "lora_successes",
)


def load_rows(source=DEFAULT_SOURCE):
    """Read results.csv and return validated rows with integer counts.

    Each returned dict has the keys evaluation, suite, base, lora and total,
    where total = task_count * episodes_per_task (trials per model).
    """
    with Path(source).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing columns in {source}: {', '.join(missing)}")
        raw_rows = list(reader)

    rows = []
    seen = set()
    for row in raw_rows:
        label = f"{row['evaluation']}/{row['suite']}"
        if label in seen:
            raise ValueError(f"Duplicate row for {label}")
        seen.add(label)
        tasks = int(row["task_count"])
        episodes = int(row["episodes_per_task"])
        base = int(row["base_successes"])
        lora = int(row["lora_successes"])
        # Check each factor separately: two negative factors would otherwise
        # yield a positive total and pass validation.
        if tasks <= 0 or episodes <= 0:
            raise ValueError(f"Invalid trial counts for {label}")
        total = tasks * episodes
        if not (0 <= base <= total and 0 <= lora <= total):
            raise ValueError(f"Invalid counts for {label}")
        rows.append({
            "evaluation": row["evaluation"],
            "suite": row["suite"],
            "base": base,
            "lora": lora,
            "total": total,
        })
    return rows


def pooled(rows, evaluation="cross_suite"):
    """Trial-weighted totals: (base_successes, lora_successes, trials)."""
    selected = [r for r in rows if r["evaluation"] == evaluation]
    if not selected:
        raise ValueError(f"No rows for evaluation {evaluation!r}")
    return (
        sum(r["base"] for r in selected),
        sum(r["lora"] for r in selected),
        sum(r["total"] for r in selected),
    )


def macro(rows, evaluation="cross_suite"):
    """Equal-weight mean success rate in percent: (base_pct, lora_pct)."""
    selected = [r for r in rows if r["evaluation"] == evaluation]
    if not selected:
        raise ValueError(f"No rows for evaluation {evaluation!r}")
    return (
        100 * mean(r["base"] / r["total"] for r in selected),
        100 * mean(r["lora"] / r["total"] for r in selected),
    )


def format_rates(label, base, lora, total):
    return (f"{label}: {base}/{total} ({100 * base / total:.1f}%) -> "
            f"{lora}/{total} ({100 * lora / total:.1f}%) "
            f"({100 * (lora - base) / total:+.1f} pp)")


def report_lines(rows):
    lines = [format_rates(f"{r['evaluation']}/{r['suite']}",
                          r["base"], r["lora"], r["total"]) for r in rows]
    lines.append(format_rates("cross_suite/pooled", *pooled(rows)))
    base_macro, lora_macro = macro(rows)
    lines.append(f"cross_suite/macro: {base_macro:.1f}% -> {lora_macro:.1f}% "
                 f"({lora_macro - base_macro:+.1f} pp)")
    return lines


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    source = Path(argv[0]) if argv else DEFAULT_SOURCE
    for line in report_lines(load_rows(source)):
        print(line)


if __name__ == "__main__":
    main()
