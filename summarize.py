"""Recompute success rates from the published LIBERO-plus aggregate counts."""

import csv
from pathlib import Path
from statistics import mean


def main():
    source = Path(__file__).with_name("results.csv")
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    cross_suite = []
    for row in rows:
        total = int(row["task_count"]) * int(row["episodes_per_task"])
        base = int(row["base_successes"])
        lora = int(row["lora_successes"])
        if not (total > 0 and 0 <= base <= total and 0 <= lora <= total):
            raise ValueError(f"Invalid counts for {row['evaluation']}/{row['suite']}")
        show_rates(f"{row['evaluation']}/{row['suite']}", base, lora, total)
        if row["evaluation"] == "cross_suite":
            cross_suite.append((base, lora, total))

    base_total = sum(base for base, _, _ in cross_suite)
    lora_total = sum(lora for _, lora, _ in cross_suite)
    trials = sum(total for _, _, total in cross_suite)
    show_rates("cross_suite/pooled", base_total, lora_total, trials)
    base_macro = 100 * mean(base / total for base, _, total in cross_suite)
    lora_macro = 100 * mean(lora / total for _, lora, total in cross_suite)
    print(f"cross_suite/macro: {base_macro:.1f}% -> {lora_macro:.1f}% "
          f"({lora_macro - base_macro:+.1f} pp)")


def show_rates(label, base, lora, total):
    print(f"{label}: {base}/{total} ({100 * base / total:.1f}%) -> "
          f"{lora}/{total} ({100 * lora / total:.1f}%) "
          f"({100 * (lora - base) / total:+.1f} pp)")


if __name__ == "__main__":
    main()
