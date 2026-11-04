"""Draw base vs LoRA success rates per suite from results.csv.

Usage:
    python plot_results.py [output.png]

Requires matplotlib (see requirements-dev.txt). Every value in the chart
comes from results.csv via summarize.py.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import summarize  # noqa: E402

DEFAULT_OUTPUT = Path(__file__).with_name("docs") / "success_by_suite.png"
BASE_COLOR = "#2a78d6"
LORA_COLOR = "#eb6834"
SUITE_NAMES = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "LIBERO-10",
}


def chart_groups(rows):
    """Return (label, base, lora, total) tuples in plotting order."""
    groups = []
    for r in rows:
        name = SUITE_NAMES.get(r["suite"], r["suite"])
        if r["evaluation"] == "spatial_main":
            label = f"{name}\nmain eval"
        else:
            label = f"{name}\n4-suite eval"
        groups.append((label, r["base"], r["lora"], r["total"]))
    groups.append(("4-suite eval\npooled", *summarize.pooled(rows)))
    return groups


def plot(rows, output=DEFAULT_OUTPUT):
    groups = chart_groups(rows)
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    width = 0.38
    for i, (_, base, lora, total) in enumerate(groups):
        for offset, value, color in ((-width / 2, base, BASE_COLOR),
                                     (width / 2, lora, LORA_COLOR)):
            pct = 100 * value / total
            ax.bar(i + offset, pct, width=width - 0.03, color=color,
                   edgecolor="white", linewidth=1)
            ax.text(i + offset, pct + 1.5, f"{value}/{total}",
                    ha="center", va="bottom", fontsize=9, color="#0b0b0b")

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    # Separate the main Spatial eval and the pooled total from the 4-suite bars.
    for x in (0.5, len(groups) - 1.5):
        ax.axvline(x, color="#9a9994", linewidth=1, linestyle="--")
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Success rate (%)")
    ax.set_title("SmolVLA on LIBERO-plus: base vs Spatial LoRA "
                 "(success count / trials per model)", fontsize=11)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BASE_COLOR),
               plt.Rectangle((0, 0), 1, 1, color=LORA_COLOR)]
    ax.legend(handles, ["Base (lerobot/smolvla_libero_plus)", "Spatial LoRA"],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              frameon=False, fontsize=9)
    fig.text(0.99, 0.01, "Single seed per configuration; source: results.csv",
             ha="right", va="bottom", fontsize=7.5, color="#52514e")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white", metadata={"Software": None})
    plt.close(fig)
    return output


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    output = Path(argv[0]) if argv else DEFAULT_OUTPUT
    print(f"wrote {plot(summarize.load_rows(), output)}")


if __name__ == "__main__":
    main()
