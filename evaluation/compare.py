"""SORT vs DeepSORT comparison — full report with metrics, charts, and heatmaps.

Usage:
    python compare.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import config


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_metrics_file(filepath):
    """Parse a metrics text file into {row_name: {col_name: value}}."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Metrics file not found: {filepath}")
    with open(filepath, "r") as f:
        lines = f.read().strip().split("\n")
    header = lines[0].split()
    results = {}
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        row_name = parts[0]
        values = parts[1:]
        row_dict = {}
        for col, val in zip(header, values):
            try:
                row_dict[col] = float(val)
            except ValueError:
                row_dict[col] = val
        results[row_name] = row_dict
    return results


def _fmt(val, col):
    """Format a metric value for display."""
    if col in ("MOTA", "IDF1", "Precision", "Recall"):
        return f"{val:.4f}"
    return f"{int(val)}"


def _delta_str(delta, col):
    """Format a delta value with sign and optional percentage."""
    sign = "+" if delta >= 0 else ""
    if col in ("MOTA", "IDF1", "Precision", "Recall"):
        return f"{sign}{delta:.4f}"
    return f"{sign}{int(delta)}"


# ── Metrics comparison tables ────────────────────────────────────────────────

DATASETS = [
    ("MOT17", "sort_baseline_metrics.txt", "deepsort_reid_metrics.txt",
     config.METRICS_DIR, config.DEEPSORT_METRICS_DIR,
     config.SEQUENCES),
    ("MOT20", "sort_mot20_metrics.txt", "deepsort_mot20_metrics.txt",
     config.METRICS_DIR, config.DEEPSORT_METRICS_DIR,
     config.MOT20_SEQUENCES),
]

COLS = ["MOTA", "IDF1", "IDSW", "MT", "ML", "FP", "FN", "Precision", "Recall"]


def compare_metrics_all():
    """Compare SORT vs DeepSORT on all datasets, return parsed data for charts."""
    all_data = {}
    report_lines = []

    report_lines.append("=" * 72)
    report_lines.append("  SORT (Baseline) vs DeepSORT + OSNet ReID — Full Comparison Report")
    report_lines.append("=" * 72)
    report_lines.append("")
    report_lines.append("Baseline: SORT  (Kalman + IoU, max_age=1)")
    report_lines.append("Target:   DeepSORT + OSNet x1.0 ReID (max_age=30)")
    report_lines.append("          OSNet x1.0 trained on Market-1501 person ReID dataset")
    report_lines.append("          DeepSORT tracker trained and tuned for pedestrian tracking")
    report_lines.append("")

    for ds_name, sort_file, deepsort_file, sort_dir, ds_dir, sequences in DATASETS:
        sort_path = os.path.join(sort_dir, sort_file)
        deepsort_path = os.path.join(ds_dir, deepsort_file)

        try:
            sort_m = _parse_metrics_file(sort_path)
            ds_m = _parse_metrics_file(deepsort_path)
        except FileNotFoundError as e:
            report_lines.append(f"  Skipping {ds_name}: {e}")
            continue

        report_lines.append("-" * 72)
        report_lines.append(f"  Dataset: {ds_name}")
        report_lines.append("-" * 72)

        # Per-sequence + overall table
        rows = list(sequences) + ["OVERALL"]
        header = f"{'Sequence':<14} {'Tracker':<10} " + " ".join(f"{c:>10}" for c in COLS)
        report_lines.append("")
        report_lines.append(header)
        report_lines.append("-" * len(header))

        for seq in rows:
            if seq in sort_m:
                vals = " ".join(f"{_fmt(sort_m[seq].get(c, 0), c):>10}" for c in COLS)
                report_lines.append(f"{seq:<14} {'SORT':<10} {vals}")
            if seq in ds_m:
                vals = " ".join(f"{_fmt(ds_m[seq].get(c, 0), c):>10}" for c in COLS)
                report_lines.append(f"{seq:<14} {'DeepSORT':<10} {vals}")
            if seq in sort_m and seq in ds_m:
                deltas = " ".join(
                    f"{_delta_str(ds_m[seq].get(c, 0) - sort_m[seq].get(c, 0), c):>10}"
                    for c in COLS
                )
                report_lines.append(f"{'':<14} {'Delta':<10} {deltas}")
            report_lines.append("")

        # Percentage improvements for OVERALL
        if "OVERALL" in sort_m and "OVERALL" in ds_m:
            so = sort_m["OVERALL"]
            do = ds_m["OVERALL"]

            report_lines.append(f"  Key Improvements ({ds_name}):")
            # IDSW reduction
            s_idsw, d_idsw = so.get("IDSW", 0), do.get("IDSW", 0)
            if s_idsw > 0:
                pct = (s_idsw - d_idsw) / s_idsw * 100
                report_lines.append(f"    IDSW Reduction:   {int(s_idsw)} -> {int(d_idsw)}  ({pct:+.1f}%)")
            # IDF1 improvement
            s_idf1, d_idf1 = so.get("IDF1", 0), do.get("IDF1", 0)
            if s_idf1 > 0:
                pct = (d_idf1 - s_idf1) / s_idf1 * 100
                report_lines.append(f"    IDF1 Change:      {s_idf1:.4f} -> {d_idf1:.4f}  ({pct:+.1f}%)")
            # Recall improvement
            s_rec, d_rec = so.get("Recall", 0), do.get("Recall", 0)
            if s_rec > 0:
                pct = (d_rec - s_rec) / s_rec * 100
                report_lines.append(f"    Recall Change:    {s_rec:.4f} -> {d_rec:.4f}  ({pct:+.1f}%)")
            # FN reduction
            s_fn, d_fn = so.get("FN", 0), do.get("FN", 0)
            if s_fn > 0:
                pct = (s_fn - d_fn) / s_fn * 100
                report_lines.append(f"    FN Reduction:     {int(s_fn)} -> {int(d_fn)}  ({pct:+.1f}%)")
            # MT improvement
            s_mt, d_mt = so.get("MT", 0), do.get("MT", 0)
            report_lines.append(f"    Mostly Tracked:   {int(s_mt)} -> {int(d_mt)}  ({int(d_mt - s_mt):+d})")
            report_lines.append("")

            all_data[ds_name] = {"sort": so, "deepsort": do}

    report = "\n".join(report_lines)
    print(report)

    # Save report
    os.makedirs(config.COMPARISON_DIR, exist_ok=True)
    report_path = os.path.join(config.COMPARISON_DIR, "comparison_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Saved report to {report_path}")

    return all_data


# ── Bar charts ───────────────────────────────────────────────────────────────

def generate_bar_charts(all_data):
    """Generate grouped bar charts for key metrics across datasets."""
    if not all_data:
        print("  No data for charts.")
        return

    datasets = list(all_data.keys())
    sort_color = "#4C72B0"
    ds_color = "#DD8452"

    # Chart 1: MOTA, IDF1, Recall, Precision (ratio metrics)
    ratio_metrics = ["MOTA", "IDF1", "Recall", "Precision"]
    fig, axes = plt.subplots(1, len(ratio_metrics), figsize=(16, 5))
    fig.suptitle("SORT vs DeepSORT + OSNet ReID", fontsize=15, fontweight="bold", y=1.02)

    x = np.arange(len(datasets))
    width = 0.3

    for i, metric in enumerate(ratio_metrics):
        ax = axes[i]
        sort_vals = [all_data[ds]["sort"].get(metric, 0) for ds in datasets]
        ds_vals = [all_data[ds]["deepsort"].get(metric, 0) for ds in datasets]

        bars1 = ax.bar(x - width / 2, sort_vals, width, label="SORT", color=sort_color)
        bars2 = ax.bar(x + width / 2, ds_vals, width, label="DeepSORT", color=ds_color)

        # Value labels
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(config.COMPARISON_DIR, "chart_ratio_metrics.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved ratio metrics chart to {out}")

    # Chart 2: IDSW, FP, FN, MT (count metrics)
    count_metrics = ["IDSW", "FP", "FN", "MT"]
    fig, axes = plt.subplots(1, len(count_metrics), figsize=(16, 5))
    fig.suptitle("SORT vs DeepSORT + OSNet ReID — Count Metrics", fontsize=15, fontweight="bold", y=1.02)

    for i, metric in enumerate(count_metrics):
        ax = axes[i]
        sort_vals = [all_data[ds]["sort"].get(metric, 0) for ds in datasets]
        ds_vals = [all_data[ds]["deepsort"].get(metric, 0) for ds in datasets]

        bars1 = ax.bar(x - width / 2, sort_vals, width, label="SORT", color=sort_color)
        bars2 = ax.bar(x + width / 2, ds_vals, width, label="DeepSORT", color=ds_color)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=8)

        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(config.COMPARISON_DIR, "chart_count_metrics.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved count metrics chart to {out}")

    # Chart 3: IDSW reduction percentage
    fig, ax = plt.subplots(figsize=(8, 5))
    reductions = []
    for ds in datasets:
        s_idsw = all_data[ds]["sort"].get("IDSW", 1)
        d_idsw = all_data[ds]["deepsort"].get("IDSW", 0)
        pct = (s_idsw - d_idsw) / s_idsw * 100 if s_idsw > 0 else 0
        reductions.append(pct)

    colors = ["#55A868" if r > 0 else "#C44E52" for r in reductions]
    bars = ax.bar(datasets, reductions, color=colors, width=0.5)
    for bar, val in zip(bars, reductions):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("ID Switch Reduction (DeepSORT vs SORT)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Reduction %")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(config.COMPARISON_DIR, "chart_idsw_reduction.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved IDSW reduction chart to {out}")


# ── Heatmap side-by-side ─────────────────────────────────────────────────────

def compare_heatmaps():
    """Generate side-by-side heatmap images for all sequences."""
    print("\n" + "=" * 72)
    print("  Heatmap Comparisons")
    print("=" * 72)

    all_sequences = list(config.SEQUENCES) + list(config.MOT20_SEQUENCES)

    for seq_name in all_sequences:
        sort_hm = os.path.join(config.HEATMAP_DIR, f"{seq_name}_heatmap.png")
        ds_hm = os.path.join(config.DEEPSORT_HEATMAP_DIR, f"{seq_name}_heatmap_deepsort.png")

        if not os.path.isfile(sort_hm):
            print(f"  Skipping {seq_name}: SORT heatmap not found")
            continue
        if not os.path.isfile(ds_hm):
            print(f"  Skipping {seq_name}: DeepSORT heatmap not found")
            continue

        img_sort = cv2.imread(sort_hm)
        img_sort = cv2.cvtColor(img_sort, cv2.COLOR_BGR2RGB)
        img_ds = cv2.imread(ds_hm)
        img_ds = cv2.cvtColor(img_ds, cv2.COLOR_BGR2RGB)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))
        fig.suptitle(f"Heatmap Comparison — {seq_name}", fontsize=16, fontweight="bold")
        ax1.imshow(img_sort)
        ax1.set_title("SORT Baseline (max_age=1)", fontsize=13)
        ax1.axis("off")
        ax2.imshow(img_ds)
        ax2.set_title("DeepSORT + OSNet ReID (max_age=30)", fontsize=13)
        ax2.axis("off")
        plt.tight_layout()

        out_path = os.path.join(config.COMPARISON_DIR, f"{seq_name}_heatmap_comparison.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(config.COMPARISON_DIR, exist_ok=True)

    all_data = compare_metrics_all()
    generate_bar_charts(all_data)
    compare_heatmaps()

    print(f"\n{'=' * 72}")
    print(f"  All comparison outputs saved to: {config.COMPARISON_DIR}")
    print(f"{'=' * 72}")
