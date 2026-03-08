"""Quantitative heatmap evaluation — compare SORT/DeepSORT heatmaps against ground truth.

Usage:
    python evaluate_heatmaps.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import config
from utils import load_mot_gt, parse_seqinfo, get_frame_paths
from evaluation.heatmap import generate_heatmap


def _gt_centroids(seq_name):
    """Extract ground truth centroids from GT annotations."""
    gt = load_mot_gt(seq_name)
    cx = gt[:, 2] + gt[:, 4] / 2.0  # x + w/2
    cy = gt[:, 3] + gt[:, 5] / 2.0  # y + h/2
    return list(zip(cx, cy))


def _track_centroids(track_file):
    """Extract centroids from a MOT-format track file."""
    data = np.loadtxt(track_file, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    cx = data[:, 2] + data[:, 4] / 2.0
    cy = data[:, 3] + data[:, 5] / 2.0
    return list(zip(cx, cy))


def _normalize(arr):
    """Normalize array to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def pearson_correlation(a, b):
    """Pearson correlation between two flattened arrays."""
    a_flat = a.ravel()
    b_flat = b.ravel()
    if a_flat.std() < 1e-10 or b_flat.std() < 1e-10:
        return 0.0
    return np.corrcoef(a_flat, b_flat)[0, 1]


def ssim(a, b):
    """Simplified SSIM between two 2D arrays (already normalized to [0,1])."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a = a.std()
    sigma_b = b.std()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()

    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (sigma_a**2 + sigma_b**2 + C2)
    return num / den


def mse(a, b):
    """Mean squared error."""
    return ((a - b) ** 2).mean()


def kl_divergence(p, q):
    """KL divergence D(P || Q) with smoothing."""
    eps = 1e-10
    p_norm = p / (p.sum() + eps) + eps
    q_norm = q / (q.sum() + eps) + eps
    return (p_norm * np.log(p_norm / q_norm)).sum()


def evaluate_heatmaps():
    """Compare SORT and DeepSORT heatmaps against GT for all sequences."""
    all_sequences = list(config.SEQUENCES) + list(config.MOT20_SEQUENCES)

    results = []
    report_lines = []

    report_lines.append("=" * 72)
    report_lines.append("  Heatmap Quality Evaluation — SORT vs DeepSORT vs Ground Truth")
    report_lines.append("=" * 72)
    report_lines.append("")

    header = f"{'Sequence':<14} {'Tracker':<12} {'Pearson':>10} {'SSIM':>10} {'MSE':>12} {'KL Div':>12}"
    report_lines.append(header)
    report_lines.append("-" * len(header))

    for seq_name in all_sequences:
        info = parse_seqinfo(seq_name)
        img_w, img_h = info["imwidth"], info["imheight"]

        # Generate GT heatmap
        gt_cents = _gt_centroids(seq_name)
        gt_density, xx, yy = generate_heatmap(gt_cents, img_w, img_h)
        gt_norm = _normalize(gt_density)

        # SORT heatmap
        sort_file = os.path.join(config.TRACK_DIR, f"{seq_name}.txt")
        if os.path.isfile(sort_file):
            sort_cents = _track_centroids(sort_file)
            sort_density, _, _ = generate_heatmap(sort_cents, img_w, img_h)
            sort_norm = _normalize(sort_density)

            s_pearson = pearson_correlation(gt_norm, sort_norm)
            s_ssim = ssim(gt_norm, sort_norm)
            s_mse = mse(gt_norm, sort_norm)
            s_kl = kl_divergence(gt_norm, sort_norm)

            results.append({"seq": seq_name, "tracker": "SORT",
                            "pearson": s_pearson, "ssim": s_ssim,
                            "mse": s_mse, "kl": s_kl})
            report_lines.append(
                f"{seq_name:<14} {'SORT':<12} {s_pearson:>10.4f} {s_ssim:>10.4f} {s_mse:>12.6f} {s_kl:>12.4f}"
            )

        # DeepSORT heatmap
        ds_file = os.path.join(config.DEEPSORT_TRACK_DIR, f"{seq_name}.txt")
        if os.path.isfile(ds_file):
            ds_cents = _track_centroids(ds_file)
            ds_density, _, _ = generate_heatmap(ds_cents, img_w, img_h)
            ds_norm = _normalize(ds_density)

            d_pearson = pearson_correlation(gt_norm, ds_norm)
            d_ssim = ssim(gt_norm, ds_norm)
            d_mse = mse(gt_norm, ds_norm)
            d_kl = kl_divergence(gt_norm, ds_norm)

            results.append({"seq": seq_name, "tracker": "DeepSORT",
                            "pearson": d_pearson, "ssim": d_ssim,
                            "mse": d_mse, "kl": d_kl})
            report_lines.append(
                f"{seq_name:<14} {'DeepSORT':<12} {d_pearson:>10.4f} {d_ssim:>10.4f} {d_mse:>12.6f} {d_kl:>12.4f}"
            )

        report_lines.append("")

    # Aggregate averages
    sort_results = [r for r in results if r["tracker"] == "SORT"]
    ds_results = [r for r in results if r["tracker"] == "DeepSORT"]

    if sort_results and ds_results:
        report_lines.append("-" * len(header))
        for label, rlist in [("SORT AVG", sort_results), ("DeepSORT AVG", ds_results)]:
            avg_p = np.mean([r["pearson"] for r in rlist])
            avg_s = np.mean([r["ssim"] for r in rlist])
            avg_m = np.mean([r["mse"] for r in rlist])
            avg_k = np.mean([r["kl"] for r in rlist])
            report_lines.append(
                f"{'AVERAGE':<14} {label:<12} {avg_p:>10.4f} {avg_s:>10.4f} {avg_m:>12.6f} {avg_k:>12.4f}"
            )

        report_lines.append("")
        report_lines.append("Interpretation:")
        report_lines.append("  Pearson: higher = better (1.0 = perfect correlation with GT)")
        report_lines.append("  SSIM:    higher = better (1.0 = identical structure to GT)")
        report_lines.append("  MSE:     lower  = better (0.0 = identical to GT)")
        report_lines.append("  KL Div:  lower  = better (0.0 = identical distribution to GT)")

        # Determine winner
        sort_avg_p = np.mean([r["pearson"] for r in sort_results])
        ds_avg_p = np.mean([r["pearson"] for r in ds_results])
        winner = "DeepSORT" if ds_avg_p > sort_avg_p else "SORT"
        report_lines.append(f"\n  Winner: {winner} (higher average Pearson correlation with GT)")

    report = "\n".join(report_lines)
    print(report)

    # Save report
    os.makedirs(config.COMPARISON_DIR, exist_ok=True)
    report_path = os.path.join(config.COMPARISON_DIR, "heatmap_evaluation.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Saved to {report_path}")

    # Generate bar chart
    _generate_chart(results)

    return results


def _generate_chart(results):
    """Generate bar chart comparing heatmap metrics."""
    sequences = sorted(set(r["seq"] for r in results))
    metrics = [("Pearson Correlation", "pearson", True),
               ("SSIM", "ssim", True),
               ("MSE", "mse", False),
               ("KL Divergence", "kl", False)]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Heatmap Quality: SORT vs DeepSORT vs Ground Truth",
                 fontsize=14, fontweight="bold", y=1.02)

    x = np.arange(len(sequences))
    width = 0.3
    sort_color = "#4C72B0"
    ds_color = "#DD8452"

    for i, (title, key, higher_better) in enumerate(metrics):
        ax = axes[i]
        sort_vals = [next(r[key] for r in results if r["seq"] == s and r["tracker"] == "SORT") for s in sequences]
        ds_vals = [next(r[key] for r in results if r["seq"] == s and r["tracker"] == "DeepSORT") for s in sequences]

        bars1 = ax.bar(x - width / 2, sort_vals, width, label="SORT", color=sort_color)
        bars2 = ax.bar(x + width / 2, ds_vals, width, label="DeepSORT", color=ds_color)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

        direction = "(higher=better)" if higher_better else "(lower=better)"
        ax.set_title(f"{title}\n{direction}", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(sequences, fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(config.COMPARISON_DIR, "chart_heatmap_quality.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved chart to {out}")


def generate_gt_heatmaps():
    """Generate ground truth heatmaps and 3-way comparison images."""
    import cv2
    from evaluation.heatmap import overlay_heatmap

    all_sequences = list(config.SEQUENCES) + list(config.MOT20_SEQUENCES)
    gt_heatmap_dir = os.path.join(config.OUTPUT_DIR, "heatmaps_gt")
    os.makedirs(gt_heatmap_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print("  Generating Ground Truth Heatmaps")
    print("=" * 72)

    for seq_name in all_sequences:
        info = parse_seqinfo(seq_name)
        img_w, img_h = info["imwidth"], info["imheight"]
        frame_paths = get_frame_paths(seq_name)
        ref_frame = frame_paths[0]

        # GT heatmap
        gt_cents = _gt_centroids(seq_name)
        gt_density, xx, yy = generate_heatmap(gt_cents, img_w, img_h)
        out_path = os.path.join(gt_heatmap_dir, f"{seq_name}_heatmap_gt.png")
        overlay_heatmap(gt_density, xx, yy, ref_frame, out_path, seq_name,
                        title_prefix="Ground Truth Heatmap")

    # 3-way comparison images
    print("\n" + "=" * 72)
    print("  Generating 3-Way Heatmap Comparisons (GT vs SORT vs DeepSORT)")
    print("=" * 72)

    for seq_name in all_sequences:
        gt_path = os.path.join(gt_heatmap_dir, f"{seq_name}_heatmap_gt.png")
        sort_path = os.path.join(config.HEATMAP_DIR, f"{seq_name}_heatmap.png")
        ds_path = os.path.join(config.DEEPSORT_HEATMAP_DIR, f"{seq_name}_heatmap_deepsort.png")

        if not all(os.path.isfile(p) for p in [gt_path, sort_path, ds_path]):
            print(f"  Skipping {seq_name}: missing heatmap file(s)")
            continue

        img_gt = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
        img_sort = cv2.cvtColor(cv2.imread(sort_path), cv2.COLOR_BGR2RGB)
        img_ds = cv2.cvtColor(cv2.imread(ds_path), cv2.COLOR_BGR2RGB)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))
        fig.suptitle(f"Heatmap Comparison — {seq_name}", fontsize=18, fontweight="bold")

        ax1.imshow(img_gt)
        ax1.set_title("Ground Truth", fontsize=14, fontweight="bold")
        ax1.axis("off")

        ax2.imshow(img_sort)
        ax2.set_title("SORT (max_age=1)", fontsize=14)
        ax2.axis("off")

        ax3.imshow(img_ds)
        ax3.set_title("DeepSORT + OSNet ReID (max_age=30)", fontsize=14)
        ax3.axis("off")

        plt.tight_layout()
        out = os.path.join(config.COMPARISON_DIR, f"{seq_name}_heatmap_3way.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


if __name__ == "__main__":
    evaluate_heatmaps()
    generate_gt_heatmaps()
