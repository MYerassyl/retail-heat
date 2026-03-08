"""Ablation study: compare different ReID models for DeepSORT.

Tests multiple ReID backbone + weight combinations and reports metrics.

Usage:
    python ablation_reid.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time

import config
from utils import ensure_dirs
from detection.detect import load_detector
from tracking.reid_embedder import ReIDEmbedder
from tracking.track_deepsort import track_sequence_deepsort
from evaluation.evaluate import evaluate_all


WEIGHTS_DIR = os.path.join(config.ROOT_DIR, "weights")

MODELS = [
    {
        "name": "osnet_x1_0 + Market-1501",
        "model_name": "osnet_x1_0",
        "model_path": os.path.join(WEIGHTS_DIR, "osnet_x1_0_market1501.pth"),
    },
    {
        "name": "osnet_ain_x1_0 + Market-1501",
        "model_name": "osnet_ain_x1_0",
        "model_path": os.path.join(WEIGHTS_DIR, "osnet_ain_x1_0_market1501.pth"),
    },
    {
        "name": "osnet_ain_x1_0 + MSMT17",
        "model_name": "osnet_ain_x1_0",
        "model_path": os.path.join(WEIGHTS_DIR, "osnet_ain_x1_0_msmt17.pth"),
    },
    {
        "name": "osnet_ibn_x1_0 + Market-1501",
        "model_name": "osnet_ibn_x1_0",
        "model_path": os.path.join(WEIGHTS_DIR, "osnet_ibn_x1_0_market1501.pth"),
    },
]


def parse_metrics_file(path):
    """Parse a metrics txt file and return the OVERALL row as a dict."""
    with open(path) as f:
        lines = f.readlines()
    header = lines[0].split()
    for line in lines[1:]:
        parts = line.split()
        if parts[0] == "OVERALL":
            vals = parts[1:]
            return {h: vals[i] for i, h in enumerate(header)}
    return None


def main():
    ensure_dirs()

    print("=" * 70)
    print("ReID Model Ablation Study")
    print(f"Tracker config: max_age={config.DEEPSORT_MAX_AGE}, "
          f"n_init={config.DEEPSORT_N_INIT}, "
          f"cosine={config.DEEPSORT_MAX_COSINE_DISTANCE}, "
          f"iou={config.DEEPSORT_MAX_IOU_DISTANCE}")
    print("=" * 70)

    # Load detector once
    print("\nLoading YOLOv8 detector...")
    detector = load_detector()

    results = []

    for model_cfg in MODELS:
        name = model_cfg["name"]
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"{'='*70}")

        t0 = time.time()

        # Load embedder with this model
        embedder = ReIDEmbedder(
            model_name=model_cfg["model_name"],
            model_path=model_cfg["model_path"],
        )

        # Track all sequences
        for seq_name in config.SEQUENCES:
            track_sequence_deepsort(
                seq_name,
                detector_model=detector,
                embedder=embedder,
            )

        # Evaluate
        metrics_file = "deepsort_reid_metrics.txt"
        evaluate_all(
            track_dir=config.DEEPSORT_TRACK_DIR,
            metrics_dir=config.DEEPSORT_METRICS_DIR,
            metrics_filename=metrics_file,
        )

        elapsed = time.time() - t0

        # Parse results
        metrics_path = os.path.join(config.DEEPSORT_METRICS_DIR, metrics_file)
        metrics = parse_metrics_file(metrics_path)
        if metrics:
            results.append({
                "name": name,
                "MOTA": metrics["MOTA"],
                "IDF1": metrics["IDF1"],
                "IDSW": metrics["IDSW"],
                "FP": metrics["FP"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "time": f"{elapsed:.1f}s",
            })

    # Print summary table
    print(f"\n{'='*70}")
    print("ABLATION RESULTS SUMMARY (OVERALL)")
    print(f"{'='*70}")
    print(f"{'Model':<35} {'MOTA':>8} {'IDF1':>8} {'IDSW':>6} {'FP':>7} {'Prec':>8} {'Rec':>8} {'Time':>8}")
    print("-" * 100)
    for r in results:
        print(f"{r['name']:<35} {r['MOTA']:>8} {r['IDF1']:>8} {r['IDSW']:>6} {r['FP']:>7} "
              f"{r['Precision']:>8} {r['Recall']:>8} {r['time']:>8}")

    # Save results
    out_path = os.path.join(config.COMPARISON_DIR, "reid_ablation.txt")
    with open(out_path, "w") as f:
        f.write(f"{'Model':<35} {'MOTA':>8} {'IDF1':>8} {'IDSW':>6} {'FP':>7} {'Prec':>8} {'Rec':>8} {'Time':>8}\n")
        f.write("-" * 100 + "\n")
        for r in results:
            f.write(f"{r['name']:<35} {r['MOTA']:>8} {r['IDF1']:>8} {r['IDSW']:>6} {r['FP']:>7} "
                    f"{r['Precision']:>8} {r['Recall']:>8} {r['time']:>8}\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
