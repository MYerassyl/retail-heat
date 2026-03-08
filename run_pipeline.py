"""End-to-end pipeline orchestrator for the SORT baseline.

Usage:
    python run_pipeline.py
"""

import time

import config
from utils import ensure_dirs
from detection.detect import load_detector
from tracking.track import track_sequence
from evaluation.evaluate import evaluate_all
from evaluation.heatmap import generate_sequence_heatmap


def main():
    start_time = time.time()

    print("=" * 60)
    print("Retail-Heat: SORT Baseline Pipeline")
    print("=" * 60)

    # Step 1: Create output directories
    print("\n[1/4] Setting up directories...")
    ensure_dirs()

    # Step 2: Load detector once (shared across sequences)
    print("\n[2/4] Loading YOLOv8m detector...")
    model = load_detector()

    # Step 3: Run detection + tracking for each sequence
    print("\n[3/4] Running detection + tracking...")
    all_centroids = {}

    for seq_name in config.SEQUENCES:
        tracks, centroids = track_sequence(seq_name, model=model)
        all_centroids[seq_name] = centroids

    # Step 4: Evaluate tracking metrics
    print("\n[4/4] Evaluating and generating heatmaps...")
    evaluate_all()

    # Step 5: Generate heatmaps
    for seq_name in config.SEQUENCES:
        centroids = all_centroids.get(seq_name, [])
        if centroids:
            generate_sequence_heatmap(seq_name, centroids)
        else:
            print(f"  Skipping heatmap for {seq_name}: no centroids")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
