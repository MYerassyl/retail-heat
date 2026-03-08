"""End-to-end pipeline orchestrator for DeepSORT + ReID.

Usage:
    python run_pipeline_deepsort.py
"""

import time

import config
from utils import ensure_dirs
from detection.detect import load_detector
from tracking.reid_embedder import ReIDEmbedder
from tracking.track_deepsort import track_sequence_deepsort
from evaluation.evaluate import evaluate_all
from evaluation.heatmap import generate_sequence_heatmap


def main():
    start_time = time.time()

    print("=" * 60)
    print("Retail-Heat: DeepSORT + ReID Pipeline")
    print("=" * 60)

    # Step 1: Create output directories
    print("\n[1/5] Setting up directories...")
    ensure_dirs()

    # Step 2: Load detector once (shared across sequences)
    print("\n[2/5] Loading YOLOv8 detector...")
    model = load_detector()

    # Step 3: Load ReID embedder once (shared across sequences)
    print("\n[3/5] Loading ReID embedder...")
    embedder = ReIDEmbedder()

    # Step 4: Run detection + DeepSORT tracking for each sequence
    print("\n[4/5] Running detection + DeepSORT tracking...")
    all_centroids = {}

    for seq_name in config.SEQUENCES:
        tracks, centroids = track_sequence_deepsort(
            seq_name,
            detector_model=model,
            embedder=embedder,
        )
        all_centroids[seq_name] = centroids

    # Step 5: Evaluate tracking metrics
    print("\n[5/5] Evaluating and generating heatmaps...")
    evaluate_all(
        track_dir=config.DEEPSORT_TRACK_DIR,
        metrics_dir=config.DEEPSORT_METRICS_DIR,
        metrics_filename="deepsort_reid_metrics.txt",
    )

    # Generate heatmaps
    for seq_name in config.SEQUENCES:
        centroids = all_centroids.get(seq_name, [])
        if centroids:
            generate_sequence_heatmap(
                seq_name,
                centroids,
                output_dir=config.DEEPSORT_HEATMAP_DIR,
                title_prefix="DeepSORT + ReID Heatmap",
                filename_suffix="_deepsort",
            )
        else:
            print(f"  Skipping heatmap for {seq_name}: no centroids")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"DeepSORT pipeline complete in {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
