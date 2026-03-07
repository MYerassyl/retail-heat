"""Run both SORT and DeepSORT pipelines on MOT20 dataset.

Usage:
    python run_pipeline_mot20.py
"""

import time

import config
from utils import ensure_dirs
from detect import load_detector
from reid_embedder import ReIDEmbedder
from track import track_sequence
from track_deepsort import track_sequence_deepsort
from evaluate import evaluate_all
from heatmap import generate_sequence_heatmap

MOT20_SEQUENCES = config.MOT20_SEQUENCES


def main():
    start_time = time.time()

    print("=" * 60)
    print("Retail-Heat: MOT20 Evaluation (SORT + DeepSORT)")
    print("=" * 60)

    ensure_dirs()

    # Load shared models
    print("\n[1/6] Loading YOLOv8 detector...")
    detector = load_detector()

    print("\n[2/6] Loading ReID embedder...")
    embedder = ReIDEmbedder()

    # --- SORT ---
    print("\n[3/6] Running SORT on MOT20...")
    sort_centroids = {}
    for seq_name in MOT20_SEQUENCES:
        tracks, centroids = track_sequence(seq_name, model=detector)
        sort_centroids[seq_name] = centroids

    print("\n[4/6] Running DeepSORT on MOT20...")
    ds_centroids = {}
    for seq_name in MOT20_SEQUENCES:
        tracks, centroids = track_sequence_deepsort(
            seq_name, detector_model=detector, embedder=embedder,
        )
        ds_centroids[seq_name] = centroids

    # --- Evaluate ---
    print("\n[5/6] Evaluating SORT on MOT20...")
    evaluate_all(
        sequences=MOT20_SEQUENCES,
        track_dir=config.TRACK_DIR,
        metrics_dir=config.METRICS_DIR,
        metrics_filename="sort_mot20_metrics.txt",
    )

    print("\n[6/6] Evaluating DeepSORT on MOT20...")
    evaluate_all(
        sequences=MOT20_SEQUENCES,
        track_dir=config.DEEPSORT_TRACK_DIR,
        metrics_dir=config.DEEPSORT_METRICS_DIR,
        metrics_filename="deepsort_mot20_metrics.txt",
    )

    # Generate heatmaps
    for seq_name in MOT20_SEQUENCES:
        if sort_centroids.get(seq_name):
            generate_sequence_heatmap(
                seq_name, sort_centroids[seq_name],
                output_dir=config.HEATMAP_DIR,
                title_prefix="SORT Heatmap (MOT20)",
            )
        if ds_centroids.get(seq_name):
            generate_sequence_heatmap(
                seq_name, ds_centroids[seq_name],
                output_dir=config.DEEPSORT_HEATMAP_DIR,
                title_prefix="DeepSORT Heatmap (MOT20)",
                filename_suffix="_deepsort",
            )

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"MOT20 pipeline complete in {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
