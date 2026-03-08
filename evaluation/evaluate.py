"""MOT evaluation using py-motmetrics."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import motmetrics as mm

import config
from utils import load_mot_gt


def _iou_distance_matrix(gt_boxes, pred_boxes, max_iou=0.5):
    """Compute IoU distance matrix (1 - IoU) between GT and predicted boxes.

    Replaces mm.distances.iou_matrix which breaks on NumPy 2.0.
    Boxes are in [x, y, w, h] format.
    Returns (len(gt), len(pred)) matrix with np.nan for entries above max_iou distance.
    """
    gt = np.asarray(gt_boxes, dtype=float)
    pred = np.asarray(pred_boxes, dtype=float)

    if gt.size == 0 or pred.size == 0:
        return np.empty((len(gt), len(pred)))

    # Convert xywh -> xyxy
    gt_xyxy = gt.copy()
    gt_xyxy[:, 2] = gt[:, 0] + gt[:, 2]
    gt_xyxy[:, 3] = gt[:, 1] + gt[:, 3]

    pred_xyxy = pred.copy()
    pred_xyxy[:, 2] = pred[:, 0] + pred[:, 2]
    pred_xyxy[:, 3] = pred[:, 1] + pred[:, 3]

    # Pairwise IoU
    n, m = len(gt_xyxy), len(pred_xyxy)
    g = np.expand_dims(gt_xyxy, 1)    # (N, 1, 4)
    p = np.expand_dims(pred_xyxy, 0)  # (1, M, 4)

    xx1 = np.maximum(g[..., 0], p[..., 0])
    yy1 = np.maximum(g[..., 1], p[..., 1])
    xx2 = np.minimum(g[..., 2], p[..., 2])
    yy2 = np.minimum(g[..., 3], p[..., 3])

    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    area_g = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
    area_p = (p[..., 2] - p[..., 0]) * (p[..., 3] - p[..., 1])
    iou = inter / np.maximum(area_g + area_p - inter, 1e-10)

    # Convert to distance (1 - IoU), mask out pairs above max distance
    dist = 1.0 - iou
    dist[dist > max_iou] = np.nan
    return dist


def load_tracks(seq_name, track_dir=None):
    """Load predicted tracks from MOT-format output file.

    Args:
        seq_name: sequence name
        track_dir: directory containing track files (defaults to config.TRACK_DIR)

    Returns numpy array with columns: [frame, id, x, y, w, h, conf, -, -, -]
    """
    if track_dir is None:
        track_dir = config.TRACK_DIR
    track_path = os.path.join(track_dir, f"{seq_name}.txt")
    if not os.path.isfile(track_path):
        raise FileNotFoundError(f"Track file not found: {track_path}")
    tracks = np.loadtxt(track_path, delimiter=",")
    if tracks.ndim == 1:
        tracks = tracks.reshape(1, -1)
    return tracks


def evaluate_sequence(seq_name, track_dir=None):
    """Evaluate tracking results against ground truth for one sequence.

    Args:
        seq_name: sequence name
        track_dir: directory containing track files (defaults to config.TRACK_DIR)

    Returns motmetrics accumulator.
    """
    print(f"\n  Evaluating {seq_name}...")

    gt = load_mot_gt(seq_name)
    pred = load_tracks(seq_name, track_dir=track_dir)

    acc = mm.MOTAccumulator(auto_id=True)

    # Get all frame numbers from both GT and predictions
    gt_frames = set(gt[:, 0].astype(int))
    pred_frames = set(pred[:, 0].astype(int))
    all_frames = sorted(gt_frames | pred_frames)

    for frame_id in all_frames:
        # GT objects in this frame: [id, x, y, w, h]
        gt_mask = gt[:, 0].astype(int) == frame_id
        gt_ids = gt[gt_mask, 1].astype(int).tolist()
        gt_boxes = gt[gt_mask, 2:6]  # x, y, w, h

        # Predicted objects in this frame: [id, x, y, w, h]
        pred_mask = pred[:, 0].astype(int) == frame_id
        pred_ids = pred[pred_mask, 1].astype(int).tolist()
        pred_boxes = pred[pred_mask, 2:6]  # x, y, w, h

        # Compute IoU distance matrix (1 - IoU)
        # Using custom implementation because mm.distances.iou_matrix
        # uses np.asfarray which was removed in NumPy 2.0
        distances = _iou_distance_matrix(
            gt_boxes, pred_boxes, max_iou=config.EVAL_IOU_THRESHOLD
        )

        acc.update(gt_ids, pred_ids, distances)

    return acc


def evaluate_all(sequences=None, track_dir=None, metrics_dir=None,
                  metrics_filename=None):
    """Evaluate all sequences and print summary metrics.

    Args:
        sequences: list of sequence names (defaults to config.SEQUENCES)
        track_dir: directory containing track files (defaults to config.TRACK_DIR)
        metrics_dir: directory to save metrics (defaults to config.METRICS_DIR)
        metrics_filename: filename for metrics output (defaults to sort_baseline_metrics.txt)

    Returns dict of per-sequence metrics.
    """
    if sequences is None:
        sequences = config.SEQUENCES
    if metrics_dir is None:
        metrics_dir = config.METRICS_DIR
    if metrics_filename is None:
        metrics_filename = "sort_baseline_metrics.txt"

    print("\n" + "=" * 60)
    print("MOT Evaluation")
    print("=" * 60)

    accumulators = []
    names = []

    for seq_name in sequences:
        try:
            acc = evaluate_sequence(seq_name, track_dir=track_dir)
            accumulators.append(acc)
            names.append(seq_name)
        except FileNotFoundError as e:
            print(f"  Skipping {seq_name}: {e}")

    if not accumulators:
        print("  No sequences to evaluate.")
        return {}

    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accumulators,
        names=names,
        metrics=[
            "mota", "idf1", "num_switches",
            "mostly_tracked", "mostly_lost",
            "num_false_positives", "num_misses",
            "precision", "recall",
        ],
        generate_overall=True,
    )

    # Rename columns for readability
    summary.columns = [
        "MOTA", "IDF1", "IDSW",
        "MT", "ML", "FP", "FN",
        "Precision", "Recall",
    ]

    print("\n" + str(summary))

    # Save metrics to file
    metrics_path = os.path.join(metrics_dir, metrics_filename)
    with open(metrics_path, "w") as f:
        f.write(str(summary))
    print(f"\n  Metrics saved to {metrics_path}")

    return summary


if __name__ == "__main__":
    evaluate_all()
