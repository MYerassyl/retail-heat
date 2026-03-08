"""Detection + SORT tracking pipeline."""

import os

import numpy as np
from tqdm import tqdm

import config
from detection.detect import load_detector, detect_sequence
from tracking.sort_tracker import Sort
from utils import ensure_dirs, get_frame_paths, save_tracks_mot_format, xyxy_to_xywh


def track_sequence(seq_name, model=None):
    """Run detection + SORT tracking on a MOT17 sequence.

    Returns:
        tracks_mot: list of dicts in MOT format
        centroids:  list of (cx, cy) tuples from tracked objects
    """
    print(f"\n{'='*60}")
    print(f"Processing sequence: {seq_name}")
    print(f"{'='*60}")

    # Load detector if not provided
    if model is None:
        model = load_detector()

    # Run detection (or load from cache)
    detections = detect_sequence(model, seq_name)

    # Initialize SORT tracker
    tracker = Sort(
        max_age=config.SORT_MAX_AGE,
        min_hits=config.SORT_MIN_HITS,
        iou_threshold=config.SORT_IOU_THRESHOLD,
    )

    tracks_mot = []
    centroids = []

    frame_paths = get_frame_paths(seq_name)
    num_frames = len(frame_paths)

    print(f"  Running SORT tracking on {num_frames} frames")
    for frame_num in tqdm(range(1, num_frames + 1), desc=f"  Tracking {seq_name}"):
        dets = detections.get(frame_num, np.empty((0, 5)))

        # Update tracker: returns (M, 5) [x1, y1, x2, y2, track_id]
        tracked = tracker.update(dets)

        for trk in tracked:
            x1, y1, x2, y2, track_id = trk
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0

            tracks_mot.append({
                "frame": frame_num,
                "id": int(track_id),
                "x": x1,
                "y": y1,
                "w": w,
                "h": h,
                "conf": 1.0,  # SORT doesn't propagate confidence
            })

            centroids.append((cx, cy))

    # Save tracks in MOT format
    track_path = os.path.join(config.TRACK_DIR, f"{seq_name}.txt")
    save_tracks_mot_format(tracks_mot, track_path)
    print(f"  Saved {len(tracks_mot)} track entries to {track_path}")
    print(f"  Collected {len(centroids)} centroids for heatmap")

    return tracks_mot, centroids
