"""Detection + DeepSORT tracking pipeline with external ReID embeddings."""

import json
import os

import cv2
import numpy as np
from tqdm import tqdm

import config
from detect import load_detector, detect_sequence
from deepsort_tracker import DeepSortTracker
from reid_embedder import ReIDEmbedder
from utils import get_frame_paths, save_tracks_mot_format


def track_sequence_deepsort(seq_name, detector_model=None, embedder=None,
                            tracker_kwargs=None, track_dir=None):
    """Run detection + DeepSORT tracking with ReID on a MOT17 sequence.

    Args:
        seq_name: MOT17 sequence name
        detector_model: pre-loaded YOLO model (loads one if None)
        embedder: pre-loaded ReIDEmbedder (loads one if None)
        tracker_kwargs: optional dict of DeepSortTracker constructor overrides
        track_dir: output directory for tracks (defaults to config.DEEPSORT_TRACK_DIR)

    Returns:
        tracks_mot: list of dicts in MOT format
        centroids: list of (cx, cy) tuples from tracked objects
    """
    if track_dir is None:
        track_dir = config.DEEPSORT_TRACK_DIR

    print(f"\n{'='*60}")
    print(f"DeepSORT Processing: {seq_name}")
    print(f"{'='*60}")

    # Load detector if not provided
    if detector_model is None:
        detector_model = load_detector()

    # Load embedder if not provided
    if embedder is None:
        embedder = ReIDEmbedder()

    # Initialize DeepSORT tracker
    kwargs = tracker_kwargs or {}
    tracker = DeepSortTracker(**kwargs)

    # Run detection (or load from cache)
    detections = detect_sequence(detector_model, seq_name)

    tracks_mot = []
    centroids = []

    frame_paths = get_frame_paths(seq_name)
    num_frames = len(frame_paths)

    print(f"  Running DeepSORT tracking on {num_frames} frames")
    for frame_num in tqdm(range(1, num_frames + 1), desc=f"  DeepSORT {seq_name}"):
        dets = detections.get(frame_num, np.empty((0, 5)))

        # Read frame for ReID crops
        frame_bgr = cv2.imread(frame_paths[frame_num - 1])

        # Extract ReID embeddings
        if len(dets) > 0:
            embeddings = embedder.extract_and_embed(frame_bgr, dets)
        else:
            embeddings = np.empty((0, 512), dtype=np.float32)

        # Update tracker
        tracked = tracker.update(dets, embeddings, frame_num)

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
                "conf": 1.0,
            })

            centroids.append((cx, cy))

    # Save tracks in MOT format
    track_path = os.path.join(track_dir, f"{seq_name}.txt")
    save_tracks_mot_format(tracks_mot, track_path)
    print(f"  Saved {len(tracks_mot)} track entries to {track_path}")
    print(f"  Collected {len(centroids)} centroids for heatmap")

    # Save occlusion log
    occlusion_log = tracker.get_occlusion_log()
    occ_path = os.path.join(config.OCCLUSION_LOG_DIR, f"{seq_name}_occlusions.json")
    with open(occ_path, "w") as f:
        json.dump(occlusion_log, f, indent=2)
    print(f"  Saved {len(occlusion_log)} occlusion events to {occ_path}")

    return tracks_mot, centroids
