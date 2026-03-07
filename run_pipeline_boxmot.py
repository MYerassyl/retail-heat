"""Run BoxMOT trackers (DeepOCSORT, BoTSORT, StrongSORT, ByteTrack) on MOT17 + MOT20.

Usage:
    python run_pipeline_boxmot.py
"""

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import config
from utils import ensure_dirs, get_frame_paths, save_tracks_mot_format
from detect import load_detector, detect_sequence
from evaluate import evaluate_all
from heatmap import generate_sequence_heatmap


# Trackers to test: (name, class_name, needs_reid)
TRACKERS = [
    ("BoTSORT", "BotSort", True),
]

REID_WEIGHTS = Path("osnet_x1_0_market1501.pt")  # stronger ReID model


def create_tracker(class_name, needs_reid):
    """Create a BoxMOT tracker instance."""
    import boxmot
    tracker_cls = getattr(boxmot, class_name)
    kwargs = dict(
        device="cpu",
        half=False,
        track_high_thresh=0.25,      # lower: keep more good detections (was 0.5)
        track_low_thresh=0.1,        # second-stage for weak detections
        new_track_thresh=0.25,       # lower: don't miss new tracks (was 0.6)
        track_buffer=30,             # frames to keep lost tracks
        match_thresh=0.8,            # association threshold
        proximity_thresh=0.5,        # IoU gate for ReID
        appearance_thresh=0.5,       # stricter ReID matching (was 0.25)
        cmc_method="sof",            # sparse optical flow (lightweight CMC)
        fuse_first_associate=False,
        with_reid=True,
    )
    if needs_reid:
        kwargs["reid_weights"] = REID_WEIGHTS
    return tracker_cls(**kwargs)


def track_sequence_boxmot(seq_name, tracker, tracker_name, detector_model, track_dir):
    """Run detection + BoxMOT tracking on a sequence."""
    print(f"\n{'='*60}")
    print(f"{tracker_name} Processing: {seq_name}")
    print(f"{'='*60}")

    detections = detect_sequence(detector_model, seq_name)
    frame_paths = get_frame_paths(seq_name)
    num_frames = len(frame_paths)

    tracks_mot = []
    centroids = []

    print(f"  Running {tracker_name} tracking on {num_frames} frames")
    for frame_num in tqdm(range(1, num_frames + 1), desc=f"  {tracker_name} {seq_name}"):
        dets = detections.get(frame_num, np.empty((0, 5)))
        frame_bgr = cv2.imread(frame_paths[frame_num - 1])

        # BoxMOT expects Nx6: [x1, y1, x2, y2, conf, cls]
        if len(dets) > 0:
            cls_col = np.zeros((len(dets), 1))  # class 0 = person
            dets_6 = np.hstack([dets, cls_col])
        else:
            dets_6 = np.empty((0, 6))

        # Update tracker — returns Nx8: [x1, y1, x2, y2, id, conf, cls, ind]
        tracked = tracker.update(dets_6, frame_bgr)

        for trk in tracked:
            x1, y1, x2, y2, track_id = trk[0], trk[1], trk[2], trk[3], int(trk[4])
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0

            tracks_mot.append({
                "frame": frame_num,
                "id": track_id,
                "x": float(x1),
                "y": float(y1),
                "w": float(w),
                "h": float(h),
                "conf": 1.0,
            })
            centroids.append((cx, cy))

    # Save tracks
    os.makedirs(track_dir, exist_ok=True)
    track_path = os.path.join(track_dir, f"{seq_name}.txt")
    save_tracks_mot_format(tracks_mot, track_path)
    print(f"  Saved {len(tracks_mot)} track entries to {track_path}")

    return tracks_mot, centroids


def main():
    start_time = time.time()

    print("=" * 60)
    print("Retail-Heat: BoxMOT Multi-Tracker Evaluation")
    print("=" * 60)

    ensure_dirs()

    print("\nLoading YOLOv8 detector...")
    detector = load_detector()

    # Test sequences: MOT17 + MOT20
    sequences = config.SEQUENCES + config.MOT20_SEQUENCES

    results = []

    for tracker_name, class_name, needs_reid in TRACKERS:
        print(f"\n{'#'*60}")
        print(f"# Tracker: {tracker_name}")
        print(f"{'#'*60}")

        t0 = time.time()
        track_dir = os.path.join(config.OUTPUT_DIR, f"tracks_{tracker_name.lower()}")
        metrics_dir = os.path.join(config.OUTPUT_DIR, f"metrics_{tracker_name.lower()}")
        os.makedirs(metrics_dir, exist_ok=True)

        for seq_name in sequences:
            tracker = create_tracker(class_name, needs_reid)
            track_sequence_boxmot(
                seq_name, tracker, tracker_name, detector, track_dir,
            )

        # Evaluate MOT17
        print(f"\n  Evaluating {tracker_name} on MOT17...")
        evaluate_all(
            sequences=config.SEQUENCES,
            track_dir=track_dir,
            metrics_dir=metrics_dir,
            metrics_filename=f"{tracker_name.lower()}_mot17_metrics.txt",
        )

        # Evaluate MOT20
        print(f"\n  Evaluating {tracker_name} on MOT20...")
        evaluate_all(
            sequences=config.MOT20_SEQUENCES,
            track_dir=track_dir,
            metrics_dir=metrics_dir,
            metrics_filename=f"{tracker_name.lower()}_mot20_metrics.txt",
        )

        elapsed = time.time() - t0
        print(f"  {tracker_name} complete in {elapsed:.1f}s")

    total = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"BoxMOT evaluation complete in {total:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
