"""Render tracking results as MP4 videos with bounding boxes and track IDs.

Usage:
    python visualize.py                    # both SORT and DeepSORT
    python visualize.py --tracker sort     # SORT only
    python visualize.py --tracker deepsort # DeepSORT only
"""

import argparse
import os
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

import config
from utils import get_frame_paths, parse_seqinfo


# Distinct colors for track IDs (BGR)
_COLORS = [
    (230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66),
    (0, 114, 178), (213, 94, 0), (204, 121, 167), (0, 0, 0),
    (127, 127, 127), (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 0),
    (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128),
]


def _color_for_id(track_id):
    return _COLORS[track_id % len(_COLORS)]


def load_tracks_by_frame(track_file):
    """Load MOT-format track file into {frame_num: [(id, x, y, w, h), ...]}."""
    data = np.loadtxt(track_file, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    by_frame = defaultdict(list)
    for row in data:
        frame = int(row[0])
        tid = int(row[1])
        x, y, w, h = row[2], row[3], row[4], row[5]
        by_frame[frame].append((tid, x, y, w, h))
    return by_frame


def render_video(seq_name, track_file, output_path, label="SORT", fps=None):
    """Render a sequence with tracking overlays as MP4."""
    frame_paths = get_frame_paths(seq_name)
    info = parse_seqinfo(seq_name)
    if fps is None:
        fps = int(info.get("framerate", 30))
    img_w, img_h = info["imwidth"], info["imheight"]

    tracks = load_tracks_by_frame(track_file)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (img_w, img_h))

    print(f"  Rendering {label} video for {seq_name} ({len(frame_paths)} frames)...")
    for idx, fpath in enumerate(tqdm(frame_paths, desc=f"  {label} {seq_name}")):
        frame_num = idx + 1
        frame = cv2.imread(fpath)

        for tid, x, y, w, h in tracks.get(frame_num, []):
            color = _color_for_id(tid)
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Label background
            txt = f"ID {tid}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, txt, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # HUD
        cv2.putText(frame, f"{label} | {seq_name} | Frame {frame_num}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        writer.write(frame)

    writer.release()
    print(f"  Saved: {output_path}")


def _draw_overlays(frame, tracks_for_frame, label, seq_name, frame_num):
    """Draw bounding boxes, IDs, and HUD on a frame (in-place)."""
    for tid, x, y, w, h in tracks_for_frame:
        color = _color_for_id(tid)
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        txt = f"ID {tid}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, txt, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"{label} | {seq_name} | Frame {frame_num}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def render_side_by_side(seq_name, sort_track_file, deepsort_track_file,
                        output_path, fps=None):
    """Render SORT and DeepSORT side by side in a single video."""
    frame_paths = get_frame_paths(seq_name)
    info = parse_seqinfo(seq_name)
    if fps is None:
        fps = int(info.get("framerate", 30))
    img_w, img_h = info["imwidth"], info["imheight"]

    sort_tracks = load_tracks_by_frame(sort_track_file)
    ds_tracks = load_tracks_by_frame(deepsort_track_file)

    # Side-by-side: double width + divider
    out_w = img_w * 2 + 4
    out_h = img_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    print(f"  Rendering side-by-side video for {seq_name} ({len(frame_paths)} frames)...")
    for idx, fpath in enumerate(tqdm(frame_paths, desc=f"  SBS {seq_name}")):
        frame_num = idx + 1
        frame_orig = cv2.imread(fpath)

        # Left: SORT
        left = frame_orig.copy()
        _draw_overlays(left, sort_tracks.get(frame_num, []),
                       "SORT", seq_name, frame_num)

        # Right: DeepSORT
        right = frame_orig.copy()
        _draw_overlays(right, ds_tracks.get(frame_num, []),
                       "DeepSORT+ReID", seq_name, frame_num)

        # Combine with white divider
        divider = np.full((img_h, 4, 3), 255, dtype=np.uint8)
        combined = np.hstack([left, divider, right])
        writer.write(combined)

    writer.release()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize tracking results as video")
    parser.add_argument("--tracker", choices=["sort", "deepsort", "botsort", "both", "all"],
                        default="both", help="Which tracker results to render")
    parser.add_argument("--dataset", choices=["mot17", "mot20", "all"],
                        default="all", help="Which dataset sequences to render")
    parser.add_argument("--side-by-side", action="store_true",
                        help="Generate SORT vs DeepSORT side-by-side comparison videos")
    args = parser.parse_args()

    video_dir = os.path.join(config.OUTPUT_DIR, "videos")
    os.makedirs(video_dir, exist_ok=True)

    sequences = []
    if args.dataset in ("mot17", "all"):
        sequences += config.SEQUENCES
    if args.dataset in ("mot20", "all"):
        sequences += config.MOT20_SEQUENCES

    for seq_name in sequences:
        if args.tracker in ("sort", "both"):
            track_file = os.path.join(config.TRACK_DIR, f"{seq_name}.txt")
            if os.path.isfile(track_file):
                out = os.path.join(video_dir, f"{seq_name}_sort.mp4")
                render_video(seq_name, track_file, out, label="SORT")
            else:
                print(f"  Skipping SORT {seq_name}: no track file")

        if args.tracker in ("deepsort", "both", "all"):
            track_file = os.path.join(config.DEEPSORT_TRACK_DIR, f"{seq_name}.txt")
            if os.path.isfile(track_file):
                out = os.path.join(video_dir, f"{seq_name}_deepsort.mp4")
                render_video(seq_name, track_file, out, label="DeepSORT")
            else:
                print(f"  Skipping DeepSORT {seq_name}: no track file")

        if args.tracker in ("botsort", "all"):
            track_file = os.path.join(config.OUTPUT_DIR, "tracks_botsort", f"{seq_name}.txt")
            if os.path.isfile(track_file):
                out = os.path.join(video_dir, f"{seq_name}_botsort.mp4")
                render_video(seq_name, track_file, out, label="BoTSORT")
            else:
                print(f"  Skipping BoTSORT {seq_name}: no track file")

    # Side-by-side comparison videos
    if args.side_by_side:
        comp_dir = os.path.join(config.OUTPUT_DIR, "comparison")
        os.makedirs(comp_dir, exist_ok=True)
        for seq_name in sequences:
            sort_file = os.path.join(config.TRACK_DIR, f"{seq_name}.txt")
            ds_file = os.path.join(config.DEEPSORT_TRACK_DIR, f"{seq_name}.txt")
            if os.path.isfile(sort_file) and os.path.isfile(ds_file):
                out = os.path.join(comp_dir, f"{seq_name}_sort_vs_deepsort.mp4")
                render_side_by_side(seq_name, sort_file, ds_file, out)
            else:
                print(f"  Skipping SBS {seq_name}: missing track file(s)")

    print(f"\nVideos saved to: {video_dir}")


if __name__ == "__main__":
    main()
