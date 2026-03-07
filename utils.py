"""I/O helpers and format conversion utilities."""

import os
import configparser

import numpy as np

import config


def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [config.OUTPUT_DIR, config.DETECTION_DIR,
              config.TRACK_DIR, config.HEATMAP_DIR, config.METRICS_DIR,
              config.DEEPSORT_TRACK_DIR, config.DEEPSORT_HEATMAP_DIR,
              config.DEEPSORT_METRICS_DIR, config.COMPARISON_DIR,
              config.OCCLUSION_LOG_DIR]:
        os.makedirs(d, exist_ok=True)


def _resolve_seq_dir(seq_name):
    """Resolve the sequence directory for MOT17 or MOT20.

    MOT17 uses '{seq_name}-FRCNN' subdirectories, MOT20 uses '{seq_name}' directly.
    """
    if seq_name.startswith("MOT20"):
        seq_dir = os.path.join(config.MOT20_DATA_DIR, seq_name)
    else:
        seq_dir = os.path.join(config.DATA_DIR, f"{seq_name}-FRCNN")
    if not os.path.isdir(seq_dir):
        raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")
    return seq_dir


def get_frame_paths(seq_name):
    """Return sorted list of frame image paths for a MOT sequence."""
    seq_dir = os.path.join(_resolve_seq_dir(seq_name), "img1")
    if not os.path.isdir(seq_dir):
        raise FileNotFoundError(f"Sequence image directory not found: {seq_dir}")
    frames = sorted(
        os.path.join(seq_dir, f)
        for f in os.listdir(seq_dir)
        if f.endswith((".jpg", ".png"))
    )
    return frames


def parse_seqinfo(seq_name):
    """Parse seqinfo.ini for a MOT sequence, return dict with metadata."""
    ini_path = os.path.join(_resolve_seq_dir(seq_name), "seqinfo.ini")
    cp = configparser.ConfigParser()
    cp.read(ini_path)
    info = dict(cp["Sequence"])
    info["seqlength"] = int(info["seqlength"])
    info["imwidth"] = int(info["imwidth"])
    info["imheight"] = int(info["imheight"])
    return info


def load_mot_gt(seq_name):
    """Load ground truth, filtering for active pedestrians.

    Returns numpy array with columns:
        [frame, id, x, y, w, h, conf, class, visibility]

    Filtering: class == 1 (pedestrian) AND conf != 0 (active).
    """
    gt_path = os.path.join(_resolve_seq_dir(seq_name), "gt", "gt.txt")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")
    gt = np.loadtxt(gt_path, delimiter=",")
    # Column indices (0-based): 0=frame, 1=id, 2=x, 3=y, 4=w, 5=h, 6=conf, 7=class, 8=visibility
    mask = (gt[:, 7] == 1) & (gt[:, 6] != 0)
    return gt[mask]


def save_tracks_mot_format(tracks, filepath):
    """Save tracks in MOT challenge format.

    Each row: frame, id, x, y, w, h, conf, -1, -1, -1
    tracks: list of dicts with keys {frame, id, x, y, w, h, conf}
    """
    with open(filepath, "w") as f:
        for t in tracks:
            f.write(f"{t['frame']},{t['id']},{t['x']:.2f},{t['y']:.2f},"
                    f"{t['w']:.2f},{t['h']:.2f},{t['conf']:.4f},-1,-1,-1\n")


def xyxy_to_xywh(bbox):
    """Convert [x1, y1, x2, y2] to [x, y, w, h] (top-left + size)."""
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    return np.array([x1, y1, x2 - x1, y2 - y1])


def xywh_to_xyxy(bbox):
    """Convert [x, y, w, h] (top-left + size) to [x1, y1, x2, y2]."""
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    return np.array([x, y, x + w, y + h])
