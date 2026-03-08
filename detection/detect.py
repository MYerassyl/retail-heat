"""YOLOv8m person detection module."""

import os

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

import config


def load_detector():
    """Load YOLOv8m pretrained model."""
    model = YOLO(config.YOLO_MODEL)
    return model


def detect_frame(model, frame_path):
    """Run YOLOv8 on a single frame.

    Returns:
        (N, 5) numpy array of [x1, y1, x2, y2, conf] for person detections.
        Returns empty (0, 5) array if no detections.
    """
    # Determine device and FP16 capability
    half = config.YOLO_HALF
    device = None
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        half = False  # MPS does not support FP16 reliably

    results = model(
        frame_path,
        conf=config.YOLO_CONF,
        iou=config.YOLO_IOU,
        classes=config.YOLO_CLASSES,
        half=half,
        device=device,
        verbose=False,
    )

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return np.empty((0, 5))

    boxes = result.boxes.xyxy.cpu().numpy()   # (N, 4)
    confs = result.boxes.conf.cpu().numpy()    # (N,)
    dets = np.hstack([boxes, confs.reshape(-1, 1)])  # (N, 5)
    return dets


def detect_sequence(model, seq_name):
    """Detect persons in all frames of a MOT17 sequence.

    Caches results as .npz file to skip re-detection on reruns.

    Returns:
        dict mapping frame_index (1-based) -> (N, 5) detections array
    """
    from utils import get_frame_paths

    cache_path = os.path.join(config.DETECTION_DIR, f"{seq_name}.npz")

    # Load from cache if available
    if os.path.isfile(cache_path):
        print(f"  Loading cached detections from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        detections = data["detections"].item()
        return detections

    frame_paths = get_frame_paths(seq_name)
    detections = {}

    print(f"  Running YOLOv8m detection on {seq_name} ({len(frame_paths)} frames)")
    for idx, fpath in enumerate(tqdm(frame_paths, desc=f"  Detecting {seq_name}")):
        frame_num = idx + 1  # MOT frames are 1-based
        dets = detect_frame(model, fpath)
        detections[frame_num] = dets

    # Cache detections
    np.savez_compressed(cache_path, detections=detections)
    print(f"  Cached detections to {cache_path}")

    return detections
