"""Central configuration for the Retail-Heat SORT baseline pipeline."""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "MOT17", "train")
MOT20_DATA_DIR = os.path.join(ROOT_DIR, "MOT20", "train")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
DETECTION_DIR = os.path.join(OUTPUT_DIR, "detections")
TRACK_DIR = os.path.join(OUTPUT_DIR, "tracks")
HEATMAP_DIR = os.path.join(OUTPUT_DIR, "heatmaps")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")

# MOT17 sequences to process
SEQUENCES = ["MOT17-09", "MOT17-11"]

# MOT20 sequences to process
MOT20_SEQUENCES = ["MOT20-01"]

# ── YOLOv8 Detection ─────────────────────────────────────────────────────────
YOLO_MODEL = "yolov8x.pt"
YOLO_CONF = 0.5
YOLO_IOU = 0.45
YOLO_CLASSES = [0]  # person class only
YOLO_HALF = True    # FP16 inference (will fallback to FP32 on MPS)

# ── SORT Tracker ──────────────────────────────────────────────────────────────
SORT_MAX_AGE = 1       # original paper default — track dies after 1 missed frame
SORT_MIN_HITS = 3      # min detections before track is reported
SORT_IOU_THRESHOLD = 0.3  # min IoU for association

# ── DeepSORT Tracker ─────────────────────────────────────────────────────────
DEEPSORT_MAX_AGE = 30           # frames to keep alive without detection (occlusion survival)
DEEPSORT_N_INIT = 3             # detections before track is confirmed
DEEPSORT_MAX_COSINE_DISTANCE = 0.3  # max cosine distance for ReID matching
DEEPSORT_NN_BUDGET = 100        # appearance gallery size per track
DEEPSORT_MAX_IOU_DISTANCE = 0.5 # max IoU distance for fallback matching

# ── ReID (OSNet) ─────────────────────────────────────────────────────────────
REID_MODEL_NAME = "osnet_x1_0"
REID_MODEL_PATH = os.path.join(ROOT_DIR, "weights", "osnet_x1_0_market1501.pth")
REID_IMAGE_SIZE = (256, 128)    # (height, width) for ReID input

# ── DeepSORT Output Paths ────────────────────────────────────────────────────
DEEPSORT_TRACK_DIR = os.path.join(OUTPUT_DIR, "tracks_deepsort")
DEEPSORT_HEATMAP_DIR = os.path.join(OUTPUT_DIR, "heatmaps_deepsort")
DEEPSORT_METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics_deepsort")
COMPARISON_DIR = os.path.join(OUTPUT_DIR, "comparison")
OCCLUSION_LOG_DIR = os.path.join(OUTPUT_DIR, "occlusion_logs")

# ── Heatmap ───────────────────────────────────────────────────────────────────
HEATMAP_COLORMAP = "inferno"
HEATMAP_ALPHA = 0.6
HEATMAP_GRID_SIZE = 200  # KDE evaluation grid resolution

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_IOU_THRESHOLD = 0.5  # max IoU distance for motmetrics
