"""SORT: Simple Online and Realtime Tracking (Bewley et al., 2016).

Implemented from scratch following the original paper.
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def convert_bbox_to_z(bbox):
    """Convert [x1, y1, x2, y2] bounding box to Kalman state [cx, cy, s, r].

    cx, cy: center coordinates
    s: scale (area)
    r: aspect ratio (w / h)
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h  # scale = area
    r = w / float(h) if h > 0 else 0
    return np.array([cx, cy, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """Convert Kalman state [cx, cy, s, r] back to [x1, y1, x2, y2] bbox.

    Optionally appends confidence score.
    """
    w = np.sqrt(x[2] * x[3]) if x[2] * x[3] > 0 else 0
    h = x[2] / w if w > 0 else 0
    x1 = x[0] - w / 2.0
    y1 = x[1] - h / 2.0
    x2 = x[0] + w / 2.0
    y2 = x[1] + h / 2.0
    if score is None:
        return np.array([x1, y1, x2, y2]).reshape((1, 4))
    else:
        return np.array([x1, y1, x2, y2, score]).reshape((1, 5))


def iou_batch(bb_test, bb_gt):
    """Compute pairwise IoU between two sets of bounding boxes.

    Args:
        bb_test: (N, 4+) array of bboxes [x1, y1, x2, y2, ...]
        bb_gt:   (M, 4+) array of bboxes [x1, y1, x2, y2, ...]

    Returns:
        (N, M) IoU matrix.
    """
    bb_gt = np.expand_dims(bb_gt, 0)   # (1, M, 4+)
    bb_test = np.expand_dims(bb_test, 1)  # (N, 1, 4+)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    union = area_test + area_gt - intersection

    iou = intersection / np.maximum(union, 1e-10)
    return iou


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """Assign detections to tracked objects using Hungarian algorithm.

    Args:
        detections: (N, 5) array [x1, y1, x2, y2, conf]
        trackers:   (M, 4) array [x1, y1, x2, y2] predicted positions
        iou_threshold: minimum IoU for valid assignment

    Returns:
        matches:           (K, 2) array of matched [detection_idx, tracker_idx]
        unmatched_detections: array of detection indices
        unmatched_trackers:   array of tracker indices
    """
    if len(trackers) == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(len(detections)),
                np.empty((0,), dtype=int))

    if len(detections) == 0:
        return (np.empty((0, 2), dtype=int),
                np.empty((0,), dtype=int),
                np.arange(len(trackers)))

    iou_matrix = iou_batch(detections, trackers)

    # Hungarian algorithm (minimize cost = maximize IoU)
    row_indices, col_indices = linear_sum_assignment(-iou_matrix)

    matched_indices = np.stack([row_indices, col_indices], axis=1)

    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # Filter out matches with low IoU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class KalmanBoxTracker:
    """Per-object Kalman filter tracker for bounding boxes.

    State vector: [cx, cy, s, r, vx, vy, vs]
    Measurement:  [cx, cy, s, r]
    """

    count = 0

    def __init__(self, bbox):
        """Initialize tracker with initial bounding box [x1, y1, x2, y2, conf]."""
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])

        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])

        # Measurement noise
        self.kf.R[2:, 2:] *= 10.0

        # Covariance matrix -- high uncertainty for unobserved velocities
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0

        # Process noise
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialize state from first detection
        self.kf.x[:4] = convert_bbox_to_z(bbox[:4])

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count

        self.conf = bbox[4] if len(bbox) > 4 else 1.0

    def update(self, bbox):
        """Update state with observed bbox [x1, y1, x2, y2, conf]."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox[:4]))
        if len(bbox) > 4:
            self.conf = bbox[4]

    def predict(self):
        """Advance state and return predicted bbox [x1, y1, x2, y2]."""
        # Prevent negative area
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return convert_x_to_bbox(self.kf.x)

    def get_state(self):
        """Return current bbox estimate [x1, y1, x2, y2]."""
        return convert_x_to_bbox(self.kf.x)


class Sort:
    """SORT multi-object tracker.

    Args:
        max_age: maximum frames to keep a track without detections
        min_hits: minimum detection hits before a track is reported
        iou_threshold: minimum IoU for detection-to-tracker assignment
    """

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """Process one frame of detections and return active tracks.

        Args:
            dets: (N, 5) array of detections [x1, y1, x2, y2, conf]

        Returns:
            (M, 5) array of active tracks [x1, y1, x2, y2, track_id]
        """
        self.frame_count += 1

        # Predict new locations of existing trackers
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)

        # Remove dead trackers
        for t in reversed(to_del):
            self.trackers.pop(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.iou_threshold
        )

        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0]])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i])
            self.trackers.append(trk)

        # Build output: only report tracks with enough hits
        ret = []
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and \
               (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
            # Remove dead tracks
            if trk.time_since_update > self.max_age:
                self.trackers.remove(trk)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
