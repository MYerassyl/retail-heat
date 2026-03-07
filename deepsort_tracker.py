"""DeepSORT tracker adapter wrapping deep-sort-realtime."""

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

import config


class DeepSortTracker:
    """DeepSORT tracker with external ReID embeddings and occlusion logging."""

    def __init__(self, max_age=None, n_init=None, max_cosine_distance=None,
                 nn_budget=None, max_iou_distance=None):
        """Initialize DeepSORT tracker.

        Args:
            max_age: max frames to keep a track alive without detection
            n_init: detections before track is confirmed
            max_cosine_distance: max cosine distance for appearance matching
            nn_budget: appearance gallery size per track
            max_iou_distance: max IoU distance for fallback matching
        """
        if max_age is None:
            max_age = config.DEEPSORT_MAX_AGE
        if n_init is None:
            n_init = config.DEEPSORT_N_INIT
        if max_cosine_distance is None:
            max_cosine_distance = config.DEEPSORT_MAX_COSINE_DISTANCE
        if nn_budget is None:
            nn_budget = config.DEEPSORT_NN_BUDGET
        if max_iou_distance is None:
            max_iou_distance = config.DEEPSORT_MAX_IOU_DISTANCE

        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            max_iou_distance=max_iou_distance,
            nms_max_overlap=1.0,  # NMS already done by YOLO
            embedder=None,        # we supply external embeddings
        )

        # Occlusion tracking: track_id -> last_seen_frame
        self._last_seen = {}
        self._occlusion_log = []

    def update(self, detections, embeddings, frame_num):
        """Update tracker with detections and ReID embeddings.

        Args:
            detections: (N, 5) array [x1, y1, x2, y2, conf]
            embeddings: (N, D) array of ReID embeddings
            frame_num: current frame number (1-based)

        Returns:
            (M, 5) numpy array [x1, y1, x2, y2, track_id] for confirmed tracks
        """
        # Convert xyxy -> ltwh format for deep-sort-realtime
        # Each detection: ([left, top, w, h], confidence, detection_class)
        raw_dets = []
        for det in detections:
            x1, y1, x2, y2, conf = det
            raw_dets.append(([x1, y1, x2 - x1, y2 - y1], conf, 0))

        embeds = embeddings if len(detections) > 0 else None
        tracks = self.tracker.update_tracks(raw_dets, embeds=embeds)

        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = int(track.track_id)
            ltrb = track.to_ltrb()  # [left, top, right, bottom]

            # Occlusion detection: track reappears after >3 frames gap
            if track_id in self._last_seen:
                gap = frame_num - self._last_seen[track_id]
                if gap > 3:
                    self._occlusion_log.append({
                        "track_id": track_id,
                        "lost_frame": self._last_seen[track_id],
                        "recovered_frame": frame_num,
                        "time_lost_frames": gap,
                    })

            self._last_seen[track_id] = frame_num
            results.append([ltrb[0], ltrb[1], ltrb[2], ltrb[3], track_id])

        if len(results) == 0:
            return np.empty((0, 5))

        return np.array(results)

    def get_occlusion_log(self):
        """Return list of occlusion recovery events."""
        return self._occlusion_log
