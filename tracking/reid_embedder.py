"""OSNet ReID feature extraction for DeepSORT appearance matching."""

import cv2
import numpy as np
import torch
import torchreid

import config


class ReIDEmbedder:
    """Wraps torchreid FeatureExtractor for person ReID embeddings."""

    def __init__(self, model_name=None, model_path=None, image_size=None):
        """Initialize OSNet feature extractor.

        Args:
            model_name: torchreid model name (defaults to config.REID_MODEL_NAME)
            model_path: path to pretrained weights (defaults to config.REID_MODEL_PATH)
            image_size: (height, width) tuple for input resize (defaults to config.REID_IMAGE_SIZE)
        """
        if model_name is None:
            model_name = config.REID_MODEL_NAME
        if model_path is None:
            model_path = config.REID_MODEL_PATH
        if image_size is None:
            image_size = config.REID_IMAGE_SIZE

        # Auto-detect device (skip MPS for torchreid compatibility)
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        print(f"  Loading ReID model: {model_name} on {device}")
        print(f"  Weights: {model_path}")
        self.extractor = torchreid.utils.FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            device=device,
            image_size=image_size,
        )
        self.image_size = image_size

    def extract_crops(self, frame_bgr, detections):
        """Crop person bounding boxes from a BGR frame.

        Args:
            frame_bgr: BGR image (H, W, 3) numpy array
            detections: (N, 4+) array with [x1, y1, x2, y2, ...] per row

        Returns:
            list of RGB numpy arrays, one per detection
        """
        h, w = frame_bgr.shape[:2]
        crops = []
        for det in detections:
            x1 = max(0, int(det[0]))
            y1 = max(0, int(det[1]))
            x2 = min(w, int(det[2]))
            y2 = min(h, int(det[3]))
            if x2 <= x1 or y2 <= y1:
                # Degenerate box — use a tiny black crop
                crop = np.zeros((self.image_size[0], self.image_size[1], 3),
                                dtype=np.uint8)
            else:
                crop = frame_bgr[y1:y2, x1:x2]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(crop)
        return crops

    def embed(self, crops):
        """Compute L2-normalized ReID embeddings from a list of RGB crops.

        Args:
            crops: list of RGB numpy arrays

        Returns:
            (N, 512) numpy array of L2-normalized embeddings
        """
        if len(crops) == 0:
            return np.empty((0, 512), dtype=np.float32)

        features = self.extractor(crops)  # returns torch.Tensor (N, D)
        features = features.cpu().numpy()

        # L2 normalize
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        features = features / norms

        return features

    def extract_and_embed(self, frame_bgr, detections):
        """Convenience: crop + embed in one call.

        Args:
            frame_bgr: BGR image (H, W, 3) numpy array
            detections: (N, 4+) array with [x1, y1, x2, y2, ...] per row

        Returns:
            (N, 512) numpy array of L2-normalized embeddings
        """
        crops = self.extract_crops(frame_bgr, detections)
        return self.embed(crops)
