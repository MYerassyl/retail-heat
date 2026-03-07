# How the Tracking Models Work

## Technical Explanation of Baseline and Target Models

---

## 1. Problem Statement

Given a video sequence, the goal is to:
1. **Detect** all pedestrians in each frame (bounding boxes)
2. **Track** them across frames — assign a consistent identity (ID) to each person over time
3. **Generate heatmaps** from the tracked positions showing where people spend the most time

The core challenge is **data association**: when you see a set of bounding boxes in frame *t* and another set in frame *t+1*, which boxes belong to the same person?

---

## 2. Shared Component: YOLOv8x Detector

Both tracking models use the same detector — **YOLOv8x** (You Only Look Once, version 8, extra-large). This CNN takes a raw image and outputs a list of bounding boxes with confidence scores for detected persons.

```
Input:  Raw video frame (1920 x 1080 pixels)
Output: List of [x1, y1, x2, y2, confidence] per detected person
```

- **Confidence threshold:** 0.5 (ignore detections below 50% confidence)
- **NMS IoU threshold:** 0.45 (suppress duplicate overlapping boxes)
- **Class filter:** Person only (class 0)

Detections are cached to disk as `.npz` files so both trackers use identical inputs — ensuring a fair comparison.

---

## 3. Baseline Model: SORT (Simple Online and Realtime Tracking)

**Paper:** Bewley et al., "Simple Online and Realtime Tracking" (2016)

SORT uses only **motion** to associate detections across frames. It has no concept of what a person looks like — it only knows where they were and where they're predicted to be.

### 3.1 Key Components

#### Kalman Filter (Motion Model)

Each tracked person has their own Kalman filter that maintains a **state vector**:

```
State:       [cx, cy, s, r, vx, vy, vs]
Measurement: [cx, cy, s, r]

Where:
  cx, cy = bounding box center coordinates
  s      = scale (area = width x height)
  r      = aspect ratio (width / height)
  vx, vy = velocity of center point
  vs     = rate of scale change
```

The Kalman filter uses a **constant velocity model** — it assumes each person moves at a roughly constant speed between frames. Given the person's position in frame *t*, it predicts where they will be in frame *t+1*.

The state transition matrix encodes this linear motion model:

```
Position_new = Position_old + Velocity
Velocity_new = Velocity_old
```

When a new detection arrives, the Kalman filter **updates** its estimate by combining the prediction with the actual measurement, weighting each by their uncertainty (covariance).

#### Hungarian Algorithm (Assignment)

Given *N* detections and *M* existing tracks, we need to find the optimal one-to-one assignment. This is done by:

1. **Compute IoU matrix** — for each (detection, track) pair, calculate the Intersection-over-Union of their bounding boxes
2. **Solve assignment** — the Hungarian algorithm finds the assignment that maximizes total IoU (minimizes cost)
3. **Apply threshold** — reject any match where IoU < 0.3 (boxes too far apart to be the same person)

```
IoU = Area(Intersection) / Area(Union)

If IoU = 1.0: boxes are identical
If IoU = 0.0: boxes don't overlap at all
```

#### Track Lifecycle

```
Detection arrives → Create new KalmanBoxTracker (tentative)
                  → After 3 consecutive detections (min_hits=3) → Confirmed (reported in output)
                  → If no matching detection for 1 frame (max_age=1) → Track deleted
```

### 3.2 Per-Frame Algorithm

```
For each frame:
  1. PREDICT:  All existing Kalman filters predict their next position
  2. MATCH:    Hungarian algorithm associates new detections to predicted positions using IoU
  3. UPDATE:   Matched tracks get their Kalman filter updated with the actual detection
  4. CREATE:   Unmatched detections spawn new tracks
  5. DELETE:   Tracks not matched for >max_age frames are removed
  6. OUTPUT:   Only tracks with time_since_update < 1 AND hit_streak >= 3 are reported
```

### 3.3 Limitation

SORT's `max_age=1` means if a person is occluded (hidden behind another person or object) for even a **single frame**, their track is immediately killed. When they reappear, they get a new ID. This causes **identity switches** — the same person gets counted as multiple different people, corrupting the heatmap.

```
Frame 10: Person A (ID=5) visible      → tracked
Frame 11: Person A hidden behind pillar → no detection → track killed
Frame 12: Person A visible again        → gets new ID=12 (identity switch!)
```

---

## 4. Target Model: DeepSORT + OSNet ReID

**Paper:** Wojke et al., "Simple Online and Realtime Tracking with a Deep Association Metric" (2017)

DeepSORT extends SORT by adding **appearance-based matching** using a deep neural network. It doesn't just know *where* a person was — it also knows what they *look like*.

### 4.1 What is ReID (Re-Identification)?

**Re-Identification (ReID)** is the task of recognizing the same person across different camera views or time gaps, based on their visual appearance.

A ReID model is a CNN that takes a cropped image of a person and produces a compact **embedding vector** (a list of numbers) that encodes their appearance. Two crops of the same person should produce similar embeddings, while crops of different people should produce dissimilar embeddings.

```
Input:  Cropped person image (256 x 128 pixels, RGB)
Output: 512-dimensional embedding vector (L2-normalized)

Same person:      cosine_distance(embedding_A, embedding_B) ≈ 0.1  (similar)
Different person:  cosine_distance(embedding_A, embedding_B) ≈ 0.7  (dissimilar)
```

The cosine distance measures the angle between two vectors:
- **0.0** = identical direction (same person)
- **1.0** = opposite direction (completely different)

### 4.2 OSNet (Omni-Scale Network)

**Paper:** Zhou et al., "Omni-Scale Feature Learning for Person Re-Identification" (2019)

Our ReID model is **OSNet x1_0** — a lightweight CNN specifically designed for person re-identification.

| Property | Value |
|---|---|
| Architecture | Omni-Scale Network (OSNet) |
| Parameters | 2.2 million |
| FLOPs | 979 million |
| Output Dimension | 512 |
| Input Size | 256 x 128 (height x width) |
| Training Dataset | Market-1501 (1,501 identities, 32,668 images) |
| Performance | 94.2% Rank-1 accuracy, 82.6% mAP on Market-1501 |

**What makes OSNet special:**

Standard CNNs capture features at a fixed scale — small convolutional filters see local patterns (texture, color) while large filters see global patterns (body shape, pose). OSNet uses **omni-scale feature aggregation** — it combines features from multiple receptive field sizes within each layer, allowing it to simultaneously capture:

- **Local features:** clothing texture, logo, patterns
- **Medium features:** body proportions, arm positions
- **Global features:** overall body shape, color distribution

This multi-scale approach makes it robust to the variations that occur in tracking: pose changes, partial occlusion, lighting shifts.

**Training on Market-1501:**

The model was trained on the Market-1501 person ReID dataset, which contains 32,668 images of 1,501 people captured by 6 different cameras in a university campus. The training objective is a **triplet loss** + **cross-entropy loss**:

- **Triplet loss:** For each anchor image, push the embedding closer to images of the same person (positive) and farther from images of different people (negative)
- **Cross-entropy loss:** Classify each image into one of 1,501 identity classes

This person-specific training is critical — switching from generic ImageNet features to Market-1501 trained features improved our ID switch reduction from 29.6% to 47.1%.

### 4.3 How DeepSORT's Matching Works

DeepSORT uses a **two-stage cascade matching** strategy:

#### Stage 1: Appearance Matching (Cosine Distance)

For each existing track, DeepSORT maintains an **appearance gallery** — a buffer of the last 100 ReID embeddings (nn_budget=100) observed for that person. When new detections arrive:

1. Extract a 512-D embedding for each new detection using OSNet
2. Compute the cosine distance between each detection's embedding and each track's gallery
3. Use the **minimum** cosine distance (most similar gallery sample) as the matching cost
4. Reject matches where cosine distance > 0.3 (max_cosine_distance)

```
Track A gallery: [emb_frame1, emb_frame5, emb_frame8, ...]  (up to 100 samples)
New detection:    emb_new

cost = min(cosine_distance(emb_new, emb_i) for emb_i in gallery_A)

If cost < 0.3 → candidate match
If cost >= 0.3 → not the same person
```

The gallery approach is powerful because it stores multiple views of the same person (front, side, back) — even if the current view doesn't match one gallery sample, it might match another.

#### Stage 2: IoU Matching (Fallback)

Detections and tracks that weren't matched by appearance go through a second round using IoU matching (same as SORT). This handles cases where the ReID embedding is unreliable (e.g., very small crops, motion blur).

#### Combined Association

```
For each frame:
  1. PREDICT:     Kalman filters predict positions for all tracks
  2. EMBED:       OSNet extracts 512-D embeddings for all detections
  3. MATCH (1st): Cosine distance matching — confirmed tracks vs detections
  4. MATCH (2nd): IoU matching — remaining unmatched tracks vs remaining detections
  5. UPDATE:      Matched tracks updated (Kalman + appearance gallery refreshed)
  6. CREATE:      Unmatched detections → new tentative tracks
  7. DELETE:      Tracks unmatched for >30 frames → removed
  8. OUTPUT:      Only confirmed tracks (n_init=3 consecutive detections) reported
```

### 4.4 Track Lifecycle (DeepSORT)

```
Detection arrives → Create new track (tentative)
                  → After 3 consecutive detections (n_init=3) → Confirmed
                  → If no detection for up to 30 frames (max_age=30) → Track survives (Kalman predicts)
                  → When person reappears → ReID embedding matches → Same ID preserved!
                  → If no detection for >30 frames → Track deleted
```

### 4.5 Occlusion Recovery Example

```
Frame 10: Person A (ID=5) detected → embedding stored in gallery
Frame 11: Person A hidden behind pillar → no detection → Kalman predicts position
Frame 12: Still hidden → Kalman continues predicting (track alive, max_age=30)
...
Frame 18: Person A reappears → new detection
          OSNet extracts embedding → cosine distance to ID=5 gallery = 0.12 (match!)
          → ID=5 preserved! No identity switch.
```

This is the fundamental advantage: SORT would have killed the track at frame 11 and assigned a new ID at frame 18. DeepSORT keeps the same ID through the occlusion.

---

## 5. Heatmap Generation

Both trackers output a list of bounding box centroids (center x, center y) for every tracked person in every frame. These centroids are converted into density heatmaps using **Gaussian Kernel Density Estimation (KDE)**.

### 5.1 How KDE Works

1. **Collect all centroids** — every (cx, cy) pair from all frames of a sequence
2. **Fit a Gaussian KDE** — place a small Gaussian "bump" at each centroid location
3. **Evaluate on a grid** — compute the sum of all Gaussian bumps at each pixel location
4. **Normalize and overlay** — map density values to a colormap (inferno) and overlay on a reference frame

```
Heatmap pixel value at point (x, y) = sum of contributions from all centroids:

  density(x, y) = (1/N) * Σ K((x - cx_i)/h, (y - cy_i)/h)

Where:
  K = Gaussian kernel function
  h = bandwidth (automatically selected by Scott's rule)
  N = total number of centroids
  (cx_i, cy_i) = centroid of the i-th tracked detection
```

Areas where many people walk produce high density values (hot spots), while areas with few or no people produce low values (cold zones).

### 5.2 Why Track Quality Affects Heatmaps

A tracker's heatmap accuracy depends directly on how well it tracks people:

- **Missed detections (high FN)** → cold spots where people actually walked → undercounting
- **False positives (high FP)** → hot spots where nobody walked → phantom traffic
- **ID switches** → don't directly affect heatmaps (a centroid is a centroid regardless of ID), but they indicate poor matching which correlates with other errors
- **Higher recall** → more centroids captured → more accurate density estimation

DeepSORT's higher recall (71.1% vs 64.6% on MOT17; 36.9% vs 25.6% on MOT20) means it captures more person locations, producing heatmaps that are closer to ground truth — confirmed by quantitative evaluation showing Pearson correlation of 0.841 vs 0.805.

---

## 6. Summary: SORT vs DeepSORT

| Aspect | SORT (Baseline) | DeepSORT + OSNet (Target) |
|---|---|---|
| **Matching** | IoU only (geometry) | Cosine distance on ReID embeddings + IoU fallback |
| **Appearance Model** | None | OSNet x1_0 (512-D embeddings) |
| **Occlusion Handling** | Track dies after 1 missed frame | Track survives up to 30 missed frames |
| **Re-identification** | Cannot re-identify after occlusion | Uses appearance similarity to re-identify |
| **Track Survival** | Conservative (max_age=1) | Extended (max_age=30) |
| **False Positives** | Very low (only reports active detections) | Higher (Kalman predictions during occlusion) |
| **Recall** | Lower (misses occluded people) | Higher (tracks through occlusion) |
| **ID Switches** | More (new ID every time person reappears) | 47-50% fewer (same ID preserved) |
| **Speed** | Faster (no CNN inference per crop) | Slower (~5 min/sequence on CPU for ReID) |
| **Best For** | Simple scenes, real-time, low compute | Dense crowds, identity consistency, accuracy |

```
SORT Pipeline:
  Frame → YOLOv8x → Detections → [Kalman Predict → IoU Match → Update] → Tracks → Heatmap

DeepSORT Pipeline:
  Frame → YOLOv8x → Detections ──→ [Kalman Predict → Cosine Match → IoU Match → Update] → Tracks → Heatmap
                  ↘ OSNet → Embeddings ↗
```
