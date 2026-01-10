"""
Tennis Ball & Person Detection and Tracking System (Enhanced)
==============================================================
Detects tennis balls and persons using YOLO, tracks with interpolation,
validates using image comparison, and filters false positives.

Enhanced features for full court wide-angle videos:
- Multi-confidence detection passes
- Image enhancement for small ball detection
- Extended interpolation range (up to 30 frames)
- Trajectory-based candidate recovery
- Kalman filter for motion prediction
- Person detection with tile-based approach for accuracy
- Person tracking with appearance-based ReID to prevent ID switches
- Post-processing ID merger to consolidate fragmented person IDs
"""

import cv2
import json
import numpy as np
import os
import math
from ultralytics import YOLO
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict
from enum import Enum


# =============================================================================
# DATA CLASSES
# =============================================================================

class DetectionSource(Enum):
    DETECTION = "detection"
    INTERPOLATED = "interpolated"


@dataclass
class BallDetection:
    """Single ball detection in a frame"""
    id: int
    x: float
    y: float
    confidence: Optional[float]
    source: DetectionSource
    validated: bool = True
    bbox: Optional[Tuple[int, int, int, int]] = None
    interpolation_method: Optional[str] = None


@dataclass
class PersonDetection:
    """Single person detection in a frame"""
    id: int
    x: float  # center x
    y: float  # center y
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    tracked: bool = True  # True if matched with existing track


@dataclass
class FrameResult:
    """All ball and person detections in a single frame"""
    frame_index: int
    timestamp_seconds: float
    balls: List[BallDetection] = field(default_factory=list)
    persons: List[PersonDetection] = field(default_factory=list)


@dataclass
class VideoInfo:
    """Video metadata"""
    filename: str
    fps: float
    total_frames: int
    duration_seconds: float
    resolution: Dict[str, int]


# =============================================================================
# BALL VALIDATOR CLASS
# =============================================================================

class BallValidator:
    """
    Validates ball detections using:
    1. Color histogram analysis (tennis ball is yellow-green)
    2. Size constraints
    """

    # Tennis ball HSV range (yellow-green)
    BALL_HSV_LOWER = np.array([20, 80, 80])
    BALL_HSV_UPPER = np.array([50, 255, 255])

    # Size constraints (in pixels, adjustable based on video resolution)
    MIN_BALL_AREA = 50
    MAX_BALL_AREA = 3000

    # Color ratio threshold
    MIN_YELLOW_RATIO = 0.05

    def __init__(self, frame_height: int = 1080):
        # Adjust size constraints based on resolution
        scale = frame_height / 1080.0
        self.min_area = int(self.MIN_BALL_AREA * scale * scale)
        self.max_area = int(self.MAX_BALL_AREA * scale * scale)

        # Store reference ball crops for histogram comparison
        self.reference_histograms: List[np.ndarray] = []

    def validate(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[bool, str]:
        """
        Validate a detection as a real tennis ball

        Returns:
            (is_valid, reason)
        """
        x1, y1, x2, y2 = bbox

        # Ensure valid coordinates
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return False, "empty_crop"

        # 1. Size validation
        area = (x2 - x1) * (y2 - y1)
        if area < self.min_area:
            return False, "too_small"
        if area > self.max_area:
            return False, "too_large"

        # 2. Color validation
        is_valid, reason = self._validate_color(crop)
        if not is_valid:
            return False, reason

        return True, "valid"

    def _validate_color(self, crop: np.ndarray) -> Tuple[bool, str]:
        """Check if crop contains tennis ball colors"""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.BALL_HSV_LOWER, self.BALL_HSV_UPPER)

        yellow_ratio = np.sum(mask > 0) / mask.size

        if yellow_ratio < self.MIN_YELLOW_RATIO:
            return False, "color_mismatch"

        return True, "color_ok"

    def add_reference(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Add a confirmed ball crop as reference for future comparisons"""
        x1, y1, x2, y2 = bbox

        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = frame[y1:y2, x1:x2]

        if crop.size > 0:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            self.reference_histograms.append(hist)

            # Keep only last 50 references
            if len(self.reference_histograms) > 50:
                self.reference_histograms.pop(0)


# =============================================================================
# TRAJECTORY TRACKER CLASS
# =============================================================================

class TrajectoryTracker:
    """
    Tracks ball trajectory and detects anomalies
    """

    def __init__(self, fps: float, max_speed_mps: float = 70.0, pixels_per_meter: float = 50.0):
        self.fps = fps
        self.max_displacement = (max_speed_mps * pixels_per_meter) / fps
        self.positions: List[Tuple[int, float, float]] = []  # (frame, x, y)

    def is_position_valid(self, frame_idx: int, x: float, y: float) -> Tuple[bool, str]:
        """
        Check if a new position is physically plausible
        """
        if not self.positions:
            return True, "first_position"

        # Find closest previous position
        last_frame, last_x, last_y = self.positions[-1]
        frame_gap = frame_idx - last_frame

        if frame_gap <= 0:
            return False, "invalid_frame_order"

        # Calculate displacement
        displacement = math.sqrt((x - last_x)**2 + (y - last_y)**2)
        max_allowed = self.max_displacement * frame_gap

        if displacement > max_allowed:
            return False, f"teleport_detected_{displacement:.1f}>{max_allowed:.1f}"

        return True, "trajectory_valid"

    def add_position(self, frame_idx: int, x: float, y: float):
        """Add confirmed position to trajectory"""
        self.positions.append((frame_idx, x, y))

    def get_recent_positions(self, n: int = 10) -> List[Tuple[int, float, float]]:
        """Get last n positions"""
        return self.positions[-n:] if self.positions else []

    def reset(self):
        """Reset trajectory tracking"""
        self.positions.clear()


# =============================================================================
# KALMAN FILTER FOR BALL TRACKING
# =============================================================================

class BallKalmanFilter:
    """
    Kalman filter for predicting ball position
    State: [x, y, vx, vy] (position and velocity)
    """

    def __init__(self):
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Observation matrix (we only observe position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Process noise covariance
        self.Q = np.eye(4, dtype=np.float32) * 0.1
        self.Q[2, 2] = 1.0  # velocity has more uncertainty
        self.Q[3, 3] = 1.0

        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float32) * 1.0

        # State and covariance
        self.x = np.zeros(4, dtype=np.float32)  # [x, y, vx, vy]
        self.P = np.eye(4, dtype=np.float32) * 100

        self.initialized = False

    def init(self, x: float, y: float):
        """Initialize filter with first observation"""
        self.x = np.array([x, y, 0, 0], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 100
        self.initialized = True

    def predict(self) -> Tuple[float, float]:
        """Predict next state"""
        if not self.initialized:
            return 0.0, 0.0

        # Predict state
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        return float(self.x[0]), float(self.x[1])

    def update(self, x: float, y: float) -> Tuple[float, float]:
        """Update with observation"""
        if not self.initialized:
            self.init(x, y)
            return x, y

        # Measurement
        z = np.array([x, y], dtype=np.float32)

        # Innovation
        y_innov = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y_innov
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return float(self.x[0]), float(self.x[1])

    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate"""
        return float(self.x[2]), float(self.x[3])

    def get_predicted_position(self, frames_ahead: int = 1) -> Tuple[float, float]:
        """Predict position n frames ahead without updating state"""
        if not self.initialized:
            return 0.0, 0.0

        x_pred = self.x.copy()
        for _ in range(frames_ahead):
            x_pred = self.F @ x_pred

        return float(x_pred[0]), float(x_pred[1])


# =============================================================================
# IMAGE ENHANCER FOR SMALL BALL DETECTION
# =============================================================================

class ImageEnhancer:
    """
    Enhances frames to improve detection of small balls in wide-angle videos
    """

    def __init__(self, enable_clahe: bool = True, enable_sharpening: bool = True):
        self.enable_clahe = enable_clahe
        self.enable_sharpening = enable_sharpening

        # CLAHE for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Sharpening kernel
        self.sharpen_kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Apply enhancement to frame"""
        enhanced = frame.copy()

        if self.enable_clahe:
            # Apply CLAHE to L channel in LAB color space
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        if self.enable_sharpening:
            # Apply sharpening
            enhanced = cv2.filter2D(enhanced, -1, self.sharpen_kernel)

        return enhanced


# =============================================================================
# PERSON DETECTOR CLASS (TILE-BASED)
# =============================================================================

class PersonDetector:
    """
    Detects persons using YOLOv8m with tile-based approach for better accuracy
    on wide-angle full court videos where persons may appear small.
    """

    PERSON_CLASS_ID = 0  # COCO class ID for person

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        conf_threshold: float = 0.5,
        use_tiles: bool = True,
        tile_overlap: float = 0.2
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.use_tiles = use_tiles
        self.tile_overlap = tile_overlap

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons in frame

        Returns:
            List of detections: [{x, y, w, h, conf, bbox}, ...]
        """
        return self._detect_single(frame)

    def _detect_single(self, frame: np.ndarray) -> List[Dict]:
        """Detect on full frame"""
        results = self.model.predict(
            frame,
            verbose=False,
            conf=self.conf_threshold,
            classes=[self.PERSON_CLASS_ID]
        )

        detections = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for i in range(len(results[0].boxes)):
                x, y, w, h = results[0].boxes.xywh[i].cpu().numpy()
                conf = float(results[0].boxes.conf[i].cpu().numpy())
                x1, y1, x2, y2 = results[0].boxes.xyxy[i].cpu().numpy().astype(int)

                detections.append({
                    "x": float(x),
                    "y": float(y),
                    "w": float(w),
                    "h": float(h),
                    "conf": conf,
                    "bbox": (int(x1), int(y1), int(x2), int(y2))
                })

        return detections

    def _detect_with_tiles(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect using 4 overlapping tiles for better accuracy on distant persons
        """
        h, w = frame.shape[:2]
        tile_h, tile_w = h // 2, w // 2

        # Calculate overlap in pixels
        overlap_h = int(tile_h * self.tile_overlap)
        overlap_w = int(tile_w * self.tile_overlap)

        # Define 4 tiles with overlap
        tiles = [
            # (x_start, y_start, x_end, y_end)
            (0, 0, tile_w + overlap_w, tile_h + overlap_h),  # Top-left
            (tile_w - overlap_w, 0, w, tile_h + overlap_h),  # Top-right
            (0, tile_h - overlap_h, tile_w + overlap_w, h),  # Bottom-left
            (tile_w - overlap_w, tile_h - overlap_h, w, h),  # Bottom-right
        ]

        all_detections = []

        for x_start, y_start, x_end, y_end in tiles:
            tile = frame[y_start:y_end, x_start:x_end]

            results = self.model.predict(
                tile,
                verbose=False,
                conf=self.conf_threshold,
                classes=[self.PERSON_CLASS_ID]
            )

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for i in range(len(results[0].boxes)):
                    x, y, w_det, h_det = results[0].boxes.xywh[i].cpu().numpy()
                    conf = float(results[0].boxes.conf[i].cpu().numpy())
                    x1, y1, x2, y2 = results[0].boxes.xyxy[i].cpu().numpy().astype(int)

                    # Convert to full frame coordinates
                    all_detections.append({
                        "x": float(x + x_start),
                        "y": float(y + y_start),
                        "w": float(w_det),
                        "h": float(h_det),
                        "conf": conf,
                        "bbox": (int(x1 + x_start), int(y1 + y_start),
                                int(x2 + x_start), int(y2 + y_start))
                    })

        # Also detect on full frame for large persons
        full_detections = self._detect_single(frame)
        all_detections.extend(full_detections)

        # Remove duplicates using NMS
        return self._nms_detections(all_detections)

    def _nms_detections(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Non-maximum suppression to remove duplicate detections"""
        if len(detections) <= 1:
            return detections

        # Sort by confidence
        sorted_dets = sorted(detections, key=lambda d: d["conf"], reverse=True)
        kept = []

        for det in sorted_dets:
            is_duplicate = False
            for kept_det in kept:
                iou = self._calculate_iou(det["bbox"], kept_det["bbox"])
                if iou > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(det)

        return kept

    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


# =============================================================================
# PERSON TRACKER CLASS (WITH APPEARANCE MATCHING)
# =============================================================================

class PersonTracker:
    """
    Tracks persons across frames using:
    1. IoU-based matching for nearby persons
    2. Appearance histogram comparison to prevent ID switches
    3. Kalman filter for motion prediction
    """

    def __init__(
        self,
        max_disappeared: int = 30,
        iou_threshold: float = 0.3,
        appearance_threshold: float = 0.6,
        max_distance: float = 200
    ):
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.appearance_threshold = appearance_threshold
        self.max_distance = max_distance

        self.next_id = 1
        self.tracks: Dict[int, Dict] = {}  # id -> track info
        self.disappeared: Dict[int, int] = {}  # id -> frames since last seen

    def update(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Update tracks with new detections

        Returns:
            List of tracked persons with IDs
        """
        if not detections:
            # Mark all tracks as disappeared
            for track_id in list(self.tracks.keys()):
                self.disappeared[track_id] = self.disappeared.get(track_id, 0) + 1
                if self.disappeared[track_id] > self.max_disappeared:
                    del self.tracks[track_id]
                    del self.disappeared[track_id]
            return []

        if not self.tracks:
            # No existing tracks - create new ones
            results = []
            for det in detections:
                track_id = self._create_track(frame, det)
                results.append({**det, "id": track_id, "tracked": True})
            return results

        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_tracks = self._match_detections(
            frame, detections
        )

        results = []

        # Update matched tracks
        for track_id, det_idx in matched:
            det = detections[det_idx]
            self._update_track(frame, track_id, det)
            self.disappeared[track_id] = 0
            results.append({**det, "id": track_id, "tracked": True})

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            track_id = self._create_track(frame, det)
            results.append({**det, "id": track_id, "tracked": True})

        # Handle disappeared tracks
        for track_id in unmatched_tracks:
            self.disappeared[track_id] = self.disappeared.get(track_id, 0) + 1
            if self.disappeared[track_id] > self.max_disappeared:
                del self.tracks[track_id]
                del self.disappeared[track_id]

        return results

    def _create_track(self, frame: np.ndarray, det: Dict) -> int:
        """Create a new track"""
        track_id = self.next_id
        self.next_id += 1

        # Extract appearance histogram
        histogram = self._extract_histogram(frame, det["bbox"])

        self.tracks[track_id] = {
            "bbox": det["bbox"],
            "center": (det["x"], det["y"]),
            "histogram": histogram,
            "velocity": (0, 0)
        }
        self.disappeared[track_id] = 0

        return track_id

    def _update_track(self, frame: np.ndarray, track_id: int, det: Dict):
        """Update existing track"""
        old_center = self.tracks[track_id]["center"]
        new_center = (det["x"], det["y"])

        # Update velocity
        velocity = (new_center[0] - old_center[0], new_center[1] - old_center[1])

        # Update histogram with exponential moving average
        new_hist = self._extract_histogram(frame, det["bbox"])
        if new_hist is not None and self.tracks[track_id]["histogram"] is not None:
            alpha = 0.3
            self.tracks[track_id]["histogram"] = (
                alpha * new_hist + (1 - alpha) * self.tracks[track_id]["histogram"]
            )
        elif new_hist is not None:
            self.tracks[track_id]["histogram"] = new_hist

        self.tracks[track_id]["bbox"] = det["bbox"]
        self.tracks[track_id]["center"] = new_center
        self.tracks[track_id]["velocity"] = velocity

    def _match_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to existing tracks using IoU and appearance

        Returns:
            (matched pairs, unmatched detection indices, unmatched track ids)
        """
        track_ids = list(self.tracks.keys())
        num_tracks = len(track_ids)
        num_dets = len(detections)

        if num_tracks == 0:
            return [], list(range(num_dets)), []

        # Compute cost matrix
        cost_matrix = np.zeros((num_tracks, num_dets))

        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            for j, det in enumerate(detections):
                # IoU score
                iou = self._calculate_iou(track["bbox"], det["bbox"])

                # Distance score
                dist = math.sqrt(
                    (track["center"][0] - det["x"])**2 +
                    (track["center"][1] - det["y"])**2
                )
                dist_score = max(0, 1 - dist / self.max_distance)

                # Appearance score
                det_hist = self._extract_histogram(frame, det["bbox"])
                if det_hist is not None and track["histogram"] is not None:
                    appearance_score = cv2.compareHist(
                        track["histogram"], det_hist, cv2.HISTCMP_CORREL
                    )
                    appearance_score = max(0, appearance_score)
                else:
                    appearance_score = 0.5

                # Combined score (higher is better)
                cost_matrix[i, j] = 0.3 * iou + 0.3 * dist_score + 0.4 * appearance_score

        # Greedy matching (Hungarian algorithm would be better but this is simpler)
        matched = []
        matched_tracks = set()
        matched_dets = set()

        # Sort by cost (descending - best matches first)
        indices = np.unravel_index(np.argsort(-cost_matrix, axis=None), cost_matrix.shape)

        for i, j in zip(indices[0], indices[1]):
            if i in matched_tracks or j in matched_dets:
                continue

            score = cost_matrix[i, j]
            track_id = track_ids[i]
            track = self.tracks[track_id]
            det = detections[j]

            # Check if match is valid
            iou = self._calculate_iou(track["bbox"], det["bbox"])
            dist = math.sqrt(
                (track["center"][0] - det["x"])**2 +
                (track["center"][1] - det["y"])**2
            )

            if iou >= self.iou_threshold or dist < self.max_distance:
                matched.append((track_id, j))
                matched_tracks.add(i)
                matched_dets.add(j)

        unmatched_dets = [j for j in range(num_dets) if j not in matched_dets]
        unmatched_tracks = [track_ids[i] for i in range(num_tracks) if i not in matched_tracks]

        return matched, unmatched_dets, unmatched_tracks

    def _extract_histogram(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract color histogram from person crop for appearance matching"""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Clamp coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        # Convert to HSV and compute histogram
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Use upper body region (more distinctive than legs)
        upper_h = crop.shape[0] // 2
        if upper_h > 10:
            hsv = hsv[:upper_h, :, :]

        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        return hist

    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


# =============================================================================
# PERSON ID MERGER (POST-PROCESSING)
# =============================================================================

class PersonIDMerger:
    """
    Post-processing class to merge fragmented person IDs.

    When tracking loses a person and re-detects them later, they get a new ID.
    This class analyzes all person tracks and merges IDs that belong to the
    same person based on:
    1. Appearance similarity (color histogram of upper body)
    2. Spatial-temporal consistency (similar positions when IDs overlap in time)
    3. Body size consistency
    """

    def __init__(
        self,
        appearance_threshold: float = 0.65,
        size_ratio_threshold: float = 0.3,
        min_samples: int = 5
    ):
        """
        Args:
            appearance_threshold: Minimum histogram correlation to consider same person
            size_ratio_threshold: Maximum allowed difference in body size ratio
            min_samples: Minimum samples needed to compute reliable appearance
        """
        self.appearance_threshold = appearance_threshold
        self.size_ratio_threshold = size_ratio_threshold
        self.min_samples = min_samples

    def merge_ids(
        self,
        frames: List[np.ndarray],
        person_results: List[List[Dict]]
    ) -> Tuple[List[List[Dict]], Dict[int, int]]:
        """
        Analyze all person detections and merge IDs belonging to same person.

        Args:
            frames: List of video frames
            person_results: List of person detections per frame

        Returns:
            (updated_person_results, id_mapping) where id_mapping maps old_id -> new_id
        """
        # Step 1: Collect appearance features for each ID
        print("    Collecting appearance features for each person ID...")
        id_features = self._collect_id_features(frames, person_results)

        if len(id_features) <= 1:
            print("    Only 1 or fewer person IDs found, no merging needed")
            return person_results, {}

        # Step 2: Compare all ID pairs and find matches
        print(f"    Comparing {len(id_features)} person IDs for similarity...")
        id_pairs_to_merge = self._find_matching_ids(id_features)

        if not id_pairs_to_merge:
            print("    No IDs to merge found")
            return person_results, {}

        # Step 3: Build merge groups using Union-Find
        merge_groups = self._build_merge_groups(id_features.keys(), id_pairs_to_merge)

        # Step 4: Create ID mapping (old_id -> new_id)
        id_mapping = self._create_id_mapping(merge_groups)

        if not id_mapping:
            print("    No ID remapping needed")
            return person_results, {}

        print(f"    Merging {len(id_mapping)} IDs into {len(set(id_mapping.values()))} unique persons")

        # Step 5: Apply mapping to results
        updated_results = self._apply_id_mapping(person_results, id_mapping)

        return updated_results, id_mapping

    def _collect_id_features(
        self,
        frames: List[np.ndarray],
        person_results: List[List[Dict]]
    ) -> Dict[int, Dict]:
        """
        Collect appearance features for each person ID.

        Returns:
            Dict mapping person_id -> {
                'histograms': list of histograms,
                'sizes': list of (width, height),
                'positions': list of (frame_idx, x, y),
                'avg_histogram': averaged histogram,
                'avg_size': (avg_width, avg_height)
            }
        """
        id_features: Dict[int, Dict] = {}

        for frame_idx, persons in enumerate(person_results):
            frame = frames[frame_idx]

            for person in persons:
                person_id = person["id"]
                bbox = person["bbox"]

                if person_id not in id_features:
                    id_features[person_id] = {
                        'histograms': [],
                        'sizes': [],
                        'positions': [],
                        'frame_ranges': []
                    }

                # Extract histogram
                hist = self._extract_appearance_histogram(frame, bbox)
                if hist is not None:
                    id_features[person_id]['histograms'].append(hist)

                # Record size
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                id_features[person_id]['sizes'].append((w, h))

                # Record position
                id_features[person_id]['positions'].append(
                    (frame_idx, person["x"], person["y"])
                )

                # Track frame range
                id_features[person_id]['frame_ranges'].append(frame_idx)

        # Compute averages
        for person_id, features in id_features.items():
            if features['histograms']:
                # Average histogram
                avg_hist = np.mean(features['histograms'], axis=0).astype(np.float32)
                cv2.normalize(avg_hist, avg_hist)
                features['avg_histogram'] = avg_hist
            else:
                features['avg_histogram'] = None

            if features['sizes']:
                avg_w = np.mean([s[0] for s in features['sizes']])
                avg_h = np.mean([s[1] for s in features['sizes']])
                features['avg_size'] = (avg_w, avg_h)
            else:
                features['avg_size'] = (0, 0)

            # Frame range
            features['first_frame'] = min(features['frame_ranges'])
            features['last_frame'] = max(features['frame_ranges'])
            features['total_detections'] = len(features['frame_ranges'])

        return id_features

    def _extract_appearance_histogram(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """Extract color histogram from person crop."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Clamp coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        # Convert to HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Use upper body (more distinctive - contains shirt/jersey)
        upper_h = crop.shape[0] // 2
        if upper_h > 10:
            hsv_upper = hsv[:upper_h, :, :]
        else:
            hsv_upper = hsv

        # Compute histogram with more bins for better discrimination
        hist = cv2.calcHist([hsv_upper], [0, 1], None, [36, 48], [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        return hist

    def _find_matching_ids(
        self,
        id_features: Dict[int, Dict]
    ) -> List[Tuple[int, int, float]]:
        """
        Compare all ID pairs and find matching ones.

        Returns:
            List of (id1, id2, similarity_score) for pairs that should be merged
        """
        matching_pairs = []
        ids = list(id_features.keys())

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                feat1 = id_features[id1]
                feat2 = id_features[id2]

                # Skip if either has too few samples
                if (feat1['total_detections'] < self.min_samples or
                    feat2['total_detections'] < self.min_samples):
                    continue

                # Skip if they significantly overlap in time (likely different persons)
                overlap = self._compute_temporal_overlap(feat1, feat2)
                if overlap > 0.5:  # More than 50% overlap = different persons
                    continue

                # Compare appearance
                appearance_sim = self._compare_appearance(feat1, feat2)
                if appearance_sim < self.appearance_threshold:
                    continue

                # Compare body size
                size_sim = self._compare_size(feat1, feat2)
                if size_sim < (1 - self.size_ratio_threshold):
                    continue

                # Combined score
                combined_score = 0.7 * appearance_sim + 0.3 * size_sim

                if combined_score >= self.appearance_threshold:
                    matching_pairs.append((id1, id2, combined_score))

        # Sort by score (highest first)
        matching_pairs.sort(key=lambda x: x[2], reverse=True)

        return matching_pairs

    def _compute_temporal_overlap(
        self,
        feat1: Dict,
        feat2: Dict
    ) -> float:
        """
        Compute how much two IDs overlap in time.
        Returns ratio of overlap frames to total frames.
        """
        frames1 = set(feat1['frame_ranges'])
        frames2 = set(feat2['frame_ranges'])

        overlap = len(frames1 & frames2)
        total = len(frames1 | frames2)

        return overlap / total if total > 0 else 0

    def _compare_appearance(self, feat1: Dict, feat2: Dict) -> float:
        """Compare appearance histograms between two IDs."""
        if feat1['avg_histogram'] is None or feat2['avg_histogram'] is None:
            return 0.0

        # Use correlation for comparison
        similarity = cv2.compareHist(
            feat1['avg_histogram'],
            feat2['avg_histogram'],
            cv2.HISTCMP_CORREL
        )

        return max(0, similarity)

    def _compare_size(self, feat1: Dict, feat2: Dict) -> float:
        """Compare body sizes between two IDs."""
        w1, h1 = feat1['avg_size']
        w2, h2 = feat2['avg_size']

        if w1 == 0 or h1 == 0 or w2 == 0 or h2 == 0:
            return 0.5  # Unknown, assume neutral

        # Compare aspect ratios and areas
        ratio1 = w1 / h1
        ratio2 = w2 / h2
        ratio_diff = abs(ratio1 - ratio2) / max(ratio1, ratio2)

        area1 = w1 * h1
        area2 = w2 * h2
        area_diff = abs(area1 - area2) / max(area1, area2)

        # Score: 1 means identical, 0 means very different
        ratio_score = 1 - min(ratio_diff, 1)
        area_score = 1 - min(area_diff, 1)

        return 0.5 * ratio_score + 0.5 * area_score

    def _build_merge_groups(
        self,
        all_ids: List[int],
        pairs_to_merge: List[Tuple[int, int, float]]
    ) -> List[List[int]]:
        """
        Build groups of IDs that should be merged using Union-Find.

        Returns:
            List of groups, each group is a list of IDs to merge
        """
        # Union-Find implementation
        parent = {id_: id_ for id_ in all_ids}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                # Keep the smaller ID as parent (for consistency)
                if px < py:
                    parent[py] = px
                else:
                    parent[px] = py

        # Apply merges
        for id1, id2, _ in pairs_to_merge:
            union(id1, id2)

        # Build groups
        groups: Dict[int, List[int]] = {}
        for id_ in all_ids:
            root = find(id_)
            if root not in groups:
                groups[root] = []
            groups[root].append(id_)

        # Only return groups with more than 1 member
        return [sorted(group) for group in groups.values() if len(group) > 1]

    def _create_id_mapping(
        self,
        merge_groups: List[List[int]]
    ) -> Dict[int, int]:
        """
        Create mapping from old IDs to new IDs.
        Each group gets the smallest ID in the group.

        Returns:
            Dict mapping old_id -> new_id (only for IDs that change)
        """
        mapping = {}

        for group in merge_groups:
            target_id = min(group)  # Use smallest ID as the canonical one
            for id_ in group:
                if id_ != target_id:
                    mapping[id_] = target_id

        return mapping

    def _apply_id_mapping(
        self,
        person_results: List[List[Dict]],
        id_mapping: Dict[int, int]
    ) -> List[List[Dict]]:
        """Apply ID mapping to all person results."""
        updated_results = []

        for frame_persons in person_results:
            updated_frame = []
            for person in frame_persons:
                updated_person = person.copy()
                old_id = person["id"]
                if old_id in id_mapping:
                    updated_person["id"] = id_mapping[old_id]
                updated_frame.append(updated_person)
            updated_results.append(updated_frame)

        return updated_results


# =============================================================================
# BALL INTERPOLATOR CLASS
# =============================================================================

class BallInterpolator:
    """
    Interpolates missing ball positions using:
    - Linear interpolation for small gaps (1-2 frames)
    - Parabolic interpolation for larger gaps (3-30 frames)
    """

    MAX_GAP = 30  # Increased from 15 for wide-angle videos
    PARABOLIC_THRESHOLD = 3

    def interpolate(
        self,
        positions: List[Optional[Tuple[float, float]]]
    ) -> List[Tuple[Optional[Tuple[float, float]], str]]:
        """
        Interpolate missing positions

        Args:
            positions: List of (x, y) or None for each frame

        Returns:
            List of ((x, y) or None, method) tuples
        """
        result = []
        n = len(positions)

        i = 0
        while i < n:
            if positions[i] is not None:
                result.append((positions[i], "detection"))
                i += 1
            else:
                # Find gap boundaries
                gap_start = i
                while i < n and positions[i] is None:
                    i += 1
                gap_end = i
                gap_size = gap_end - gap_start

                # Get boundary positions
                prev_pos = positions[gap_start - 1] if gap_start > 0 else None
                next_pos = positions[gap_end] if gap_end < n else None

                if prev_pos is None or next_pos is None or gap_size > self.MAX_GAP:
                    # Cannot interpolate
                    for _ in range(gap_size):
                        result.append((None, "no_interpolation"))
                elif gap_size < self.PARABOLIC_THRESHOLD:
                    # Linear interpolation
                    interp_positions = self._linear_interpolate(prev_pos, next_pos, gap_size)
                    for pos in interp_positions:
                        result.append((pos, "linear"))
                else:
                    # Parabolic interpolation
                    context = self._get_context_positions(positions, gap_start, gap_end)
                    interp_positions = self._parabolic_interpolate(
                        context, gap_start, gap_size
                    )
                    for pos in interp_positions:
                        result.append((pos, "parabolic"))

        return result

    def _linear_interpolate(
        self,
        prev: Tuple[float, float],
        next_pos: Tuple[float, float],
        gap_size: int
    ) -> List[Tuple[float, float]]:
        """Simple linear interpolation"""
        positions = []
        for i in range(gap_size):
            ratio = (i + 1) / (gap_size + 1)
            x = prev[0] + ratio * (next_pos[0] - prev[0])
            y = prev[1] + ratio * (next_pos[1] - prev[1])
            positions.append((x, y))
        return positions

    def _get_context_positions(
        self,
        positions: List[Optional[Tuple[float, float]]],
        gap_start: int,
        gap_end: int
    ) -> List[Tuple[int, float, float]]:
        """Get context positions around the gap for parabolic fit"""
        context = []

        # Get 3 positions before gap
        for i in range(gap_start - 1, max(-1, gap_start - 4), -1):
            if positions[i] is not None:
                context.insert(0, (i, positions[i][0], positions[i][1]))
                if len([c for c in context if c[0] < gap_start]) >= 3:
                    break

        # Get 3 positions after gap
        for i in range(gap_end, min(len(positions), gap_end + 4)):
            if positions[i] is not None:
                context.append((i, positions[i][0], positions[i][1]))
                if len([c for c in context if c[0] >= gap_end]) >= 3:
                    break

        return context

    def _parabolic_interpolate(
        self,
        context: List[Tuple[int, float, float]],
        gap_start: int,
        gap_size: int
    ) -> List[Tuple[float, float]]:
        """Parabolic interpolation using polynomial fit"""
        if len(context) < 3:
            # Fall back to linear if not enough context
            if len(context) >= 2:
                prev = (context[0][1], context[0][2])
                next_pos = (context[-1][1], context[-1][2])
                return self._linear_interpolate(prev, next_pos, gap_size)
            return [(0.0, 0.0)] * gap_size

        try:
            t = [c[0] for c in context]
            x = [c[1] for c in context]
            y = [c[2] for c in context]

            # Fit polynomials (degree 2 for parabola)
            degree = min(2, len(context) - 1)
            x_coeffs = np.polyfit(t, x, degree)
            y_coeffs = np.polyfit(t, y, degree)

            # Generate interpolated positions
            positions = []
            for i in range(gap_size):
                frame = gap_start + i
                interp_x = float(np.polyval(x_coeffs, frame))
                interp_y = float(np.polyval(y_coeffs, frame))
                positions.append((interp_x, interp_y))

            return positions

        except Exception:
            # Fall back to linear
            if len(context) >= 2:
                prev = (context[0][1], context[0][2])
                next_pos = (context[-1][1], context[-1][2])
                return self._linear_interpolate(prev, next_pos, gap_size)
            return [(0.0, 0.0)] * gap_size


# =============================================================================
# MAIN BALL TRACKER CLASS
# =============================================================================

class TennisBallTracker:
    """
    Main class for tennis ball and person detection and tracking

    Enhanced features:
    - Multi-pass detection with different confidence thresholds
    - Image enhancement for small ball detection
    - Kalman filter for trajectory prediction
    - Candidate recovery using predicted positions
    - Person detection with tile-based approach
    - Person tracking with appearance-based ReID
    """

    # Multi-pass confidence thresholds (high to low)
    CONF_THRESHOLDS = [0.3, 0.2, 0.15]

    def __init__(
        self,
        model_path: str = "models/ball_best.pt",
        person_model_path: str = "yolov8m.pt",
        conf_threshold: float = 0.15,  # Lower base threshold
        person_conf_threshold: float = 0.5,
        batch_size: int = 16,
        enable_validation: bool = True,
        enable_enhancement: bool = True,
        enable_kalman: bool = True,
        enable_person_detection: bool = True,
        use_person_tiles: bool = True,
        enable_person_id_merge: bool = True
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.batch_size = batch_size
        self.enable_validation = enable_validation
        self.enable_enhancement = enable_enhancement
        self.enable_kalman = enable_kalman
        self.enable_person_detection = enable_person_detection
        self.enable_person_id_merge = enable_person_id_merge

        self.interpolator = BallInterpolator()
        self.validator: Optional[BallValidator] = None
        self.tracker: Optional[TrajectoryTracker] = None
        self.kalman: Optional[BallKalmanFilter] = None
        self.enhancer: Optional[ImageEnhancer] = None

        # Person detection components
        self.person_detector: Optional[PersonDetector] = None
        self.person_tracker: Optional[PersonTracker] = None
        self.person_id_merger: Optional[PersonIDMerger] = None
        if enable_person_detection:
            self.person_detector = PersonDetector(
                model_path=person_model_path,
                conf_threshold=person_conf_threshold,
                use_tiles=use_person_tiles
            )
            self.person_tracker = PersonTracker()
            if enable_person_id_merge:
                self.person_id_merger = PersonIDMerger()

    def process_video(
        self,
        video_path: str,
        output_json_path: str,
        output_video_path: str
    ) -> Dict:
        """
        Main processing pipeline (Enhanced)

        Args:
            video_path: Path to input video
            output_json_path: Path to save JSON results
            output_video_path: Path to save annotated video

        Returns:
            Processing results dictionary
        """
        # 1. Get video info
        video_info = self._get_video_info(video_path)

        # Initialize components
        self.validator = BallValidator(video_info.resolution["height"])
        self.tracker = TrajectoryTracker(video_info.fps)
        self.kalman = BallKalmanFilter() if self.enable_kalman else None
        self.enhancer = ImageEnhancer() if self.enable_enhancement else None

        # 2. Read all frames
        print(f"Reading video: {video_path}")
        frames = self._read_video(video_path)
        print(f"Total frames: {len(frames)}")

        # 3. Multi-pass ball detection with enhancement
        print("Detecting balls (multi-pass with enhancement)...")
        raw_detections = self._detect_balls_enhanced(frames)

        # 4. Extract primary positions with Kalman-assisted selection
        print("Extracting positions with trajectory prediction...")
        raw_positions = self._extract_positions_with_kalman(raw_detections)

        # 5. Recover missing detections using predicted positions
        print("Recovering missing detections...")
        recovered_positions = self._recover_missing_detections(
            frames, raw_positions, raw_detections
        )

        # 6. Interpolate remaining missing positions
        print("Interpolating remaining gaps...")
        interpolated = self.interpolator.interpolate(recovered_positions)

        # 7. Validate and filter
        print("Validating detections...")
        validated_results = self._validate_and_filter(frames, interpolated, raw_detections)

        # 8. Detect and track persons
        person_results = []
        id_mapping = {}
        if self.enable_person_detection and self.person_detector and self.person_tracker:
            print("Detecting and tracking persons...")
            person_results = self._detect_and_track_persons(frames)

            # 8.5. Post-process: Merge fragmented person IDs
            if self.enable_person_id_merge and self.person_id_merger:
                print("Merging fragmented person IDs...")
                person_results, id_mapping = self.person_id_merger.merge_ids(
                    frames, person_results
                )
                if id_mapping:
                    print(f"  ID mapping: {id_mapping}")

        # 9. Build frame results (balls + persons)
        frame_results = self._build_frame_results(
            validated_results, video_info.fps, person_results
        )

        # 10. Calculate statistics
        stats = self._calculate_statistics(frame_results, raw_detections, id_mapping)

        # 11. Save JSON
        print(f"Saving JSON to: {output_json_path}")
        self._save_json(video_info, frame_results, stats, output_json_path)

        # 12. Create annotated video
        print(f"Creating annotated video: {output_video_path}")
        self._create_annotated_video(frames, frame_results, video_info.fps, output_video_path)

        print("Processing complete!")

        return {
            "video_info": asdict(video_info),
            "statistics": stats,
            "output_json": output_json_path,
            "output_video": output_video_path
        }

    def _detect_and_track_persons(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """
        Detect and track persons in all frames

        Returns:
            List of tracked person lists per frame
        """
        all_person_results = []

        for i, frame in enumerate(frames):
            # Detect persons
            detections = self.person_detector.detect(frame)

            # Track persons (assign IDs)
            tracked = self.person_tracker.update(frame, detections)

            all_person_results.append(tracked)

            if (i + 1) % 500 == 0 or i == len(frames) - 1:
                print(f"  Processed {i + 1}/{len(frames)} frames for persons")

        return all_person_results

    def _get_video_info(self, video_path: str) -> VideoInfo:
        """Extract video metadata"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        return VideoInfo(
            filename=os.path.basename(video_path),
            fps=fps,
            total_frames=total_frames,
            duration_seconds=total_frames / fps if fps > 0 else 0,
            resolution={"width": width, "height": height}
        )

    def _read_video(self, video_path: str) -> List[np.ndarray]:
        """Read all frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def _detect_balls_enhanced(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """
        Enhanced detection with:
        1. Multi-pass with different confidence thresholds
        2. Image enhancement for difficult frames
        3. Merge detections from all passes
        """
        all_detections = [[] for _ in range(len(frames))]

        # Pass 1: Normal detection with low confidence
        print("  Pass 1: Standard detection...")
        pass1_detections = self._detect_balls_batch(frames, self.conf_threshold)

        for i, dets in enumerate(pass1_detections):
            all_detections[i].extend(dets)

        # Pass 2: Enhanced frames for missing detections
        if self.enhancer:
            missing_indices = [i for i, dets in enumerate(pass1_detections) if not dets]

            if missing_indices:
                print(f"  Pass 2: Enhanced detection for {len(missing_indices)} missing frames...")

                # Process in batches
                for batch_start in range(0, len(missing_indices), self.batch_size):
                    batch_indices = missing_indices[batch_start:batch_start + self.batch_size]
                    enhanced_batch = [self.enhancer.enhance(frames[i]) for i in batch_indices]

                    results = self.model.predict(
                        enhanced_batch,
                        verbose=False,
                        conf=self.conf_threshold * 0.8  # Slightly lower threshold for enhanced
                    )

                    for j, res in enumerate(results):
                        frame_idx = batch_indices[j]
                        if res.boxes is not None and len(res.boxes) > 0:
                            for k in range(len(res.boxes)):
                                x, y, w, h = res.boxes.xywh[k].cpu().numpy()
                                conf = float(res.boxes.conf[k].cpu().numpy())
                                x1, y1, x2, y2 = res.boxes.xyxy[k].cpu().numpy().astype(int)

                                all_detections[frame_idx].append({
                                    "x": float(x),
                                    "y": float(y),
                                    "w": float(w),
                                    "h": float(h),
                                    "conf": conf,
                                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                                    "enhanced": True
                                })

        # Remove duplicates (same position from different passes)
        for i in range(len(all_detections)):
            all_detections[i] = self._remove_duplicate_detections(all_detections[i])

        return all_detections

    def _remove_duplicate_detections(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Remove duplicate detections based on IoU"""
        if len(detections) <= 1:
            return detections

        # Sort by confidence (highest first)
        sorted_dets = sorted(detections, key=lambda d: d["conf"], reverse=True)
        kept = []

        for det in sorted_dets:
            is_duplicate = False
            for kept_det in kept:
                iou = self._calculate_iou(det["bbox"], kept_det["bbox"])
                if iou > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(det)

        return kept

    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _detect_balls_batch(self, frames: List[np.ndarray], conf: float) -> List[List[Dict]]:
        """Detect balls in frames using batch inference"""
        all_detections = []

        for i in range(0, len(frames), self.batch_size):
            batch = frames[i:i + self.batch_size]

            results = self.model.predict(
                batch,
                verbose=False,
                conf=conf
            )

            for res in results:
                frame_detections = []
                if res.boxes is not None and len(res.boxes) > 0:
                    for j in range(len(res.boxes)):
                        x, y, w, h = res.boxes.xywh[j].cpu().numpy()
                        conf_score = float(res.boxes.conf[j].cpu().numpy())
                        x1, y1, x2, y2 = res.boxes.xyxy[j].cpu().numpy().astype(int)

                        frame_detections.append({
                            "x": float(x),
                            "y": float(y),
                            "w": float(w),
                            "h": float(h),
                            "conf": conf_score,
                            "bbox": (int(x1), int(y1), int(x2), int(y2))
                        })

                all_detections.append(frame_detections)

            # Progress
            progress = min(i + self.batch_size, len(frames))
            if progress % 500 == 0 or progress == len(frames):
                print(f"    Processed {progress}/{len(frames)} frames")

        return all_detections

    def _extract_positions_with_kalman(
        self,
        detections: List[List[Dict]]
    ) -> List[Optional[Tuple[float, float]]]:
        """
        Extract positions using Kalman filter for trajectory prediction
        When multiple detections exist, choose the one closest to predicted position
        """
        positions = []
        kalman = BallKalmanFilter()

        for frame_idx, frame_dets in enumerate(detections):
            if not frame_dets:
                # No detection - use Kalman prediction for tracking state
                if kalman.initialized:
                    kalman.predict()
                positions.append(None)
                continue

            if not kalman.initialized:
                # First detection - just use highest confidence
                best = max(frame_dets, key=lambda d: d["conf"])
                kalman.init(best["x"], best["y"])
                positions.append((best["x"], best["y"]))
            else:
                # Predict next position
                pred_x, pred_y = kalman.predict()

                # Find detection closest to prediction
                best_det = None
                best_score = float('inf')

                for det in frame_dets:
                    dist = math.sqrt((det["x"] - pred_x)**2 + (det["y"] - pred_y)**2)
                    # Score combines distance and confidence
                    score = dist - det["conf"] * 50  # Weight confidence

                    if score < best_score:
                        best_score = score
                        best_det = det

                if best_det:
                    kalman.update(best_det["x"], best_det["y"])
                    positions.append((best_det["x"], best_det["y"]))
                else:
                    positions.append(None)

        return positions

    def _recover_missing_detections(
        self,
        frames: List[np.ndarray],  # noqa: F841 - kept for future ROI detection
        positions: List[Optional[Tuple[float, float]]],
        detections: List[List[Dict]]
    ) -> List[Optional[Tuple[float, float]]]:
        """
        Try to recover missing detections by:
        1. Using Kalman predicted positions to search in low-confidence detections
        2. Running targeted detection on predicted ROI
        """
        recovered = list(positions)  # Copy
        kalman = BallKalmanFilter()

        # Build trajectory from known positions
        for i, pos in enumerate(positions):
            if pos is not None:
                if not kalman.initialized:
                    kalman.init(pos[0], pos[1])
                else:
                    kalman.predict()
                    kalman.update(pos[0], pos[1])

        # Reset and try to recover
        kalman = BallKalmanFilter()
        recovery_count = 0

        for i, pos in enumerate(positions):
            if pos is not None:
                if not kalman.initialized:
                    kalman.init(pos[0], pos[1])
                else:
                    kalman.predict()
                    kalman.update(pos[0], pos[1])
            elif kalman.initialized:
                # Missing detection - try to recover
                pred_x, pred_y = kalman.predict()

                # Look for low-confidence detections near predicted position
                search_radius = 100  # pixels

                for det in detections[i]:
                    dist = math.sqrt((det["x"] - pred_x)**2 + (det["y"] - pred_y)**2)
                    if dist < search_radius:
                        recovered[i] = (det["x"], det["y"])
                        kalman.update(det["x"], det["y"])
                        recovery_count += 1
                        break

        if recovery_count > 0:
            print(f"  Recovered {recovery_count} missing detections using trajectory prediction")

        return recovered

    def _detect_balls(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """
        Detect balls in all frames using batch inference

        Returns:
            List of detection lists per frame, each detection has:
            {x, y, w, h, conf, bbox}
        """
        all_detections = []

        for i in range(0, len(frames), self.batch_size):
            batch = frames[i:i + self.batch_size]

            results = self.model.predict(
                batch,
                verbose=False,
                conf=self.conf_threshold
            )

            for res in results:
                frame_detections = []
                if res.boxes is not None and len(res.boxes) > 0:
                    for j in range(len(res.boxes)):
                        x, y, w, h = res.boxes.xywh[j].cpu().numpy()
                        conf = float(res.boxes.conf[j].cpu().numpy())
                        x1, y1, x2, y2 = res.boxes.xyxy[j].cpu().numpy().astype(int)

                        frame_detections.append({
                            "x": float(x),
                            "y": float(y),
                            "w": float(w),
                            "h": float(h),
                            "conf": conf,
                            "bbox": (int(x1), int(y1), int(x2), int(y2))
                        })

                all_detections.append(frame_detections)

            # Progress
            progress = min(i + self.batch_size, len(frames))
            if progress % 500 == 0 or progress == len(frames):
                print(f"  Processed {progress}/{len(frames)} frames")

        return all_detections

    def _extract_primary_positions(
        self,
        detections: List[List[Dict]]
    ) -> List[Optional[Tuple[float, float]]]:
        """
        Extract primary (highest confidence) position per frame
        """
        positions = []
        for frame_dets in detections:
            if frame_dets:
                # Get highest confidence detection
                best = max(frame_dets, key=lambda d: d["conf"])
                positions.append((best["x"], best["y"]))
            else:
                positions.append(None)
        return positions

    def _validate_and_filter(
        self,
        frames: List[np.ndarray],
        interpolated: List[Tuple[Optional[Tuple[float, float]], str]],
        raw_detections: List[List[Dict]]
    ) -> List[Dict]:
        """
        Validate interpolated positions and filter false positives
        """
        results = []

        for i, ((pos, method), frame_dets) in enumerate(zip(interpolated, raw_detections)):
            result = {
                "frame_index": i,
                "position": pos,
                "method": method,
                "validated": True,
                "all_detections": frame_dets
            }

            if pos is None:
                result["validated"] = False
                results.append(result)
                continue

            # Validate using trajectory
            if self.tracker:
                is_valid, reason = self.tracker.is_position_valid(i, pos[0], pos[1])
                if not is_valid:
                    result["validated"] = False
                    result["rejection_reason"] = reason
                    # Reset tracker on teleport to allow recovery
                    self.tracker.reset()
                    results.append(result)
                    continue

            # Validate using image comparison (for detections only)
            if method == "detection" and self.enable_validation and self.validator:
                # Find the detection for this position
                for det in frame_dets:
                    if abs(det["x"] - pos[0]) < 1 and abs(det["y"] - pos[1]) < 1:
                        is_valid, reason = self.validator.validate(frames[i], det["bbox"])
                        if not is_valid:
                            result["validated"] = False
                            result["rejection_reason"] = reason
                        else:
                            # Add to reference templates
                            self.validator.add_reference(frames[i], det["bbox"])
                        break

            # Update tracker with valid position
            if result["validated"] and self.tracker:
                self.tracker.add_position(i, pos[0], pos[1])

            results.append(result)

        return results

    def _build_frame_results(
        self,
        validated_results: List[Dict],
        fps: float,
        person_results: List[List[Dict]] = None
    ) -> List[FrameResult]:
        """Build final frame results with ball and person detections"""
        frame_results = []
        ball_id = 1  # Primary ball tracking

        for i, res in enumerate(validated_results):
            frame = FrameResult(
                frame_index=res["frame_index"],
                timestamp_seconds=res["frame_index"] / fps
            )

            # Add primary tracked ball
            if res["validated"] and res["position"] is not None:
                pos = res["position"]

                # Get bbox and confidence if available from detections
                bbox = None
                conf = None
                for det in res.get("all_detections", []):
                    if abs(det["x"] - pos[0]) < 1 and abs(det["y"] - pos[1]) < 1:
                        bbox = det["bbox"]
                        conf = det["conf"]
                        break

                ball = BallDetection(
                    id=ball_id,
                    x=pos[0],
                    y=pos[1],
                    confidence=conf,
                    source=DetectionSource.DETECTION if res["method"] == "detection" else DetectionSource.INTERPOLATED,
                    validated=True,
                    bbox=bbox,
                    interpolation_method=res["method"] if res["method"] != "detection" else None
                )
                frame.balls.append(ball)

            # Also add all other detections (for multi-ball requirement)
            secondary_id = 2
            for det in res.get("all_detections", []):
                # Skip if already added as primary
                if res["position"] and abs(det["x"] - res["position"][0]) < 1 and abs(det["y"] - res["position"][1]) < 1:
                    continue

                ball = BallDetection(
                    id=secondary_id,
                    x=det["x"],
                    y=det["y"],
                    confidence=det["conf"],
                    source=DetectionSource.DETECTION,
                    validated=True,
                    bbox=det["bbox"]
                )
                frame.balls.append(ball)
                secondary_id += 1

            # Add person detections
            if person_results and i < len(person_results):
                for person in person_results[i]:
                    frame.persons.append(PersonDetection(
                        id=person["id"],
                        x=person["x"],
                        y=person["y"],
                        bbox=person["bbox"],
                        confidence=person["conf"],
                        tracked=person.get("tracked", True)
                    ))

            frame_results.append(frame)

        return frame_results

    def _calculate_statistics(
        self,
        frame_results: List[FrameResult],
        raw_detections: List[List[Dict]],
        id_mapping: Dict[int, int] = None
    ) -> Dict:
        """Calculate processing statistics for both balls and persons"""
        # Ball statistics
        total_detections = sum(1 for res in frame_results if any(
            b.source == DetectionSource.DETECTION for b in res.balls
        ))
        interpolated_frames = sum(1 for res in frame_results if any(
            b.source == DetectionSource.INTERPOLATED for b in res.balls
        ))

        total_raw_detections = sum(len(dets) for dets in raw_detections)
        total_kept = sum(len(res.balls) for res in frame_results)
        rejected = max(0, total_raw_detections - total_kept)

        total_frames = len(frame_results)
        frames_with_ball = sum(1 for res in frame_results if res.balls)

        all_confs = [b.confidence for res in frame_results for b in res.balls if b.confidence]

        # Person statistics
        frames_with_persons = sum(1 for res in frame_results if res.persons)
        unique_person_ids = set()
        for res in frame_results:
            for p in res.persons:
                unique_person_ids.add(p.id)

        # ID merge statistics
        ids_merged = len(id_mapping) if id_mapping else 0
        original_ids = len(unique_person_ids) + ids_merged

        return {
            "ball": {
                "total_frames": total_frames,
                "frames_with_detection": total_detections,
                "interpolated_frames": interpolated_frames,
                "rejected_false_positives": rejected,
                "detection_rate": round(frames_with_ball / total_frames, 4) if total_frames > 0 else 0,
                "average_confidence": round(sum(all_confs) / len(all_confs), 4) if all_confs else 0
            },
            "person": {
                "frames_with_persons": frames_with_persons,
                "unique_persons_detected": len(unique_person_ids),
                "person_ids": sorted(list(unique_person_ids)),
                "ids_before_merge": original_ids,
                "ids_merged": ids_merged,
                "id_mapping": id_mapping if id_mapping else {}
            }
        }

    def _save_json(
        self,
        video_info: VideoInfo,
        frame_results: List[FrameResult],
        stats: Dict,
        output_path: str
    ):
        """Save results to JSON file"""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Convert to serializable format
        output = {
            "video_info": asdict(video_info),
            "detection_config": {
                "ball": {
                    "model_path": "models/ball_best.pt",
                    "confidence_threshold": self.conf_threshold,
                    "max_interpolation_gap": BallInterpolator.MAX_GAP,
                    "validation_enabled": self.enable_validation
                },
                "person": {
                    "model_path": "yolov8m.pt",
                    "enabled": self.enable_person_detection,
                    "tile_based_detection": True
                }
            },
            "frames": [
                {
                    "frame_index": fr.frame_index,
                    "timestamp_seconds": round(fr.timestamp_seconds, 4),
                    "balls": [
                        {
                            "id": b.id,
                            "x": round(b.x, 2),
                            "y": round(b.y, 2),
                            "confidence": round(b.confidence, 4) if b.confidence else None,
                            "source": b.source.value,
                            "validated": b.validated,
                            "bbox": {"x1": b.bbox[0], "y1": b.bbox[1], "x2": b.bbox[2], "y2": b.bbox[3]} if b.bbox else None,
                            "interpolation_method": b.interpolation_method
                        }
                        for b in fr.balls
                    ],
                    "persons": [
                        {
                            "id": p.id,
                            "x": round(p.x, 2),
                            "y": round(p.y, 2),
                            "confidence": round(p.confidence, 4),
                            "bbox": {"x1": p.bbox[0], "y1": p.bbox[1], "x2": p.bbox[2], "y2": p.bbox[3]},
                            "tracked": p.tracked
                        }
                        for p in fr.persons
                    ]
                }
                for fr in frame_results
            ],
            "statistics": stats
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    def _create_annotated_video(
        self,
        frames: List[np.ndarray],
        frame_results: List[FrameResult],
        fps: float,
        output_path: str
    ):
        """Create annotated video with ball and person detections"""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Colors for different person IDs
        person_colors = [
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (255, 255, 0),  # Cyan
            (128, 0, 128),  # Purple
            (0, 128, 128),  # Teal
            (128, 128, 0),  # Olive
        ]

        try:
            for i, (frame, result) in enumerate(zip(frames, frame_results)):
                annotated = frame.copy()

                # Draw persons first (so balls appear on top)
                for person in result.persons:
                    # Get color based on ID
                    color = person_colors[(person.id - 1) % len(person_colors)]

                    # Draw bounding box
                    x1, y1, x2, y2 = person.bbox
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                    # Draw ID label with background
                    label = f"Person {person.id}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10),
                                 (x1 + label_size[0] + 10, y1), color, -1)
                    cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Draw balls
                for ball in result.balls:
                    x, y = int(ball.x), int(ball.y)

                    # Color based on source
                    if ball.source == DetectionSource.DETECTION:
                        color = (0, 255, 0)  # Green for detection
                    else:
                        color = (0, 165, 255)  # Orange for interpolated (BGR)

                    # Draw ball marker
                    cv2.circle(annotated, (x, y), 10, color, 2)
                    cv2.circle(annotated, (x, y), 3, color, -1)

                    # Draw bbox if available
                    if ball.bbox:
                        bx1, by1, bx2, by2 = ball.bbox
                        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 1)

                    # Label
                    src_label = "DET" if ball.source == DetectionSource.DETECTION else "INT"
                    label = f"Ball [{src_label}]"
                    if ball.confidence:
                        label += f" {ball.confidence:.2f}"
                    cv2.putText(annotated, label, (x + 15, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Frame info
                cv2.putText(annotated, f"Frame: {result.frame_index}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated, f"Frame: {result.frame_index}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

                # Person count info
                if result.persons:
                    person_info = f"Persons: {len(result.persons)}"
                    cv2.putText(annotated, person_info, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(annotated, person_info, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

                out.write(annotated)

                if (i + 1) % 500 == 0 or i == len(frames) - 1:
                    print(f"  Writing frame {i + 1}/{len(frames)}")

        finally:
            out.release()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point"""
    # Configuration - Hardcoded paths as requested
    VIDEO_PATH = "1765199807.mp4"
    MODEL_PATH = "models/ball_best.pt"
    PERSON_MODEL_PATH = "yolov8m.pt"  # Will be auto-downloaded if not exists

    # Output paths - JSON filename matches video filename
    video_basename = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    OUTPUT_JSON = f"output/{video_basename}.json"
    OUTPUT_VIDEO = f"output/{video_basename}_annotated.mp4"

    print("=" * 60)
    print("TENNIS BALL & PERSON DETECTION SYSTEM (ENHANCED)")
    print("=" * 60)
    print(f"Input video: {VIDEO_PATH}")
    print(f"Ball model: {MODEL_PATH}")
    print(f"Person model: {PERSON_MODEL_PATH}")
    print(f"Output JSON: {OUTPUT_JSON}")
    print(f"Output video: {OUTPUT_VIDEO}")
    print()
    print("Enhanced features enabled:")
    print("  - Multi-pass ball detection with image enhancement")
    print("  - Kalman filter for ball trajectory prediction")
    print("  - Extended interpolation (up to 30 frames)")
    print("  - Trajectory-based candidate recovery")
    print("  - Person detection (full frame)")
    print("  - Person tracking with appearance-based ReID")
    print("  - Post-processing ID merger (consolidate fragmented IDs)")
    print("=" * 60)

    # Create tracker with enhanced settings
    tracker = TennisBallTracker(
        model_path=MODEL_PATH,
        person_model_path=PERSON_MODEL_PATH,
        conf_threshold=0.15,          # Lower threshold for better recall
        person_conf_threshold=0.5,    # Person detection threshold
        batch_size=16,
        enable_validation=True,
        enable_enhancement=True,      # CLAHE + sharpening for small balls
        enable_kalman=True,           # Kalman filter for trajectory
        enable_person_detection=True, # Enable person detection
        use_person_tiles=False,       # Full frame detection (not tile-based)
        enable_person_id_merge=True   # Merge fragmented person IDs
    )

    # Process video
    result = tracker.process_video(
        video_path=VIDEO_PATH,
        output_json_path=OUTPUT_JSON,
        output_video_path=OUTPUT_VIDEO
    )

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"JSON output: {result['output_json']}")
    print(f"Video output: {result['output_video']}")

    print("\nBall Statistics:")
    for key, value in result['statistics']['ball'].items():
        print(f"  {key}: {value}")

    print("\nPerson Statistics:")
    person_stats = result['statistics']['person']
    print(f"  frames_with_persons: {person_stats['frames_with_persons']}")
    print(f"  unique_persons_detected: {person_stats['unique_persons_detected']}")
    print(f"  person_ids: {person_stats['person_ids']}")
    print(f"  ids_before_merge: {person_stats['ids_before_merge']}")
    print(f"  ids_merged: {person_stats['ids_merged']}")
    if person_stats['id_mapping']:
        print(f"  id_mapping: {person_stats['id_mapping']}")


if __name__ == "__main__":
    main()
