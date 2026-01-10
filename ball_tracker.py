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
import time
from ultralytics import YOLO
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict
from enum import Enum

# Import MatchAnalyzer for combined pipeline
from match_analyzer import MatchAnalyzer


def format_time(seconds: float) -> str:
    """Format seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


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


@dataclass
class ShotEvent:
    """Information about a single shot/hit event"""

    frame_index: int
    timestamp_seconds: float
    person_id: int
    ball_position: Tuple[float, float]  # (x, y)
    person_bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    # Pose angles (degrees)
    left_shoulder_angle: Optional[float] = None
    right_shoulder_angle: Optional[float] = None
    left_knee_angle: Optional[float] = None
    right_knee_angle: Optional[float] = None
    # Court position info
    ball_in_court: bool = True
    ball_zone: str = "unknown"
    ball_side: str = "unknown"
    keypoints: Optional[Dict] = None


@dataclass
class CourtPosition:
    """Ball/Person position relative to court"""

    x: float  # Original video x
    y: float  # Original video y
    is_in_court: bool = False
    court_zone: str = "unknown"  # "service_box", "baseline", "near_net", "out"
    side: str = "unknown"  # "top", "bottom"


# =============================================================================
# COURT ALIGNER CLASS
# =============================================================================


class CourtAligner:
    """
    Aligns ball/person positions to court coordinates.
    Determines if ball is IN/OUT and which zone it's in.
    """

    def __init__(self, court_data: str | Dict = "data.json"):
        """
        Initialize court aligner with court coordinates.

        Args:
            court_data: Either a path to JSON file (str) or court coordinates dict directly
        """
        if isinstance(court_data, dict):
            # Direct dict input
            self.court_points = court_data.copy()
            print("Loaded court data from dict")
        else:
            # File path input
            self.court_points = self._load_court_data(court_data)
        self._setup_court_boundaries()

    def _load_court_data(self, path: str) -> Dict:
        """Load court coordinates from JSON file"""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            print(f"Loaded court data from: {path}")
            return data
        except FileNotFoundError:
            print(f"Warning: Court data file not found: {path}. Using default court.")
            return self._default_court_data()

    def _default_court_data(self) -> Dict:
        """Return default court coordinates"""
        return {
            "top_left": {"x": 100, "y": 150},
            "top_center": {"x": 400, "y": 450},
            "top_right": {"x": 700, "y": 150},
            "center_left": {"x": 100, "y": 750},
            "center_right": {"x": 700, "y": 750},
            "bottom_left": {"x": 100, "y": 1050},
            "bottom_center": {"x": 400, "y": 1050},
            "bottom_right": {"x": 700, "y": 1050},
        }

    def _setup_court_boundaries(self):
        """Setup court polygon and zone boundaries"""
        # Court boundary polygon (4 corners in order)
        self.court_polygon = np.array(
            [
                [
                    self.court_points["top_left"]["x"],
                    self.court_points["top_left"]["y"],
                ],
                [
                    self.court_points["top_right"]["x"],
                    self.court_points["top_right"]["y"],
                ],
                [
                    self.court_points["bottom_right"]["x"],
                    self.court_points["bottom_right"]["y"],
                ],
                [
                    self.court_points["bottom_left"]["x"],
                    self.court_points["bottom_left"]["y"],
                ],
            ],
            dtype=np.float32,
        )

        # Net line y coordinate (divides top and bottom half)
        self.net_y = self.court_points["center_left"]["y"]

        # Service box boundaries
        self.top_service_y = self.court_points.get("top_center", {}).get(
            "y", self.net_y - 200
        )
        self.bottom_service_y = self.court_points.get("bottom_center", {}).get(
            "y", self.net_y + 200
        )

        # Court boundaries
        self.court_left = min(
            self.court_points["top_left"]["x"], self.court_points["bottom_left"]["x"]
        )
        self.court_right = max(
            self.court_points["top_right"]["x"], self.court_points["bottom_right"]["x"]
        )
        self.court_top = min(
            self.court_points["top_left"]["y"], self.court_points["top_right"]["y"]
        )
        self.court_bottom = max(
            self.court_points["bottom_left"]["y"],
            self.court_points["bottom_right"]["y"],
        )

    def is_point_in_court(self, x: float, y: float) -> bool:
        """
        Check if point is inside court boundaries using ray casting algorithm.

        Args:
            x, y: Point coordinates

        Returns:
            True if point is inside court
        """
        point = np.array([x, y])
        return self._point_in_polygon(point, self.court_polygon)

    def _point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """
        Ray casting algorithm for point-in-polygon test.

        Args:
            point: [x, y] coordinates
            polygon: Nx2 array of polygon vertices

        Returns:
            True if point is inside polygon
        """
        x, y = point
        n = len(polygon)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            if ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi
            ):
                inside = not inside
            j = i

        return inside

    def get_court_zone(self, x: float, y: float) -> Tuple[str, str]:
        """
        Determine which zone the ball is in.

        Args:
            x, y: Ball coordinates

        Returns:
            (zone, side) tuple
            - zone: "service_box", "baseline", "near_net", "out"
            - side: "top", "bottom", "unknown"
        """
        if not self.is_point_in_court(x, y):
            return ("out", "unknown")

        # Determine side (above or below net)
        if y < self.net_y:
            side = "top"
            # Zone determination for top half
            if y < self.top_service_y:
                zone = "baseline"
            elif abs(y - self.net_y) < 100:  # Near net threshold
                zone = "near_net"
            else:
                zone = "service_box"
        else:
            side = "bottom"
            # Zone determination for bottom half
            if y > self.bottom_service_y:
                zone = "baseline"
            elif abs(y - self.net_y) < 100:  # Near net threshold
                zone = "near_net"
            else:
                zone = "service_box"

        return (zone, side)

    def align_position(self, x: float, y: float) -> CourtPosition:
        """
        Create CourtPosition with all alignment info.

        Args:
            x, y: Original video coordinates

        Returns:
            CourtPosition with in_court status and zone info
        """
        is_in = self.is_point_in_court(x, y)
        zone, side = self.get_court_zone(x, y)

        return CourtPosition(x=x, y=y, is_in_court=is_in, court_zone=zone, side=side)

    def get_court_info(self) -> Dict:
        """Return court coordinates for JSON output"""
        return {
            "top_left": self.court_points.get("top_left"),
            "top_right": self.court_points.get("top_right"),
            "center_left": self.court_points.get("center_left"),
            "center_right": self.court_points.get("center_right"),
            "bottom_left": self.court_points.get("bottom_left"),
            "bottom_right": self.court_points.get("bottom_right"),
            "net_y": self.net_y,
        }


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

    def validate(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Tuple[bool, str]:
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

    def __init__(
        self, fps: float, max_speed_mps: float = 70.0, pixels_per_meter: float = 50.0
    ):
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
        displacement = math.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)
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
        self.F = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )

        # Observation matrix (we only observe position)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)

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
        self.sharpen_kernel = np.array(
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32
        )

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
    Detects persons using yolov8s with tile-based approach for better accuracy
    on wide-angle full court videos where persons may appear small.
    """

    PERSON_CLASS_ID = 0  # COCO class ID for person

    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        conf_threshold: float = 0.5,
        use_tiles: bool = True,
        tile_overlap: float = 0.2,
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

    def detect_batch(self, frames: List[np.ndarray], batch_size: int = 32) -> List[List[Dict]]:
        """
        Detect persons in multiple frames using batch inference for better performance.

        Args:
            frames: List of frames to process
            batch_size: Number of frames to process in each batch

        Returns:
            List of detection lists, one per frame
        """
        all_detections = []

        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]

            # Batch inference - much faster than single frame
            results = self.model.predict(
                batch,
                verbose=False,
                conf=self.conf_threshold,
                classes=[self.PERSON_CLASS_ID],
            )

            # Process results with optimized GPU transfer
            for res in results:
                frame_detections = []
                if res.boxes is not None and len(res.boxes) > 0:
                    # Batch transfer all tensors to CPU at once (OPTIMIZED)
                    boxes_xywh = res.boxes.xywh.cpu().numpy()
                    boxes_conf = res.boxes.conf.cpu().numpy()
                    boxes_xyxy = res.boxes.xyxy.cpu().numpy().astype(int)

                    for j in range(len(res.boxes)):
                        x, y, w, h = boxes_xywh[j]
                        conf = float(boxes_conf[j])
                        x1, y1, x2, y2 = boxes_xyxy[j]

                        frame_detections.append({
                            "x": float(x),
                            "y": float(y),
                            "w": float(w),
                            "h": float(h),
                            "conf": conf,
                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        })

                all_detections.append(frame_detections)

        return all_detections

    def _detect_single(self, frame: np.ndarray) -> List[Dict]:
        """Detect on full frame"""
        results = self.model.predict(
            frame,
            verbose=False,
            conf=self.conf_threshold,
            classes=[self.PERSON_CLASS_ID],
        )

        detections = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            # Batch transfer all tensors to CPU at once (OPTIMIZED)
            boxes_xywh = results[0].boxes.xywh.cpu().numpy()
            boxes_conf = results[0].boxes.conf.cpu().numpy()
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int)

            for i in range(len(results[0].boxes)):
                x, y, w, h = boxes_xywh[i]
                conf = float(boxes_conf[i])
                x1, y1, x2, y2 = boxes_xyxy[i]

                detections.append(
                    {
                        "x": float(x),
                        "y": float(y),
                        "w": float(w),
                        "h": float(h),
                        "conf": conf,
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    }
                )

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
                classes=[self.PERSON_CLASS_ID],
            )

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                # Batch transfer all tensors to CPU at once (OPTIMIZED)
                boxes_xywh = results[0].boxes.xywh.cpu().numpy()
                boxes_conf = results[0].boxes.conf.cpu().numpy()
                boxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int)

                for i in range(len(results[0].boxes)):
                    x, y, w_det, h_det = boxes_xywh[i]
                    conf = float(boxes_conf[i])
                    x1, y1, x2, y2 = boxes_xyxy[i]

                    # Convert to full frame coordinates
                    all_detections.append(
                        {
                            "x": float(x + x_start),
                            "y": float(y + y_start),
                            "w": float(w_det),
                            "h": float(h_det),
                            "conf": conf,
                            "bbox": (
                                int(x1 + x_start),
                                int(y1 + y_start),
                                int(x2 + x_start),
                                int(y2 + y_start),
                            ),
                        }
                    )

        # Also detect on full frame for large persons
        full_detections = self._detect_single(frame)
        all_detections.extend(full_detections)

        # Remove duplicates using NMS
        return self._nms_detections(all_detections)

    def _nms_detections(
        self, detections: List[Dict], iou_threshold: float = 0.5
    ) -> List[Dict]:
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

    def _calculate_iou(
        self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
    ) -> float:
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
        max_distance: float = 200,
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
            "velocity": (0, 0),
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
        self, frame: np.ndarray, detections: List[Dict]
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
                    (track["center"][0] - det["x"]) ** 2
                    + (track["center"][1] - det["y"]) ** 2
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
                cost_matrix[i, j] = (
                    0.3 * iou + 0.3 * dist_score + 0.4 * appearance_score
                )

        # Greedy matching (Hungarian algorithm would be better but this is simpler)
        matched = []
        matched_tracks = set()
        matched_dets = set()

        # Sort by cost (descending - best matches first)
        indices = np.unravel_index(
            np.argsort(-cost_matrix, axis=None), cost_matrix.shape
        )

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
                (track["center"][0] - det["x"]) ** 2
                + (track["center"][1] - det["y"]) ** 2
            )

            if iou >= self.iou_threshold or dist < self.max_distance:
                matched.append((track_id, j))
                matched_tracks.add(i)
                matched_dets.add(j)

        unmatched_dets = [j for j in range(num_dets) if j not in matched_dets]
        unmatched_tracks = [
            track_ids[i] for i in range(num_tracks) if i not in matched_tracks
        ]

        return matched, unmatched_dets, unmatched_tracks

    def _extract_histogram(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
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

    def _calculate_iou(
        self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
    ) -> float:
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
# PERSON ID MERGER (POST-PROCESSING) - COLOR-BASED
# =============================================================================


class PersonIDMerger:
    """
    Post-processing class to merge fragmented person IDs.

    When tracking loses a person and re-detects them later, they get a new ID.
    This class analyzes all person tracks and merges IDs that belong to the
    same person based on CLOTHING COLOR:
    1. Dominant colors of upper body (shirt/jersey)
    2. Dominant colors of lower body (pants/shorts)
    3. Color distribution matching

    Two persons with different clothing colors will NOT be merged.
    """

    def __init__(
        self,
        color_similarity_threshold: float = 0.85,  # Very strict - must match clothing
        min_samples: int = 10,
        n_dominant_colors: int = 3,
    ):
        """
        Args:
            color_similarity_threshold: Minimum color similarity to consider same person (strict)
            min_samples: Minimum samples needed to compute reliable appearance
            n_dominant_colors: Number of dominant colors to extract
        """
        self.color_similarity_threshold = color_similarity_threshold
        self.min_samples = min_samples
        self.n_dominant_colors = n_dominant_colors

    def merge_ids(
        self, frames: List[np.ndarray], person_results: List[List[Dict]]
    ) -> Tuple[List[List[Dict]], Dict[int, int]]:
        """
        Analyze all person detections and merge IDs belonging to same person.
        Uses strict color matching to avoid merging different people.
        """
        # Step 1: Collect color features for each ID
        print("    Collecting clothing color features for each person ID...")
        id_features = self._collect_color_features(frames, person_results)

        if len(id_features) <= 1:
            print("    Only 1 or fewer person IDs found, no merging needed")
            return person_results, {}

        # Step 2: Compare all ID pairs using strict color matching
        print(f"    Comparing {len(id_features)} person IDs by clothing color...")
        id_pairs_to_merge = self._find_color_matching_ids(id_features)

        if not id_pairs_to_merge:
            print("    No IDs with matching clothing colors found")
            return person_results, {}

        # Step 3: Build merge groups using Union-Find
        merge_groups = self._build_merge_groups(id_features.keys(), id_pairs_to_merge)

        # Step 4: Create ID mapping (old_id -> new_id)
        id_mapping = self._create_id_mapping(merge_groups)

        if not id_mapping:
            print("    No ID remapping needed")
            return person_results, {}

        print(f"    Merging {len(id_mapping)} IDs based on clothing color match")

        # Step 5: Apply mapping to results
        updated_results = self._apply_id_mapping(person_results, id_mapping)

        return updated_results, id_mapping

    def _collect_color_features(
        self, frames: List[np.ndarray], person_results: List[List[Dict]]
    ) -> Dict[int, Dict]:
        """
        Collect clothing color features for each person ID.
        Extracts dominant colors from upper body (shirt) and lower body (pants).
        """
        id_features: Dict[int, Dict] = {}

        for frame_idx, persons in enumerate(person_results):
            frame = frames[frame_idx]

            for person in persons:
                person_id = person["id"]
                bbox = person["bbox"]

                if person_id not in id_features:
                    id_features[person_id] = {
                        "upper_colors": [],  # List of dominant colors from upper body
                        "lower_colors": [],  # List of dominant colors from lower body
                        "upper_histograms": [],
                        "lower_histograms": [],
                        "frame_ranges": [],
                        "positions": [],
                    }

                # Extract color features
                upper_colors, lower_colors, upper_hist, lower_hist = (
                    self._extract_clothing_colors(frame, bbox)
                )

                if upper_colors is not None:
                    id_features[person_id]["upper_colors"].append(upper_colors)
                    id_features[person_id]["upper_histograms"].append(upper_hist)
                if lower_colors is not None:
                    id_features[person_id]["lower_colors"].append(lower_colors)
                    id_features[person_id]["lower_histograms"].append(lower_hist)

                id_features[person_id]["frame_ranges"].append(frame_idx)
                id_features[person_id]["positions"].append(
                    (frame_idx, person["x"], person["y"])
                )

        # Compute average/representative colors for each ID
        for person_id, features in id_features.items():
            features["total_detections"] = len(features["frame_ranges"])

            # Average upper body histogram
            if features["upper_histograms"]:
                avg_upper = np.mean(features["upper_histograms"], axis=0).astype(
                    np.float32
                )
                cv2.normalize(avg_upper, avg_upper)
                features["avg_upper_hist"] = avg_upper
                features["dominant_upper"] = self._get_average_dominant_colors(
                    features["upper_colors"]
                )
            else:
                features["avg_upper_hist"] = None
                features["dominant_upper"] = None

            # Average lower body histogram
            if features["lower_histograms"]:
                avg_lower = np.mean(features["lower_histograms"], axis=0).astype(
                    np.float32
                )
                cv2.normalize(avg_lower, avg_lower)
                features["avg_lower_hist"] = avg_lower
                features["dominant_lower"] = self._get_average_dominant_colors(
                    features["lower_colors"]
                )
            else:
                features["avg_lower_hist"] = None
                features["dominant_lower"] = None

        return id_features

    def _extract_clothing_colors(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """
        Extract dominant colors from upper body (shirt) and lower body (pants).

        Returns:
            (upper_dominant_colors, lower_dominant_colors, upper_hist, lower_hist)
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Clamp coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None, None, None, None

        crop = frame[y1:y2, x1:x2]
        crop_h, crop_w = crop.shape[:2]

        if crop.size == 0 or crop_h < 20:
            return None, None, None, None

        # Split into upper (shirt) and lower (pants) regions
        # Upper: 20% to 50% of height (avoid head)
        # Lower: 50% to 80% of height (avoid feet)
        upper_start = int(crop_h * 0.2)
        upper_end = int(crop_h * 0.5)
        lower_start = int(crop_h * 0.5)
        lower_end = int(crop_h * 0.8)

        # Also crop horizontally to focus on body center (avoid arms at sides)
        center_start = int(crop_w * 0.25)
        center_end = int(crop_w * 0.75)

        upper_region = crop[upper_start:upper_end, center_start:center_end]
        lower_region = crop[lower_start:lower_end, center_start:center_end]

        upper_colors = None
        lower_colors = None
        upper_hist = None
        lower_hist = None

        # Extract upper body colors
        if (
            upper_region.size > 0
            and upper_region.shape[0] > 5
            and upper_region.shape[1] > 5
        ):
            upper_colors = self._extract_dominant_colors(upper_region)
            upper_hsv = cv2.cvtColor(upper_region, cv2.COLOR_BGR2HSV)
            upper_hist = cv2.calcHist(
                [upper_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256]
            )
            cv2.normalize(upper_hist, upper_hist)

        # Extract lower body colors
        if (
            lower_region.size > 0
            and lower_region.shape[0] > 5
            and lower_region.shape[1] > 5
        ):
            lower_colors = self._extract_dominant_colors(lower_region)
            lower_hsv = cv2.cvtColor(lower_region, cv2.COLOR_BGR2HSV)
            lower_hist = cv2.calcHist(
                [lower_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256]
            )
            cv2.normalize(lower_hist, lower_hist)

        return upper_colors, lower_colors, upper_hist, lower_hist

    def _extract_dominant_colors(self, region: np.ndarray) -> np.ndarray:
        """
        Extract dominant colors from a region using k-means clustering.
        Returns colors in HSV format for better color comparison.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Reshape to list of pixels
        pixels = hsv.reshape(-1, 3).astype(np.float32)

        # Remove very dark or very bright pixels (likely shadows or highlights)
        mask = (pixels[:, 2] > 30) & (pixels[:, 2] < 250)
        pixels = pixels[mask]

        if len(pixels) < self.n_dominant_colors * 10:
            # Not enough valid pixels
            return np.zeros((self.n_dominant_colors, 3), dtype=np.float32)

        # K-means clustering to find dominant colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, self.n_dominant_colors, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )

        # Sort by frequency (most common first)
        _, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(-counts)
        sorted_centers = centers[sorted_indices]

        return sorted_centers

    def _get_average_dominant_colors(
        self, color_samples: List[np.ndarray]
    ) -> np.ndarray:
        """Average dominant colors across multiple samples."""
        if not color_samples:
            return np.zeros((self.n_dominant_colors, 3), dtype=np.float32)

        # Stack all samples
        stacked = np.array(color_samples)
        # Average across samples for each dominant color
        return np.mean(stacked, axis=0).astype(np.float32)

    def _find_color_matching_ids(
        self, id_features: Dict[int, Dict]
    ) -> List[Tuple[int, int, float]]:
        """
        Compare all ID pairs using strict clothing color matching.
        Only pairs with very similar clothing colors will be merged.
        """
        matching_pairs = []
        ids = list(id_features.keys())

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                feat1 = id_features[id1]
                feat2 = id_features[id2]

                # Skip if either has too few samples
                if (
                    feat1["total_detections"] < self.min_samples
                    or feat2["total_detections"] < self.min_samples
                ):
                    continue

                # CRITICAL: Skip if they overlap in time - definitely different persons
                overlap = self._compute_temporal_overlap(feat1, feat2)
                if overlap > 0.1:  # Even 10% overlap means different persons
                    continue

                # Compare clothing colors strictly
                color_score = self._compare_clothing_colors(feat1, feat2)

                # Print debug info
                # print(f"    ID {id1} vs ID {id2}: color_score={color_score:.3f}")

                if color_score >= self.color_similarity_threshold:
                    matching_pairs.append((id1, id2, color_score))

        # Sort by score (highest first)
        matching_pairs.sort(key=lambda x: x[2], reverse=True)

        return matching_pairs

    def _compare_clothing_colors(self, feat1: Dict, feat2: Dict) -> float:
        """
        Compare clothing colors between two IDs.
        Requires BOTH upper and lower body colors to match.
        """
        scores = []

        # Compare upper body (shirt) histogram
        if feat1["avg_upper_hist"] is not None and feat2["avg_upper_hist"] is not None:
            upper_hist_sim = cv2.compareHist(
                feat1["avg_upper_hist"], feat2["avg_upper_hist"], cv2.HISTCMP_CORREL
            )
            upper_hist_sim = max(0, upper_hist_sim)

            # Also compare dominant colors
            if (
                feat1["dominant_upper"] is not None
                and feat2["dominant_upper"] is not None
            ):
                upper_color_sim = self._compare_dominant_colors(
                    feat1["dominant_upper"], feat2["dominant_upper"]
                )
                # Combine histogram and dominant color similarity
                upper_score = 0.5 * upper_hist_sim + 0.5 * upper_color_sim
            else:
                upper_score = upper_hist_sim

            scores.append(upper_score)

        # Compare lower body (pants) histogram
        if feat1["avg_lower_hist"] is not None and feat2["avg_lower_hist"] is not None:
            lower_hist_sim = cv2.compareHist(
                feat1["avg_lower_hist"], feat2["avg_lower_hist"], cv2.HISTCMP_CORREL
            )
            lower_hist_sim = max(0, lower_hist_sim)

            # Also compare dominant colors
            if (
                feat1["dominant_lower"] is not None
                and feat2["dominant_lower"] is not None
            ):
                lower_color_sim = self._compare_dominant_colors(
                    feat1["dominant_lower"], feat2["dominant_lower"]
                )
                lower_score = 0.5 * lower_hist_sim + 0.5 * lower_color_sim
            else:
                lower_score = lower_hist_sim

            scores.append(lower_score)

        if not scores:
            return 0.0

        # Return MINIMUM score - both upper and lower must match well
        # This prevents merging people with same shirt but different pants (or vice versa)
        return min(scores)

    def _compare_dominant_colors(
        self, colors1: np.ndarray, colors2: np.ndarray
    ) -> float:
        """
        Compare two sets of dominant colors.
        Uses HSV distance with special handling for Hue (circular).
        """
        if colors1 is None or colors2 is None:
            return 0.0

        total_similarity = 0.0
        n_colors = min(len(colors1), len(colors2), self.n_dominant_colors)

        for i in range(n_colors):
            c1 = colors1[i]
            c2 = colors2[i]

            # Hue distance (circular, max 90 for opposite colors)
            h_diff = min(abs(c1[0] - c2[0]), 180 - abs(c1[0] - c2[0]))
            h_sim = 1.0 - (h_diff / 90.0)

            # Saturation difference
            s_diff = abs(c1[1] - c2[1]) / 255.0
            s_sim = 1.0 - s_diff

            # Value difference
            v_diff = abs(c1[2] - c2[2]) / 255.0
            v_sim = 1.0 - v_diff

            # Weighted combination (Hue is most important for color matching)
            color_sim = 0.5 * h_sim + 0.3 * s_sim + 0.2 * v_sim
            total_similarity += color_sim

        return total_similarity / n_colors if n_colors > 0 else 0.0

    def _compute_temporal_overlap(self, feat1: Dict, feat2: Dict) -> float:
        """
        Compute how much two IDs overlap in time.
        Any overlap strongly suggests they are different persons.
        """
        frames1 = set(feat1["frame_ranges"])
        frames2 = set(feat2["frame_ranges"])

        overlap = len(frames1 & frames2)
        min_frames = min(len(frames1), len(frames2))

        return overlap / min_frames if min_frames > 0 else 0

    def _build_merge_groups(
        self, all_ids, pairs_to_merge: List[Tuple[int, int, float]]
    ) -> List[List[int]]:
        """Build groups of IDs that should be merged using Union-Find."""
        parent = {id_: id_ for id_ in all_ids}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                if px < py:
                    parent[py] = px
                else:
                    parent[px] = py

        for id1, id2, _ in pairs_to_merge:
            union(id1, id2)

        groups: Dict[int, List[int]] = {}
        for id_ in all_ids:
            root = find(id_)
            if root not in groups:
                groups[root] = []
            groups[root].append(id_)

        return [sorted(group) for group in groups.values() if len(group) > 1]

    def _create_id_mapping(self, merge_groups: List[List[int]]) -> Dict[int, int]:
        """Create mapping from old IDs to new IDs."""
        mapping = {}
        for group in merge_groups:
            target_id = min(group)
            for id_ in group:
                if id_ != target_id:
                    mapping[id_] = target_id
        return mapping

    def _apply_id_mapping(
        self, person_results: List[List[Dict]], id_mapping: Dict[int, int]
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
        self, positions: List[Optional[Tuple[float, float]]]
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
                    interp_positions = self._linear_interpolate(
                        prev_pos, next_pos, gap_size
                    )
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
        self, prev: Tuple[float, float], next_pos: Tuple[float, float], gap_size: int
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
        gap_end: int,
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
        self, context: List[Tuple[int, float, float]], gap_start: int, gap_size: int
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
# POSE ANALYZER CLASS
# =============================================================================


class PoseAnalyzer:
    """
    Analyzes player pose using YOLOv8 Pose estimation.
    Calculates shoulder and knee angles from keypoints.
    """

    # COCO Keypoint indices
    KEYPOINTS = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }

    def __init__(self, model_path: str = "yolov8s-pose.pt"):
        """Initialize pose analyzer with YOLOv8 pose model"""
        self.model = YOLO(model_path)
        print(f"Loaded pose model: {model_path}")

    def analyze(self, person_crop: np.ndarray) -> Dict:
        """
        Analyze pose from a cropped person image.

        Args:
            person_crop: BGR image of cropped person

        Returns:
            Dict with angle measurements in degrees
        """
        if person_crop is None or person_crop.size == 0:
            return self._empty_angles()

        try:
            results = self.model(person_crop, verbose=False)

            if not results or len(results) == 0:
                return self._empty_angles()

            if results[0].keypoints is None or len(results[0].keypoints) == 0:
                return self._empty_angles()

            keypoints = results[0].keypoints.xy[0].cpu().numpy()

            angles = {
                "left_shoulder_angle": self._calc_shoulder_angle(keypoints, "left"),
                "right_shoulder_angle": self._calc_shoulder_angle(keypoints, "right"),
                "left_knee_angle": self._calc_knee_angle(keypoints, "left"),
                "right_knee_angle": self._calc_knee_angle(keypoints, "right"),
            }
            return angles

        except Exception as e:
            print(f"Pose analysis error: {e}")
            return self._empty_angles()

    def _empty_angles(self) -> Dict:
        """Return empty angles dict"""
        return {
            "left_shoulder_angle": None,
            "right_shoulder_angle": None,
            "left_knee_angle": None,
            "right_knee_angle": None,
        }

    def _calc_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate angle at p2 between three points (in degrees).

        Args:
            p1, p2, p3: Points as numpy arrays [x, y]

        Returns:
            Angle in degrees
        """
        v1 = p1 - p2
        v2 = p3 - p2

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            return None

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        return float(np.degrees(np.arccos(cos_angle)))

    def _calc_shoulder_angle(self, keypoints: np.ndarray, side: str) -> Optional[float]:
        """
        Calculate shoulder angle: elbow - shoulder - hip

        Args:
            keypoints: Array of 17 keypoints [x, y]
            side: 'left' or 'right'

        Returns:
            Angle in degrees or None if keypoints not detected
        """
        if side == "left":
            elbow_idx, shoulder_idx, hip_idx = 7, 5, 11
        else:
            elbow_idx, shoulder_idx, hip_idx = 8, 6, 12

        elbow = keypoints[elbow_idx]
        shoulder = keypoints[shoulder_idx]
        hip = keypoints[hip_idx]

        # Check if keypoints are valid (not zero)
        if np.allclose(elbow, 0) or np.allclose(shoulder, 0) or np.allclose(hip, 0):
            return None

        return self._calc_angle(elbow, shoulder, hip)

    def _calc_knee_angle(self, keypoints: np.ndarray, side: str) -> Optional[float]:
        """
        Calculate knee angle: hip - knee - ankle

        Args:
            keypoints: Array of 17 keypoints [x, y]
            side: 'left' or 'right'

        Returns:
            Angle in degrees or None if keypoints not detected
        """
        if side == "left":
            hip_idx, knee_idx, ankle_idx = 11, 13, 15
        else:
            hip_idx, knee_idx, ankle_idx = 12, 14, 16

        hip = keypoints[hip_idx]
        knee = keypoints[knee_idx]
        ankle = keypoints[ankle_idx]

        # Check if keypoints are valid (not zero)
        if np.allclose(hip, 0) or np.allclose(knee, 0) or np.allclose(ankle, 0):
            return None

        return self._calc_angle(hip, knee, ankle)


# =============================================================================
# SHOT DETECTOR CLASS
# =============================================================================


class ShotDetector:
    """
    Detects shot events when:
    1. Ball position is near a person AND
    2. Ball changes direction (velocity vector changes significantly)

    Analyzes player pose at each shot moment.
    """

    def __init__(
        self,
        pose_analyzer: PoseAnalyzer,
        court_aligner: CourtAligner = None,
        proximity_threshold: float = 50.0,
        min_frames_between_shots: int = 15,
        direction_change_threshold: float = 45.0,
        velocity_window: int = 3,
    ):
        """
        Initialize shot detector.

        Args:
            pose_analyzer: PoseAnalyzer instance for pose estimation
            court_aligner: CourtAligner instance for court position analysis
            proximity_threshold: Max distance (pixels) for ball to be considered "touching" person
            min_frames_between_shots: Cooldown frames between shots for same person
            direction_change_threshold: Minimum angle change (degrees) to consider as direction change
            velocity_window: Number of frames to use for velocity calculation
        """
        self.pose_analyzer = pose_analyzer
        self.court_aligner = court_aligner
        self.proximity_threshold = proximity_threshold
        self.min_frames_between_shots = min_frames_between_shots
        self.direction_change_threshold = direction_change_threshold
        self.velocity_window = velocity_window
        self.last_shot_frame = {}  # {person_id: frame_index}

    def detect_shots(
        self, frames: List[np.ndarray], frame_results: List[FrameResult], fps: float
    ) -> List[ShotEvent]:
        """
        Detect all shot events in the video.

        Args:
            frames: List of video frames (BGR images)
            frame_results: List of FrameResult with ball and person detections
            fps: Video frame rate

        Returns:
            List of ShotEvent objects
        """
        shots = []

        # Extract ball positions for velocity calculation
        ball_positions = self._extract_ball_positions(frame_results)

        # Detect direction changes
        direction_changes = self._detect_direction_changes(ball_positions)

        for frame_idx in direction_changes:
            frame_result = frame_results[frame_idx]

            # Skip if no persons detected at this frame
            if not frame_result.persons:
                continue

            # Get ball position at direction change
            ball = frame_result.balls[0] if frame_result.balls else None
            if ball is None:
                continue

            ball_pos = np.array([ball.x, ball.y])

            # Find the person closest to the ball (within threshold)
            closest_person = None
            min_distance = float("inf")

            for person in frame_result.persons:
                if self._is_ball_touching_person(ball_pos, person.bbox):
                    # Calculate distance to person center
                    person_center = np.array(
                        [
                            (person.bbox[0] + person.bbox[2]) / 2,
                            (person.bbox[1] + person.bbox[3]) / 2,
                        ]
                    )
                    distance = np.linalg.norm(ball_pos - person_center)

                    if distance < min_distance:
                        min_distance = distance
                        closest_person = person

            # Register shot if person found and cooldown passed
            if closest_person is not None:
                if self._can_register_shot(closest_person.id, frame_idx):
                    shot = self._create_shot_event(
                        frame_idx, fps, closest_person, ball, frames[frame_idx]
                    )
                    shots.append(shot)
                    self.last_shot_frame[closest_person.id] = frame_idx

        return shots

    def _extract_ball_positions(
        self, frame_results: List[FrameResult]
    ) -> List[Optional[Tuple[float, float]]]:
        """
        Extract ball positions from all frames.

        Returns:
            List of (x, y) tuples or None for frames without ball
        """
        positions = []
        for fr in frame_results:
            if fr.balls:
                positions.append((fr.balls[0].x, fr.balls[0].y))
            else:
                positions.append(None)
        return positions

    def _detect_direction_changes(
        self, positions: List[Optional[Tuple[float, float]]]
    ) -> List[int]:
        """
        Detect frames where ball changes direction.

        A direction change is detected when the angle between
        incoming velocity vector and outgoing velocity vector
        exceeds the threshold.

        Args:
            positions: List of ball positions (x, y) or None

        Returns:
            List of frame indices where direction changes occur
        """
        direction_change_frames = []
        n = len(positions)
        w = self.velocity_window

        for i in range(w, n - w):
            # Get positions before and after current frame
            pos_before = self._get_valid_positions(positions, i - w, i)
            pos_after = self._get_valid_positions(positions, i, i + w)

            if len(pos_before) < 2 or len(pos_after) < 2:
                continue

            # Calculate incoming velocity (average direction before)
            vel_in = self._calculate_velocity(pos_before)
            # Calculate outgoing velocity (average direction after)
            vel_out = self._calculate_velocity(pos_after)

            if vel_in is None or vel_out is None:
                continue

            # Calculate angle between velocities
            angle_change = self._angle_between_vectors(vel_in, vel_out)

            # Check if direction change exceeds threshold
            if angle_change >= self.direction_change_threshold:
                direction_change_frames.append(i)

        return direction_change_frames

    def _get_valid_positions(
        self, positions: List[Optional[Tuple[float, float]]], start: int, end: int
    ) -> List[Tuple[float, float]]:
        """Get non-None positions in range [start, end)"""
        valid = []
        for i in range(start, end):
            if 0 <= i < len(positions) and positions[i] is not None:
                valid.append(positions[i])
        return valid

    def _calculate_velocity(
        self, positions: List[Tuple[float, float]]
    ) -> Optional[np.ndarray]:
        """
        Calculate average velocity vector from a list of positions.

        Returns:
            Velocity vector [vx, vy] or None if can't calculate
        """
        if len(positions) < 2:
            return None

        # Use first and last position for overall direction
        p_start = np.array(positions[0])
        p_end = np.array(positions[-1])

        velocity = p_end - p_start
        norm = np.linalg.norm(velocity)

        if norm < 1e-6:
            return None

        return velocity / norm  # Normalize

    def _angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calculate angle between two vectors in degrees.

        Args:
            v1, v2: Normalized velocity vectors

        Returns:
            Angle in degrees (0-180)
        """
        # Dot product of normalized vectors gives cos(angle)
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.degrees(np.arccos(dot))
        return angle

    def _is_ball_touching_person(
        self, ball_pos: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> bool:
        """
        Check if ball position is within or near person bounding box.

        Args:
            ball_pos: Ball position [x, y]
            bbox: Person bounding box (x1, y1, x2, y2)

        Returns:
            True if ball is touching/near person
        """
        x1, y1, x2, y2 = bbox

        # Expand bbox by proximity threshold
        x1_exp = x1 - self.proximity_threshold
        y1_exp = y1 - self.proximity_threshold
        x2_exp = x2 + self.proximity_threshold
        y2_exp = y2 + self.proximity_threshold

        return x1_exp <= ball_pos[0] <= x2_exp and y1_exp <= ball_pos[1] <= y2_exp

    def _can_register_shot(self, person_id: int, frame_idx: int) -> bool:
        """
        Check if enough frames have passed since last shot for this person.

        Args:
            person_id: ID of the person
            frame_idx: Current frame index

        Returns:
            True if shot can be registered
        """
        if person_id not in self.last_shot_frame:
            return True
        return (
            frame_idx - self.last_shot_frame[person_id] >= self.min_frames_between_shots
        )

    def _create_shot_event(
        self,
        frame_idx: int,
        fps: float,
        person: PersonDetection,
        ball: BallDetection,
        frame: np.ndarray,
    ) -> ShotEvent:
        """
        Create a ShotEvent with pose analysis.

        Args:
            frame_idx: Frame index
            fps: Video FPS
            person: PersonDetection object
            ball: BallDetection object
            frame: Video frame (BGR image)

        Returns:
            ShotEvent with pose angles
        """
        x1, y1, x2, y2 = person.bbox

        # Ensure bbox is within frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))

        # Crop person from frame
        person_crop = frame[y1:y2, x1:x2]

        # Analyze pose
        angles = self.pose_analyzer.analyze(person_crop)

        # Get court position info
        ball_in_court = True
        ball_zone = "unknown"
        ball_side = "unknown"
        if self.court_aligner:
            court_pos = self.court_aligner.align_position(ball.x, ball.y)
            ball_in_court = court_pos.is_in_court
            ball_zone = court_pos.court_zone
            ball_side = court_pos.side

        return ShotEvent(
            frame_index=frame_idx,
            timestamp_seconds=frame_idx / fps,
            person_id=person.id,
            ball_position=(ball.x, ball.y),
            person_bbox=person.bbox,
            left_shoulder_angle=angles.get("left_shoulder_angle"),
            right_shoulder_angle=angles.get("right_shoulder_angle"),
            left_knee_angle=angles.get("left_knee_angle"),
            right_knee_angle=angles.get("right_knee_angle"),
            ball_in_court=ball_in_court,
            ball_zone=ball_zone,
            ball_side=ball_side,
        )


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
        person_model_path: str = "yolov8s.pt",
        conf_threshold: float = 0.15,  # Lower base threshold
        person_conf_threshold: float = 0.5,
        batch_size: int = 32,
        enable_validation: bool = True,
        enable_enhancement: bool = True,
        enable_kalman: bool = True,
        enable_person_detection: bool = True,
        use_person_tiles: bool = True,
        enable_person_id_merge: bool = True,
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
                use_tiles=use_person_tiles,
            )
            self.person_tracker = PersonTracker()
            if enable_person_id_merge:
                self.person_id_merger = PersonIDMerger()

    def process_video(
        self, video_path: str, output_json_path: str, output_video_path: str,
        court_data: Dict = None
    ) -> Dict:
        """
        Main processing pipeline (Enhanced)

        Args:
            video_path: Path to input video
            output_json_path: Path to save JSON results
            output_video_path: Path to save annotated video
            court_data: Optional dict with court coordinates for alignment.
                        If None, court alignment is disabled.

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
        validated_results = self._validate_and_filter(
            frames, interpolated, raw_detections
        )

        # 8. Detect and track persons
        person_results = []
        id_mapping = {}
        if (
            self.enable_person_detection
            and self.person_detector
            and self.person_tracker
        ):
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

        # 9.5. Shot Detection & Pose Analysis with Court Alignment
        shots = []
        court_aligner = None
        if self.enable_person_detection and frame_results:
            print("Detecting shots and analyzing poses...")
            # Initialize court aligner if court_data provided
            if court_data:
                court_aligner = CourtAligner(court_data)
            pose_analyzer = PoseAnalyzer()
            shot_detector = ShotDetector(pose_analyzer, court_aligner)
            shots = shot_detector.detect_shots(frames, frame_results, video_info.fps)
            print(f"  Detected {len(shots)} shots")

        # 10. Calculate statistics
        stats = self._calculate_statistics(
            frame_results, raw_detections, id_mapping, shots
        )

        # 11. Save JSON
        print(f"Saving JSON to: {output_json_path}")
        self._save_json(
            video_info, frame_results, stats, output_json_path, shots, court_aligner
        )

        # 12. Create annotated video
        print(f"Creating annotated video: {output_video_path}")
        self._create_annotated_video(
            frames, frame_results, video_info.fps, output_video_path, shots
        )

        print("Processing complete!")

        return {
            "video_info": asdict(video_info),
            "statistics": stats,
            "output_json": output_json_path,
            "output_video": output_video_path,
        }

    def _detect_and_track_persons(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """
        Detect and track persons in all frames using batch detection (OPTIMIZED)

        Returns:
            List of tracked person lists per frame
        """
        # STEP 1: Batch detect all persons (MUCH FASTER than frame-by-frame)
        print("  Batch detecting persons...")
        all_detections = self.person_detector.detect_batch(frames, batch_size=self.batch_size)

        # STEP 2: Track persons frame by frame (requires sequential processing)
        print("  Tracking persons...")
        all_person_results = []

        for i, (frame, detections) in enumerate(zip(frames, all_detections)):
            # Track persons (assign IDs)
            tracked = self.person_tracker.update(frame, detections)
            all_person_results.append(tracked)

            if (i + 1) % 500 == 0 or i == len(frames) - 1:
                print(f"    Tracked {i + 1}/{len(frames)} frames")

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
            resolution={"width": width, "height": height},
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
                print(
                    f"  Pass 2: Enhanced detection for {len(missing_indices)} missing frames..."
                )

                # Process in batches
                for batch_start in range(0, len(missing_indices), self.batch_size):
                    batch_indices = missing_indices[
                        batch_start : batch_start + self.batch_size
                    ]
                    enhanced_batch = [
                        self.enhancer.enhance(frames[i]) for i in batch_indices
                    ]

                    results = self.model.predict(
                        enhanced_batch,
                        verbose=False,
                        conf=self.conf_threshold
                        * 0.8,  # Slightly lower threshold for enhanced
                    )

                    for j, res in enumerate(results):
                        frame_idx = batch_indices[j]
                        if res.boxes is not None and len(res.boxes) > 0:
                            # Batch transfer all tensors to CPU at once (OPTIMIZED)
                            boxes_xywh = res.boxes.xywh.cpu().numpy()
                            boxes_conf = res.boxes.conf.cpu().numpy()
                            boxes_xyxy = res.boxes.xyxy.cpu().numpy().astype(int)

                            for k in range(len(res.boxes)):
                                x, y, w, h = boxes_xywh[k]
                                conf = float(boxes_conf[k])
                                x1, y1, x2, y2 = boxes_xyxy[k]

                                all_detections[frame_idx].append(
                                    {
                                        "x": float(x),
                                        "y": float(y),
                                        "w": float(w),
                                        "h": float(h),
                                        "conf": conf,
                                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                                        "enhanced": True,
                                    }
                                )

        # Remove duplicates (same position from different passes)
        for i in range(len(all_detections)):
            all_detections[i] = self._remove_duplicate_detections(all_detections[i])

        return all_detections

    def _remove_duplicate_detections(
        self, detections: List[Dict], iou_threshold: float = 0.5
    ) -> List[Dict]:
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

    def _calculate_iou(
        self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]
    ) -> float:
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

    def _detect_balls_batch(
        self, frames: List[np.ndarray], conf: float
    ) -> List[List[Dict]]:
        """Detect balls in frames using batch inference"""
        all_detections = []

        for i in range(0, len(frames), self.batch_size):
            batch = frames[i : i + self.batch_size]

            results = self.model.predict(batch, verbose=False, conf=conf)

            for res in results:
                frame_detections = []
                if res.boxes is not None and len(res.boxes) > 0:
                    # Batch transfer all tensors to CPU at once (OPTIMIZED)
                    boxes_xywh = res.boxes.xywh.cpu().numpy()
                    boxes_conf = res.boxes.conf.cpu().numpy()
                    boxes_xyxy = res.boxes.xyxy.cpu().numpy().astype(int)

                    for j in range(len(res.boxes)):
                        x, y, w, h = boxes_xywh[j]
                        conf_score = float(boxes_conf[j])
                        x1, y1, x2, y2 = boxes_xyxy[j]

                        frame_detections.append(
                            {
                                "x": float(x),
                                "y": float(y),
                                "w": float(w),
                                "h": float(h),
                                "conf": conf_score,
                                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                            }
                        )

                all_detections.append(frame_detections)

            # Progress
            progress = min(i + self.batch_size, len(frames))
            if progress % 500 == 0 or progress == len(frames):
                print(f"    Processed {progress}/{len(frames)} frames")

        return all_detections

    def _extract_positions_with_kalman(
        self, detections: List[List[Dict]]
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
                best_score = float("inf")

                for det in frame_dets:
                    dist = math.sqrt(
                        (det["x"] - pred_x) ** 2 + (det["y"] - pred_y) ** 2
                    )
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
        detections: List[List[Dict]],
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
                    dist = math.sqrt(
                        (det["x"] - pred_x) ** 2 + (det["y"] - pred_y) ** 2
                    )
                    if dist < search_radius:
                        recovered[i] = (det["x"], det["y"])
                        kalman.update(det["x"], det["y"])
                        recovery_count += 1
                        break

        if recovery_count > 0:
            print(
                f"  Recovered {recovery_count} missing detections using trajectory prediction"
            )

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
            batch = frames[i : i + self.batch_size]

            results = self.model.predict(batch, verbose=False, conf=self.conf_threshold)

            for res in results:
                frame_detections = []
                if res.boxes is not None and len(res.boxes) > 0:
                    # Batch transfer all tensors to CPU at once (OPTIMIZED)
                    boxes_xywh = res.boxes.xywh.cpu().numpy()
                    boxes_conf = res.boxes.conf.cpu().numpy()
                    boxes_xyxy = res.boxes.xyxy.cpu().numpy().astype(int)

                    for j in range(len(res.boxes)):
                        x, y, w, h = boxes_xywh[j]
                        conf = float(boxes_conf[j])
                        x1, y1, x2, y2 = boxes_xyxy[j]

                        frame_detections.append(
                            {
                                "x": float(x),
                                "y": float(y),
                                "w": float(w),
                                "h": float(h),
                                "conf": conf,
                                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                            }
                        )

                all_detections.append(frame_detections)

            # Progress
            progress = min(i + self.batch_size, len(frames))
            if progress % 500 == 0 or progress == len(frames):
                print(f"  Processed {progress}/{len(frames)} frames")

        return all_detections

    def _extract_primary_positions(
        self, detections: List[List[Dict]]
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
        raw_detections: List[List[Dict]],
    ) -> List[Dict]:
        """
        Validate interpolated positions and filter false positives
        """
        results = []

        for i, ((pos, method), frame_dets) in enumerate(
            zip(interpolated, raw_detections)
        ):
            result = {
                "frame_index": i,
                "position": pos,
                "method": method,
                "validated": True,
                "all_detections": frame_dets,
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
                        is_valid, reason = self.validator.validate(
                            frames[i], det["bbox"]
                        )
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
        person_results: List[List[Dict]] = None,
    ) -> List[FrameResult]:
        """Build final frame results with ball and person detections"""
        frame_results = []
        ball_id = 1  # Primary ball tracking

        for i, res in enumerate(validated_results):
            frame = FrameResult(
                frame_index=res["frame_index"],
                timestamp_seconds=res["frame_index"] / fps,
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
                    source=(
                        DetectionSource.DETECTION
                        if res["method"] == "detection"
                        else DetectionSource.INTERPOLATED
                    ),
                    validated=True,
                    bbox=bbox,
                    interpolation_method=(
                        res["method"] if res["method"] != "detection" else None
                    ),
                )
                frame.balls.append(ball)

            # Also add all other detections (for multi-ball requirement)
            secondary_id = 2
            for det in res.get("all_detections", []):
                # Skip if already added as primary
                if (
                    res["position"]
                    and abs(det["x"] - res["position"][0]) < 1
                    and abs(det["y"] - res["position"][1]) < 1
                ):
                    continue

                ball = BallDetection(
                    id=secondary_id,
                    x=det["x"],
                    y=det["y"],
                    confidence=det["conf"],
                    source=DetectionSource.DETECTION,
                    validated=True,
                    bbox=det["bbox"],
                )
                frame.balls.append(ball)
                secondary_id += 1

            # Add person detections
            if person_results and i < len(person_results):
                for person in person_results[i]:
                    frame.persons.append(
                        PersonDetection(
                            id=person["id"],
                            x=person["x"],
                            y=person["y"],
                            bbox=person["bbox"],
                            confidence=person["conf"],
                            tracked=person.get("tracked", True),
                        )
                    )

            frame_results.append(frame)

        return frame_results

    def _calculate_statistics(
        self,
        frame_results: List[FrameResult],
        raw_detections: List[List[Dict]],
        id_mapping: Dict[int, int] = None,
        shots: List[ShotEvent] = None,
    ) -> Dict:
        """Calculate processing statistics for balls, persons, and shots"""
        # Ball statistics
        total_detections = sum(
            1
            for res in frame_results
            if any(b.source == DetectionSource.DETECTION for b in res.balls)
        )
        interpolated_frames = sum(
            1
            for res in frame_results
            if any(b.source == DetectionSource.INTERPOLATED for b in res.balls)
        )

        total_raw_detections = sum(len(dets) for dets in raw_detections)
        total_kept = sum(len(res.balls) for res in frame_results)
        rejected = max(0, total_raw_detections - total_kept)

        total_frames = len(frame_results)
        frames_with_ball = sum(1 for res in frame_results if res.balls)

        all_confs = [
            b.confidence for res in frame_results for b in res.balls if b.confidence
        ]

        # Person statistics
        frames_with_persons = sum(1 for res in frame_results if res.persons)
        unique_person_ids = set()
        for res in frame_results:
            for p in res.persons:
                unique_person_ids.add(p.id)

        # ID merge statistics
        ids_merged = len(id_mapping) if id_mapping else 0
        original_ids = len(unique_person_ids) + ids_merged

        # Shot statistics
        shots = shots or []
        shots_by_person = {}
        for shot in shots:
            if shot.person_id not in shots_by_person:
                shots_by_person[shot.person_id] = 0
            shots_by_person[shot.person_id] += 1

        return {
            "ball": {
                "total_frames": total_frames,
                "frames_with_detection": total_detections,
                "interpolated_frames": interpolated_frames,
                "rejected_false_positives": rejected,
                "detection_rate": (
                    round(frames_with_ball / total_frames, 4) if total_frames > 0 else 0
                ),
                "average_confidence": (
                    round(sum(all_confs) / len(all_confs), 4) if all_confs else 0
                ),
            },
            "person": {
                "frames_with_persons": frames_with_persons,
                "unique_persons_detected": len(unique_person_ids),
                "person_ids": sorted(list(unique_person_ids)),
                "ids_before_merge": original_ids,
                "ids_merged": ids_merged,
                "id_mapping": id_mapping if id_mapping else {},
            },
            "shots": {"total_shots": len(shots), "shots_by_person": shots_by_person},
        }

    def _save_json(
        self,
        video_info: VideoInfo,
        frame_results: List[FrameResult],
        stats: Dict,
        output_path: str,
        shots: List[ShotEvent] = None,
        court_aligner: CourtAligner = None,
    ):
        """Save results to JSON file including shot events and court info"""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Get court info if available
        court_info = court_aligner.get_court_info() if court_aligner else None

        # Convert to serializable format
        output = {
            "video_info": asdict(video_info),
            "court_info": court_info,
            "detection_config": {
                "ball": {
                    "model_path": "models/ball_best.pt",
                    "confidence_threshold": self.conf_threshold,
                    "max_interpolation_gap": BallInterpolator.MAX_GAP,
                    "validation_enabled": self.enable_validation,
                },
                "person": {
                    "model_path": "yolov8s.pt",
                    "enabled": self.enable_person_detection,
                    "tile_based_detection": True,
                },
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
                            "confidence": (
                                round(b.confidence, 4) if b.confidence else None
                            ),
                            "source": b.source.value,
                            "validated": b.validated,
                            "bbox": (
                                {
                                    "x1": b.bbox[0],
                                    "y1": b.bbox[1],
                                    "x2": b.bbox[2],
                                    "y2": b.bbox[3],
                                }
                                if b.bbox
                                else None
                            ),
                            "interpolation_method": b.interpolation_method,
                        }
                        for b in fr.balls
                    ],
                    "persons": [
                        {
                            "id": p.id,
                            "x": round(p.x, 2),
                            "y": round(p.y, 2),
                            "confidence": round(p.confidence, 4),
                            "bbox": {
                                "x1": p.bbox[0],
                                "y1": p.bbox[1],
                                "x2": p.bbox[2],
                                "y2": p.bbox[3],
                            },
                            "tracked": p.tracked,
                        }
                        for p in fr.persons
                    ],
                }
                for fr in frame_results
            ],
            "shots": [
                {
                    "frame_index": s.frame_index,
                    "timestamp_seconds": round(s.timestamp_seconds, 4),
                    "person_id": s.person_id,
                    "ball_position": {
                        "x": round(s.ball_position[0], 2),
                        "y": round(s.ball_position[1], 2),
                    },
                    "ball_in_court": s.ball_in_court,
                    "ball_zone": s.ball_zone,
                    "ball_side": s.ball_side,
                    "person_bbox": {
                        "x1": s.person_bbox[0],
                        "y1": s.person_bbox[1],
                        "x2": s.person_bbox[2],
                        "y2": s.person_bbox[3],
                    },
                    "pose_angles": {
                        "left_shoulder": (
                            round(s.left_shoulder_angle, 2)
                            if s.left_shoulder_angle
                            else None
                        ),
                        "right_shoulder": (
                            round(s.right_shoulder_angle, 2)
                            if s.right_shoulder_angle
                            else None
                        ),
                        "left_knee": (
                            round(s.left_knee_angle, 2) if s.left_knee_angle else None
                        ),
                        "right_knee": (
                            round(s.right_knee_angle, 2) if s.right_knee_angle else None
                        ),
                    },
                }
                for s in (shots or [])
            ],
            "statistics": stats,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    def _create_annotated_video(
        self,
        frames: List[np.ndarray],
        frame_results: List[FrameResult],
        fps: float,
        output_path: str,
        shots: List[ShotEvent] = None,
    ):
        """Create annotated video with ball, person detections and shot events"""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Build shot lookup by frame index for quick access
        shots = shots or []
        shot_by_frame = {}
        for shot in shots:
            if shot.frame_index not in shot_by_frame:
                shot_by_frame[shot.frame_index] = []
            shot_by_frame[shot.frame_index].append(shot)

        # Track recent shots for display duration (show for N frames after shot)
        shot_display_duration = int(fps * 1.5)  # 1.5 seconds
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Colors for different person IDs
        person_colors = [
            (255, 0, 0),  # Blue
            (0, 0, 255),  # Red
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
                    label_size, _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        annotated,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0] + 10, y1),
                        color,
                        -1,
                    )
                    cv2.putText(
                        annotated,
                        label,
                        (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

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
                    src_label = (
                        "DET" if ball.source == DetectionSource.DETECTION else "INT"
                    )
                    label = f"Ball [{src_label}]"
                    if ball.confidence:
                        label += f" {ball.confidence:.2f}"
                    cv2.putText(
                        annotated,
                        label,
                        (x + 15, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )

                # Frame info
                cv2.putText(
                    annotated,
                    f"Frame: {result.frame_index}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    annotated,
                    f"Frame: {result.frame_index}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    1,
                )

                # Person count info
                if result.persons:
                    person_info = f"Persons: {len(result.persons)}"
                    cv2.putText(
                        annotated,
                        person_info,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        annotated,
                        person_info,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        1,
                    )

                # Draw shot events (check recent shots within display duration)
                active_shots = []
                for frame_idx in range(max(0, i - shot_display_duration), i + 1):
                    if frame_idx in shot_by_frame:
                        for shot in shot_by_frame[frame_idx]:
                            active_shots.append((shot, i - frame_idx))

                for shot, frames_ago in active_shots:
                    # Color based on IN/OUT status
                    if shot.ball_in_court:
                        shot_color = (0, 255, 0)  # Green for IN
                        status_text = "IN"
                    else:
                        shot_color = (0, 0, 255)  # Red for OUT
                        status_text = "OUT"

                    x1, y1, x2, y2 = shot.person_bbox

                    # Draw highlight box around person who made the shot
                    thickness = 4 if frames_ago < 5 else 2
                    cv2.rectangle(
                        annotated,
                        (int(x1) - 5, int(y1) - 5),
                        (int(x2) + 5, int(y2) + 5),
                        shot_color,
                        thickness,
                    )

                    # Draw "SHOT!" text and IN/OUT status above the person
                    if frames_ago < shot_display_duration // 2:
                        # Main shot label with status
                        shot_label = f"SHOT! [{status_text}]"
                        label_size, _ = cv2.getTextSize(
                            shot_label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3
                        )
                        label_x = int((x1 + x2) / 2 - label_size[0] / 2)
                        label_y = int(y1) - 20

                        # Background for shot label
                        cv2.rectangle(
                            annotated,
                            (label_x - 5, label_y - label_size[1] - 5),
                            (label_x + label_size[0] + 5, label_y + 5),
                            shot_color,
                            -1,
                        )
                        cv2.putText(
                            annotated,
                            shot_label,
                            (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 0),
                            3,
                        )

                        # Zone info below shot label
                        if shot.ball_zone != "unknown":
                            zone_label = f"{shot.ball_zone} ({shot.ball_side})"
                            zone_size, _ = cv2.getTextSize(
                                zone_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                            )
                            zone_x = int((x1 + x2) / 2 - zone_size[0] / 2)
                            zone_y = label_y + 25
                            cv2.putText(
                                annotated,
                                zone_label,
                                (zone_x, zone_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                shot_color,
                                2,
                            )

                    # Draw pose angles if available (show for longer)
                    if frames_ago < shot_display_duration:
                        angle_y = int(y2) + 25
                        angle_x = int(x1)

                        # Prepare angle info
                        angle_lines = []
                        if shot.left_shoulder_angle is not None:
                            angle_lines.append(
                                f"L.Shoulder: {shot.left_shoulder_angle:.1f}"
                            )
                        if shot.right_shoulder_angle is not None:
                            angle_lines.append(
                                f"R.Shoulder: {shot.right_shoulder_angle:.1f}"
                            )
                        if shot.left_knee_angle is not None:
                            angle_lines.append(f"L.Knee: {shot.left_knee_angle:.1f}")
                        if shot.right_knee_angle is not None:
                            angle_lines.append(f"R.Knee: {shot.right_knee_angle:.1f}")

                        # Draw angle info box
                        if angle_lines:
                            box_width = 180
                            box_height = len(angle_lines) * 22 + 10
                            cv2.rectangle(
                                annotated,
                                (angle_x, angle_y),
                                (angle_x + box_width, angle_y + box_height),
                                (0, 0, 0),
                                -1,
                            )
                            cv2.rectangle(
                                annotated,
                                (angle_x, angle_y),
                                (angle_x + box_width, angle_y + box_height),
                                shot_color,
                                2,
                            )

                            for idx, line in enumerate(angle_lines):
                                cv2.putText(
                                    annotated,
                                    line,
                                    (angle_x + 5, angle_y + 20 + idx * 22),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    1,
                                )

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
    PERSON_MODEL_PATH = "yolov8s.pt"  # Will be auto-downloaded if not exists
    ID_SAN = "court_001"  # Court/field ID

    # Output paths - output/{id_san}/{video_name}/
    video_basename = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    output_dir = f"output/{ID_SAN}/{video_basename}"
    os.makedirs(output_dir, exist_ok=True)
    OUTPUT_JSON = f"{output_dir}/{video_basename}.json"
    OUTPUT_VIDEO = f"{output_dir}/{video_basename}_annotated.mp4"

    print("=" * 60)
    print("TENNIS BALL & PERSON DETECTION SYSTEM (ENHANCED)")
    print("=" * 60)
    print(f"Input video: {VIDEO_PATH}")
    print(f"Court ID: {ID_SAN}")
    print(f"Ball model: {MODEL_PATH}")
    print(f"Person model: {PERSON_MODEL_PATH}")
    print(f"Output directory: {output_dir}")
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
        conf_threshold=0.15,  # Lower threshold for better recall
        person_conf_threshold=0.5,  # Person detection threshold
        batch_size=32,
        enable_validation=True,
        enable_enhancement=True,  # CLAHE + sharpening for small balls
        enable_kalman=True,  # Kalman filter for trajectory
        enable_person_detection=True,  # Enable person detection
        use_person_tiles=False,  # Full frame detection (not tile-based)
        enable_person_id_merge=True,  # Merge fragmented person IDs
    )

    # Court coordinates for alignment (8 points defining the court)
    court_data = {
        "top_left": {"x": 171, "y": 325},
        "top_center": {"x": 271, "y": 221},
        "top_right": {"x": 363, "y": 139},
        "center_right": {"x": 559, "y": 128},
        "bottom_right": {"x": 981, "y": 154},
        "bottom_center": {"x": 887, "y": 343},
        "bottom_left": {"x": 715, "y": 643},
        "center_left": {"x": 294, "y": 406},
        "net_y": 406  # center_left.y = net line
    }

    # =========================================================================
    # STEP 1: Ball & Person Tracking
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: BALL & PERSON TRACKING")
    print("=" * 60)

    step1_start = time.time()

    # Process video
    result = tracker.process_video(
        video_path=VIDEO_PATH,
        output_json_path=OUTPUT_JSON,
        output_video_path=OUTPUT_VIDEO,
        court_data=court_data,
    )

    step1_time = time.time() - step1_start

    print("\n" + "-" * 60)
    print("TRACKING COMPLETE")
    print("-" * 60)
    print(f"Time elapsed: {format_time(step1_time)}")
    print(f"JSON output: {result['output_json']}")
    print(f"Video output: {result['output_video']}")

    print("\nBall Statistics:")
    for key, value in result["statistics"]["ball"].items():
        print(f"  {key}: {value}")

    print("\nPerson Statistics:")
    person_stats = result["statistics"]["person"]
    print(f"  frames_with_persons: {person_stats['frames_with_persons']}")
    print(f"  unique_persons_detected: {person_stats['unique_persons_detected']}")
    print(f"  person_ids: {person_stats['person_ids']}")
    print(f"  ids_before_merge: {person_stats['ids_before_merge']}")
    print(f"  ids_merged: {person_stats['ids_merged']}")
    if person_stats["id_mapping"]:
        print(f"  id_mapping: {person_stats['id_mapping']}")

    # =========================================================================
    # STEP 2: Match Analysis
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: MATCH ANALYSIS")
    print("=" * 60)

    step2_start = time.time()

    # Analysis output directory: analysis/{id_san}/{video_name}/
    analysis_dir = f"analysis/{ID_SAN}/{video_basename}"

    # Run match analyzer
    analyzer = MatchAnalyzer(
        tracking_json_path=OUTPUT_JSON,
        video_path=VIDEO_PATH,
        meme_json_path="meme.json",
        court_data=court_data,
        output_dir=analysis_dir
    )

    analysis_result = analyzer.analyze()

    step2_time = time.time() - step2_start
    total_time = step1_time + step2_time

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    print(f"\nTime Summary:")
    print(f"  Step 1 (Tracking):  {format_time(step1_time)}")
    print(f"  Step 2 (Analysis):  {format_time(step2_time)}")
    print(f"  Total:              {format_time(total_time)}")

    print(f"\nOutput locations:")
    print(f"  Tracking JSON: {OUTPUT_JSON}")
    print(f"  Annotated Video: {OUTPUT_VIDEO}")
    print(f"  Analysis JSON: {analysis_dir}/{video_basename}_analysis.json")
    print(f"  Player Images: {analysis_dir}/player_images/")
    print(f"  Heatmaps: {analysis_dir}/heatmaps/")
    print(f"  Highlights: {analysis_dir}/highlights/")

    print(f"\nPlayers analyzed: {len(analysis_result['players'])}")
    print(f"Rallies detected: {len(analysis_result['rallies'])}")

    for pid, player in analysis_result['players'].items():
        print(f"\nPlayer {pid}:")
        print(f"  Score: {player['score']}")
        print(f"  In court rate: {player['shots_in_court_rate']:.1%}")
        print(f"  Max speed: {player['max_ball_speed_ms']:.1f} m/s")
        print(f"  Highlights: {len(player['highlights'])}")


if __name__ == "__main__":
    main()
