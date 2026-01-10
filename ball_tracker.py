"""
Tennis Ball Detection and Tracking System
==========================================
Detects tennis balls using YOLO, tracks with interpolation,
validates using image comparison, and filters false positives.
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
class FrameResult:
    """All ball detections in a single frame"""
    frame_index: int
    timestamp_seconds: float
    balls: List[BallDetection] = field(default_factory=list)


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
# BALL INTERPOLATOR CLASS
# =============================================================================

class BallInterpolator:
    """
    Interpolates missing ball positions using:
    - Linear interpolation for small gaps (1-2 frames)
    - Parabolic interpolation for larger gaps (3-15 frames)
    """

    MAX_GAP = 15
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
    Main class for tennis ball detection and tracking
    """

    def __init__(
        self,
        model_path: str = "models/ball_best.pt",
        conf_threshold: float = 0.3,
        batch_size: int = 16,
        enable_validation: bool = True
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.batch_size = batch_size
        self.enable_validation = enable_validation

        self.interpolator = BallInterpolator()
        self.validator: Optional[BallValidator] = None
        self.tracker: Optional[TrajectoryTracker] = None

    def process_video(
        self,
        video_path: str,
        output_json_path: str,
        output_video_path: str
    ) -> Dict:
        """
        Main processing pipeline

        Args:
            video_path: Path to input video
            output_json_path: Path to save JSON results
            output_video_path: Path to save annotated video

        Returns:
            Processing results dictionary
        """
        # 1. Get video info
        video_info = self._get_video_info(video_path)

        # Initialize validator and tracker
        self.validator = BallValidator(video_info.resolution["height"])
        self.tracker = TrajectoryTracker(video_info.fps)

        # 2. Read all frames
        print(f"Reading video: {video_path}")
        frames = self._read_video(video_path)
        print(f"Total frames: {len(frames)}")

        # 3. Detect balls in all frames
        print("Detecting balls...")
        raw_detections = self._detect_balls(frames)

        # 4. Extract primary positions (best detection per frame)
        raw_positions = self._extract_primary_positions(raw_detections)

        # 5. Interpolate missing positions
        print("Interpolating missing positions...")
        interpolated = self.interpolator.interpolate(raw_positions)

        # 6. Validate and filter
        print("Validating detections...")
        validated_results = self._validate_and_filter(frames, interpolated, raw_detections)

        # 7. Build frame results
        frame_results = self._build_frame_results(validated_results, video_info.fps)

        # 8. Calculate statistics
        stats = self._calculate_statistics(frame_results, raw_detections)

        # 9. Save JSON
        print(f"Saving JSON to: {output_json_path}")
        self._save_json(video_info, frame_results, stats, output_json_path)

        # 10. Create annotated video
        print(f"Creating annotated video: {output_video_path}")
        self._create_annotated_video(frames, frame_results, video_info.fps, output_video_path)

        print("Processing complete!")

        return {
            "video_info": asdict(video_info),
            "statistics": stats,
            "output_json": output_json_path,
            "output_video": output_video_path
        }

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
        fps: float
    ) -> List[FrameResult]:
        """Build final frame results with ball detections"""
        frame_results = []
        ball_id = 1  # Primary ball tracking

        for res in validated_results:
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

            frame_results.append(frame)

        return frame_results

    def _calculate_statistics(
        self,
        frame_results: List[FrameResult],
        raw_detections: List[List[Dict]]
    ) -> Dict:
        """Calculate processing statistics"""
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

        return {
            "total_frames": total_frames,
            "frames_with_detection": total_detections,
            "interpolated_frames": interpolated_frames,
            "rejected_false_positives": rejected,
            "detection_rate": round(frames_with_ball / total_frames, 4) if total_frames > 0 else 0,
            "average_confidence": round(sum(all_confs) / len(all_confs), 4) if all_confs else 0
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
                "model_path": "models/ball_best.pt",
                "confidence_threshold": self.conf_threshold,
                "max_interpolation_gap": BallInterpolator.MAX_GAP,
                "validation_enabled": self.enable_validation
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
        """Create annotated video with ball detections"""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        try:
            for i, (frame, result) in enumerate(zip(frames, frame_results)):
                annotated = frame.copy()

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
                        x1, y1, x2, y2 = ball.bbox
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

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

    # Output paths - JSON filename matches video filename
    video_basename = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    OUTPUT_JSON = f"output/{video_basename}.json"
    OUTPUT_VIDEO = f"output/{video_basename}_annotated.mp4"

    print("=" * 60)
    print("TENNIS BALL DETECTION AND TRACKING SYSTEM")
    print("=" * 60)
    print(f"Input video: {VIDEO_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(f"Output JSON: {OUTPUT_JSON}")
    print(f"Output video: {OUTPUT_VIDEO}")
    print("=" * 60)

    # Create tracker
    tracker = TennisBallTracker(
        model_path=MODEL_PATH,
        conf_threshold=0.3,
        batch_size=16,
        enable_validation=True
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
    print("\nStatistics:")
    for key, value in result['statistics'].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
