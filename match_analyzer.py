"""
Match Analyzer - Phân tích trận đấu Tennis
==========================================

Phân tích trận đấu từ kết quả tracking JSON:
1. Highlight Generator: Cắt video highlight theo cú đánh + phân loại meme
2. Player Analyzer: Phân tích chi tiết từng người chơi
3. Match Summary: Tổng hợp thống kê trận đấu
"""

import os
import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import cv2
import numpy as np


# ==============================================================================
# BASE URL
# ==============================================================================
BASE_URL = "https://download-linevision.ngrok.app"


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class AccuracyStats:
    """Độ chính xác"""
    in_court_count: int = 0
    out_court_count: int = 0
    net_fault_count: int = 0
    in_court_rate: float = 0.0
    out_court_rate: float = 0.0
    net_fault_rate: float = 0.0


@dataclass
class ServeStats:
    """Thống kê giao bóng"""
    in_court_count: int = 0
    out_court_count: int = 0
    net_fault_count: int = 0
    in_court_rate: float = 0.0
    out_court_rate: float = 0.0
    net_fault_rate: float = 0.0
    avg_speed_ms: float = 0.0
    max_speed_ms: float = 0.0


@dataclass
class ReturnStats:
    """Thống kê trả bóng"""
    in_court_count: int = 0
    out_court_count: int = 0
    net_fault_count: int = 0
    in_court_rate: float = 0.0
    out_court_rate: float = 0.0
    net_fault_rate: float = 0.0


@dataclass
class DriveStats:
    """Thống kê cú drive"""
    avg_speed_ms: float = 0.0
    max_speed_ms: float = 0.0


@dataclass
class BallDensityStats:
    """Mật độ bóng đánh sang sân đối phương"""
    short_count: int = 0       # 0-1/4 sân (gần lưới)
    medium_count: int = 0      # 1/4-1/2 sân
    long_count: int = 0        # 1/2-3/4 sân
    very_long_count: int = 0   # 3/4-1 sân (baseline)


@dataclass
class Rally:
    """Một lượt rally (chuỗi cú đánh liên tục)"""
    rally_id: int
    start_frame: int
    end_frame: int
    shots: List[Dict] = field(default_factory=list)
    first_hitter_id: int = 0


@dataclass
class Highlight:
    """Thông tin một highlight video"""
    highlight_id: int
    person_id: int
    rally_id: int
    shot_index: int
    start_time: float
    end_time: float
    video_path: str = ""
    video_url: str = ""
    meme_item: Dict = field(default_factory=dict)


@dataclass
class PlayerStats:
    """Thống kê chi tiết một người chơi"""
    person_id: int
    image_path: str = ""
    image_url: str = ""
    score: float = 0.0

    # Tỉ lệ đánh bóng
    shots_in_court: int = 0
    shots_out_court: int = 0
    shots_in_court_rate: float = 0.0
    shots_out_court_rate: float = 0.0

    # Tốc độ bóng
    avg_ball_speed_ms: float = 0.0
    max_ball_speed_ms: float = 0.0

    # Góc pose
    avg_shoulder_angle: float = 0.0
    avg_knee_angle: float = 0.0

    # Chỉ số chi tiết
    accuracy: AccuracyStats = field(default_factory=AccuracyStats)
    serve: ServeStats = field(default_factory=ServeStats)
    return_stats: ReturnStats = field(default_factory=ReturnStats)
    drive: DriveStats = field(default_factory=DriveStats)
    ball_density: BallDensityStats = field(default_factory=BallDensityStats)
    heatmap_path: str = ""
    heatmap_url: str = ""


# ==============================================================================
# RALLY DETECTOR
# ==============================================================================

class RallyDetector:
    """Phát hiện các rally (chuỗi cú đánh liên tục)"""

    def __init__(self, max_gap_seconds: float = 5.0):
        """
        Args:
            max_gap_seconds: Khoảng cách tối đa giữa 2 cú đánh để coi là cùng 1 rally
        """
        self.max_gap_seconds = max_gap_seconds

    def detect_rallies(self, shots: List[Dict]) -> List[Rally]:
        """
        Nhóm các shots thành rallies.
        - Shots cách nhau > max_gap_seconds -> rally mới
        - Shot đầu tiên của rally = giao bóng
        """
        if not shots:
            return []

        rallies = []
        current_rally_shots = []

        for shot in shots:
            if not current_rally_shots:
                current_rally_shots.append(shot)
            else:
                time_gap = shot['timestamp_seconds'] - current_rally_shots[-1]['timestamp_seconds']
                if time_gap > self.max_gap_seconds:
                    # Tạo rally mới
                    rallies.append(self._create_rally(current_rally_shots, len(rallies)))
                    current_rally_shots = [shot]
                else:
                    current_rally_shots.append(shot)

        # Rally cuối cùng
        if current_rally_shots:
            rallies.append(self._create_rally(current_rally_shots, len(rallies)))

        return rallies

    def _create_rally(self, shots: List[Dict], rally_id: int) -> Rally:
        """Tạo Rally object từ danh sách shots"""
        return Rally(
            rally_id=rally_id,
            start_frame=shots[0]['frame_index'],
            end_frame=shots[-1]['frame_index'],
            shots=shots,
            first_hitter_id=shots[0]['person_id']
        )


# ==============================================================================
# BALL SPEED CALCULATOR
# ==============================================================================

class BallSpeedCalculator:
    """Tính tốc độ bóng (m/s) từ pixel displacement"""

    # Sân tennis chuẩn: 23.77m x 10.97m (doubles)
    COURT_LENGTH_M = 23.77
    COURT_WIDTH_M = 10.97

    def __init__(self, court_data: Dict, fps: float):
        self.court_data = court_data
        self.fps = fps
        self.pixels_per_meter = self._calc_pixels_per_meter()

    def _calc_pixels_per_meter(self) -> float:
        """Tính tỉ lệ pixels/meter dựa trên court_data"""
        # Khoảng cách pixel giữa top_left và bottom_left (chiều dọc sân)
        tl = self.court_data['top_left']
        bl = self.court_data['bottom_left']
        court_length_px = math.sqrt((bl['x'] - tl['x'])**2 + (bl['y'] - tl['y'])**2)
        return court_length_px / self.COURT_LENGTH_M

    def calc_speed(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """
        Tính tốc độ bóng giữa 2 frame liên tiếp.
        Returns: speed in m/s
        """
        displacement_px = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        displacement_m = displacement_px / self.pixels_per_meter
        speed_ms = displacement_m * self.fps
        return speed_ms

    def calc_speeds_for_shots(self, frames_data: List[Dict], shots: List[Dict]) -> Dict[int, float]:
        """
        Tính tốc độ bóng cho mỗi shot dựa trên ball displacement
        Returns: {shot_frame_index: speed_ms}
        """
        shot_speeds = {}

        # Tạo dict frame_index -> ball position
        ball_positions = {}
        for frame in frames_data:
            if frame.get('balls'):
                ball = frame['balls'][0]
                ball_positions[frame['frame_index']] = (ball['x'], ball['y'])

        for shot in shots:
            frame_idx = shot['frame_index']
            # Tìm frame trước và sau để tính velocity
            prev_idx = frame_idx - 1
            next_idx = frame_idx + 1

            if prev_idx in ball_positions and frame_idx in ball_positions:
                x1, y1 = ball_positions[prev_idx]
                x2, y2 = ball_positions[frame_idx]
                speed = self.calc_speed(x1, y1, x2, y2)
                shot_speeds[frame_idx] = speed
            elif frame_idx in ball_positions and next_idx in ball_positions:
                x1, y1 = ball_positions[frame_idx]
                x2, y2 = ball_positions[next_idx]
                speed = self.calc_speed(x1, y1, x2, y2)
                shot_speeds[frame_idx] = speed
            else:
                shot_speeds[frame_idx] = 0.0

        return shot_speeds


# ==============================================================================
# HEATMAP GENERATOR
# ==============================================================================

class HeatmapGenerator:
    """Tạo heatmap phạm vi hoạt động trên sân giả lập"""

    # Kích thước ảnh sân chuẩn
    COURT_IMG_WIDTH = 400
    COURT_IMG_HEIGHT = 800

    def __init__(self, court_data: Dict, output_dir: str):
        self.court_data = court_data
        self.output_dir = output_dir
        self._setup_transform()

    def _setup_transform(self):
        """Setup perspective transform matrix"""
        # Điểm nguồn (từ video)
        src_points = np.float32([
            [self.court_data['top_left']['x'], self.court_data['top_left']['y']],
            [self.court_data['top_right']['x'], self.court_data['top_right']['y']],
            [self.court_data['bottom_right']['x'], self.court_data['bottom_right']['y']],
            [self.court_data['bottom_left']['x'], self.court_data['bottom_left']['y']]
        ])

        # Điểm đích (sân chuẩn)
        margin = 20
        dst_points = np.float32([
            [margin, margin],
            [self.COURT_IMG_WIDTH - margin, margin],
            [self.COURT_IMG_WIDTH - margin, self.COURT_IMG_HEIGHT - margin],
            [margin, self.COURT_IMG_HEIGHT - margin]
        ])

        self.transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    def _draw_court(self) -> np.ndarray:
        """Vẽ sân tennis giả lập"""
        court = np.ones((self.COURT_IMG_HEIGHT, self.COURT_IMG_WIDTH, 3), dtype=np.uint8) * 76
        court[:, :, 1] = 153  # Green tint
        court[:, :, 2] = 76

        margin = 20
        white = (255, 255, 255)

        # Outer boundary
        cv2.rectangle(court, (margin, margin),
                      (self.COURT_IMG_WIDTH - margin, self.COURT_IMG_HEIGHT - margin),
                      white, 2)

        # Net line (middle)
        net_y = self.COURT_IMG_HEIGHT // 2
        cv2.line(court, (margin, net_y), (self.COURT_IMG_WIDTH - margin, net_y), white, 2)

        # Service boxes
        service_depth = int((self.COURT_IMG_HEIGHT - 2 * margin) * 0.21)  # ~21% from net
        center_x = self.COURT_IMG_WIDTH // 2

        # Top service box
        cv2.line(court, (center_x, margin), (center_x, net_y - service_depth), white, 1)
        cv2.line(court, (margin, net_y - service_depth),
                 (self.COURT_IMG_WIDTH - margin, net_y - service_depth), white, 1)

        # Bottom service box
        cv2.line(court, (center_x, net_y + service_depth), (center_x, self.COURT_IMG_HEIGHT - margin), white, 1)
        cv2.line(court, (margin, net_y + service_depth),
                 (self.COURT_IMG_WIDTH - margin, net_y + service_depth), white, 1)

        return court

    def _transform_position(self, x: float, y: float) -> Tuple[int, int]:
        """Transform vị trí từ video sang tọa độ sân chuẩn"""
        point = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.transform_matrix)
        tx, ty = int(transformed[0][0][0]), int(transformed[0][0][1])
        # Clamp to court bounds
        tx = max(0, min(self.COURT_IMG_WIDTH - 1, tx))
        ty = max(0, min(self.COURT_IMG_HEIGHT - 1, ty))
        return tx, ty

    def generate_heatmap(self, person_id: int, positions: List[Tuple[float, float]]) -> str:
        """
        Tạo heatmap từ danh sách vị trí người chơi.
        Returns: Path của hình ảnh heatmap
        """
        # 1. Vẽ sân tennis
        court_img = self._draw_court()

        # 2. Tạo heatmap data
        heatmap_data = np.zeros((self.COURT_IMG_HEIGHT, self.COURT_IMG_WIDTH), dtype=np.float32)

        for x, y in positions:
            tx, ty = self._transform_position(x, y)
            # Add gaussian blob
            cv2.circle(heatmap_data, (tx, ty), 15, 1.0, -1)

        # Gaussian blur for smooth heatmap
        heatmap_data = cv2.GaussianBlur(heatmap_data, (31, 31), 0)

        # Normalize
        if heatmap_data.max() > 0:
            heatmap_data = heatmap_data / heatmap_data.max()

        # 3. Convert to color heatmap
        heatmap_colored = cv2.applyColorMap(
            (heatmap_data * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        # 4. Blend với sân
        alpha = 0.6
        result = cv2.addWeighted(court_img, alpha, heatmap_colored, 1 - alpha, 0)

        # 5. Save
        os.makedirs(os.path.join(self.output_dir, "heatmaps"), exist_ok=True)
        filename = f"heatmap_person_{person_id}.png"
        filepath = os.path.join(self.output_dir, "heatmaps", filename)
        cv2.imwrite(filepath, result)

        return filepath

    def get_heatmap_url(self, person_id: int) -> str:
        """Get URL for heatmap - includes full path from output_dir"""
        # output_dir có dạng: analysis/court_001/1765189807
        # URL cần có dạng: BASE_URL/analysis/court_001/1765189807/heatmaps/heatmap_person_1.png
        return f"{BASE_URL}/{self.output_dir}/heatmaps/heatmap_person_{person_id}.png"


# ==============================================================================
# HIGHLIGHT GENERATOR
# ==============================================================================

class HighlightGenerator:
    """Cắt video highlight và phân loại meme"""

    def __init__(self,
                 video_path: str,
                 output_dir: str,
                 meme_data: List[Dict],
                 base_output_dir: str = None,  # Thư mục gốc (analysis/court_001/xxx)
                 padding_seconds: float = 3.0,
                 use_ffmpeg: bool = True,  # Dùng FFmpeg để nén tốt hơn
                 crf: int = 28):  # Chất lượng nén (18-28, cao hơn = file nhỏ hơn)
        self.video_path = video_path
        self.output_dir = output_dir  # Thư mục highlights (analysis/court_001/xxx/highlights)
        self.base_output_dir = base_output_dir or output_dir  # Thư mục gốc để tạo URL
        self.meme_data = meme_data
        self.padding_seconds = padding_seconds
        self.use_ffmpeg = use_ffmpeg
        self.crf = crf  # Constant Rate Factor: 18=high quality, 28=smaller file

        os.makedirs(output_dir, exist_ok=True)

    def generate_highlights(self,
                            rallies: List[Rally],
                            player_stats: Dict[int, PlayerStats],
                            all_shots: List[Dict],
                            shot_speeds: Dict[int, float]) -> Dict[int, List[Highlight]]:
        """
        Tạo highlight video cho từng person.
        Returns: {person_id: [Highlight, ...]}
        """
        # Mở video
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps

        highlights_by_person = defaultdict(list)
        highlight_counter = defaultdict(int)

        # Tìm các extreme values để phân loại meme
        extremes = self._find_extremes(all_shots, player_stats, shot_speeds)

        for rally in rallies:
            for shot_idx, shot in enumerate(rally.shots):
                person_id = shot['person_id']
                frame_idx = shot['frame_index']
                timestamp = shot['timestamp_seconds']

                # Tính start/end time
                start_time = max(0, timestamp - self.padding_seconds)
                end_time = min(duration, timestamp + self.padding_seconds)

                # Tạo output directory cho person
                person_dir = os.path.join(self.output_dir, f"person_{person_id}")
                os.makedirs(person_dir, exist_ok=True)

                # Filename
                highlight_id = highlight_counter[person_id]
                filename = f"highlight_{highlight_id:03d}.mp4"
                filepath = os.path.join(person_dir, filename)

                # Cắt video
                self._cut_video_segment(cap, filepath, start_time, end_time, fps, frame_width, frame_height)

                # Phân loại meme
                meme_item = self._classify_meme(shot, player_stats.get(person_id), extremes, shot_speeds)

                # Tạo highlight object
                highlight = Highlight(
                    highlight_id=highlight_id,
                    person_id=person_id,
                    rally_id=rally.rally_id,
                    shot_index=shot_idx,
                    start_time=start_time,
                    end_time=end_time,
                    video_path=filepath,
                    video_url=f"{BASE_URL}/{self.base_output_dir}/highlights/person_{person_id}/{filename}",
                    meme_item=meme_item
                )

                highlights_by_person[person_id].append(highlight)
                highlight_counter[person_id] += 1

        cap.release()
        return dict(highlights_by_person)

    def _cut_video_segment(self, cap: cv2.VideoCapture, output_path: str,
                           start_time: float, end_time: float,
                           fps: float, width: int, height: int):
        """Cắt một đoạn video với nén tối ưu bằng FFmpeg"""
        import subprocess
        import shutil

        if self.use_ffmpeg and shutil.which('ffmpeg'):
            # Dùng FFmpeg để nén với H.264 - giữ nguyên resolution
            duration = end_time - start_time
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-i', self.video_path,
                '-t', str(duration),
                '-c:v', 'libx264',
                '-crf', str(self.crf),  # Chất lượng nén (18-28)
                '-preset', 'fast',  # Tốc độ encode (fast = cân bằng)
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',  # Cho phép stream
                output_path
            ]
            try:
                subprocess.run(cmd, capture_output=True, check=True)
                return
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass  # Fallback to OpenCV

        # Fallback: OpenCV với codec tốt nhất có thể
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()

    def _find_extremes(self, all_shots: List[Dict],
                       player_stats: Dict[int, PlayerStats],
                       shot_speeds: Dict[int, float]) -> Dict:
        """Tìm các giá trị cực đại để phân loại meme"""
        extremes = {
            'max_speed_frame': None,
            'max_speed_value': 0,
            'min_knee_frame': None,
            'min_knee_value': 180,
            'max_shoulder_frame': None,
            'max_shoulder_value': 0,
            'min_shoulder_frame': None,
            'min_shoulder_value': 180,
            'best_in_court_person': None,
            'worst_in_court_person': None
        }

        # Tốc độ cao nhất
        for frame_idx, speed in shot_speeds.items():
            if speed > extremes['max_speed_value']:
                extremes['max_speed_value'] = speed
                extremes['max_speed_frame'] = frame_idx

        # Góc gối, vai
        for shot in all_shots:
            pose = shot.get('pose_angles', {})

            # Knee angle
            left_knee = pose.get('left_knee')
            right_knee = pose.get('right_knee')
            knee_angles = [a for a in [left_knee, right_knee] if a is not None]
            if knee_angles:
                min_knee = min(knee_angles)
                if min_knee < extremes['min_knee_value']:
                    extremes['min_knee_value'] = min_knee
                    extremes['min_knee_frame'] = shot['frame_index']

            # Shoulder angle
            left_shoulder = pose.get('left_shoulder')
            right_shoulder = pose.get('right_shoulder')
            shoulder_angles = [a for a in [left_shoulder, right_shoulder] if a is not None]
            if shoulder_angles:
                max_shoulder = max(shoulder_angles)
                if max_shoulder > extremes['max_shoulder_value']:
                    extremes['max_shoulder_value'] = max_shoulder
                    extremes['max_shoulder_frame'] = shot['frame_index']

                min_shoulder = min(shoulder_angles)
                if min_shoulder < extremes['min_shoulder_value']:
                    extremes['min_shoulder_value'] = min_shoulder
                    extremes['min_shoulder_frame'] = shot['frame_index']

        # In court rate
        if player_stats:
            best_rate = 0
            worst_rate = 1
            for pid, stats in player_stats.items():
                if stats.shots_in_court_rate > best_rate:
                    best_rate = stats.shots_in_court_rate
                    extremes['best_in_court_person'] = pid
                if stats.shots_in_court_rate < worst_rate:
                    worst_rate = stats.shots_in_court_rate
                    extremes['worst_in_court_person'] = pid

        return extremes

    def _classify_meme(self, shot: Dict, player_stats: Optional[PlayerStats],
                       extremes: Dict, shot_speeds: Dict[int, float]) -> Dict:
        """Phân loại shot thuộc meme category nào"""
        frame_idx = shot['frame_index']
        pose = shot.get('pose_angles', {})

        # Mặc định không có meme
        default_meme = {}

        # 1. Cú đánh mạnh nhất
        if extremes['max_speed_frame'] == frame_idx:
            for meme in self.meme_data:
                if meme.get('name_category') == 'Cú đánh mạnh nhất':
                    return meme

        # 2. Khụy gối sâu nhất
        if extremes['min_knee_frame'] == frame_idx:
            for meme in self.meme_data:
                if meme.get('name_category') == 'Khụy gối sâu nhất':
                    return meme

        # 3. Mở vai rộng nhất
        if extremes['max_shoulder_frame'] == frame_idx:
            for meme in self.meme_data:
                if meme.get('name_category') == 'Mở vai rộng nhất':
                    return meme

        # 4. Mở vai hẹp nhất
        if extremes['min_shoulder_frame'] == frame_idx:
            for meme in self.meme_data:
                if meme.get('name_category') == 'Mở vai hẹp nhất':
                    return meme

        # 5. Cú đánh bay lên trời (bóng out và bay lên)
        if shot.get('ball_zone') == 'out' and shot.get('ball_side') == 'unknown':
            ball_y = shot.get('ball_position', {}).get('y', 0)
            if ball_y < 200:  # Ball flew up
                for meme in self.meme_data:
                    if meme.get('name_category') == 'Cú đánh bay lên trời':
                        return meme

        # 6. Tỉ lệ bóng trong sân cao nhất (person level)
        if player_stats and extremes['best_in_court_person'] == shot['person_id']:
            for meme in self.meme_data:
                if meme.get('name_category') == 'Tỉ lệ bóng trong sân cao nhất':
                    return meme

        # 7. Tỉ lệ bóng ngoài sân cao nhất
        if player_stats and extremes['worst_in_court_person'] == shot['person_id']:
            for meme in self.meme_data:
                if meme.get('name_category') == 'Tỉ lệ bóng ngoài sân cao nhất':
                    return meme

        return default_meme


# ==============================================================================
# PLAYER ANALYZER
# ==============================================================================

class PlayerAnalyzer:
    """Phân tích chi tiết từng người chơi"""

    def __init__(self,
                 tracking_data: Dict,
                 video_path: str,
                 court_data: Dict,
                 output_dir: str,
                 skip_video: bool = False):
        self.tracking_data = tracking_data
        self.video_path = video_path
        self.court_data = court_data
        self.output_dir = output_dir
        self.skip_video = skip_video or (video_path is None)

        self.speed_calculator = BallSpeedCalculator(
            court_data, tracking_data['video_info']['fps']
        )
        self.heatmap_generator = HeatmapGenerator(court_data, output_dir)

        # Tính shot speeds một lần
        self.shot_speeds = self.speed_calculator.calc_speeds_for_shots(
            tracking_data.get('frames', []),
            tracking_data.get('shots', [])
        )

    def analyze_player(self, person_id: int, rallies: List[Rally]) -> PlayerStats:
        """Phân tích một người chơi"""
        # 1. Crop hình ảnh người chơi
        image_path = self._crop_player_image(person_id)

        # 2. Lọc shots của person này
        player_shots = self._get_player_shots(person_id)

        if not player_shots:
            return PlayerStats(
                person_id=person_id,
                image_path=image_path,
                image_url=f"{BASE_URL}/{self.output_dir}/player_images/person_{person_id}.jpg"
            )

        # 3. Phân loại giao bóng vs trả bóng
        serves, returns = self._classify_shots(person_id, rallies)

        # 4. Tính các chỉ số
        accuracy = self._calc_accuracy(player_shots)
        serve_stats = self._calc_serve_stats(serves)
        return_stats = self._calc_return_stats(returns)
        drive_stats = self._calc_drive_stats(player_shots)
        ball_density = self._calc_ball_density(player_shots)

        # 5. Tạo heatmap
        positions = self._get_player_positions(person_id)
        heatmap_path = self.heatmap_generator.generate_heatmap(person_id, positions)
        heatmap_url = self.heatmap_generator.get_heatmap_url(person_id)

        # 6. Tính góc trung bình
        avg_shoulder = self._calc_avg_shoulder_angle(player_shots)
        avg_knee = self._calc_avg_knee_angle(player_shots)

        # 7. Tính điểm tổng hợp
        score = self._calculate_score(accuracy, serve_stats, return_stats, drive_stats)

        return PlayerStats(
            person_id=person_id,
            image_path=image_path,
            image_url=f"{BASE_URL}/{self.output_dir}/player_images/person_{person_id}.jpg",
            score=score,
            shots_in_court=accuracy.in_court_count,
            shots_out_court=accuracy.out_court_count,
            shots_in_court_rate=accuracy.in_court_rate,
            shots_out_court_rate=accuracy.out_court_rate,
            avg_ball_speed_ms=drive_stats.avg_speed_ms,
            max_ball_speed_ms=drive_stats.max_speed_ms,
            avg_shoulder_angle=avg_shoulder,
            avg_knee_angle=avg_knee,
            accuracy=accuracy,
            serve=serve_stats,
            return_stats=return_stats,
            drive=drive_stats,
            ball_density=ball_density,
            heatmap_path=heatmap_path,
            heatmap_url=heatmap_url
        )

    def _crop_player_image(self, person_id: int) -> str:
        """Crop hình ảnh người chơi từ video"""
        # Bỏ qua nếu không có video
        if self.skip_video or not self.video_path:
            return ""

        # Tìm frame có person này với confidence cao nhất
        best_frame_idx = None
        best_confidence = 0
        best_bbox = None

        for frame in self.tracking_data.get('frames', []):
            for person in frame.get('persons', []):
                if person['id'] == person_id and person['confidence'] > best_confidence:
                    best_confidence = person['confidence']
                    best_frame_idx = frame['frame_index']
                    best_bbox = person['bbox']

        if best_frame_idx is None or best_bbox is None:
            return ""

        # Mở video và đọc frame
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return ""

        # Crop
        x1, y1 = best_bbox['x1'], best_bbox['y1']
        x2, y2 = best_bbox['x2'], best_bbox['y2']

        # Mở rộng bbox một chút
        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)

        cropped = frame[y1:y2, x1:x2]

        # Save
        os.makedirs(os.path.join(self.output_dir, "player_images"), exist_ok=True)
        filename = f"person_{person_id}.jpg"
        filepath = os.path.join(self.output_dir, "player_images", filename)
        cv2.imwrite(filepath, cropped)

        return filepath

    def _get_player_shots(self, person_id: int) -> List[Dict]:
        """Lấy danh sách shots của một người"""
        return [s for s in self.tracking_data['shots'] if s['person_id'] == person_id]

    def _get_player_positions(self, person_id: int) -> List[Tuple[float, float]]:
        """Lấy danh sách vị trí người chơi trong các frame"""
        positions = []
        for frame in self.tracking_data['frames']:
            for person in frame.get('persons', []):
                if person['id'] == person_id:
                    positions.append((person['x'], person['y']))
        return positions

    def _classify_shots(self, person_id: int, rallies: List[Rally]) -> Tuple[List[Dict], List[Dict]]:
        """Phân loại giao bóng vs trả bóng"""
        serves = []
        returns = []

        for rally in rallies:
            if not rally.shots:
                continue

            first_shot = rally.shots[0]

            for shot in rally.shots:
                if shot['person_id'] == person_id:
                    if shot['frame_index'] == first_shot['frame_index']:
                        serves.append(shot)
                    else:
                        returns.append(shot)

        return serves, returns

    def _calc_accuracy(self, shots: List[Dict]) -> AccuracyStats:
        """Tính độ chính xác"""
        in_count = sum(1 for s in shots if s.get('ball_in_court', False))
        out_count = sum(1 for s in shots if not s.get('ball_in_court', True))
        total = len(shots)

        # Net fault: ball_zone == 'near_net' và out
        net_fault = sum(1 for s in shots
                        if s.get('ball_zone') == 'near_net' and not s.get('ball_in_court', True))

        return AccuracyStats(
            in_court_count=in_count,
            out_court_count=out_count,
            net_fault_count=net_fault,
            in_court_rate=in_count / total if total > 0 else 0.0,
            out_court_rate=out_count / total if total > 0 else 0.0,
            net_fault_rate=net_fault / total if total > 0 else 0.0
        )

    def _calc_serve_stats(self, serves: List[Dict]) -> ServeStats:
        """Tính thống kê giao bóng"""
        accuracy = self._calc_accuracy(serves)

        # Tốc độ
        speeds = [self.shot_speeds.get(s['frame_index'], 0) for s in serves]
        speeds = [s for s in speeds if s > 0]

        return ServeStats(
            in_court_count=accuracy.in_court_count,
            out_court_count=accuracy.out_court_count,
            net_fault_count=accuracy.net_fault_count,
            in_court_rate=accuracy.in_court_rate,
            out_court_rate=accuracy.out_court_rate,
            net_fault_rate=accuracy.net_fault_rate,
            avg_speed_ms=sum(speeds) / len(speeds) if speeds else 0.0,
            max_speed_ms=max(speeds) if speeds else 0.0
        )

    def _calc_return_stats(self, returns: List[Dict]) -> ReturnStats:
        """Tính thống kê trả bóng"""
        accuracy = self._calc_accuracy(returns)
        return ReturnStats(
            in_court_count=accuracy.in_court_count,
            out_court_count=accuracy.out_court_count,
            net_fault_count=accuracy.net_fault_count,
            in_court_rate=accuracy.in_court_rate,
            out_court_rate=accuracy.out_court_rate,
            net_fault_rate=accuracy.net_fault_rate
        )

    def _calc_drive_stats(self, shots: List[Dict]) -> DriveStats:
        """Tính thống kê drive"""
        speeds = [self.shot_speeds.get(s['frame_index'], 0) for s in shots]
        speeds = [s for s in speeds if s > 0]

        return DriveStats(
            avg_speed_ms=sum(speeds) / len(speeds) if speeds else 0.0,
            max_speed_ms=max(speeds) if speeds else 0.0
        )

    def _calc_ball_density(self, shots: List[Dict]) -> BallDensityStats:
        """
        Phân loại mật độ bóng sang sân đối phương.
        Dựa trên ball_zone và khoảng cách từ net.
        """
        short_count = 0
        medium_count = 0
        long_count = 0
        very_long_count = 0

        net_y = self.court_data.get('net_y', self.court_data['center_left']['y'])

        for shot in shots:
            if not shot.get('ball_in_court', False):
                continue

            ball_pos = shot.get('ball_position', {})
            ball_y = ball_pos.get('y', net_y)
            ball_side = shot.get('ball_side', 'unknown')

            # Tính khoảng cách từ net
            if ball_side == 'top':
                # Top half: từ net đến top_left
                baseline_y = self.court_data['top_left']['y']
                half_length = abs(net_y - baseline_y)
                distance_from_net = abs(ball_y - net_y)
            else:
                # Bottom half: từ net đến bottom_left
                baseline_y = self.court_data['bottom_left']['y']
                half_length = abs(baseline_y - net_y)
                distance_from_net = abs(ball_y - net_y)

            # Chia thành 4 phần
            ratio = distance_from_net / half_length if half_length > 0 else 0

            if ratio < 0.25:
                short_count += 1
            elif ratio < 0.5:
                medium_count += 1
            elif ratio < 0.75:
                long_count += 1
            else:
                very_long_count += 1

        return BallDensityStats(
            short_count=short_count,
            medium_count=medium_count,
            long_count=long_count,
            very_long_count=very_long_count
        )

    def _calc_avg_shoulder_angle(self, shots: List[Dict]) -> float:
        """Tính góc vai trung bình"""
        angles = []
        for shot in shots:
            pose = shot.get('pose_angles', {})
            left = pose.get('left_shoulder')
            right = pose.get('right_shoulder')
            if left is not None:
                angles.append(left)
            if right is not None:
                angles.append(right)
        return sum(angles) / len(angles) if angles else 0.0

    def _calc_avg_knee_angle(self, shots: List[Dict]) -> float:
        """Tính góc gối trung bình"""
        angles = []
        for shot in shots:
            pose = shot.get('pose_angles', {})
            left = pose.get('left_knee')
            right = pose.get('right_knee')
            if left is not None:
                angles.append(left)
            if right is not None:
                angles.append(right)
        return sum(angles) / len(angles) if angles else 0.0

    def _calculate_score(self, accuracy: AccuracyStats, serve: ServeStats,
                         return_stats: ReturnStats, drive: DriveStats) -> float:
        """
        Tính điểm tổng hợp cho người chơi.
        Score = 40% accuracy + 30% serve + 20% return + 10% speed
        """
        # Accuracy score (0-100)
        accuracy_score = accuracy.in_court_rate * 100

        # Serve score
        serve_score = serve.in_court_rate * 70 + min(serve.max_speed_ms / 50 * 30, 30)

        # Return score
        return_score = return_stats.in_court_rate * 100

        # Speed score (normalize to 0-100, assuming max 50 m/s)
        speed_score = min(drive.max_speed_ms / 50 * 100, 100)

        total = (accuracy_score * 0.4 + serve_score * 0.3 +
                 return_score * 0.2 + speed_score * 0.1)

        return round(total, 1)


# ==============================================================================
# MATCH ANALYZER (Main Class)
# ==============================================================================

class MatchAnalyzer:
    """Class chính điều phối phân tích trận đấu"""

    def __init__(self,
                 tracking_json_path: str,
                 video_path: str = None,
                 meme_json_path: str = None,
                 court_data: Dict = None,
                 output_dir: str = "analysis",
                 skip_highlights: bool = False):
        self.tracking_data = self._load_json(tracking_json_path)
        self.video_path = video_path
        self.meme_data = self._load_json(meme_json_path) if meme_json_path else []
        self.output_dir = output_dir
        self.skip_highlights = skip_highlights or (video_path is None)

        # Lấy court_data từ tracking nếu không truyền vào
        if court_data:
            self.court_data = court_data
        else:
            self.court_data = self.tracking_data.get('court_info', {})

        # Tạo output directories
        os.makedirs(output_dir, exist_ok=True)
        if not self.skip_highlights:
            os.makedirs(os.path.join(output_dir, "highlights"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "player_images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "heatmaps"), exist_ok=True)

    def _load_json(self, path: str) -> any:
        """Load JSON file"""
        if not path or not os.path.exists(path):
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def analyze(self) -> Dict:
        """
        Main analysis pipeline.
        Returns: Analysis result dict
        """
        fps = self.tracking_data['video_info']['fps']
        shots = self.tracking_data.get('shots', [])

        if not shots:
            print("WARNING: No shots found in tracking data")
            return {}

        print("=" * 60)
        print("MATCH ANALYZER")
        print("=" * 60)

        # 1. Detect rallies
        print("\n[1/6] Detecting rallies...")
        rally_detector = RallyDetector(max_gap_seconds=5.0)
        rallies = rally_detector.detect_rallies(shots)
        print(f"  Found {len(rallies)} rallies")

        # 2. Get unique player IDs (chỉ lấy những người có shot)
        player_ids = sorted(set(shot['person_id'] for shot in shots if shot.get('person_id') is not None))
        print(f"  Active players: {player_ids}")

        # 3. Analyze each player
        print("\n[2/6] Analyzing players...")
        player_analyzer = PlayerAnalyzer(
            self.tracking_data, self.video_path,
            self.court_data, self.output_dir,
            skip_video=self.skip_highlights
        )
        player_stats = {}
        for pid in player_ids:
            print(f"  Analyzing player {pid}...")
            player_stats[pid] = player_analyzer.analyze_player(pid, rallies)

        # 4. Generate highlights (nếu có video)
        highlights = {}
        if not self.skip_highlights and self.video_path:
            print("\n[3/6] Generating highlights...")
            highlight_generator = HighlightGenerator(
                self.video_path,
                os.path.join(self.output_dir, "highlights"),
                self.meme_data,
                base_output_dir=self.output_dir,  # Truyền thư mục gốc để tạo URL đúng
                padding_seconds=3.0
            )
            highlights = highlight_generator.generate_highlights(
                rallies, player_stats, shots, player_analyzer.shot_speeds
            )
            total_highlights = sum(len(h) for h in highlights.values())
            print(f"  Generated {total_highlights} highlight videos")
        else:
            print("\n[3/6] Skipping highlights (no video)")

        # 5. Generate match summary
        print("\n[4/6] Generating match summary...")
        match_summary = self._generate_match_summary(player_stats)

        # 6. Build final result
        print("\n[5/6] Building result...")

        # Build video_info - chỉ giữ các fields chuẩn (loại bỏ source_videos nếu có)
        video_info = {
            "filename": self.tracking_data['video_info'].get('filename', ''),
            "fps": self.tracking_data['video_info'].get('fps', 25),
            "total_frames": self.tracking_data['video_info'].get('total_frames', 0),
            "duration_seconds": self.tracking_data['video_info'].get('duration_seconds', 0),
            "resolution": self.tracking_data['video_info'].get('resolution', {})
        }

        result = {
            "video_info": video_info,
            "court_info": self.court_data,
            "rallies": [
                {
                    "rally_id": r.rally_id,
                    "start_frame": r.start_frame,
                    "end_frame": r.end_frame,
                    "first_hitter_id": r.first_hitter_id,
                    "shot_count": len(r.shots)
                }
                for r in rallies
            ],
            "players": {
                str(pid): {
                    **self._player_stats_to_dict(stats),
                    "highlights": [self._highlight_to_dict(h) for h in highlights.get(pid, [])]
                }
                for pid, stats in player_stats.items()
            },
            "match_summary": match_summary
        }

        # 7. Save result
        print("\n[6/6] Saving result...")
        # Lấy basename từ video_path hoặc tracking filename
        if self.video_path:
            video_basename = os.path.splitext(os.path.basename(self.video_path))[0]
        else:
            video_basename = os.path.splitext(self.tracking_data['video_info'].get('filename', 'segment'))[0]

        output_path = os.path.join(self.output_dir, f"{video_basename}_analysis.json")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n  Analysis saved to: {output_path}")
        print("=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)

        return result

    def _player_stats_to_dict(self, stats: PlayerStats) -> Dict:
        """Convert PlayerStats to dict"""
        return {
            "person_id": stats.person_id,
            "image_url": stats.image_url,
            "score": stats.score,
            "shots_in_court": stats.shots_in_court,
            "shots_out_court": stats.shots_out_court,
            "shots_in_court_rate": round(stats.shots_in_court_rate, 4),
            "shots_out_court_rate": round(stats.shots_out_court_rate, 4),
            "avg_ball_speed_ms": round(stats.avg_ball_speed_ms, 2),
            "max_ball_speed_ms": round(stats.max_ball_speed_ms, 2),
            "avg_shoulder_angle": round(stats.avg_shoulder_angle, 2),
            "avg_knee_angle": round(stats.avg_knee_angle, 2),
            "accuracy": asdict(stats.accuracy),
            "serve": asdict(stats.serve),
            "return_stats": asdict(stats.return_stats),
            "drive": asdict(stats.drive),
            "ball_density": asdict(stats.ball_density),
            "heatmap_url": stats.heatmap_url
        }

    def _highlight_to_dict(self, highlight: Highlight) -> Dict:
        """Convert Highlight to dict"""
        return {
            "highlight_id": highlight.highlight_id,
            "person_id": highlight.person_id,
            "rally_id": highlight.rally_id,
            "start_time": round(highlight.start_time, 2),
            "end_time": round(highlight.end_time, 2),
            "video_url": highlight.video_url,
            "meme_item": highlight.meme_item
        }

    def _generate_match_summary(self, player_stats: Dict[int, PlayerStats]) -> Dict:
        """Tạo tổng hợp thống kê trận đấu"""
        if not player_stats:
            return {}

        # Tìm người có tốc độ cao nhất
        fastest_player = max(player_stats.values(), key=lambda p: p.max_ball_speed_ms)

        # Tổng số shots
        total_in = sum(p.shots_in_court for p in player_stats.values())
        total_out = sum(p.shots_out_court for p in player_stats.values())
        total = total_in + total_out

        return {
            "fastest_player": {
                "person_id": fastest_player.person_id,
                "image_url": fastest_player.image_url,
                "max_ball_speed_ms": round(fastest_player.max_ball_speed_ms, 2),
                "avg_shoulder_angle": round(fastest_player.avg_shoulder_angle, 2),
                "avg_knee_angle": round(fastest_player.avg_knee_angle, 2)
            },
            "overall_stats": {
                "total_shots": total,
                "total_in_court": total_in,
                "total_out_court": total_out,
                "in_court_rate": round(total_in / total, 4) if total > 0 else 0,
                "out_court_rate": round(total_out / total, 4) if total > 0 else 0
            },
            "player_comparison": {
                str(pid): {
                    "in_court_rate": round(stats.shots_in_court_rate, 4),
                    "avg_ball_speed_ms": round(stats.avg_ball_speed_ms, 2),
                    "avg_shoulder_angle": round(stats.avg_shoulder_angle, 2),
                    "avg_knee_angle": round(stats.avg_knee_angle, 2)
                }
                for pid, stats in player_stats.items()
            }
        }


# ==============================================================================
# SEGMENT MERGER - Merge nhiều video thành 1 segment
# ==============================================================================

class PersonMatcher:
    """
    So khớp người chơi giữa các videos dựa trên:
    1. Vị trí trên sân (top half vs bottom half)
    2. Số lượng shots
    """

    def __init__(self, court_data: Dict):
        self.court_data = court_data
        # Tính đường giữa sân (net line)
        self.net_y = court_data.get('net_y', court_data.get('center_left', {}).get('y', 400))

    def extract_person_features_from_frames(self, person_id: int, frames: List[Dict], video_name: str, shots: List[Dict]) -> Dict:
        """
        Trích xuất đặc điểm của người chơi từ frames (vị trí thực của người).

        Returns:
            {
                'avg_y': float,  # Vị trí trung bình theo Y
                'court_side': str,  # 'top' hoặc 'bottom'
                'shot_count': int,
                'video': str,
                'person_id': int
            }
        """
        # Lấy vị trí người từ frames
        positions = []
        for frame in frames:
            for person in frame.get('persons', []):
                if person.get('id') == person_id:
                    # Dùng y của center của bbox
                    bbox = person.get('bbox', {})
                    if bbox:
                        center_y = (bbox.get('y1', 0) + bbox.get('y2', 0)) / 2
                        positions.append(center_y)
                    elif 'y' in person:
                        positions.append(person['y'])

        # Đếm số shots của person này
        shot_count = sum(1 for s in shots if s.get('person_id') == person_id)

        if not positions:
            avg_y = self.net_y  # Default nếu không có data
        else:
            avg_y = sum(positions) / len(positions)

        return {
            'avg_y': avg_y,
            'court_side': 'top' if avg_y < self.net_y else 'bottom',
            'shot_count': shot_count,
            'video': video_name,
            'person_id': person_id
        }

    def extract_person_features(self, person_data: Dict, video_name: str) -> Dict:
        """
        Trích xuất đặc điểm của người chơi từ shots (fallback).

        Returns:
            {
                'avg_y': float,  # Vị trí trung bình theo Y
                'court_side': str,  # 'top' hoặc 'bottom'
                'shot_count': int,
                'video': str
            }
        """
        positions = []
        for shot in person_data.get('shots', []):
            if 'hitter_position' in shot:
                pos = shot['hitter_position']
                if pos and 'y' in pos:
                    positions.append(pos['y'])

        if not positions:
            # Fallback: dùng ball position để ước lượng
            for shot in person_data.get('shots', []):
                if 'ball_position' in shot:
                    ball_pos = shot['ball_position']
                    if ball_pos and 'y' in ball_pos:
                        positions.append(ball_pos['y'])

        avg_y = sum(positions) / len(positions) if positions else self.net_y

        return {
            'avg_y': avg_y,
            'court_side': 'top' if avg_y < self.net_y else 'bottom',
            'shot_count': len(person_data.get('shots', [])),
            'video': video_name,
            'person_id': person_data.get('person_id')
        }

    def match_persons_across_videos(self,
                                     all_persons: List[Dict],
                                     min_shot_ratio: float = 0.1) -> Dict[str, int]:
        """
        So khớp người chơi giữa các videos.

        Chiến lược:
        1. Trong mỗi video, xác định main players dựa trên số shots (>= min_shot_ratio * max_shots)
        2. Phân loại main players dựa trên vị trí tương đối (người có avg_y lớn hơn = gần camera = bottom)
        3. Gán global_id: bottom=1, top=2, ... (sắp xếp theo Y giảm dần)
        4. Người còn lại gán theo vị trí gần main player nào nhất

        Args:
            all_persons: List các person features từ tất cả videos
            min_shot_ratio: Tỉ lệ tối thiểu so với người có nhiều shots nhất để được coi là main player

        Returns:
            Mapping {f"{video}_{person_id}": global_person_id}
        """
        if not all_persons:
            return {}

        mapping = {}

        # Group by video
        from collections import defaultdict
        by_video = defaultdict(list)
        for p in all_persons:
            by_video[p['video']].append(p)

        # Process each video
        for video_name, persons in by_video.items():
            if not persons:
                continue

            # Sort by shot count descending
            sorted_persons = sorted(persons, key=lambda p: p.get('shot_count', 0), reverse=True)

            # Xác định ngưỡng shots để được coi là main player
            max_shots = sorted_persons[0].get('shot_count', 0)
            min_shots_threshold = max_shots * min_shot_ratio

            # Lọc main players dựa trên shot count threshold
            main_players = [p for p in sorted_persons if p.get('shot_count', 0) >= min_shots_threshold]

            # Đảm bảo ít nhất có 1 main player (người có nhiều shots nhất)
            if not main_players and sorted_persons:
                main_players = [sorted_persons[0]]

            # Sắp xếp main players theo avg_y giảm dần (Y lớn = gần camera = bottom)
            main_players_sorted = sorted(main_players, key=lambda p: p['avg_y'], reverse=True)

            # Gán global_id cho main players: bottom (Y lớn nhất) = 1, tiếp theo = 2, ...
            main_player_ids = {}  # {local_person_id: global_id}
            for idx, player in enumerate(main_players_sorted):
                global_id = idx + 1  # 1, 2, 3, ...
                mapping[f"{video_name}_{player['person_id']}"] = global_id
                main_player_ids[player['person_id']] = global_id

            # Các người không phải main player: gán theo vị trí gần main player nào nhất
            non_main_players = [p for p in sorted_persons if p['person_id'] not in main_player_ids]

            for p in non_main_players:
                if not main_players_sorted:
                    # Fallback: gán = 1
                    mapping[f"{video_name}_{p['person_id']}"] = 1
                    continue

                # Tìm main player gần nhất theo vị trí Y
                min_dist = float('inf')
                nearest_global_id = 1
                for main_p in main_players_sorted:
                    dist = abs(p['avg_y'] - main_p['avg_y'])
                    if dist < min_dist:
                        min_dist = dist
                        nearest_global_id = main_player_ids[main_p['person_id']]

                mapping[f"{video_name}_{p['person_id']}"] = nearest_global_id

        return mapping

    def get_global_person_id(self,
                              video_name: str,
                              local_person_id: int,
                              mapping: Dict[str, int]) -> int:
        """Lấy global person ID từ mapping"""
        key = f"{video_name}_{local_person_id}"
        return mapping.get(key, local_person_id)


class SegmentMerger:
    """
    Merge nhiều video tracking results theo khoảng thời gian thành 1 segment analysis.

    Ví dụ: Merge các video từ timestamp 1765180000 - 1765200000

    QUAN TRỌNG: Sử dụng PersonMatcher để nhận diện đúng người chơi giữa các videos
    dựa trên vị trí trên sân (không phụ thuộc vào person_id từ tracking).

    court_info sẽ được tự động lấy từ file tracking JSON đầu tiên.
    """

    def __init__(self,
                 id_san: str,
                 start_timestamp: int,
                 end_timestamp: int,
                 output_base_dir: str = "output",
                 segment_output_dir: str = "segments",
                 min_shot_ratio: float = 0.1):
        """
        Args:
            id_san: ID sân (ví dụ: "court_001")
            start_timestamp: Timestamp bắt đầu
            end_timestamp: Timestamp kết thúc
            output_base_dir: Thư mục chứa tracking results (default: "output")
            segment_output_dir: Thư mục lưu segment analysis (default: "segments")
            min_shot_ratio: Tỉ lệ tối thiểu so với người có nhiều shots nhất để được coi là main player
                           Ví dụ: 0.1 = 10%, nếu người cao nhất có 100 shots thì người có >= 10 shots là main
                           Tăng giá trị này để có ít main players hơn
        """
        self.id_san = id_san
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.output_base_dir = output_base_dir
        self.segment_output_dir = segment_output_dir
        self.min_shot_ratio = min_shot_ratio
        self.court_data = None  # Sẽ load từ tracking JSON
        self.person_matcher = None  # Sẽ khởi tạo sau khi có court_data

    def merge(self) -> Dict:
        """
        Merge tất cả videos trong khoảng thời gian đã chỉ định.

        Returns:
            Merged analysis result
        """
        # Tìm videos trong khoảng thời gian
        video_names = self._find_videos_in_range()

        if not video_names:
            print(f"No videos found in time range {self.start_timestamp} - {self.end_timestamp}")
            return {}

        # Tạo segment name từ trung bình timestamps
        avg_timestamp = (self.start_timestamp + self.end_timestamp) // 2
        segment_name = str(avg_timestamp)

        return self._merge_videos(video_names, segment_name)

    def _find_videos_in_range(self) -> List[str]:
        """Tìm các videos trong khoảng thời gian"""
        court_dir = os.path.join(self.output_base_dir, self.id_san)

        if not os.path.exists(court_dir):
            print(f"ERROR: Court directory not found: {court_dir}")
            return []

        video_names = []
        for folder in sorted(os.listdir(court_dir)):
            folder_path = os.path.join(court_dir, folder)
            if os.path.isdir(folder_path):
                try:
                    folder_ts = int(folder)
                    if self.start_timestamp <= folder_ts <= self.end_timestamp:
                        # Kiểm tra có JSON file không
                        json_path = os.path.join(folder_path, f"{folder}.json")
                        if os.path.exists(json_path):
                            video_names.append(folder)
                except ValueError:
                    pass

        return video_names

    def _merge_videos(self,
                      video_names: List[str],
                      segment_name: str) -> Dict:
        """
        Merge nhiều video tracking results thành 1 segment.
        Output format giống file tracking gốc (output/xxx.json).
        Sau đó chạy MatchAnalyzer để tạo file analysis.

        Args:
            video_names: Danh sách tên video (không có extension)
            segment_name: Tên segment output

        Returns:
            Analysis result (từ MatchAnalyzer)
        """
        print(f"=" * 60)
        print(f"SEGMENT MERGER: {segment_name}")
        print(f"=" * 60)
        print(f"Court ID: {self.id_san}")
        print(f"Time range: {self.start_timestamp} - {self.end_timestamp}")
        print(f"Videos to merge: {len(video_names)}")
        for v in video_names:
            print(f"  - {v}")
        print()

        # 1. Load tất cả tracking JSONs
        all_tracking_data = []
        total_duration = 0
        total_frames = 0

        for video_name in video_names:
            json_path = os.path.join(
                self.output_base_dir, self.id_san, video_name, f"{video_name}.json"
            )
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_tracking_data.append({
                        'video_name': video_name,
                        'data': data
                    })
                    total_duration += data.get('video_info', {}).get('duration_seconds', 0)
                    total_frames += data.get('video_info', {}).get('total_frames', 0)
                print(f"  Loaded: {video_name}")
            else:
                print(f"  WARNING: Not found - {json_path}")

        if not all_tracking_data:
            print("ERROR: No tracking data found!")
            return {}

        # Lấy court_info từ tracking JSON đầu tiên
        self.court_data = all_tracking_data[0]['data'].get('court_info', {})
        if not self.court_data:
            print("WARNING: No court_info found in tracking JSON, using default")
            self.court_data = {
                "top_left": {"x": 171, "y": 325},
                "center_left": {"x": 294, "y": 406},
                "net_y": 406
            }

        # Khởi tạo PersonMatcher với court_data
        self.person_matcher = PersonMatcher(self.court_data)

        print(f"\nTotal videos loaded: {len(all_tracking_data)}")
        print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        print(f"Court info loaded from: {all_tracking_data[0]['video_name']}")
        print()

        # 2. Tạo person mapping
        print("[1/5] Matching persons across videos...")
        person_mapping = self._build_person_mapping(all_tracking_data)
        print(f"  Person mapping created: {len(set(person_mapping.values()))} unique players identified")

        # 3. Merge frames và shots từ tất cả videos (với global person_id)
        print("\n[2/5] Merging frames and shots...")
        merged_frames = []
        merged_shots = []
        time_offset = 0
        frame_offset = 0

        for item in all_tracking_data:
            video_name = item['video_name']
            data = item['data']
            fps = data.get('video_info', {}).get('fps', 25)

            # Merge frames
            for frame in data.get('frames', []):
                merged_frame = frame.copy()
                merged_frame['original_video'] = video_name
                merged_frame['original_frame_index'] = frame.get('frame_index', 0)
                merged_frame['frame_index'] = frame.get('frame_index', 0) + frame_offset
                merged_frame['timestamp_seconds'] = frame.get('timestamp_seconds', 0) + time_offset

                # Map person IDs trong frame
                if 'persons' in merged_frame:
                    for person in merged_frame['persons']:
                        local_pid = person.get('id')
                        if local_pid is not None:
                            global_pid = self.person_matcher.get_global_person_id(
                                video_name, local_pid, person_mapping
                            )
                            person['id'] = global_pid
                            person['original_id'] = local_pid

                merged_frames.append(merged_frame)

            # Merge shots
            for shot in data.get('shots', []):
                merged_shot = shot.copy()
                merged_shot['original_video'] = video_name
                merged_shot['original_frame'] = shot.get('frame_index', 0)
                merged_shot['original_person_id'] = shot.get('person_id')
                merged_shot['frame_index'] = shot.get('frame_index', 0) + frame_offset
                merged_shot['timestamp_seconds'] = shot.get('timestamp_seconds', 0) + time_offset

                # Map to global person_id
                local_pid = shot.get('person_id')
                if local_pid is not None:
                    global_pid = self.person_matcher.get_global_person_id(
                        video_name, local_pid, person_mapping
                    )
                    merged_shot['person_id'] = global_pid

                merged_shots.append(merged_shot)

            video_duration = data.get('video_info', {}).get('duration_seconds', 0)
            video_frames = data.get('video_info', {}).get('total_frames', 0)
            time_offset += video_duration
            frame_offset += video_frames

        print(f"  Total merged frames: {len(merged_frames)}")
        print(f"  Total merged shots: {len(merged_shots)}")

        # 4. Tạo output directory - segments/{id_san}/{segment_name}/
        output_dir = os.path.join(self.segment_output_dir, self.id_san, segment_name)
        os.makedirs(output_dir, exist_ok=True)

        # 5. Tạo merged video_info
        first_video = all_tracking_data[0]['data'].get('video_info', {})
        merged_video_info = {
            'filename': f"{segment_name}.mp4",
            'fps': first_video.get('fps', 25),
            'total_frames': total_frames,
            'duration_seconds': total_duration,
            'resolution': first_video.get('resolution', {}),
            'source_videos': video_names
        }

        # 6. Tính statistics
        print("\n[3/5] Computing statistics...")
        statistics = self._compute_tracking_statistics(merged_shots, merged_frames)

        # 7. Build tracking result (giống format output/xxx.json)
        tracking_result = {
            'video_info': merged_video_info,
            'court_info': self.court_data,
            'detection_config': all_tracking_data[0]['data'].get('detection_config', {}),
            'frames': merged_frames,
            'shots': merged_shots,
            'statistics': statistics,
            'segment_info': {
                'segment_name': segment_name,
                'id_san': self.id_san,
                'start_timestamp': self.start_timestamp,
                'end_timestamp': self.end_timestamp,
                'source_videos': video_names,
                'person_mapping': {k: v for k, v in person_mapping.items()}
            }
        }

        # 8. Save tracking result
        print("\n[4/5] Saving merged tracking JSON...")
        tracking_output_path = os.path.join(output_dir, f"{segment_name}.json")
        with open(tracking_output_path, 'w', encoding='utf-8') as f:
            json.dump(tracking_result, f, ensure_ascii=False, indent=2)
        print(f"  Tracking saved to: {tracking_output_path}")

        # 9. Run MatchAnalyzer để tạo analysis và merge highlights
        print("\n[5/5] Running MatchAnalyzer...")
        analysis_result = self._run_analysis(
            tracking_output_path, output_dir, segment_name,
            video_names=video_names, person_mapping=person_mapping
        )

        print(f"\nSegment merge completed!")
        print(f"  Tracking: {tracking_output_path}")
        print(f"  Analysis: {os.path.join(output_dir, f'{segment_name}_analysis.json')}")
        print("=" * 60)

        return analysis_result

    def _compute_tracking_statistics(self, shots: List[Dict], frames: List[Dict]) -> Dict:
        """Tính statistics cho merged tracking data"""
        total_shots = len(shots)
        shots_in_court = sum(1 for s in shots if s.get('ball_zone') not in ['out', None])
        shots_out_court = sum(1 for s in shots if s.get('ball_zone') == 'out')

        # Count by person
        person_shots = {}
        for shot in shots:
            pid = shot.get('person_id')
            if pid is not None:
                if pid not in person_shots:
                    person_shots[pid] = {'total': 0, 'in_court': 0, 'out_court': 0}
                person_shots[pid]['total'] += 1
                if shot.get('ball_zone') not in ['out', None]:
                    person_shots[pid]['in_court'] += 1
                elif shot.get('ball_zone') == 'out':
                    person_shots[pid]['out_court'] += 1

        return {
            'total_frames': len(frames),
            'total_shots': total_shots,
            'shots_in_court': shots_in_court,
            'shots_out_court': shots_out_court,
            'in_court_rate': round(shots_in_court / total_shots, 4) if total_shots > 0 else 0,
            'person_statistics': person_shots
        }

    def _run_analysis(self, tracking_json_path: str, output_dir: str, segment_name: str,
                       video_names: List[str] = None, person_mapping: Dict[str, int] = None) -> Dict:
        """Chạy MatchAnalyzer trên merged tracking data và merge highlights từ analysis gốc"""
        try:
            # Load meme.json nếu có
            meme_json_path = os.path.join(os.path.dirname(os.path.dirname(output_dir)), "meme.json")
            if not os.path.exists(meme_json_path):
                meme_json_path = "meme.json"

            # Không có video file cho segment, nên bỏ qua việc tạo highlights
            # Chỉ chạy phân tích thống kê
            analyzer = MatchAnalyzer(
                tracking_json_path=tracking_json_path,
                video_path=None,  # Không có video
                meme_json_path=meme_json_path if os.path.exists(meme_json_path) else None,
                court_data=self.court_data,
                output_dir=output_dir,
                skip_highlights=True  # Bỏ qua highlights vì không có video
            )

            result = analyzer.analyze()

            # Merge highlights từ analysis gốc
            if video_names and person_mapping:
                result = self._merge_highlights_from_analysis(
                    result, video_names, person_mapping
                )

                # Re-save kết quả với highlights
                output_path = os.path.join(output_dir, f"{segment_name}_analysis.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

            return result

        except Exception as e:
            print(f"  WARNING: MatchAnalyzer failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: tạo analysis đơn giản
            return self._create_simple_analysis(tracking_json_path, output_dir, segment_name)

    def _merge_highlights_from_analysis(self,
                                         result: Dict,
                                         video_names: List[str],
                                         person_mapping: Dict[str, int]) -> Dict:
        """
        Merge highlights từ các file analysis gốc vào segment analysis.

        Args:
            result: Analysis result từ MatchAnalyzer
            video_names: Danh sách video names đã merge
            person_mapping: Mapping {video_personId: global_personId}

        Returns:
            Updated result với highlights
        """
        print("  Merging highlights from original analysis files...")

        # Khởi tạo highlights cho mỗi player
        highlights_by_player = defaultdict(list)
        highlight_counter = defaultdict(int)

        # Load analysis từ mỗi video gốc
        for video_name in video_names:
            analysis_path = os.path.join(
                "analysis", self.id_san, video_name, f"{video_name}_analysis.json"
            )

            if not os.path.exists(analysis_path):
                print(f"    WARNING: Analysis not found: {analysis_path}")
                continue

            with open(analysis_path, 'r', encoding='utf-8') as f:
                original_analysis = json.load(f)

            # Lấy highlights từ mỗi player trong analysis gốc
            for local_pid_str, player_data in original_analysis.get('players', {}).items():
                local_pid = int(local_pid_str)

                # Map local person_id sang global person_id
                mapping_key = f"{video_name}_{local_pid}"
                global_pid = person_mapping.get(mapping_key, local_pid)

                # Lấy highlights của player này
                original_highlights = player_data.get('highlights', [])

                for hl in original_highlights:
                    # Tạo highlight mới với global person_id và highlight_id mới
                    # Giữ nguyên format như file analysis gốc (không thêm fields mới)
                    new_highlight = {
                        'highlight_id': highlight_counter[global_pid],
                        'person_id': global_pid,
                        'rally_id': hl.get('rally_id', 0),
                        'start_time': hl.get('start_time', 0),
                        'end_time': hl.get('end_time', 0),
                        'video_url': hl.get('video_url', ''),
                        'meme_item': hl.get('meme_item', {})
                    }

                    highlights_by_player[global_pid].append(new_highlight)
                    highlight_counter[global_pid] += 1

            print(f"    Loaded highlights from: {video_name}")

        # Gắn highlights vào result
        for pid_str, player_data in result.get('players', {}).items():
            pid = int(pid_str)
            player_data['highlights'] = highlights_by_player.get(pid, [])

        total_highlights = sum(len(hl) for hl in highlights_by_player.values())
        print(f"  Total highlights merged: {total_highlights}")

        return result

    def _create_simple_analysis(self, tracking_json_path: str, output_dir: str, segment_name: str) -> Dict:
        """Tạo analysis đơn giản khi MatchAnalyzer fail"""
        print("  Creating simple analysis fallback...")

        # Load tracking data
        with open(tracking_json_path, 'r', encoding='utf-8') as f:
            tracking_data = json.load(f)

        shots = tracking_data.get('shots', [])
        video_info = tracking_data.get('video_info', {})

        # Tính statistics đơn giản
        player_stats = {}
        for shot in shots:
            pid = shot.get('person_id')
            if pid is None:
                continue
            if pid not in player_stats:
                player_stats[pid] = {
                    'person_id': pid,
                    'total_shots': 0,
                    'shots_in_court': 0,
                    'shots_out_court': 0
                }
            player_stats[pid]['total_shots'] += 1
            if shot.get('ball_zone') not in ['out', None]:
                player_stats[pid]['shots_in_court'] += 1
            elif shot.get('ball_zone') == 'out':
                player_stats[pid]['shots_out_court'] += 1

        # Tính rates
        for pid, stats in player_stats.items():
            total = stats['total_shots']
            stats['shots_in_court_rate'] = round(stats['shots_in_court'] / total, 4) if total > 0 else 0
            stats['shots_out_court_rate'] = round(stats['shots_out_court'] / total, 4) if total > 0 else 0

        # Build result
        result = {
            'video_info': video_info,
            'court_info': self.court_data,
            'rallies': [],
            'players': {str(pid): stats for pid, stats in player_stats.items()},
            'match_summary': {
                'total_shots': len(shots),
                'total_in_court': sum(s['shots_in_court'] for s in player_stats.values()),
                'total_out_court': sum(s['shots_out_court'] for s in player_stats.values()),
                'player_count': len(player_stats)
            }
        }

        # Save
        output_path = os.path.join(output_dir, f"{segment_name}_analysis.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"  Simple analysis saved to: {output_path}")
        return result

    def _build_person_mapping(self, all_tracking_data: List[Dict]) -> Dict[str, int]:
        """
        Xây dựng mapping từ (video, local_person_id) -> global_person_id
        dựa trên vị trí trên sân.
        """
        # Thu thập thông tin tất cả persons từ các videos
        all_person_features = []

        for item in all_tracking_data:
            video_name = item['video_name']
            data = item['data']
            frames = data.get('frames', [])
            shots = data.get('shots', [])

            # Tìm tất cả unique person_ids có trong shots
            person_ids_with_shots = set(s.get('person_id') for s in shots if s.get('person_id') is not None)

            # Trích xuất features cho mỗi person từ frames (vị trí thực)
            for person_id in person_ids_with_shots:
                features = self.person_matcher.extract_person_features_from_frames(
                    person_id, frames, video_name, shots
                )
                all_person_features.append(features)

        # Tạo mapping (sử dụng min_shot_ratio từ config)
        mapping = self.person_matcher.match_persons_across_videos(
            all_person_features,
            min_shot_ratio=self.min_shot_ratio
        )

        # Log mapping details
        for key, global_id in mapping.items():
            video, local_id = key.rsplit('_', 1)
            side = 'BOTTOM' if global_id == 1 else 'TOP'
            print(f"    {video} person_{local_id} -> Player {global_id} ({side})")

        return mapping

    def _merge_persons_with_mapping(self,
                                     all_tracking_data: List[Dict],
                                     person_mapping: Dict[str, int]) -> Dict:
        """
        Merge person data từ nhiều videos sử dụng global person_id mapping.
        """
        merged = {}

        for item in all_tracking_data:
            data = item['data']
            video_name = item['video_name']

            # Lấy persons từ shots
            for shot in data.get('shots', []):
                local_person_id = shot.get('person_id')
                if local_person_id is not None:
                    # Map to global person_id
                    global_person_id = self.person_matcher.get_global_person_id(
                        video_name, local_person_id, person_mapping
                    )

                    if global_person_id not in merged:
                        merged[global_person_id] = {
                            'person_id': global_person_id,
                            'court_side': 'bottom' if global_person_id == 1 else 'top',
                            'appearances': [],
                            'source_person_ids': [],  # Track original IDs
                            'total_shots': 0,
                            'shots_in_court': 0,
                            'shots_out_court': 0,
                            'all_shoulder_angles': [],
                            'all_knee_angles': [],
                            'all_speeds': []
                        }

                    # Track source
                    source_key = f"{video_name}_{local_person_id}"
                    if source_key not in merged[global_person_id]['source_person_ids']:
                        merged[global_person_id]['source_person_ids'].append(source_key)

                    merged[global_person_id]['appearances'].append(video_name)
                    merged[global_person_id]['total_shots'] += 1

                    # Ball zone
                    ball_zone = shot.get('ball_zone', 'unknown')
                    if ball_zone == 'in':
                        merged[global_person_id]['shots_in_court'] += 1
                    elif ball_zone == 'out':
                        merged[global_person_id]['shots_out_court'] += 1

                    # Pose angles
                    pose = shot.get('pose_at_hit', {})
                    if pose:
                        angles = pose.get('angles', {})
                        shoulder = angles.get('left_shoulder_angle') or angles.get('right_shoulder_angle')
                        knee = angles.get('left_knee_angle') or angles.get('right_knee_angle')
                        if shoulder and shoulder > 0:
                            merged[global_person_id]['all_shoulder_angles'].append(shoulder)
                        if knee and knee > 0:
                            merged[global_person_id]['all_knee_angles'].append(knee)

        return merged

    def _compute_segment_stats(self, shots: List[Dict],
                                persons: Dict) -> Dict:
        """Tính toán statistics cho segment"""

        # Player stats
        player_stats = {}
        for person_id, person_data in persons.items():
            total = person_data['total_shots']
            in_court = person_data['shots_in_court']
            out_court = person_data['shots_out_court']

            shoulder_angles = person_data['all_shoulder_angles']
            knee_angles = person_data['all_knee_angles']

            player_stats[person_id] = {
                'person_id': person_id,
                'court_side': person_data.get('court_side', 'unknown'),
                'source_person_ids': person_data.get('source_person_ids', []),
                'total_shots': total,
                'shots_in_court': in_court,
                'shots_out_court': out_court,
                'shots_in_court_rate': round(in_court / total, 4) if total > 0 else 0,
                'shots_out_court_rate': round(out_court / total, 4) if total > 0 else 0,
                'avg_shoulder_angle': round(sum(shoulder_angles) / len(shoulder_angles), 2) if shoulder_angles else 0,
                'avg_knee_angle': round(sum(knee_angles) / len(knee_angles), 2) if knee_angles else 0,
                'appearances_count': len(set(person_data['appearances']))
            }

        # Match summary
        total_shots = len(shots)
        total_in = sum(p['shots_in_court'] for p in player_stats.values())
        total_out = sum(p['shots_out_court'] for p in player_stats.values())

        match_summary = {
            'total_shots': total_shots,
            'total_in_court': total_in,
            'total_out_court': total_out,
            'in_court_rate': round(total_in / total_shots, 4) if total_shots > 0 else 0,
            'out_court_rate': round(total_out / total_shots, 4) if total_shots > 0 else 0,
            'player_count': len(player_stats)
        }

        # Best player (highest in_court_rate)
        if player_stats:
            best_player = max(player_stats.values(), key=lambda p: p['shots_in_court_rate'])
            match_summary['best_accuracy_player'] = {
                'person_id': best_player['person_id'],
                'in_court_rate': best_player['shots_in_court_rate'],
                'total_shots': best_player['total_shots']
            }

        # Rallies summary (nhóm shots theo thời gian)
        rallies = self._detect_rallies(shots)
        rallies_summary = {
            'total_rallies': len(rallies),
            'avg_shots_per_rally': round(sum(r['shot_count'] for r in rallies) / len(rallies), 2) if rallies else 0,
            'longest_rally': max((r['shot_count'] for r in rallies), default=0)
        }

        return {
            'players': player_stats,
            'match_summary': match_summary,
            'rallies_summary': rallies_summary
        }

    def _detect_rallies(self, shots: List[Dict], max_gap_seconds: float = 5.0) -> List[Dict]:
        """Phát hiện rallies từ shots"""
        if not shots:
            return []

        # Sort by timestamp
        sorted_shots = sorted(shots, key=lambda s: s.get('timestamp_seconds', 0))

        rallies = []
        current_rally = {
            'start_time': sorted_shots[0].get('timestamp_seconds', 0),
            'shots': [sorted_shots[0]],
            'shot_count': 1
        }

        for i in range(1, len(sorted_shots)):
            prev_time = sorted_shots[i-1].get('timestamp_seconds', 0)
            curr_time = sorted_shots[i].get('timestamp_seconds', 0)

            if curr_time - prev_time <= max_gap_seconds:
                current_rally['shots'].append(sorted_shots[i])
                current_rally['shot_count'] += 1
            else:
                current_rally['end_time'] = prev_time
                current_rally['duration'] = prev_time - current_rally['start_time']
                rallies.append(current_rally)

                current_rally = {
                    'start_time': curr_time,
                    'shots': [sorted_shots[i]],
                    'shot_count': 1
                }

        # Last rally
        current_rally['end_time'] = sorted_shots[-1].get('timestamp_seconds', 0)
        current_rally['duration'] = current_rally['end_time'] - current_rally['start_time']
        rallies.append(current_rally)

        return rallies


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Configuration
    VIDEO_PATH = "1765199807.mp4"
    MEME_JSON = "meme.json"
    ID_SAN = "court_001"  # Court/field ID

    # Derive paths from video name and id_san
    video_basename = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

    # Input: output/{id_san}/{video_name}/{video_name}.json
    TRACKING_JSON = f"output/{ID_SAN}/{video_basename}/{video_basename}.json"

    # Output: analysis/{id_san}/{video_name}/
    OUTPUT_DIR = f"analysis/{ID_SAN}/{video_basename}"

    # Court data
    COURT_DATA = {
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

    print(f"Court ID: {ID_SAN}")
    print(f"Video: {VIDEO_PATH}")
    print(f"Tracking JSON: {TRACKING_JSON}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Run analyzer
    analyzer = MatchAnalyzer(
        tracking_json_path=TRACKING_JSON,
        video_path=VIDEO_PATH,
        meme_json_path=MEME_JSON,
        court_data=COURT_DATA,
        output_dir=OUTPUT_DIR
    )

    result = analyzer.analyze()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Players analyzed: {len(result['players'])}")
    print(f"Rallies detected: {len(result['rallies'])}")

    for pid, player in result['players'].items():
        print(f"\nPlayer {pid}:")
        print(f"  Score: {player['score']}")
        print(f"  In court rate: {player['shots_in_court_rate']:.1%}")
        print(f"  Max speed: {player['max_ball_speed_ms']:.1f} m/s")
        print(f"  Highlights: {len(player['highlights'])}")


# ==============================================================================
# SEGMENT MERGER - Ví dụ sử dụng
# ==============================================================================
# Uncomment và chạy để merge nhiều videos thành 1 segment

if __name__ == "__main__":
    # Merge videos theo khoảng thời gian timestamp
    # court_info sẽ tự động được lấy từ file tracking JSON đầu tiên
    #
    # Output:
    #   segments/court_001/segment_1765180000_1765200000/
    #   ├── segment_1765180000_1765200000.json          # Tracking format (giống output/)
    #   └── segment_1765180000_1765200000_analysis.json # Analysis format (giống analysis/)
    merger = SegmentMerger(
        id_san="court_001",
        start_timestamp=1765180000,
        end_timestamp=1765200000
    )
    result = merger.merge()

    if result:
        print(f"\nMerge completed!")
        print(f"Players: {len(result.get('players', {}))}")
        print(f"Rallies: {len(result.get('rallies', []))}")

        for pid, player in result.get('players', {}).items():
            rate = player.get('shots_in_court_rate', 0)
            print(f"  Player {pid}: {rate:.1%} in court")
