"""
API Phân tích Video Tennis Realtime
====================================

API endpoints:
- POST /analyze/upload: Upload và phân tích video (hỗ trợ xử lý song song)
- GET /analysis/{path}: Download file từ thư mục analysis
- GET /segments/{path}: Download file từ thư mục segments

Chạy server:
    uvicorn api_analyzer:app --host 0.0.0.0 --port 8000 --reload

Hoặc chạy với nhiều workers để xử lý song song:
    uvicorn api_analyzer:app --host 0.0.0.0 --port 8000 --workers 4
"""

import os
import json
import shutil
import asyncio
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from match_analyzer import MatchAnalyzer
from ball_tracker import TennisBallTracker


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# URL của file server để tải file
FILE_SERVER_URL = "https://download-linevision.ngrok.app"

# Callback URL để gửi kết quả về server
CALLBACK_URL = "http://linevision.asia/save_json_realtime"

# Số giờ trước khi xóa dữ liệu
CLEANUP_HOURS = 72  # 3 ngày

app = FastAPI(
    title="LineVision Tennis Analyzer API",
    description="API phân tích video tennis realtime - hỗ trợ xử lý song song",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool để xử lý các tác vụ nặng song song
# max_workers = số request có thể xử lý đồng thời (giới hạn 3)
executor = ThreadPoolExecutor(max_workers=3)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def convert_paths_to_urls(result: Dict) -> Dict:
    """Convert local paths to URLs using file server."""
    def replace_path(obj: Any) -> Any:
        if isinstance(obj, str):
            # Replace local paths with URLs
            if obj.startswith("analysis/"):
                return f"{FILE_SERVER_URL}/{obj}"
            elif obj.startswith("output/"):
                return f"{FILE_SERVER_URL}/{obj}"
            elif obj.startswith("segments/"):
                return f"{FILE_SERVER_URL}/{obj}"
        elif isinstance(obj, dict):
            return {k: replace_path(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_path(item) for item in obj]
        return obj

    return replace_path(result)


def send_callback(file_name: str, court_id: str, result: Dict) -> bool:
    """Send analysis result to callback server."""
    callback_data = {
        "file_name": file_name,
        "court_id": court_id,
        "data": result,
    }
    try:
        response = requests.post(
            CALLBACK_URL,
            headers={"Content-Type": "application/json"},
            json=callback_data,
            timeout=30,
        )
        print(f"[API] Callback response: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"[API] Callback failed: {e}")
        return False


def validate_video(video_path: str) -> bool:
    """
    Kiểm tra video có hợp lệ không.
    Returns: True nếu video hợp lệ
    """
    import cv2

    if not os.path.exists(video_path):
        print(f"[API] Video file not found: {video_path}")
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[API] Cannot open video: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        print(f"[API] Invalid video metadata: fps={fps}, frames={frame_count}, size={width}x{height}")
        return False

    print(f"[API] Video validated: {fps}fps, {frame_count} frames, {width}x{height}")
    return True


def run_tracking(video_path: str, id_san: str, court_data: Dict) -> str:
    """
    Chạy ball tracking trên video.
    Returns: Path to tracking JSON
    Raises: ValueError nếu video không hợp lệ
    """
    # Validate video first
    if not validate_video(video_path):
        raise ValueError(f"Invalid video file: {video_path}")

    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = f"output/{id_san}/{video_basename}"
    os.makedirs(output_dir, exist_ok=True)

    # Output paths
    tracking_json_path = os.path.join(output_dir, f"{video_basename}.json")
    output_video_path = os.path.join(output_dir, f"{video_basename}_tracked.mp4")

    # Create tracker and process video
    tracker = TennisBallTracker()
    tracker.process_video(
        video_path=video_path,
        output_json_path=tracking_json_path,
        output_video_path=output_video_path,
        court_data=court_data
    )

    return tracking_json_path


def run_analysis(tracking_json_path: str, video_path: str, id_san: str,
                 court_data: Dict, meme_json_path: str = "meme.json") -> Dict:
    """
    Chạy phân tích trên tracking data.
    Returns: Analysis result
    """
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = f"analysis/{id_san}/{video_basename}"

    analyzer = MatchAnalyzer(
        tracking_json_path=tracking_json_path,
        video_path=video_path,
        meme_json_path=meme_json_path,
        court_data=court_data,
        output_dir=output_dir
    )

    result = analyzer.analyze()
    return result


def process_video_sync(video_path: str, id_san: str, court_data_dict: Dict,
                       meme_json_path: str, original_filename: str) -> Dict:
    """
    Xử lý video đồng bộ (chạy trong thread pool).
    Hàm này sẽ được gọi trong thread riêng để không block event loop.
    """
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    print(f"[API] Processing video: {video_basename}")

    # Run tracking
    print(f"[API] [{video_basename}] Running tracking...")
    tracking_json_path = run_tracking(video_path, id_san, court_data_dict)

    # Run analysis
    print(f"[API] [{video_basename}] Running analysis...")
    result = run_analysis(
        tracking_json_path=tracking_json_path,
        video_path=video_path,
        id_san=id_san,
        court_data=court_data_dict,
        meme_json_path=meme_json_path
    )

    analysis_path = f"analysis/{id_san}/{video_basename}/{video_basename}_analysis.json"

    # Add metadata
    result["request_id"] = video_basename
    result["file_name"] = original_filename
    result["court_id"] = id_san
    result["timestamp"] = datetime.now().isoformat()
    result["expires_at"] = (datetime.now() + timedelta(hours=CLEANUP_HOURS)).isoformat()


    # Convert numpy types to native Python types
    result = convert_numpy_types(result)

    # Callback to server with analysis results
    print(f"[API] [{video_basename}] Sending callback to server...")
    callback_success = send_callback(original_filename, id_san, result)

    print(f"[API] [{video_basename}] Completed! Callback: {'OK' if callback_success else 'FAILED'}")

    return {
        "success": True,
        "message": "Analysis completed successfully",
        "video_name": video_basename,
        "tracking_path": tracking_json_path,
        "analysis_path": analysis_path,
        "callback_sent": callback_success,
        "analysis": result
    }


# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "LineVision Tennis Analyzer API",
        "version": "1.0.0",
        "concurrent_workers": executor._max_workers
    }


@app.post("/analyze/upload")
async def analyze_uploaded_video(
    video: UploadFile = File(...),
    id_san: str = Form(...),
    court_data: str = Form(...),  # JSON string
    meme_json_path: str = Form("meme.json")
):
    """
    Upload và phân tích video.

    Hỗ trợ xử lý nhiều request song song (tối đa 4 video cùng lúc).

    Input (multipart/form-data):
    - video: File video upload
    - id_san: ID sân
    - court_data: JSON string của court coordinates
    - meme_json_path: Đường dẫn meme.json (optional)
    """
    try:
        # Parse court_data JSON
        court_data_dict = json.loads(court_data)

        # Ensure net_y exists
        if "net_y" not in court_data_dict:
            court_data_dict["net_y"] = court_data_dict.get("center_left", {}).get("y", 400)

        # Save uploaded video to uploads folder
        temp_dir = f"uploads/{id_san}"
        os.makedirs(temp_dir, exist_ok=True)

        video_path = os.path.join(temp_dir, video.filename)
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        original_filename = video.filename
        print(f"[API] Uploaded video saved to: {video_path}")

        # Chạy xử lý trong thread pool để không block các request khác
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            process_video_sync,
            video_path,
            id_san,
            court_data_dict,
            meme_json_path,
            original_filename
        )

        return result

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid court_data JSON format")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
