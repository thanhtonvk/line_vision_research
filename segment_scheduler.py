"""
Segment Scheduler - Tự động merge và phân tích segment theo khung giờ
======================================================================

Chạy tự động 24/7, xử lý vào cuối mỗi khung giờ:
- 5-6h, 6-7h, ... 22-23h, 23-24h

Khi giờ hiện tại là X:05, sẽ xử lý khung giờ (X-1)h đến Xh.
Ví dụ: 9:05 sẽ xử lý khung 8h-9h (start=8h, end=9h)

Tự động đọc danh sách court từ thư mục output.

Cách chạy:
    python segment_scheduler.py

Chạy thủ công cho một khung giờ cụ thể:
    python segment_scheduler.py --manual --hour 9 --date 2024-01-17
"""

import os
import sys
import time
import json
import argparse
import schedule
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Any

from match_analyzer import SegmentMerger


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Khung giờ hoạt động (5h sáng đến 24h đêm)
START_HOUR = 5
END_HOUR = 24

# Thư mục chứa tracking results
OUTPUT_BASE_DIR = "output"

# Thư mục lưu segment results
SEGMENT_OUTPUT_DIR = "segments"

# Tỉ lệ shot tối thiểu để là main player
MIN_SHOT_RATIO = 0.5

# Log file
LOG_FILE = "segment_scheduler.log"

# Callback URL để gửi kết quả về server
CALLBACK_URL = "http://linevision.asia/save_json"

# URL của file server để tải file
FILE_SERVER_URL = "https://download-linevision.ngrok.app"

# Số giờ trước khi xóa dữ liệu
CLEANUP_HOURS = 72  # 3 ngày

# Số lần retry callback
MAX_CALLBACK_RETRIES = 3


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def log(message: str):
    """Ghi log ra console và file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_message + "\n")


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





def send_callback_with_retry(file_name: str, court_id: str, result: dict) -> bool:
    """
    Gửi callback với retry logic.

    Args:
        file_name: Tên file gốc
        court_id: ID sân
        result: Kết quả phân tích

    Returns:
        True nếu gửi thành công, False nếu thất bại
    """
    callback_data = {
        "file_name": file_name,
        "court_id": court_id,
        "data": result,
    }

    for attempt in range(MAX_CALLBACK_RETRIES):
        try:
            log(f"  [CALLBACK] Sending to {CALLBACK_URL} (attempt {attempt + 1}/{MAX_CALLBACK_RETRIES})")
            response = requests.post(
                CALLBACK_URL,
                json=callback_data,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            if response.status_code == 200:
                log(f"  [CALLBACK] Success!")
                return True
            else:
                log(f"  [CALLBACK] HTTP error {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            log(f"  [CALLBACK] Request error (attempt {attempt + 1}): {e}")

        if attempt < MAX_CALLBACK_RETRIES - 1:
            wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
            log(f"  [CALLBACK] Waiting {wait_time}s before retry...")
            time.sleep(wait_time)

    log(f"  [CALLBACK] Failed after {MAX_CALLBACK_RETRIES} attempts")
    return False


def get_court_ids() -> List[str]:
    """
    Tự động đọc danh sách court_id từ thư mục output.
    Các thư mục con của output chính là court_id.

    Returns:
        List các court_id
    """
    if not os.path.exists(OUTPUT_BASE_DIR):
        return []

    court_ids = []
    for folder in os.listdir(OUTPUT_BASE_DIR):
        folder_path = os.path.join(OUTPUT_BASE_DIR, folder)
        if os.path.isdir(folder_path):
            court_ids.append(folder)

    return sorted(court_ids)


def get_timestamp_range_for_hour(date: datetime, start_hour: int) -> tuple:
    """
    Lấy khoảng timestamp cho một khung giờ cụ thể.

    Args:
        date: Ngày cần xử lý
        start_hour: Giờ bắt đầu (ví dụ: 8 = khung 8h-9h)

    Returns:
        (start_timestamp, end_timestamp)

    Ví dụ:
        start_hour=8 -> khung 8:00-9:00
        start_hour=23 -> khung 23:00-24:00 (0:00 ngày hôm sau)
    """
    # Tạo datetime cho giờ bắt đầu
    start_dt = date.replace(hour=start_hour, minute=0, second=0, microsecond=0)

    # Giờ kết thúc = start + 1 giờ
    end_dt = start_dt + timedelta(hours=1)

    # Convert sang Unix timestamp
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    return start_ts, end_ts


def find_videos_in_range(id_san: str, start_ts: int, end_ts: int) -> List[str]:
    """
    Tìm các videos trong khoảng thời gian.

    Returns:
        List tên video (timestamp)
    """
    court_dir = os.path.join(OUTPUT_BASE_DIR, id_san)

    if not os.path.exists(court_dir):
        return []

    videos = []
    for folder in os.listdir(court_dir):
        folder_path = os.path.join(court_dir, folder)
        if os.path.isdir(folder_path):
            try:
                folder_ts = int(folder)
                if start_ts <= folder_ts <= end_ts:
                    json_path = os.path.join(folder_path, f"{folder}.json")
                    if os.path.exists(json_path):
                        videos.append(folder)
            except ValueError:
                pass

    return sorted(videos)


def process_hour_segment(id_san: str, date: datetime, start_hour: int) -> Optional[dict]:
    """
    Xử lý segment cho một khung giờ cụ thể.

    Args:
        id_san: ID sân
        date: Ngày cần xử lý
        start_hour: Giờ bắt đầu (ví dụ: 8 = khung 8h-9h)

    Returns:
        Analysis result hoặc None nếu không có video
    """
    start_ts, end_ts = get_timestamp_range_for_hour(date, start_hour)
    end_hour = start_hour + 1

    log(f"Processing {id_san} - {start_hour}:00-{end_hour}:00 ({date.strftime('%Y-%m-%d')})")
    log(f"  Timestamp range: {start_ts} - {end_ts}")

    # Kiểm tra có video không
    videos = find_videos_in_range(id_san, start_ts, end_ts)

    if not videos:
        log(f"  No videos found in this time range")
        return None

    log(f"  Found {len(videos)} videos: {videos}")

    # Chạy SegmentMerger
    try:
        merger = SegmentMerger(
            id_san=id_san,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            output_base_dir=OUTPUT_BASE_DIR,
            segment_output_dir=SEGMENT_OUTPUT_DIR,
            min_shot_ratio=MIN_SHOT_RATIO
        )

        result = merger.merge()

        if result:
            avg_ts = (start_ts + end_ts) // 2
            log(f"  SUCCESS: Segment created at {SEGMENT_OUTPUT_DIR}/{id_san}/{avg_ts}/")

            # Tạo request_id và file_name cho segment
            request_id = f"{id_san}_{start_hour:02d}h_{end_hour:02d}h_{date.strftime('%Y%m%d')}"
            original_filename = f"segment_{request_id}.json"

            # Add metadata
            result["request_id"] = request_id
            result["file_name"] = original_filename
            result["court_id"] = id_san
            result["timestamp"] = datetime.now().isoformat()
            result["expires_at"] = (datetime.now() + timedelta(hours=CLEANUP_HOURS)).isoformat()
            result["segment_info"] = {
                "start_hour": start_hour,
                "end_hour": end_hour,
                "date": date.strftime("%Y-%m-%d"),
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
                "video_count": len(videos)
            }


            # Convert numpy types to native Python types
            result = convert_numpy_types(result)

            # Callback to server
            log(f"  Sending callback to server...")
            callback_success = send_callback_with_retry(original_filename, id_san, result)

            if callback_success:
                log(f"  Callback sent successfully")
            else:
                log(f"  Callback failed")

            return result
        else:
            log(f"  WARNING: Merge returned empty result")
            return None

    except Exception as e:
        log(f"  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def process_all_courts_for_hour(start_hour: int, date: datetime = None, court_ids: List[str] = None):
    """
    Xử lý tất cả sân cho một khung giờ.

    Args:
        start_hour: Giờ bắt đầu (ví dụ: 8 = khung 8h-9h)
        date: Ngày cần xử lý (mặc định là hôm nay)
        court_ids: Danh sách court cần xử lý (mặc định: đọc từ output/)
    """
    if date is None:
        date = datetime.now()

    # Tự động lấy danh sách court từ output nếu không chỉ định
    if court_ids is None:
        court_ids = get_court_ids()

    if not court_ids:
        log(f"No courts found in {OUTPUT_BASE_DIR}/")
        return {}

    end_hour = start_hour + 1

    log("=" * 60)
    log(f"SEGMENT SCHEDULER - Processing hour {start_hour}:00-{end_hour}:00")
    log(f"Courts: {court_ids}")
    log("=" * 60)

    results = {}
    for id_san in court_ids:
        result = process_hour_segment(id_san, date, start_hour)
        results[id_san] = result

    # Summary
    success_count = sum(1 for r in results.values() if r is not None)
    log(f"\nSummary: {success_count}/{len(court_ids)} courts processed successfully")
    log("=" * 60 + "\n")

    return results


def scheduled_job():
    """
    Job chạy theo schedule - xử lý khung giờ vừa kết thúc.

    Ví dụ: Chạy lúc 9:05 -> xử lý khung 8h-9h (start=8, end=9)
    """
    now = datetime.now()
    current_hour = now.hour

    # Giờ bắt đầu của khung giờ vừa kết thúc = current_hour - 1
    # Ví dụ: 9:05 -> start_hour = 8 (khung 8h-9h)
    start_hour = current_hour - 1

    # Xử lý trường hợp 0h -> start_hour = 23 (ngày hôm trước)
    if start_hour < 0:
        start_hour = 23
        now = now - timedelta(days=1)

    if START_HOUR <= start_hour < END_HOUR:
        log(f"Current time: {current_hour}:05 -> Processing previous hour: {start_hour}:00-{start_hour+1}:00")
        process_all_courts_for_hour(start_hour, now)
    else:
        log(f"Skipping hour {start_hour} (outside operating hours {START_HOUR}:00-{END_HOUR}:00)")


def run_scheduler():
    """
    Chạy scheduler daemon 24/7.
    Schedule job chạy vào đầu mỗi giờ trong khung giờ hoạt động.
    Tự động đọc danh sách court từ thư mục output.
    """
    court_ids = get_court_ids()

    log("=" * 60)
    log("SEGMENT SCHEDULER STARTED - RUNNING 24/7")
    log(f"Operating hours: {START_HOUR}:00 - {END_HOUR}:00 (khung {START_HOUR}h-{END_HOUR}h)")
    log(f"Auto-detect courts from: {OUTPUT_BASE_DIR}/")
    log(f"Current courts: {court_ids if court_ids else 'None (will check each hour)'}")
    log("=" * 60 + "\n")

    # Schedule chạy vào phút thứ 5 của mỗi giờ
    # (để đảm bảo video cuối cùng của giờ trước đã được tracking xong)
    # Ví dụ: 9:05 -> xử lý khung 8h-9h
    schedule.every().hour.at(":05").do(scheduled_job)

    log("Scheduler running 24/7. Press Ctrl+C to stop.")
    log(f"Job runs at XX:05 -> processes previous hour (XX-1):00 to XX:00")
    log(f"Example: 9:05 -> processes 8:00-9:00\n")

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check mỗi phút
    except KeyboardInterrupt:
        log("\nScheduler stopped by user.")
        sys.exit(0)


def run_all_hours_today():
    """
    Chạy tất cả khung giờ đã qua trong ngày hôm nay.
    """
    now = datetime.now()
    current_hour = now.hour

    log("=" * 60)
    log("RUNNING ALL HOURS FOR TODAY")
    log("=" * 60 + "\n")

    for hour in range(START_HOUR, min(current_hour, END_HOUR)):
        process_all_courts_for_hour(hour, now)
        print()  # Blank line between hours


def run_date_range(start_date: datetime, end_date: datetime):
    """
    Chạy cho một khoảng ngày.

    Args:
        start_date: Ngày bắt đầu
        end_date: Ngày kết thúc
    """
    log("=" * 60)
    log(f"RUNNING DATE RANGE: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    log("=" * 60 + "\n")

    current_date = start_date
    while current_date <= end_date:
        log(f"\n>>> Processing date: {current_date.strftime('%Y-%m-%d')}")

        for hour in range(START_HOUR, END_HOUR):
            process_all_courts_for_hour(hour, current_date)

        current_date += timedelta(days=1)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Segment Scheduler - Tự động merge và phân tích segment theo khung giờ (chạy 24/7)"
    )

    parser.add_argument(
        "--manual", "-m",
        action="store_true",
        help="Chạy thủ công cho một khung giờ cụ thể"
    )

    parser.add_argument(
        "--hour",
        type=int,
        help="Giờ cần xử lý (dùng với --manual)"
    )

    parser.add_argument(
        "--date",
        type=str,
        help="Ngày cần xử lý (YYYY-MM-DD, dùng với --manual)"
    )

    parser.add_argument(
        "--today", "-t",
        action="store_true",
        help="Chạy tất cả khung giờ đã qua trong ngày hôm nay"
    )

    parser.add_argument(
        "--start-date",
        type=str,
        help="Ngày bắt đầu (YYYY-MM-DD, dùng với --end-date)"
    )

    parser.add_argument(
        "--end-date",
        type=str,
        help="Ngày kết thúc (YYYY-MM-DD, dùng với --start-date)"
    )

    args = parser.parse_args()

    # Chạy theo mode
    if args.manual:
        if args.hour is None:
            print("Error: --hour is required with --manual")
            sys.exit(1)

        date = datetime.now()
        if args.date:
            date = datetime.strptime(args.date, "%Y-%m-%d")

        process_all_courts_for_hour(args.hour, date)

    elif args.today:
        run_all_hours_today()

    elif args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        run_date_range(start_date, end_date)

    else:
        # Mặc định: chạy daemon 24/7
        run_scheduler()


if __name__ == "__main__":
    main()
