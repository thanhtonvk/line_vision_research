"""
Data Cleaner - Tự động xóa dữ liệu cũ hơn N ngày
================================================================

Xóa dữ liệu trong các thư mục: uploads, segments, output, analysis
Dựa trên Unix timestamp trong tên thư mục/file.

Cách chạy:
    python data_cleaner.py                    # Xóa dữ liệu > 3 ngày (mặc định)
    python data_cleaner.py --days 7           # Xóa dữ liệu > 7 ngày
    python data_cleaner.py --dry-run          # Chỉ hiển thị, không xóa thật
    python data_cleaner.py --daemon           # Chạy tự động hàng ngày lúc 3:00 sáng
"""

import os
import sys
import time
import shutil
import argparse
import schedule
from datetime import datetime, timedelta
from typing import List, Tuple

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Số ngày giữ lại dữ liệu (mặc định: 3 ngày)
DEFAULT_RETENTION_DAYS = 3

# Các thư mục cần dọn dẹp
CLEANUP_DIRS = ["uploads", "segments", "output", "analysis"]

# Log file
LOG_FILE = "data_cleaner.log"

# Giờ chạy tự động (3:00 sáng)
AUTO_RUN_TIME = "03:00"


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


def get_cutoff_timestamp(days: int) -> int:
    """
    Lấy timestamp ngưỡng để xóa.
    Dữ liệu có timestamp < cutoff sẽ bị xóa.

    Args:
        days: Số ngày giữ lại

    Returns:
        Unix timestamp
    """
    cutoff_date = datetime.now() - timedelta(days=days)
    return int(cutoff_date.timestamp())


def is_timestamp_folder(name: str) -> bool:
    """Kiểm tra tên có phải là Unix timestamp không"""
    try:
        ts = int(name)
        # Timestamp hợp lệ: từ năm 2020 đến 2100
        return 1577836800 <= ts <= 4102444800
    except ValueError:
        return False


def is_timestamp_file(name: str) -> bool:
    """Kiểm tra file có tên là timestamp không (ví dụ: 1765189807.mp4)"""
    base_name = os.path.splitext(name)[0]
    return is_timestamp_folder(base_name)


def get_timestamp_from_name(name: str) -> int:
    """Lấy timestamp từ tên file/folder"""
    base_name = os.path.splitext(name)[0]
    return int(base_name)


def format_timestamp(ts: int) -> str:
    """Format timestamp thành datetime string"""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def format_size(size_bytes: int) -> str:
    """Format kích thước file"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def get_folder_size(folder_path: str) -> int:
    """Tính tổng kích thước thư mục"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size


# ==============================================================================
# CLEANUP FUNCTIONS
# ==============================================================================

def find_old_data(base_dir: str, cutoff_ts: int) -> Tuple[List[str], List[str]]:
    """
    Tìm các thư mục và file cũ cần xóa.

    Cấu trúc:
    - output/court_001/1765189807/  -> thư mục timestamp
    - segments/court_001/1765190000/ -> thư mục timestamp
    - analysis/court_001/1765189807/ -> thư mục timestamp
    - uploads/court_001/1765189807.mp4 -> file timestamp

    Args:
        base_dir: Thư mục gốc (output, segments, analysis, uploads)
        cutoff_ts: Timestamp ngưỡng

    Returns:
        (folders_to_delete, files_to_delete)
    """
    folders_to_delete = []
    files_to_delete = []

    if not os.path.exists(base_dir):
        return folders_to_delete, files_to_delete

    # Duyệt qua các court
    for court_id in os.listdir(base_dir):
        court_path = os.path.join(base_dir, court_id)
        if not os.path.isdir(court_path):
            continue

        # Duyệt qua các item trong court
        for item in os.listdir(court_path):
            item_path = os.path.join(court_path, item)

            if os.path.isdir(item_path):
                # Thư mục: kiểm tra tên có phải timestamp không
                if is_timestamp_folder(item):
                    ts = get_timestamp_from_name(item)
                    if ts < cutoff_ts:
                        folders_to_delete.append(item_path)

            elif os.path.isfile(item_path):
                # File: kiểm tra tên có phải timestamp không
                if is_timestamp_file(item):
                    ts = get_timestamp_from_name(item)
                    if ts < cutoff_ts:
                        files_to_delete.append(item_path)

    return folders_to_delete, files_to_delete


def cleanup_directory(base_dir: str, cutoff_ts: int, dry_run: bool = False) -> Tuple[int, int, int]:
    """
    Dọn dẹp một thư mục.

    Args:
        base_dir: Thư mục gốc
        cutoff_ts: Timestamp ngưỡng
        dry_run: Nếu True, chỉ hiển thị không xóa thật

    Returns:
        (số folder xóa, số file xóa, tổng kích thước xóa bytes)
    """
    folders, files = find_old_data(base_dir, cutoff_ts)

    total_size = 0
    folders_deleted = 0
    files_deleted = 0

    # Xóa folders
    for folder_path in folders:
        folder_size = get_folder_size(folder_path)
        total_size += folder_size
        ts = get_timestamp_from_name(os.path.basename(folder_path))

        log(f"  {'[DRY-RUN] ' if dry_run else ''}DELETE folder: {folder_path}")
        log(f"    Timestamp: {format_timestamp(ts)} | Size: {format_size(folder_size)}")

        if not dry_run:
            try:
                shutil.rmtree(folder_path)
                folders_deleted += 1
            except Exception as e:
                log(f"    ERROR: {str(e)}")

    # Xóa files
    for file_path in files:
        file_size = os.path.getsize(file_path)
        total_size += file_size
        ts = get_timestamp_from_name(os.path.basename(file_path))

        log(f"  {'[DRY-RUN] ' if dry_run else ''}DELETE file: {file_path}")
        log(f"    Timestamp: {format_timestamp(ts)} | Size: {format_size(file_size)}")

        if not dry_run:
            try:
                os.remove(file_path)
                files_deleted += 1
            except Exception as e:
                log(f"    ERROR: {str(e)}")

    if dry_run:
        return len(folders), len(files), total_size
    return folders_deleted, files_deleted, total_size


def run_cleanup(days: int = DEFAULT_RETENTION_DAYS, dry_run: bool = False):
    """
    Chạy dọn dẹp tất cả các thư mục.

    Args:
        days: Số ngày giữ lại
        dry_run: Nếu True, chỉ hiển thị không xóa thật
    """
    cutoff_ts = get_cutoff_timestamp(days)
    cutoff_date = datetime.fromtimestamp(cutoff_ts)

    log("=" * 70)
    log(f"DATA CLEANER - {'DRY RUN' if dry_run else 'CLEANUP'}")
    log("=" * 70)
    log(f"Retention period: {days} days")
    log(f"Cutoff date: {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Cutoff timestamp: {cutoff_ts}")
    log(f"Directories: {CLEANUP_DIRS}")
    log("=" * 70 + "\n")

    total_folders = 0
    total_files = 0
    total_size = 0

    for dir_name in CLEANUP_DIRS:
        log(f"\n>>> Processing: {dir_name}/")

        if not os.path.exists(dir_name):
            log(f"  Directory not found, skipping...")
            continue

        folders, files, size = cleanup_directory(dir_name, cutoff_ts, dry_run)
        total_folders += folders
        total_files += files
        total_size += size

        log(f"  Subtotal: {folders} folders, {files} files, {format_size(size)}")

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"{'Would delete' if dry_run else 'Deleted'}:")
    log(f"  - Folders: {total_folders}")
    log(f"  - Files: {total_files}")
    log(f"  - Total size: {format_size(total_size)}")
    log("=" * 70 + "\n")


def scheduled_cleanup():
    """Job chạy theo schedule"""
    log("\n>>> Scheduled cleanup triggered")
    run_cleanup(days=DEFAULT_RETENTION_DAYS, dry_run=False)


def run_daemon():
    """Chạy daemon tự động dọn dẹp hàng ngày"""
    log("=" * 70)
    log("DATA CLEANER DAEMON STARTED")
    log(f"Auto cleanup at: {AUTO_RUN_TIME} daily")
    log(f"Retention period: {DEFAULT_RETENTION_DAYS} days")
    log("=" * 70 + "\n")

    # Schedule chạy hàng ngày
    schedule.every().day.at(AUTO_RUN_TIME).do(scheduled_cleanup)

    log(f"Daemon running. Press Ctrl+C to stop.")
    log(f"Next cleanup at: {AUTO_RUN_TIME}\n")

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        log("\nDaemon stopped by user.")
        sys.exit(0)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Data Cleaner - Tự động xóa dữ liệu cũ"
    )

    parser.add_argument(
        "--days", "-d",
        type=int,
        default=DEFAULT_RETENTION_DAYS,
        help=f"Số ngày giữ lại dữ liệu (mặc định: {DEFAULT_RETENTION_DAYS})"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ hiển thị dữ liệu sẽ bị xóa, không xóa thật"
    )

    parser.add_argument(
        "--daemon",
        action="store_true",
        help=f"Chạy daemon tự động dọn dẹp hàng ngày lúc {AUTO_RUN_TIME}"
    )

    args = parser.parse_args()

    if args.daemon:
        run_daemon()
    else:
        run_cleanup(days=args.days, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
