# Terminal 1: API Analyzer
uvicorn api_analyzer:app --host 0.0.0.0 --port 8000 --workers 4

# Terminal 2: API Download
uvicorn api_download:app --host 0.0.0.0 --port 8001

# Terminal 3: Segment Scheduler
python segment_scheduler.py

# Terminal 4: Data Cleaner (daemon)
python data_cleaner.py --daemon
