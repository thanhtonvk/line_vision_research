"""
API Download Files
==================

API endpoints để download các file đã phân tích:
- GET /analysis/{path}: Download file từ thư mục analysis
- GET /segments/{path}: Download file từ thư mục segments
- GET /output/{path}: Download file từ thư mục output

Chạy server:
    uvicorn api_download:app --host 0.0.0.0 --port 8001
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="LineVision Download API",
    description="API download files phân tích tennis",
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


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "LineVision Download API",
        "version": "1.0.0",
        "endpoints": {
            "analysis": "/analysis/{path}",
            "segments": "/segments/{path}",
            "output": "/output/{path}"
        }
    }


# ==============================================================================
# STATIC FILE ENDPOINTS
# ==============================================================================

# Tạo thư mục nếu chưa có
os.makedirs("analysis", exist_ok=True)
os.makedirs("segments", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Mount static files
app.mount("/analysis", StaticFiles(directory="analysis"), name="analysis")
app.mount("/segments", StaticFiles(directory="segments"), name="segments")
app.mount("/output", StaticFiles(directory="output"), name="output")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
