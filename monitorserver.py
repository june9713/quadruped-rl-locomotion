import os
import mimetypes
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI(title="Video Monitor Server", description="비디오 스트리밍 서버")

# 비디오 폴더 경로 설정
VIDEO_DIR = Path("eval_videos_standing")

# 템플릿 설정
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """메인 페이지 - 비디오 목록 표시"""
    videos = get_video_list()
    return templates.TemplateResponse("index.html", {"request": request, "videos": videos})

@app.get("/api/videos")
async def get_videos():
    """비디오 목록을 JSON으로 반환"""
    return {"videos": get_video_list()}

def get_video_list() -> List[dict]:
    """비디오 폴더에서 비디오 파일 목록을 가져옴"""
    videos = []
    if VIDEO_DIR.exists():
        for file_path in VIDEO_DIR.glob("*.mp4"):
            videos.append({
                "filename": file_path.name,
                "size": file_path.stat().st_size,
                "path": str(file_path)
            })
    return videos

@app.get("/stream/{filename}")
async def stream_video(filename: str, request: Request):
    """비디오 파일을 스트리밍으로 제공"""
    video_path = VIDEO_DIR / filename
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="비디오 파일을 찾을 수 없습니다")
    
    # 파일 크기 확인
    file_size = video_path.stat().st_size
    
    # Range 헤더 처리
    range_header = request.headers.get("range")
    
    if range_header:
        # 부분 요청 처리
        start, end = parse_range_header(range_header, file_size)
        return create_partial_response(video_path, start, end, file_size)
    else:
        # 전체 파일 스트리밍
        return create_full_response(video_path, file_size)

def parse_range_header(range_header: str, file_size: int) -> tuple:
    """Range 헤더를 파싱하여 시작과 끝 위치를 반환"""
    try:
        range_str = range_header.replace("bytes=", "")
        start_str, end_str = range_str.split("-")
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
        return start, end
    except:
        return 0, file_size - 1

def create_partial_response(video_path: Path, start: int, end: int, file_size: int):
    """부분 응답 생성"""
    content_length = end - start + 1
    
    def iterfile():
        with open(video_path, "rb") as f:
            f.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk_size = min(8192, remaining)
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
                remaining -= len(chunk)
    
    return StreamingResponse(
        iterfile(),
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(content_length)
        }
    )

def create_full_response(video_path: Path, file_size: int):
    """전체 파일 응답 생성"""
    def iterfile():
        with open(video_path, "rb") as f:
            while chunk := f.read(8192):
                yield chunk
    
    return StreamingResponse(
        iterfile(),
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size)
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8898)
