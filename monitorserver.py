import os
import mimetypes
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# --- 설정 ---
# 비디오 파일이 있는 폴더 이름
VIDEO_DIR_NAME = "training_videos"
# 템플릿 파일이 있는 폴더 이름
TEMPLATE_DIR_NAME = "templates"

# 현재 파일 위치를 기준으로 절대 경로 생성
BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / VIDEO_DIR_NAME
TEMPLATE_DIR = BASE_DIR / TEMPLATE_DIR_NAME

# FastAPI 앱 생성
app = FastAPI(title="Video Monitor Server", description="비디오 스트리밍 서버")

# 디렉토리 존재 여부 확인 및 생성
if not VIDEO_DIR.is_dir():
    print(f"경고: 비디오 디렉토리 '{VIDEO_DIR}'를 찾을 수 없습니다. 폴더를 생성합니다.")
    VIDEO_DIR.mkdir(exist_ok=True)

if not TEMPLATE_DIR.is_dir():
    print(f"정보: 템플릿 디렉토리 '{TEMPLATE_DIR}'를 찾을 수 없습니다. 폴더를 생성합니다.")
    TEMPLATE_DIR.mkdir(exist_ok=True)

# 템플릿 설정
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


def get_video_list() -> List[dict]:
    """
    비디오 폴더에서 .mp4 파일 목록을 가져와 최신순으로 정렬합니다.
    """
    videos = []
    if VIDEO_DIR.exists():
        # glob으로 mp4 파일을 가져온 후, 수정 시간을 기준으로 내림차순 정렬
        video_paths = sorted(
            VIDEO_DIR.glob("*.mp4"), 
            key=lambda p: p.stat().st_mtime, 
            reverse=True
        )
        for file_path in video_paths:
            videos.append({
                "filename": file_path.name,
                "size": file_path.stat().st_size,
                "path": str(file_path)
            })
    return videos

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    메인 페이지를 렌더링하고 비디오 목록을 전달합니다.
    """
    videos = get_video_list()
    # index.html 파일이 없으면 기본 안내 메시지를 표시
    index_template_path = TEMPLATE_DIR / "index.html"
    if not index_template_path.exists():
        return HTMLResponse(content="<h1>서버 실행 중</h1><p>'templates/index.html' 파일을 생성해주세요.</p>", status_code=200)
    return templates.TemplateResponse("index.html", {"request": request, "videos": videos, "VIDEO_DIR_NAME": VIDEO_DIR_NAME})

@app.get("/api/videos")
async def get_videos_api():
    """
    비디오 목록을 JSON API 형태로 반환합니다.
    """
    return {"videos": get_video_list()}


def parse_range_header(range_header: str, file_size: int) -> tuple[int, int]:
    """
    Range 헤더를 파싱하여 시작과 끝 위치를 반환합니다.
    """
    try:
        range_str = range_header.strip().lower().replace("bytes=", "")
        start_str, end_str = range_str.split("-")
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1

        # 유효성 검사
        if start >= file_size or end >= file_size or start > end:
            raise ValueError("Invalid range")
        
        return start, end
    except ValueError:
        raise HTTPException(status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE)


def create_partial_response(video_path: Path, start: int, end: int, file_size: int):
    """
    HTTP 206 Partial Content 응답을 생성합니다.
    """
    content_length = end - start + 1
    
    def iterfile():
        with open(video_path, "rb") as f:
            f.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk_size = min(65536, remaining) # 64KB chunk
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
                remaining -= len(chunk)
    
    return StreamingResponse(
        iterfile(),
        status_code=status.HTTP_206_PARTIAL_CONTENT,
        media_type="video/mp4",
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
        }
    )

def create_full_response(video_path: Path, file_size: int):
    """
    전체 파일을 스트리밍하는 응답을 생성합니다. (주로 다운로드용)
    """
    def iterfile():
        with open(video_path, "rb") as f:
            while chunk := f.read(65536): # 64KB chunk
                yield chunk
    
    return StreamingResponse(
        iterfile(),
        media_type="video/mp4",
        headers={
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
        }
    )

@app.get("/stream/{filename}")
async def stream_video(filename: str, request: Request):
    """
    요청 헤더를 분석하여 비디오 파일을 전체 또는 부분 스트리밍으로 제공합니다.
    """
    video_path = VIDEO_DIR / filename
    
    if not video_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="비디오 파일을 찾을 수 없습니다.")
    
    file_size = video_path.stat().st_size
    range_header = request.headers.get("range")
    
    if range_header:
        start, end = parse_range_header(range_header, file_size)
        return create_partial_response(video_path, start, end, file_size)
    else:
        # Range 헤더가 없는 경우, 전체 파일을 스트리밍합니다.
        # 브라우저는 보통 이 응답을 받고 Range 요청을 다시 보냅니다.
        return create_full_response(video_path, file_size)


if __name__ == "__main__":
    # uvicorn.run에 '파일이름:앱객체' 형식으로 전달
    # 파일 이름이 monitorserver.py이므로 "monitorserver:app"으로 변경
    uvicorn.run("monitorserver:app", host="0.0.0.0", port=8898, reload=True)
