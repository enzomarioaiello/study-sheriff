import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from src.dashboard.state import shared_state
from src.pipeline_runner import run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_ROOT = PROJECT_ROOT / "frontend"

_pipeline_lock = threading.Lock()
_pipeline_thread = None


def _start_pipeline_once():
    global _pipeline_thread
    with _pipeline_lock:
        if _pipeline_thread and _pipeline_thread.is_alive():
            return
        _pipeline_thread = threading.Thread(
            target=run_pipeline,
            args=(shared_state,),
            name="study-sheriff-pipeline",
            daemon=True,
        )
        _pipeline_thread.start()


@asynccontextmanager
async def lifespan(app):
    _start_pipeline_once()
    yield


app = FastAPI(title="Study Sheriff Dashboard", lifespan=lifespan)
app.mount("/css", StaticFiles(directory=FRONTEND_ROOT / "css"), name="css")
app.mount("/js", StaticFiles(directory=FRONTEND_ROOT / "js"), name="js")


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = FRONTEND_ROOT / "html" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/video_feed")
async def video_feed():
    async def frames():
        while True:
            frame = shared_state.get_frame()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame)).encode("ascii") + b"\r\n\r\n"
                + frame
                + b"\r\n"
            )
            await asyncio.sleep(0.08)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/state")
def api_state():
    return JSONResponse(shared_state.get_snapshot())


@app.get("/health")
def health():
    snapshot = shared_state.get_snapshot()
    return {"ok": True, "status": snapshot.get("status", "unknown")}
