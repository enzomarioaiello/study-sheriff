import base64
import threading
from datetime import datetime, timezone

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


_BLANK_JPEG = base64.b64decode(
    b"/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8X"
    b"GBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARC"
    b"AABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEG"
    b"E1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWG"
    b"h4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQ"
    b"EBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRC"
    b"hYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmq"
    b"srO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD4cooor0zzz//Z"
)
_UNSET = object()


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


class DashboardState:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame_bytes = None
        self._metadata = {
            "current_class": "unknown",
            "focus_score": 0,
            "person_count": 0,
            "fps": 0.0,
            "latency_ms": 0.0,
            "status": "starting",
            "error_message": None,
            "updated_at": _now_iso(),
        }

    def update_frame(self, frame_bgr, jpeg_quality=70, max_width=854):
        if cv2 is None or np is None or frame_bgr is None:
            return

        frame = frame_bgr
        height, width = frame.shape[:2]
        if width > max_width:
            scale = max_width / float(width)
            frame = cv2.resize(
                frame,
                (max_width, max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not ok:
            return

        with self._lock:
            self._frame_bytes = encoded.tobytes()
            self._metadata["updated_at"] = _now_iso()

    def update_metadata(
        self,
        current_class=_UNSET,
        focus_score=_UNSET,
        person_count=_UNSET,
        fps=_UNSET,
        latency_ms=_UNSET,
        status=_UNSET,
        error_message=_UNSET,
    ):
        updates = {
            "current_class": current_class,
            "focus_score": focus_score,
            "person_count": person_count,
            "fps": fps,
            "latency_ms": latency_ms,
            "status": status,
            "error_message": error_message,
        }
        with self._lock:
            for key, value in updates.items():
                if value is not _UNSET:
                    self._metadata[key] = value
            self._metadata["updated_at"] = _now_iso()

    def get_snapshot(self):
        with self._lock:
            return dict(self._metadata)

    def get_frame(self):
        with self._lock:
            frame = self._frame_bytes
            status = self._metadata.get("status") or "starting"
            error_message = self._metadata.get("error_message")
        if frame is not None:
            return frame
        return self._fallback_frame(status, error_message)

    def set_error(self, status, error_message):
        self.update_metadata(status=status, error_message=str(error_message))

    def _fallback_frame(self, status, error_message=None):
        if cv2 is None or np is None:
            return _BLANK_JPEG

        frame = np.zeros((480, 854, 3), dtype=np.uint8)
        frame[:] = (35, 45, 56)
        title = "Waiting for camera..."
        detail = error_message or status
        cv2.putText(
            frame,
            title,
            (52, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            str(detail)[:80],
            (52, 272),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (180, 205, 230),
            1,
            cv2.LINE_AA,
        )
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            return _BLANK_JPEG
        return encoded.tobytes()


shared_state = DashboardState()
