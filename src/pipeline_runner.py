import math
import os
import time


DEFAULT_MODEL = os.path.join("models", "yolov8s_pose.hef")


def score_for_label(label):
    text = str(label or "unknown").lower()
    if any(term in text for term in ("active", "desk", "work", "study", "writing", "typing", "reading")):
        return 100
    if "phone" in text:
        return 40
    if any(term in text for term in ("talking", "turned", "neighbour", "neighbor")):
        return 50
    if any(term in text for term in ("resting", "rest", "head_down", "sleep")):
        return 20
    if any(term in text for term in ("walking", "absent", "standing", "no_person")):
        return 0
    if "unknown" in text:
        return 30
    if "person" in text:
        return 30
    return 30


def compute_focus_score(labels):
    labels = [label for label in labels if label is not None]
    if not labels:
        return 0
    scores = [score_for_label(label) for label in labels]
    return round(sum(scores) / len(scores))


def run_pipeline(shared_state):
    if os.environ.get("STUDY_SHERIFF_MOCK") == "1":
        run_mock_pipeline(shared_state)
        return

    try:
        from src.pose.pose_live import run_pose_pipeline

        model = os.environ.get("STUDY_SHERIFF_MODEL", DEFAULT_MODEL)
        run_pose_pipeline(model=model, on_result=lambda result: _publish_result(shared_state, result))
    except Exception as exc:
        status = _status_for_error(exc)
        shared_state.set_error(status, exc)
        run_mock_pipeline(shared_state, error_message=f"{status}: {exc}")


def run_mock_pipeline(shared_state, error_message=None):
    shared_state.update_metadata(
        current_class="mock",
        focus_score=30,
        person_count=1,
        fps=0.0,
        latency_ms=0.0,
        status="mock",
        error_message=error_message,
    )

    try:
        import cv2
        import numpy as np
    except ImportError:
        _run_metadata_only_mock(shared_state, error_message)
        return

    start = time.time()
    frame_idx = 0
    while True:
        now = time.time()
        elapsed = now - start
        activity = _mock_activity(elapsed)
        focus_score = compute_focus_score([activity])
        fps = 12.5 + math.sin(elapsed / 2.0) * 1.5
        latency_ms = 1000.0 / max(fps, 1.0)

        frame = np.zeros((480, 854, 3), dtype=np.uint8)
        frame[:] = (28, 42, 54)
        x = int(240 + math.sin(elapsed) * 130)
        y = int(245 + math.cos(elapsed * 0.7) * 36)
        cv2.rectangle(frame, (x - 82, y - 150), (x + 82, y + 118), (46, 204, 113), 2)
        cv2.circle(frame, (x, y - 105), 32, (74, 144, 226), -1)
        cv2.line(frame, (x, y - 70), (x, y + 50), (245, 245, 245), 4)
        cv2.line(frame, (x, y - 35), (x - 65, y + 20), (245, 245, 245), 3)
        cv2.line(frame, (x, y - 35), (x + 65, y + 20), (245, 245, 245), 3)
        cv2.putText(frame, "StudySheriff mock camera", (32, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"class: {activity}", (32, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (220, 235, 248), 2)
        cv2.putText(frame, f"focus: {focus_score}  fps: {fps:.1f}", (32, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (220, 235, 248), 2)
        if error_message:
            cv2.putText(frame, str(error_message)[:80], (32, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (110, 190, 255), 1)

        shared_state.update_frame(frame)
        shared_state.update_metadata(
            current_class=activity,
            focus_score=focus_score,
            person_count=1,
            fps=round(fps, 1),
            latency_ms=round(latency_ms, 1),
            status="mock",
            error_message=error_message,
        )
        frame_idx += 1
        time.sleep(0.08)


def _publish_result(shared_state, result):
    frame = result.get("annotated_frame")
    if frame is None:
        frame = result.get("frame")
    person_count = int(result.get("person_count") or 0)
    labels = result.get("per_person_classes") or result.get("classes") or []
    if isinstance(labels, str):
        labels = [labels]
    current_class = result.get("current_class")
    if not current_class:
        current_class = labels[0] if labels else ("person" if person_count else "no_person")

    focus_score = result.get("focus_score")
    if focus_score is None:
        focus_score = compute_focus_score(labels or [current_class])

    persons = result.get("persons") or []

    if frame is not None:
        shared_state.update_frame(frame)
    shared_state.update_metadata(
        current_class=current_class,
        focus_score=focus_score,
        person_count=person_count,
        persons=persons,
        fps=round(float(result.get("fps") or 0.0), 1),
        latency_ms=round(float(result.get("latency_ms") or 0.0), 1),
        status="running",
        error_message=None,
    )


def _mock_activity(elapsed):
    sequence = ("active desk work", "reading", "phone use", "talking", "resting")
    return sequence[int(elapsed // 5) % len(sequence)]


def _run_metadata_only_mock(shared_state, error_message):
    start = time.time()
    while True:
        activity = _mock_activity(time.time() - start)
        shared_state.update_metadata(
            current_class=activity,
            focus_score=compute_focus_score([activity]),
            person_count=1,
            fps=0.0,
            latency_ms=0.0,
            status="mock",
            error_message=error_message or "OpenCV is unavailable; serving fallback frame.",
        )
        time.sleep(0.5)


def _status_for_error(exc):
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(term in text for term in ("picamera", "camera", "libcamera")):
        return "camera_error"
    if any(term in text for term in ("hailo", "npu", "hef", "vdevice")):
        return "npu_error"
    return "pipeline_error"
