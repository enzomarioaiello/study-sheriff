import os


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
    try:
        from src.pose.pose_live import run_pose_pipeline

        model = os.environ.get("STUDY_SHERIFF_MODEL", DEFAULT_MODEL)
        run_pose_pipeline(model=model, on_result=lambda result: _publish_result(shared_state, result))
    except Exception as exc:
        status = _status_for_error(exc)
        shared_state.set_error(status, exc)


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

    if frame is not None:
        shared_state.update_frame(frame)
    shared_state.update_metadata(
        current_class=current_class,
        focus_score=focus_score,
        person_count=person_count,
        fps=round(float(result.get("fps") or 0.0), 1),
        latency_ms=round(float(result.get("latency_ms") or 0.0), 1),
        status="running",
        error_message=None,
    )


def _status_for_error(exc):
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(term in text for term in ("picamera", "camera", "libcamera")):
        return "camera_error"
    if any(term in text for term in ("hailo", "npu", "hef", "vdevice")):
        return "npu_error"
    return "pipeline_error"
