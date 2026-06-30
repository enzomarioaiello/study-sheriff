// The single source of truth for dashboard data.
// Your Python/WebSocket code can update it through window.updateDashboard(...).
export const dashboardState = {
  people: [],
  frozen: false,
  fps: 0,
  latencyMs: 0,
  currentClass: "unknown",
  focusScore: 0,
  personCount: 0,
  status: "starting",
  errorMessage: "",
  updatedAt: null,
  frameUpdatedAt: null,
  frameAgeSeconds: null,
  frameStaleAfterSeconds: 3,
  oddActive: false,
  history: [],
  maxHistory: 24,
  lastHistoryAt: 0,
};

function normalizeActivity(label) {
  const text = String(label || "unknown").toLowerCase();
  if (["active", "desk", "work", "study", "writing", "typing", "reading"].some((term) => text.includes(term))) {
    return "deskwork";
  }
  if (text.includes("phone")) return "phone";
  if (["talking", "turned", "neighbour", "neighbor"].some((term) => text.includes(term))) {
    return "talking";
  }
  if (["resting", "rest", "head_down", "sleep"].some((term) => text.includes(term))) {
    return "resting";
  }
  if (["walking", "absent", "standing", "no_person"].some((term) => text.includes(term))) {
    return "absent";
  }
  return "unknown";
}

function buildPeopleFromMetadata(data) {
  if (data.status === "camera_stale") return [];

  const count = Number(data.person_count || 0);
  if (count <= 0) return [];

  const activity = normalizeActivity(data.current_class);
  const now = Date.now();
  return Array.from({ length: count }, (_, index) => {
    const id = `P${index + 1}`;
    const previous = dashboardState.people.find((person) => person.id === id);
    const sameActivity = previous && previous.activity === activity;
    return {
      id,
      activity,
      conf: activity === "unknown" ? 0.3 : 1,
      since: sameActivity ? previous.since : now,
    };
  });
}

export function applyDashboardUpdate(data = {}) {
  const hasBackendState =
    data.current_class !== undefined ||
    data.focus_score !== undefined ||
    data.person_count !== undefined ||
    data.status !== undefined;

  if (hasBackendState) {
    dashboardState.currentClass = data.current_class || "unknown";
    dashboardState.focusScore = Number(data.focus_score || 0);
    dashboardState.personCount = Number(data.person_count || 0);
    dashboardState.fps = Number(data.fps || 0);
    dashboardState.latencyMs = Number(data.latency_ms || 0);
    dashboardState.status = data.status || "unknown";
    dashboardState.errorMessage = data.error_message || "";
    dashboardState.updatedAt = data.updated_at || null;
    dashboardState.frameUpdatedAt = data.frame_updated_at || null;
    dashboardState.frameAgeSeconds =
      data.frame_age_seconds === null || data.frame_age_seconds === undefined
        ? null
        : Number(data.frame_age_seconds);
    dashboardState.frameStaleAfterSeconds = Number(data.frame_stale_after_seconds || 3);
    dashboardState.people = buildPeopleFromMetadata(data);
    dashboardState.frozen = [
      "camera_stale",
      "camera_error",
      "npu_error",
      "pipeline_error",
      "dashboard_error",
    ].includes(dashboardState.status);
  } else {
    if (data.people !== undefined) dashboardState.people = data.people;
    if (data.frozen !== undefined) dashboardState.frozen = data.frozen;
    if (data.fps !== undefined) dashboardState.fps = data.fps;
  }

  dashboardState.oddActive = dashboardState.people.some(
    (person) => person.activity === "unknown"
  );
}
