// The single source of truth for dashboard data.
// Your Python/WebSocket code can update it through window.updateDashboard(...).
export const dashboardState = {
  people: [],
  frozen: false,
  fps: 0,
  oddActive: false,
  history: [],
  maxHistory: 24,
};

export function applyDashboardUpdate(data = {}) {
  if (data.people !== undefined) dashboardState.people = data.people;
  if (data.frozen !== undefined) dashboardState.frozen = data.frozen;
  if (data.fps !== undefined) dashboardState.fps = data.fps;

  dashboardState.oddActive = dashboardState.people.some(
    (person) => person.activity === "unknown"
  );
}
