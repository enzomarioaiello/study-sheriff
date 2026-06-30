import { dashboardState } from "./state.js";

function formatUpdatedAt(value) {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleTimeString();
}

export function renderHealth() {
  const frameAge =
    dashboardState.frameAgeSeconds === null
      ? "--"
      : `${dashboardState.frameAgeSeconds.toFixed(1)} s`;
  const rows = [
    ["Pipeline status", dashboardState.status || "unknown"],
    ["Current class", dashboardState.currentClass || "unknown"],
    ["Person count", String(dashboardState.personCount || 0)],
    ["Focus score", dashboardState.frozen ? "--" : `${Math.round(dashboardState.focusScore || 0)}%`],
    ["Pipeline FPS", dashboardState.frozen ? "0.0" : dashboardState.fps.toFixed(1)],
    ["Inference latency", `${dashboardState.latencyMs.toFixed(1)} ms/frame`],
    ["Last frame", formatUpdatedAt(dashboardState.frameUpdatedAt)],
    ["Frame age", frameAge],
    ["Last update", formatUpdatedAt(dashboardState.updatedAt)],
  ];

  if (dashboardState.errorMessage) {
    rows.push(["Error", dashboardState.errorMessage]);
  }

  document.getElementById("health-table").innerHTML = rows
    .map(
      ([label, value]) => `
        <tr>
          <td style="color:var(--text2);">${label}</td>
          <td class="right">${value}</td>
        </tr>
      `
    )
    .join("");
}
