import { CLASSES, COLORS, LABELS, UNKNOWN_COLOR } from "./config.js";
import { dashboardState } from "./state.js";

function formatDuration(since) {
  if (!Number.isFinite(since)) return "--";
  const seconds = Math.max(0, Math.floor((Date.now() - since) / 1000));
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

export function renderMetrics() {
  const counts = {
    deskwork: 0,
    talking: 0,
    phone: 0,
    resting: 0,
    absent: 0,
    unknown: 0,
  };

  dashboardState.people.forEach((person) => {
    counts[person.activity] = (counts[person.activity] || 0) + 1;
  });

  const peopleList = document.getElementById("people-activity-list");
  peopleList.innerHTML = dashboardState.people.length
    ? dashboardState.people
        .map((person) => {
          const label = LABELS[person.activity] || "Unknown activity";
          const color = COLORS[person.activity] || UNKNOWN_COLOR;
          return `
            <div class="person-activity">
              <span class="person-id">${person.id}</span>
              <span class="person-state">
                <span class="dotsq" style="background:${color};"></span>
                ${label}
              </span>
              <span class="person-duration">${formatDuration(person.since)}</span>
            </div>
          `;
        })
        .join("")
    : '<p class="empty-activity">No people detected</p>';

  document.getElementById("metric-grid").innerHTML = CLASSES.map(
    (activity) => `
      <div class="metric">
        <div class="lbl">
          <span class="dotsq" style="background:${COLORS[activity]};"></span>
          ${LABELS[activity]}
        </div>
        <div class="val">${counts[activity]}</div>
      </div>
    `
  ).join("");
}
