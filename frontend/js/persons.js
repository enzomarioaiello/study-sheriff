import { COLORS, LABELS, UNKNOWN_COLOR } from "./config.js";
import { dashboardState } from "./state.js";

function formatElapsedTime(milliseconds) {
  const totalSeconds = Math.floor(milliseconds / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds < 10 ? "0" : ""}${seconds}`;
}

export function renderPersonTable() {
  document.getElementById("person-count-label").textContent =
    dashboardState.frozen ? "feed stale" : `${dashboardState.people.length} detected`;

  const now = Date.now();
  const personTable = document.getElementById("person-table");

  if (dashboardState.frozen) {
    personTable.innerHTML = `
      <tr>
        <td colspan="4" style="color:var(--text2);">
          Occupancy and activity results paused until a new frame arrives.
        </td>
      </tr>
    `;
    return;
  }

  personTable.innerHTML = dashboardState.people
    .map((person) => {
      const isUnknown = person.activity === "unknown";
      const color = isUnknown ? UNKNOWN_COLOR : COLORS[person.activity];
      const label = isUnknown ? "Unknown" : LABELS[person.activity];

      return `
        <tr>
          <td>${person.id}</td>
          <td>
            <span class="pname">
              <span class="dotsq" style="background:${color};"></span>
              ${label}
            </span>
          </td>
          <td>${Math.round(person.conf * 100)}%</td>
          <td class="right" style="color:var(--text2);">
            ${formatElapsedTime(now - person.since)}
          </td>
        </tr>
      `;
    })
    .join("");
}
