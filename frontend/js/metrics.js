import { CLASSES, COLORS, LABELS } from "./config.js";
import { dashboardState } from "./state.js";
import { updateChart } from "./chart.js";

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

  const score = Math.max(0, Math.min(100, Math.round(dashboardState.focusScore || 0)));
  document.getElementById("focus-score").textContent = dashboardState.frozen ? "--" : `${score}%`;

  const focusBar = document.getElementById("focus-bar");
  focusBar.style.width = dashboardState.frozen ? "0%" : `${score}%`;
  focusBar.style.background =
    score >= 60 ? "#1D9E75" : score >= 35 ? "#BA7517" : "#A32D2D";

  const now = Date.now();
  if (!dashboardState.frozen && now - dashboardState.lastHistoryAt >= 1000) {
    dashboardState.history.push(counts);
    dashboardState.lastHistoryAt = now;
    if (dashboardState.history.length > dashboardState.maxHistory) {
      dashboardState.history.shift();
    }
  }

  updateChart();
}
