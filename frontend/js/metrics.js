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

  const total = dashboardState.people.length;
  const focusWeight =
    counts.deskwork * 1.0 +
    counts.talking * 0.5 +
    counts.phone * 0.1 +
    counts.resting * 0.2 +
    counts.absent * 0 +
    (counts.unknown || 0) * 0.3;

  const score = total ? Math.round((focusWeight / total) * 100) : 0;
  document.getElementById("focus-score").textContent = `${score}%`;

  const focusBar = document.getElementById("focus-bar");
  focusBar.style.width = `${score}%`;
  focusBar.style.background =
    score >= 60 ? "#1D9E75" : score >= 35 ? "#BA7517" : "#A32D2D";

  dashboardState.history.push(counts);
  if (dashboardState.history.length > dashboardState.maxHistory) {
    dashboardState.history.shift();
  }

  updateChart();
}
