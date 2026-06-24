import { renderCamera } from "./camera.js";
import { updateChart } from "./chart.js";
import { renderMetrics } from "./metrics.js";
import { renderPersonTable } from "./persons.js";
import { renderHealth } from "./health.js";
import { renderAlert } from "./alert.js";
import { applyDashboardUpdate } from "./state.js";

function renderClock() {
  document.getElementById("clock").textContent = new Date().toLocaleTimeString();
}

function renderDashboard() {
  renderClock();
  renderCamera();
  renderMetrics();
  renderPersonTable();
  renderHealth();
  renderAlert();
}

window.addEventListener("resize", updateChart);

// Your Python WebSocket code can call:
// window.updateDashboard({ people, frozen, fps })
//
// people: [{ id, x, y, activity, conf, since }, ...]
window.updateDashboard = (data) => {
  applyDashboardUpdate(data);
  renderDashboard();
};

renderDashboard();
setInterval(renderDashboard, 1000);
