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

window.updateDashboard = (data) => {
  applyDashboardUpdate(data);
  renderDashboard();
};

async function pollState() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    window.updateDashboard(await response.json());
  } catch (error) {
    window.updateDashboard({
      status: "dashboard_error",
      error_message: error.message,
      updated_at: new Date().toISOString(),
    });
  }
}

renderDashboard();
pollState();
setInterval(renderDashboard, 1000);
setInterval(pollState, 400);
