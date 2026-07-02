import { renderCamera } from "./camera.js";
import { renderMetrics } from "./metrics.js";
import { renderHealth } from "./health.js";
import { renderAlert } from "./alert.js";
import { applyDashboardUpdate } from "./state.js";
import { initializeDialogs } from "./dialogs.js";

function renderDashboard() {
  renderCamera();
  renderMetrics();
  renderHealth();
  renderAlert();
}

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

initializeDialogs();
renderDashboard();
pollState();
setInterval(renderDashboard, 1000);
setInterval(pollState, 400);
