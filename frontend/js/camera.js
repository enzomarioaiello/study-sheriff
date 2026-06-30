import { dashboardState } from "./state.js";

export function renderCamera() {
  const freezeBanner = document.getElementById("freeze-banner");
  const freezeText = document.getElementById("freeze-text");
  const recDot = document.getElementById("rec-dot");
  const liveLabel = document.getElementById("live-label");
  const fpsBadge = document.getElementById("fps-badge");
  const runState = document.getElementById("run-state");

  const status = dashboardState.status || "unknown";
  const hasError = Boolean(dashboardState.errorMessage);
  const showBanner = dashboardState.frozen || hasError;
  const framesAreRunning = !dashboardState.frozen && status === "running";

  freezeBanner.style.display = showBanner ? "flex" : "none";
  freezeText.textContent =
    status === "camera_stale"
      ? "Camera Offline / Feed Stale"
      : dashboardState.errorMessage || "Camera signal lost - last frame frozen";
  recDot.style.background =
    status === "running" ? "#22c55e" : "#888780";
  liveLabel.textContent = status === "camera_stale" ? "OFFLINE" : status.toUpperCase();
  fpsBadge.textContent = dashboardState.frozen
    ? "FROZEN"
    : `${dashboardState.fps.toFixed(1)} FPS`;

  runState.classList.toggle("running", framesAreRunning);
  runState.textContent = framesAreRunning ? "RUNNING" : "NOT RUNNING";
}
