import { dashboardState } from "./state.js";

export function renderCamera() {
  const freezeBanner = document.getElementById("freeze-banner");
  const freezeText = document.getElementById("freeze-text");
  const recDot = document.getElementById("rec-dot");
  const liveLabel = document.getElementById("live-label");
  const fpsBadge = document.getElementById("fps-badge");
  const frameFreshness = document.getElementById("frame-freshness");

  const status = dashboardState.status || "unknown";
  const hasError = Boolean(dashboardState.errorMessage);
  const showBanner = dashboardState.frozen || hasError;
  const frameAge = dashboardState.frameAgeSeconds;
  const hasFrameAge = frameAge !== null && frameAge !== undefined && Number.isFinite(frameAge);

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

  frameFreshness.classList.toggle("stale", status === "camera_stale");
  frameFreshness.textContent =
    status === "camera_stale"
      ? `STALE ${hasFrameAge ? frameAge.toFixed(1) : "--"}s`
      : `FRAME OK ${hasFrameAge ? frameAge.toFixed(1) : "--"}s`;
}
