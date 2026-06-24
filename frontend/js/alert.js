import { dashboardState } from "./state.js";

export function renderAlert() {
  const alertBox = document.getElementById("alert-box");
  const alertText = document.getElementById("alert-text");

  if (dashboardState.frozen) {
    alertBox.classList.add("show");
    alertText.textContent =
      "Camera disconnect detected — feed frozen on last frame. Check the Pi camera ribbon cable and rpicam-vid process.";
  } else if (dashboardState.oddActive) {
    alertBox.classList.add("show");
    alertText.textContent =
      "Unknown activity rate elevated — one or more people are below the classifier confidence threshold. Review for a new behavior class.";
  } else {
    alertBox.classList.remove("show");
  }
}
