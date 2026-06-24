import { dashboardState } from "./state.js";

export function renderHealth() {
  const rows = [
    ["NPU model", "yolov8s_pose.hef + classifier"],
    [
      "Inference latency",
      dashboardState.frozen
        ? "--"
        : `${(62 + Math.random() * 8).toFixed(0)} ms/frame`,
    ],
    ["Pipeline FPS", dashboardState.frozen ? "0.0" : dashboardState.fps.toFixed(1)],
    [
      "Camera",
      dashboardState.frozen ? "Disconnected" : "Camera Module 3 OK",
    ],
    [
      "Unknown rate (5 min)",
      `${(
        dashboardState.oddActive
          ? 18 + Math.random() * 6
          : 2 + Math.random() * 2
      ).toFixed(0)}%`,
    ],
  ];

  document.getElementById("health-table").innerHTML = rows
    .map(
      ([label, value]) => `
        <tr>
          <td style="color:var(--text2);">${label}</td>
          <td class="right">${value}</td>
        </tr>
      `
    )
    .join("");
}
