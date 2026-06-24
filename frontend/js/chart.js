import { CLASSES, COLORS, LABELS } from "./config.js";
import { dashboardState } from "./state.js";

export function updateChart() {
  const canvas = document.getElementById("trendChart");
  const wrap = canvas.parentElement;
  const devicePixelRatio = window.devicePixelRatio || 1;
  const width = wrap.clientWidth;
  const height = wrap.clientHeight;

  canvas.width = width * devicePixelRatio;
  canvas.height = height * devicePixelRatio;

  const context = canvas.getContext("2d");
  context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  context.clearRect(0, 0, width, height);

  const paddingLeft = 24;
  const paddingRight = 6;
  const paddingTop = 8;
  const paddingBottom = 18;
  const plotWidth = width - paddingLeft - paddingRight;
  const plotHeight = height - paddingTop - paddingBottom;
  const maxY = Math.max(1, dashboardState.people.length);
  const pointCount = dashboardState.history.length;

  if (pointCount < 2) {
    buildLegend();
    return;
  }

  context.strokeStyle = "rgba(82,97,107,0.18)";
  context.fillStyle = "#64748b";
  context.font = "11px -apple-system, sans-serif";
  context.textAlign = "right";

  for (let gridY = 0; gridY <= maxY; gridY += 1) {
    const y = paddingTop + plotHeight - (gridY / maxY) * plotHeight;
    context.beginPath();
    context.moveTo(paddingLeft, y);
    context.lineTo(width - paddingRight, y);
    context.stroke();
    context.fillText(String(gridY), paddingLeft - 6, y + 3);
  }

  const xAt = (index) =>
    paddingLeft + (index / (pointCount - 1)) * plotWidth;
  const yAt = (value) => paddingTop + plotHeight - (value / maxY) * plotHeight;

  const runningTotals = new Array(pointCount).fill(0);

  CLASSES.forEach((activity) => {
    context.beginPath();

    for (let index = 0; index < pointCount; index += 1) {
      const value = dashboardState.history[index][activity] || 0;
      const top = runningTotals[index] + value;
      const x = xAt(index);
      const y = yAt(top);

      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }

    for (let index = pointCount - 1; index >= 0; index -= 1) {
      context.lineTo(xAt(index), yAt(runningTotals[index]));
    }

    context.closePath();
    context.fillStyle = `${COLORS[activity]}cc`;
    context.fill();

    for (let index = 0; index < pointCount; index += 1) {
      runningTotals[index] += dashboardState.history[index][activity] || 0;
    }
  });

  context.fillStyle = "#64748b";
  context.textAlign = "left";
  context.fillText(`-${(pointCount - 1) * 2}s`, paddingLeft, height - 4);
  context.textAlign = "right";
  context.fillText("now", width - paddingRight, height - 4);

  buildLegend();
}

function buildLegend() {
  const legend = document.getElementById("trend-legend");

  if (legend.dataset.built) return;

  legend.dataset.built = "1";
  legend.innerHTML = CLASSES.map(
    (activity) => `
      <span class="item">
        <span class="dotsq" style="background:${COLORS[activity]};"></span>
        ${LABELS[activity]}
      </span>
    `
  ).join("");
}
