import { COLORS, UNKNOWN_COLOR } from "./config.js";
import { dashboardState } from "./state.js";

export function renderCamera() {
  const svg = document.getElementById("camsvg");
  const freezeBanner = document.getElementById("freeze-banner");

  freezeBanner.style.display = dashboardState.frozen ? "flex" : "none";
  document.getElementById("rec-dot").style.background = dashboardState.frozen
    ? "#888780"
    : "#A32D2D";
  document.getElementById("fps-badge").textContent = dashboardState.frozen
    ? "FROZEN"
    : `${dashboardState.fps.toFixed(1)} FPS`;

  const parts = [];
  parts.push('<rect x="0" y="0" width="640" height="400" fill="#09213d"/>');
  parts.push('<rect x="40" y="320" width="180" height="14" fill="#163554"/>');
  parts.push('<rect x="430" y="40" width="14" height="200" fill="#163554"/>');

  for (let gridX = 0; gridX <= 640; gridX += 80) {
    parts.push(
      `<line x1="${gridX}" y1="0" x2="${gridX}" y2="400" stroke="#2b4e72" stroke-width="1"/>`
    );
  }

  for (let gridY = 0; gridY <= 400; gridY += 80) {
    parts.push(
      `<line x1="0" y1="${gridY}" x2="640" y2="${gridY}" stroke="#2b4e72" stroke-width="1"/>`
    );
  }

  dashboardState.people.forEach((person) => {
    const isUnknown = person.activity === "unknown";
    const color = isUnknown ? UNKNOWN_COLOR : COLORS[person.activity];
    const boxWidth = 46;
    const boxHeight = 70;
    const boxX = person.x - boxWidth / 2;
    const boxY = person.y - boxHeight / 2;

    parts.push(
      `<rect x="${boxX}" y="${boxY}" width="${boxWidth}" height="${boxHeight}" fill="none" stroke="${color}" stroke-width="2" rx="3"/>`
    );
    parts.push(
      `<circle cx="${person.x}" cy="${person.y - boxHeight / 2 + 10}" r="6" fill="none" stroke="${color}" stroke-width="1.5"/>`
    );
    parts.push(
      `<line x1="${person.x}" y1="${person.y - boxHeight / 2 + 16}" x2="${person.x}" y2="${person.y + 8}" stroke="${color}" stroke-width="1.5"/>`
    );
    parts.push(
      `<line x1="${person.x}" y1="${person.y - 12}" x2="${person.x - 14}" y2="${person.y + 4}" stroke="${color}" stroke-width="1.5"/>`
    );
    parts.push(
      `<line x1="${person.x}" y1="${person.y - 12}" x2="${person.x + 14}" y2="${person.y + 4}" stroke="${color}" stroke-width="1.5"/>`
    );
    parts.push(
      `<line x1="${person.x}" y1="${person.y + 8}" x2="${person.x - 12}" y2="${person.y + boxHeight / 2}" stroke="${color}" stroke-width="1.5"/>`
    );
    parts.push(
      `<line x1="${person.x}" y1="${person.y + 8}" x2="${person.x + 12}" y2="${person.y + boxHeight / 2}" stroke="${color}" stroke-width="1.5"/>`
    );

    const label = isUnknown ? "unknown" : person.activity;
    const textWidth = label.length * 6.4 + 10;
    parts.push(
      `<rect x="${boxX}" y="${boxY - 16}" width="${textWidth}" height="15" fill="${color}"/>`
    );
    parts.push(
      `<text x="${boxX + 5}" y="${boxY - 5}" font-size="11" fill="#ffffff" font-family="-apple-system,sans-serif">${person.id} ${label}</text>`
    );
  });

  svg.innerHTML = parts.join("");
}
