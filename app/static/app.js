"use strict";

const $ = (id) => document.getElementById(id);
const fmt = (value, digits = 3) => (value == null ? "–" : Number(value).toFixed(digits));
const clamp01 = (value) => Math.min(1, Math.max(0, value));

const ICONS = {
  dup: '<svg viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  notdup: '<svg viewBox="0 0 24 24"><path d="M7 7l10 10M17 7L7 17" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>',
};

let models = [];

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function setStatus(state, text) {
  const status = $("status");
  status.classList.remove("ready", "error");
  if (state) status.classList.add(state);
  $("status-text").textContent = text;
}

function renderRows(results) {
  const bySlug = new Map((results || []).map((r) => [r.slug, r]));
  $("model-rows").innerHTML = models
    .map((model) => {
      const result = bySlug.get(model.slug);
      const tag = model.is_best ? '<span class="tag">Best</span>' : "";
      let score = '<span class="pending">Waiting for a pair</span>';
      let prediction = '<span class="pending">–</span>';
      if (result) {
        const kind = result.is_duplicate ? "dup" : "notdup";
        score = `
          <div class="score-cell">
            <span class="score-num">${fmt(result.score)}</span>
            <div class="bar" title="Threshold ${fmt(result.threshold)}">
              <div class="bar-fill ${kind}" style="width:${clamp01(result.score) * 100}%"></div>
              <div class="bar-tick" style="left:calc(${clamp01(result.threshold) * 100}% - 1px)"></div>
            </div>
          </div>`;
        prediction = `<span class="badge ${kind}">${escapeHtml(result.label)}</span>`;
      }
      return `
        <tr class="${model.is_best ? "best" : ""}">
          <td><span class="model-name">${escapeHtml(model.name)}</span>${tag}</td>
          <td>${score}</td>
          <td class="num col-threshold">${fmt(model.threshold)}</td>
          <td>${prediction}</td>
          <td class="num">${fmt(model.validation_f1)}</td>
        </tr>`;
    })
    .join("");
}

function renderVerdict(payload) {
  const best = payload.best;
  const kind = best.is_duplicate ? "dup" : "notdup";
  const verdict = $("verdict");
  verdict.classList.remove("dup", "notdup");
  verdict.classList.add(kind);
  verdict.hidden = false;

  $("verdict-icon").innerHTML = ICONS[kind];
  $("verdict-label").textContent = best.is_duplicate ? "Duplicate" : "Not a duplicate";
  const relation = best.is_duplicate ? "above" : "below";
  $("verdict-text").textContent =
    `${best.name} scored this pair ${fmt(best.score)}, ${relation} its decision threshold of ` +
    `${fmt(best.threshold)}. Scored by ${payload.results.length} models in ${payload.elapsed_ms} ms.`;

  $("meter-value").textContent = fmt(best.score);
  $("meter-fill").style.width = `${clamp01(best.score) * 100}%`;
  $("meter-threshold").style.left = `calc(${clamp01(best.threshold) * 100}% - 1px)`;
  $("meter-threshold-label").textContent = `threshold ${fmt(best.threshold)}`;

  const agreeing = payload.results.filter((r) => r.is_duplicate === best.is_duplicate).length;
  const agreement = $("agreement");
  agreement.textContent = `${agreeing} of ${payload.results.length} models agree`;
  agreement.hidden = false;
}

function showError(message) {
  const error = $("form-error");
  error.textContent = message;
  error.hidden = !message;
}

async function compare(event) {
  if (event) event.preventDefault();
  const question1 = $("q1").value.trim();
  const question2 = $("q2").value.trim();
  if (!question1 || !question2) {
    showError("Enter both questions to compare them.");
    (question1 ? $("q2") : $("q1")).focus();
    return;
  }
  showError("");
  const button = $("submit");
  button.disabled = true;
  button.querySelector(".btn-label").textContent = "Comparing…";
  try {
    const response = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question1, question2 }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
    renderVerdict(payload);
    renderRows(payload.results);
  } catch (error) {
    showError(typeof error.message === "string" ? error.message : "Something went wrong. Try again.");
  } finally {
    button.disabled = false;
    button.querySelector(".btn-label").textContent = "Compare";
  }
}

function reset() {
  $("q1").value = "";
  $("q2").value = "";
  $("verdict").hidden = true;
  $("agreement").hidden = true;
  showError("");
  renderRows(null);
  $("q1").focus();
}

// A laptop GPU sleeps after ~10 s idle and needs over a second to wake. While this
// page is visible, a tiny keep-alive every 5 s keeps comparisons instant.
function keepAwake() {
  if (document.visibilityState === "visible") {
    fetch("/api/warm", { method: "POST" }).catch(() => {});
  }
}

async function init() {
  const form = $("compare-form");
  form.addEventListener("submit", compare);
  $("clear").addEventListener("click", reset);
  $("swap").addEventListener("click", () => {
    [$("q1").value, $("q2").value] = [$("q2").value, $("q1").value];
  });
  document.querySelectorAll(".chip").forEach((chip) =>
    chip.addEventListener("click", () => {
      $("q1").value = chip.dataset.q1;
      $("q2").value = chip.dataset.q2;
      compare();
    }),
  );
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") compare(event);
  });

  try {
    const response = await fetch("/api/models");
    const payload = await response.json();
    models = payload.models;
    renderRows(null);
    setStatus("ready", `${models.length} models loaded`);
    $("privacy-note").textContent = payload.hosted
      ? "Questions are scored on the server and are not stored."
      : "Runs locally. No data leaves this computer.";
    if (payload.keep_awake) {
      setInterval(keepAwake, 5000);
      document.addEventListener("visibilitychange", keepAwake);
    }
  } catch {
    setStatus("error", "Server not reachable");
  }
}

init();
