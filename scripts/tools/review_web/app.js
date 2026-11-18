// Review page: loads /api/words, shows one word at a time and posts answers to /api/answer.
"use strict";

const $ = (id) => document.getElementById(id);
const STATUS_TEXT = { unreviewed: "not reviewed", correct: "correct", wrong: "wrong", unsure: "not sure" };

let words = [];
let shown = [];      // words that pass the filter
let pos = 0;         // index in `shown`
let busy = false;

async function load() {
  const res = await fetch("/api/words", { cache: "no-store" });
  const data = await res.json();
  words = data.words;
  fillSelect($("f-category"), [...new Set(words.map((w) => w.category))].sort());
  fillSelect($("f-group"), [...new Set(words.map((w) => w.group_name))].sort());
  $("no-clips-count").textContent = data.no_clips.length;
  $("no-clips-body").innerHTML = data.no_clips.map((w) =>
    `<tr><td>${esc(w.id)}</td><td class="bn">${esc(w.bangla)}</td><td>${esc(w.english)}</td>` +
    `<td>${esc(w.source)}</td><td>${esc(w.category)}</td></tr>`).join("");
  applyFilter(null);
  const first = shown.findIndex((w) => w.status === "unreviewed");
  pos = first >= 0 ? first : 0;
  render();
}

function fillSelect(select, values) {
  for (const v of values) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = v;
    select.appendChild(o);
  }
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function applyFilter(keepId) {
  const st = $("f-status").value, cat = $("f-category").value, grp = $("f-group").value;
  shown = words.filter((w) => (st === "all" || w.status === st) && (cat === "all" || w.category === cat) &&
    (grp === "all" || w.group_name === grp));
  const i = keepId ? shown.findIndex((w) => w.id === keepId) : -1;
  pos = i >= 0 ? i : Math.min(pos, Math.max(0, shown.length - 1));
}

function progress() {
  const done = words.filter((w) => w.status !== "unreviewed").length;
  $("bar").style.width = words.length ? `${(100 * done) / words.length}%` : "0";
  $("progress-text").textContent = `${done} of ${words.length} reviewed`;
}

function render() {
  progress();
  const w = shown[pos];
  $("main").hidden = !w;
  $("empty").hidden = !!w;
  $("position").textContent = w ? `${pos + 1} / ${shown.length} in view` : "";
  hideReason();
  if (!w) return;

  $("group").textContent = w.group_name;
  $("bangla").textContent = w.bangla;
  $("english").textContent = w.english;
  const status = $("status");
  status.textContent = STATUS_TEXT[w.status] + (w.note ? ` (${w.note})` : "");
  status.className = `status ${w.status}`;
  $("wid").textContent = `${w.id}${w.core_id ? ` (core ${w.core_id})` : ""}`;
  $("source").textContent = w.source;
  $("folder").textContent = w.folder || "-";
  $("category").textContent = w.category;
  $("clips").textContent = `${w.good_clips} of ${w.clips}`;
  $("signers").textContent = w.signers || "-";

  const c = w.consistency;
  if (c.ratio !== "") {
    const score = Math.max(0, 1 - Number(c.ratio));
    $("consistency").textContent =
      `${score.toFixed(2)} (within / other words ${Number(c.ratio).toFixed(2)}, ` +
      `nearest clip same word ${Math.round(100 * c.nearest_same)}%, chance ${(100 * c.chance).toFixed(1)}%, ` +
      `${c.clips_used} clips)`;
  } else {
    $("consistency").textContent = "not computed";
  }

  const warn = [];
  if (w.unclear) warn.push(`Unclear: ${w.unclear}`);
  if (w.matches) warn.push(`Same meaning as ${w.matches}`);
  if (c.warn) warn.push(`Clips look inconsistent: ${c.warn}`);
  $("warnings").innerHTML = warn.map((t) => `<div class="warning">${esc(t)}</div>`).join("");

  $("edit-bangla").value = w.bangla;
  $("edit-english").value = w.english;

  const video = $("video");
  if (w.video) {
    const src = `/media/${encodeURI(w.video)}`;
    if (video.getAttribute("src") !== src) video.setAttribute("src", src);
    video.hidden = false;
    $("no-video").hidden = true;
  } else {
    video.removeAttribute("src");
    video.load();
    video.hidden = true;
    $("no-video").hidden = false;
  }
  if (w.sheet) {
    const src = `/media/${encodeURI(w.sheet)}`;
    $("sheet").src = src;
    $("sheet-link").href = src;
    $("sheet").hidden = false;
    $("sheet-title").textContent = `Contact sheet ${w.sheet.split("/").pop()}${w.sheet_note ? `, ${w.sheet_note}` : ""}` +
      " (click to open full size)";
  } else {
    $("sheet").removeAttribute("src");
    $("sheet").hidden = true;
    $("sheet-title").textContent = "No contact sheet";
  }
}

function say(text, error) {
  const m = $("message");
  m.textContent = text;
  m.className = error ? "message error" : "message";
}

async function send(decision, reason) {
  const w = shown[pos];
  if (!w || busy) return;
  busy = true;
  const body = { id: w.id, decision, reason: reason || "" };
  const bangla = $("edit-bangla").value.trim(), english = $("edit-english").value.trim();
  if (bangla !== w.bangla) body.bangla = bangla;
  if (english !== w.english) body.english = english;
  try {
    const res = await fetch("/api/answer", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || res.statusText);
    Object.assign(w, { bangla: data.row.bangla, english: data.row.english, verified: data.row.verified,
      status: decision, note: decision === "unsure" ? "unsure" + (reason ? `: ${reason}` : "") : (reason || "") });
    say(`Saved ${w.id}: ${STATUS_TEXT[decision]} (verified = ${data.row.verified}).`);
    nextAfterAnswer(w.id);
  } catch (e) {
    say(`Not saved: ${e.message}`, true);
  } finally {
    busy = false;
  }
}

function nextAfterAnswer(id) {
  applyFilter(id);
  // the answered word may have left the filter: continue with the next unreviewed word in view
  let i = shown.findIndex((w) => w.id === id);
  const start = i >= 0 ? i + 1 : pos;
  const next = shown.findIndex((w, k) => k >= start && w.status === "unreviewed");
  if (next >= 0) pos = next;
  else if (i >= 0 && i + 1 < shown.length) pos = i + 1;
  else pos = Math.min(start, Math.max(0, shown.length - 1));
  render();
}

function move(step) {
  if (!shown.length) return;
  pos = Math.min(Math.max(0, pos + step), shown.length - 1);
  say("");
  render();
}

function askReason() {
  $("reason-box").hidden = false;
  $("reason").value = "";
  $("reason").focus();
}

function hideReason() {
  $("reason-box").hidden = true;
}

$("b-correct").onclick = () => send("correct");
$("b-unsure").onclick = () => send("unsure");
$("b-wrong").onclick = askReason;
$("b-back").onclick = () => move(-1);
$("b-next").onclick = () => move(1);
$("show-no-clips").onclick = () => $("no-clips").showModal();
for (const id of ["f-status", "f-category", "f-group"]) {
  $(id).onchange = () => {
    applyFilter(shown[pos]?.id);
    render();
  };
}

$("reason").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    send("wrong", $("reason").value.trim());
  } else if (e.key === "Escape") {
    hideReason();
  }
  e.stopPropagation();
});

document.addEventListener("keydown", (e) => {
  const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName);
  if (typing || e.ctrlKey || e.metaKey || e.altKey || $("no-clips").open) return;
  const k = e.key.toLowerCase();
  if (k === "y") send("correct");
  else if (k === "n") { e.preventDefault(); askReason(); }
  else if (k === "s") send("unsure");
  else if (e.key === "ArrowRight") move(1);
  else if (e.key === "ArrowLeft") move(-1);
  else return;
  e.preventDefault();
});

load().catch((e) => say(`Could not load the words: ${e.message}`, true));
