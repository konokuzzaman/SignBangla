// SignBangla app page. Speech -> Sign: push to talk or typed text -> /api/speech/* -> one sequence (JSON) that both
// views play from one clock: the 3D mannequin (iframe, studio3d in embed mode) and the 2D skeleton (skeleton.js).
import { SkeletonView } from '/skeleton.js';

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

async function api(path, body) {
  const res = await fetch(path, body === undefined ? { cache: 'no-store' }
    : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  return res.json();
}

// ---------------------------------------------------------------- status
let status = null;
function chip(label, ok, message) {
  return `<span class="chip ${ok ? 'ok' : 'off'}" title="${esc(message || '')}">${esc(label)}</span>`;
}
function showStatus(st) {
  const s = st.speech_to_sign;
  const times = Object.entries(st.load_times || {}).map(([k, v]) => `${k} ${v} s`).join(', ');
  const total = st.load_times && st.load_times.total !== undefined ? `loaded in ${st.load_times.total} s` : 'load times';
  $('status').innerHTML = [
    chip(`Dictionary ${s.dictionary.words} words`, s.dictionary.words > 0),
    chip('Microphone', s.microphone.ok, s.microphone.message),
    chip(`Speech recognition${s.speech.device ? ' (' + s.speech.device + ')' : ''}`, s.speech.ok, s.speech.message),
    chip('Camera', st.sign_to_text.ok, st.sign_to_text.message),
    chip('Sign model', st.sign_to_text.model.ok, st.sign_to_text.model.message || st.sign_to_text.model.name),
    chip('Voice', st.sign_to_text.speech_output.ok, st.sign_to_text.speech_output.message),
    times ? `<span class="chip" title="${esc(times)}">${esc(total)}</span>` : '',
  ].join('');
  const sign = st.sign_to_text;
  $('vocab').textContent = sign.model.ok ? `Camera vocabulary: ${sign.model.vocabulary} words` : '';
  const signProblems = [];
  if (!sign.ok) signProblems.push(sign.message);
  if (!sign.speech_output.ok && sign.speech_output.message) signProblems.push(`Voice: ${sign.speech_output.message}`);
  $('sign-message').hidden = !signProblems.length;
  $('sign-message').textContent = signProblems.join(' · ');
  if (sign.ok) {
    $('video').src = '/video.mjpg';
    $('video').hidden = false;
    $('camera-off').hidden = true;
  } else {
    $('camera-off').textContent = sign.message || 'Camera off';
  }
  const problems = [];
  if (!s.microphone.ok) problems.push(`Microphone: ${s.microphone.message}`);
  if (!s.speech.ok) problems.push(`Speech recognition: ${s.speech.message}`);
  if (problems.length) {
    speechMessage(`${problems.join(' · ')}. Typed text still works.`);
    $('talk').disabled = true;
  }
}
function speechMessage(text) {
  $('speech-message').hidden = !text;
  $('speech-message').textContent = text || '';
}

// ---------------------------------------------------------------- player: one clock, two views
const player = { data: null, start: 0, speed: 1, view: '3d', last: -1, ready3d: false, pending: null };
const skeleton = { view: null };
const frame3d = $('viewer3d');

window.addEventListener('message', (e) => {
  if (e.origin !== location.origin || !e.data) return;
  if (e.data.type === 'ready') {
    player.ready3d = true;
    if (player.data) send3d({ type: 'load', data: player.data });
  }
});
function send3d(msg) {
  if (player.ready3d && frame3d.contentWindow) frame3d.contentWindow.postMessage(msg, location.origin);
}

function load(data) {
  player.data = data;
  $('stage-empty').hidden = !!data;
  skeleton.view.setData(data);
  send3d({ type: 'load', data });
  buildCaption(data);
  replay();
}
function replay() {
  player.start = performance.now();
  player.last = -1;
}
function frameAt(now) {
  const d = player.data;
  return Math.min(d.frames.length - 1, Math.floor(((now - player.start) / 1000) * d.fps * player.speed));
}
function setSpeed(speed) {
  if (player.data) {  // keep the current position
    const i = frameAt(performance.now());
    player.start = performance.now() - (i / (player.data.fps * speed)) * 1000;
  }
  player.speed = speed;
  for (const b of $('speeds').children) b.classList.toggle('on', Number(b.dataset.speed) === speed);
}
function setView(view) {
  player.view = view;
  $('view-3d').classList.toggle('on', view === '3d');
  $('view-2d').classList.toggle('on', view === '2d');
  frame3d.classList.toggle('hide', view !== '3d');
  $('skeleton').classList.toggle('hide', view !== '2d');
  if (view === '2d') skeleton.view.resize();
  else send3d({ type: 'resize' });
  player.last = -1;  // draw the current frame in the new view now
}

let captionSpans = [];
function buildCaption(data) {
  const box = $('caption');
  box.replaceChildren();
  captionSpans = (data ? data.tokens : []).map((t) => {
    const s = document.createElement('span');
    s.textContent = t.text;
    if (t.color) s.style.color = t.color;  // red: no sign, orange: predicted from the context
    box.appendChild(s);
    return s;
  });
}

function tick(now) {
  requestAnimationFrame(tick);
  const d = player.data;
  if (!d || !d.frames.length) return;
  const i = frameAt(now);
  if (i === player.last) return;
  player.last = i;
  // both views get every frame, so a switch shows the same moment at once
  skeleton.view.draw(i);
  send3d({ type: 'frame', i });
  const w = d.word[i];
  captionSpans.forEach((s, k) => s.classList.toggle('on', k === w));
}

// ---------------------------------------------------------------- results
const STATUS_CLASS = { sign: '', synonym: 'synonym', marker: 'marker', predicted: 'predicted', no_sign: 'nosign',
  unknown: 'nosign', dropped: 'dropped' };
function showResult(out) {
  if (!out.ok) {
    speechMessage(out.message || 'Something went wrong.');
    showTimings(out.timings);
    return;
  }
  speechMessage(out.message || '');
  $('heard').textContent = out.text;
  const g = out.gloss;
  const words = g.tokens.map((t, k) => {
    const title = t.note ? ` title="${esc(t.note)}"` : (t.key ? '' : ' title="no sign yet"');
    const word = `<span class="w ${STATUS_CLASS[t.status] || ''}"${title}>${esc(t.caption)}</span>`;
    if (t.status !== 'predicted') return word;
    // a prediction: one click to keep it, undo it, or take one of the next best words
    const alts = t.alternatives.map((a) =>
      `<button type="button" class="mini" data-k="${k}" data-a="alternative" data-w="${esc(a)}">${esc(a)}</button>`).join('');
    return `<span class="pred">${word}<button type="button" class="mini" data-k="${k}" data-a="accept" title="keep">✓</button>` +
      `<button type="button" class="mini" data-k="${k}" data-a="undo" title="undo">✗</button>${alts}</span>`;
  });
  const dropped = g.dropped.map((t) => `<span class="w dropped" title="not signed in BdSL">${esc(t.text)}</span>`);
  $('signs').innerHTML = words.concat(dropped).join('') || '-';
  if (out.sequence) load(out.sequence);
  showTimings(out.timings);
  refreshHistory();
}
const timings = { speech: '', sign: '' };
function showTimings(t) {
  if (!t) return;
  const parts = [];
  if (t.speech_s !== undefined) parts.push(`speech recognised in ${t.speech_s} s`);
  if (t.text_s !== undefined) parts.push(`sign words in ${t.text_s} s`);
  if (t.animation_s !== undefined) parts.push(`animation in ${t.animation_s} s`);
  timings.speech = parts.join(' · ');
  renderTimings();
}
function renderTimings() {
  $('timings').textContent = [timings.sign, timings.speech].filter(Boolean).join('  |  ');
}

// ---------------------------------------------------------------- Sign -> Text
const sign = { selected: -1, handled: 0, items: [], words: [], lastEvent: '' };

function renderSign(v) {
  if (!v || !v.ok) return;
  const cam = v.camera;
  const ind = $('indicator');
  ind.textContent = cam.enabled ? (cam.state === 'signing' ? 'signing …' : 'ready') : 'off';
  ind.classList.toggle('signing', cam.state === 'signing');
  const framing = cam.framing ? ` · framing: ${cam.framing.hint}` : '';
  $('last-event').textContent = (cam.last || '') + (cam.fps ? ` · ${cam.fps} fps` : '') + framing;
  const grew = v.items.length > sign.items.length;
  const newPrediction = v.items.findLastIndex((it, k) => it.prediction &&
    (!sign.items[k] || !sign.items[k].prediction || sign.items[k].prediction.word !== it.prediction.word));
  sign.items = v.items;
  if (newPrediction >= 0) sign.selected = newPrediction;  // show a new suggestion at once
  else if (grew || sign.selected >= v.items.length) sign.selected = v.items.length - 1;
  $('items').innerHTML = v.items.map((it, k) =>
    `<button type="button" data-k="${k}" class="${k === sign.selected ? 'sel' : ''} ${it.source === 'hand' ? 'hand' : ''}` +
    ` ${it.prediction ? 'predicted' : ''}" title="${it.prediction ? 'predicted from the other words; recognised: ' +
      esc(it.prediction.top1) : (it.source === 'hand' ? 'added by hand' : 'from the camera')}">${esc(it.word)}</button>`).join('');
  const it = v.items[sign.selected];
  $('sel-word').textContent = it ? `for “${it.own}”` : '';
  let html = '';
  if (it && it.prediction) {
    const p = it.prediction;
    html += `<div class="predbox">${p.word === '…' ? 'Unclear: no confident word from the context' :
      `Context suggests <b class="bn">${esc(p.word)}</b>${p.kind === 'gap' ? ' (not in the top 5)' : ''}`}` +
      ` <button type="button" class="mini" data-act="accept" ${p.word === '…' ? 'disabled' : ''}>✓ keep</button>` +
      `<button type="button" class="mini" data-act="reject">✗ undo (${esc(p.top1)})</button>` +
      p.alternatives.map((a) => `<button type="button" class="mini" data-w="${esc(a)}">${esc(a)}</button>`).join('') + '</div>';
  }
  html += it && it.candidates.length ? it.candidates.map(([w, s]) =>
    `<button type="button" data-w="${esc(w)}" class="${w === it.word ? 'on' : ''}">${esc(w)}<small>${Math.round(s * 100)}%</small></button>`).join('')
    : (it ? '<span class="small">added by hand: no other candidates</span>' : '');
  $('top5').innerHTML = html;
  $('sentence').textContent = v.sentence || '';
  $('question').checked = v.question;
  const t = [];
  if (cam.recognise_s !== undefined) t.push(`sign recognised in ${cam.recognise_s} s`);
  if (v.items.length) t.push(`context ${v.predict_s} s, sentence built in ${v.build_s} s`);
  timings.sign = t.join(' · ');
  for (const f of v.finished) {
    if (!f.done || f.id <= sign.handled) continue;
    sign.handled = f.id;
    if (f.audio) {
      $('voice').src = f.audio;
      $('voice').play().catch(() => {});
      $('spoken').textContent = `spoken in ${f.speech_s} s`;
    } else {
      $('spoken').textContent = f.message || '';
    }
    refreshHistory();
  }
  renderTimings();
}

async function signPost(path, body = {}) {
  const v = await api(path, body);
  if (v.ok === false && v.message) $('spoken').textContent = v.message;
  renderSign(v.items ? v : null);
  if (path.endsWith('/done')) refreshHistory();
  return v;
}
$('items').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b) return;
  sign.selected = Number(b.dataset.k);
  renderSign(Object.assign({ ok: true }, lastSignView));
});
$('top5').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (!b || sign.selected < 0) return;
  if (b.dataset.act) signPost(`/api/sign/${b.dataset.act}`, { index: sign.selected });
  else signPost('/api/sign/replace', { index: sign.selected, word: b.dataset.w });
});
$('done').addEventListener('click', () => signPost('/api/sign/done'));
$('undo').addEventListener('click', () => signPost('/api/sign/undo'));
$('clear').addEventListener('click', () => signPost('/api/sign/clear'));
$('question').addEventListener('change', (e) => signPost('/api/sign/question', { on: e.target.checked }));

function showWordButtons() {
  const q = $('word-search').value.trim().toLowerCase();
  const list = sign.words.filter((w) => !q || w.word.toLowerCase().includes(q) || (w.english || '').toLowerCase().includes(q));
  $('word-buttons').innerHTML = list.slice(0, 60).map((w) =>
    `<button type="button" data-w="${esc(w.word)}" title="${esc(w.english || '')}">${esc(w.word)}</button>`).join('')
    + (list.length > 60 ? `<span class="small">… ${list.length - 60} more, type to search</span>` : '');
}
$('word-search').addEventListener('input', showWordButtons);
$('word-search').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    const first = $('word-buttons').querySelector('button');
    if (first) signPost('/api/sign/add', { word: first.dataset.w });
    $('word-search').value = '';
    showWordButtons();
  }
});
$('word-buttons').addEventListener('click', (e) => {
  const b = e.target.closest('button');
  if (b) signPost('/api/sign/add', { word: b.dataset.w });
});

let lastSignView = null;
async function pollSign(ms) {
  try {
    const v = await api(`/api/sign/state?since=${sign.handled}`);
    if (v.ok) {
      lastSignView = v;
      renderSign(v);
    }
  } catch (e) {
    // the server has stopped: keep the page as it is
  }
  setTimeout(() => pollSign(ms), ms);
}
async function refreshHistory() {
  const items = await api('/api/history');
  $('history').innerHTML = items.slice().reverse().map((h) =>
    `<li><span class="t">${esc(h.time)}</span><span class="d">${h.direction === 'speech' ? 'Speech → Sign' : 'Sign → Text'}</span>` +
    `<span class="bn">${esc(h.text)}</span><span class="x bn">${esc(h.detail)}</span></li>`).join('');
}

$('signs').addEventListener('click', async (e) => {
  const b = e.target.closest('button[data-a]');
  if (!b) return;
  showResult(await api('/api/speech/choose', { index: Number(b.dataset.k), action: b.dataset.a, word: b.dataset.w || '' }));
});

// ---------------------------------------------------------------- test sentences
async function openTests() {
  const out = await api('/api/tests');
  const box = $('tests-body');
  if (!out.ok) {
    $('tests-summary').textContent = out.message;
    box.innerHTML = '';
  } else {
    const s = out.summary;
    $('tests-summary').textContent = `${s.sentences} sentences · ${s.fully_signed} fully signed · ` +
      `mean coverage ${s.mean_coverage_pct}% · saved to ${out.saved}. Click a sentence to sign it.`;
    box.innerHTML = out.rows.map((r) =>
      `<tr data-s="${esc(r.sentence)}"><td class="bn">${esc(r.sentence)}</td><td>${r.coverage_pct}%</td>` +
      `<td class="bn">${esc(r.signs)}</td><td class="bn nosign">${esc(r.missing)}</td></tr>`).join('');
  }
  $('tests').showModal();
}
$('open-tests').addEventListener('click', openTests);
$('tests-body').addEventListener('click', async (e) => {
  const tr = e.target.closest('tr[data-s]');
  if (!tr) return;
  $('tests').close();
  $('text').value = tr.dataset.s;
  showResult(await api('/api/speech/text', { text: tr.dataset.s }));
});

// ---------------------------------------------------------------- input
$('text-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = $('text').value.trim();
  if (!text) return;
  showResult(await api('/api/speech/text', { text }));
});

let talking = false, busy = false;
async function talkStart() {
  if (talking || busy || $('talk').disabled) return;
  talking = true;
  $('talk').classList.add('recording');
  $('talk-label').textContent = 'Listening ... release to stop';
  const out = await api('/api/speech/start', {});
  if (!out.ok) {
    talking = false;
    $('talk').classList.remove('recording');
    $('talk-label').textContent = 'Hold to speak';
    speechMessage(out.message);
  }
}
async function talkStop() {
  if (!talking) return;
  talking = false;
  busy = true;
  $('talk').classList.remove('recording');
  $('talk').classList.add('busy');
  $('talk-label').textContent = 'Recognising ...';
  try {
    showResult(await api('/api/speech/stop', {}));
  } finally {
    busy = false;
    $('talk').classList.remove('busy');
    $('talk-label').textContent = 'Hold to speak';
  }
}
const talk = $('talk');
talk.addEventListener('pointerdown', (e) => { talk.setPointerCapture(e.pointerId); talkStart(); });
talk.addEventListener('pointerup', talkStop);
talk.addEventListener('pointercancel', talkStop);
const typing = () => ['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName);
window.addEventListener('keydown', (e) => {
  if (typing()) return;
  if (e.code === 'Space' && !e.repeat) { e.preventDefault(); talkStart(); }
  if (e.key === 'r' || e.key === 'R') replay();
});
window.addEventListener('keyup', (e) => {
  if (e.code === 'Space' && !typing()) { e.preventDefault(); talkStop(); }
});

$('replay').addEventListener('click', replay);
$('view-3d').addEventListener('click', () => setView('3d'));
$('view-2d').addEventListener('click', () => setView('2d'));

// ---------------------------------------------------------------- start
status = await api('/api/status');
skeleton.view = new SkeletonView($('skeleton'), status.skeleton);
for (const s of status.speeds) {
  const b = document.createElement('button');
  b.type = 'button';
  b.dataset.speed = s;
  b.textContent = `${s}x`;
  b.addEventListener('click', () => setSpeed(s));
  $('speeds').appendChild(b);
}
setSpeed(status.speeds.includes(1) ? 1 : status.speeds[0]);
showStatus(status);
if (new URLSearchParams(location.search).get('view') === '2d') setView('2d');
const last = await api('/api/speech/last');
if (last.ok) showResult(last);
else refreshHistory();
requestAnimationFrame(tick);

const words = await api('/api/sign/words');
const seen = new Set();
for (const group of Object.values(words.groups || {})) {
  for (const w of group) {
    if (!seen.has(w.word)) {
      seen.add(w.word);
      sign.words.push(w);
    }
  }
}
showWordButtons();
pollSign(status.sign_to_text.poll_ms || 250);
