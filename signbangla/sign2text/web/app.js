// Module 3 screen: sign words -> Bangla sentence. The server builds every sentence (build_sentence /
// build_from_candidates); this page only keeps the word list and shows the result.
const $ = (id) => document.getElementById(id);
const state = {
  items: [],            // {word, candidates?: [[word, score]], top?: [{word, english, score}], camera?: true}
  question: false,
  selected: -1,         // the camera item whose top-5 is shown
  lastEvent: 0,
  config: null,
};

async function post(url, body) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v; else if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v);
  }
  for (const c of children) e.append(c);
  return e;
}

function message(text) { $('message').textContent = text || ''; }

// ---------------------------------------------------------------- the sentence
let pending = 0;
async function rebuild() {
  const id = ++pending;
  const items = state.items.map((i) => (i.candidates ? { word: i.word, candidates: i.candidates } : { word: i.word }));
  let r = { sentence: '', trace: [], unknown: [], known: [] };
  if (items.length) {
    try { r = await post('/api/build', { items, is_question: state.question }); } catch (e) { message(e.message); return; }
  }
  if (id !== pending) return; // a newer change is on its way
  const s = $('sentence');
  s.textContent = r.sentence || 'Add words to build a sentence';
  s.classList.toggle('empty', !r.sentence);
  const chips = $('chips');
  chips.replaceChildren(...state.items.map((it, k) => el('span', {
    class: `chip${r.known[k] === false ? ' unknown' : ''}${it.camera ? ' camera' : ''}${k === state.selected ? ' selected' : ''}`,
    title: r.known[k] === false ? 'Not in the lexicon: kept as it is' : (it.camera ? 'From the camera: click to see its top-5' : ''),
    onclick: () => { if (it.camera) { state.selected = k; showTop(); rebuild(); } },
  }, it.word)));
  $('trace').replaceChildren(...r.trace.map((t) => el('li', {}, t)));
  $('unknown').textContent = r.unknown.length ? `Unknown words (kept as they are, never guessed): ${r.unknown.join(' ')}` : '';
}

function add(word) { state.items.push({ word }); message(''); rebuild(); }

// ---------------------------------------------------------------- words mode
function renderGroups(filter = '') {
  const f = filter.trim().toLowerCase();
  const groups = state.config.groups;
  $('groups').replaceChildren(...Object.entries(groups).map(([name, words]) => {
    const shown = words.filter((w) => !f || w.word.includes(f) || (w.english || '').toLowerCase().includes(f));
    if (!shown.length) return '';
    return el('div', { class: 'group' }, el('h3', {}, `${name} (${shown.length})`),
      el('div', { class: 'words' }, ...shown.map((w) => {
        const b = el('button', { type: 'button', class: 'word', title: w.english || '', onclick: () => add(w.word) }, w.word);
        if (w.has_sign) b.append(el('span', { class: 'sign', title: 'has a sign' }, '●'));
        return b;
      })));
  }));
}

async function example(text) {
  const { words } = await post('/api/split', { text });
  state.items = words.map((word) => ({ word }));
  state.selected = -1;
  showTop();
  rebuild();
}

// ---------------------------------------------------------------- camera mode
function showTop() {
  const it = state.items[state.selected];
  if (!it || !it.top) {
    $('top5').replaceChildren(el('span', { class: 'hint' },
      'Sign in front of the camera; the best word is added by itself. Click another word here to replace it.'));
    return;
  }
  $('top5').replaceChildren(...it.top.map((t) => el('button', {
    type: 'button', class: t.word === it.word ? 'current' : '',
    onclick: () => { it.word = t.word; showTop(); rebuild(); },
  }, el('span', {}, el('span', { class: 'bn' }, t.word), el('span', { class: 'en' }, t.english)),
  el('span', { class: 'score' }, `${(100 * t.score).toFixed(1)}%`))));
}

function showCamera(c) {
  $('camera-message').textContent = c.message || (c.requested ? '' :
    'Camera mode is off. Start with: python scripts\\52_sentence_screen.py --camera 0');
  const ind = $('indicator');
  ind.textContent = c.state === 'signing' ? 'signing…' : 'ready';
  ind.className = `indicator ${c.state === 'signing' ? 'signing' : 'ready'}`;
  $('camera-last').textContent = c.enabled ? `${c.fps} frames per second${c.last ? ' · ' + c.last : ''}` : '';
}

async function pollCamera() {
  try {
    const c = await (await fetch(`/api/camera?since=${state.lastEvent}`, { cache: 'no-store' })).json();
    showCamera(c);
    for (const e of c.events) {
      state.lastEvent = Math.max(state.lastEvent, e.id);
      if (!e.top.length) continue;
      state.items.push({ word: e.top[0].word, top: e.top, candidates: e.top.map((t) => [t.word, t.score]), camera: true });
      state.selected = state.items.length - 1;
      showTop();
      rebuild();
    }
  } catch (e) { /* the server has stopped */ }
  setTimeout(pollCamera, state.config.poll_ms);
}

function setMode(mode) {
  const camera = mode === 'camera';
  $('tab-words').setAttribute('aria-selected', String(!camera));
  $('tab-camera').setAttribute('aria-selected', String(camera));
  $('words-mode').hidden = camera;
  $('camera-mode').hidden = !camera;
  if (camera && state.config.camera.requested && !$('video').src) $('video').src = '/video.mjpg';
}

// ---------------------------------------------------------------- buttons
$('undo').onclick = () => { state.items.pop(); state.selected = Math.min(state.selected, state.items.length - 1); showTop(); rebuild(); };
$('clear').onclick = () => { state.items = []; state.selected = -1; showTop(); message(''); rebuild(); };
$('question').onclick = () => {
  state.question = !state.question;
  $('question').setAttribute('aria-pressed', String(state.question));
  rebuild();
};
$('speak').onclick = async () => {
  const text = $('sentence').classList.contains('empty') ? '' : $('sentence').textContent;
  message('');
  const r = await post('/api/speak', { text });
  if (!r.ok) { message(r.message); return; }
  new Audio(r.url).play().catch((e) => message(`Could not play the audio: ${e.message}`));
};
$('search').oninput = (e) => renderGroups(e.target.value);
$('tab-words').onclick = () => setMode('words');
$('tab-camera').onclick = () => setMode('camera');

(async function start() {
  state.config = await (await fetch('/api/config')).json();
  $('camera-note').textContent = state.config.note;
  $('examples').replaceChildren(...state.config.examples.map((x) =>
    el('button', { type: 'button', onclick: () => example(x) }, x)));
  renderGroups();
  showCamera(state.config.camera);
  setMode(state.config.camera.requested ? 'camera' : 'words');
  rebuild();
  pollCamera();
})();
