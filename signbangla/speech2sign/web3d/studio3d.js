// studio3d viewer: a 3D mannequin playing the current sentence from the local server.
// /settings.json: camera, floor and colours (speech2sign.studio3d in configs/config.yaml)
// /sequence.json: { version, fps, frames: [T][67][3] (x right, y up, z towards the camera), word: [T], tokens }
// /version:       { version } - checked every poll_ms, a new version is loaded and played by itself
import * as THREE from 'three';

const settings = await (await fetch('/settings.json')).json();
const C = settings.colors;
const stage = document.getElementById('stage');
const caption = document.getElementById('caption');
const replay = document.getElementById('replay');

const root = document.documentElement.style;
for (const [name, key] of [['--background', 'background'], ['--caption-background', 'caption_background'],
  ['--caption-border', 'caption_border'], ['--caption-text', 'caption_text'],
  ['--highlight-background', 'highlight_background'], ['--highlight-text', 'highlight_text']]) {
  root.setProperty(name, C[key]);
}
root.setProperty('--caption-font-size', settings.caption_font_size + 'px');

// ---- renderer, scene, fixed front camera
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(C.background);
scene.fog = new THREE.Fog(C.background, 14, 24);
const cam = settings.camera;
const camera = new THREE.PerspectiveCamera(cam.fov, 1, 0.1, 60);
const target = new THREE.Vector3(...cam.target);

scene.add(new THREE.HemisphereLight(0xffffff, 0xb9c2d0, 1.5));
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.position.set(2.2, 4.5, 5);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3, near: 0.5, far: 14 });
sun.shadow.radius = 7;
sun.shadow.bias = -0.0005;
scene.add(sun);

const FLOOR = settings.floor_y;
const floor = new THREE.Mesh(new THREE.CircleGeometry(7, 64),
  new THREE.MeshStandardMaterial({ color: C.floor, roughness: 1 }));
floor.rotation.x = -Math.PI / 2;
floor.position.y = FLOOR;
floor.receiveShadow = true;
scene.add(floor);
const wall = new THREE.Mesh(new THREE.PlaneGeometry(30, 14),
  new THREE.MeshStandardMaterial({ color: C.background, roughness: 1 }));
wall.position.set(0, 2, -7);
wall.receiveShadow = true;
scene.add(wall);

// ---- mannequin parts
const mat = (color, roughness = 0.55) => new THREE.MeshStandardMaterial({ color, roughness, metalness: 0.05 });
const M = {
  limb: mat(C.limbs), joint: mat(C.joints, 0.35), torso: mat(C.torso), head: mat(C.head), face: mat(C.face),
  right: mat(C.right_hand, 0.45), left: mat(C.left_hand, 0.45),
};
const figure = new THREE.Group();
figure.visible = false; // until the first sentence arrives
scene.add(figure);
const sphere = new THREE.SphereGeometry(1, 24, 16);
const UP = new THREE.Vector3(0, 1, 0);

// a tapered capsule from a (radius r0) to b (radius r1); returns a function that places it
function bone(m, r0, r1) {
  const g = new THREE.Group();
  const cyl = new THREE.Mesh(new THREE.CylinderGeometry(r1, r0, 1, 20, 1, true), m);
  const s0 = new THREE.Mesh(sphere, m), s1 = new THREE.Mesh(sphere, m);
  s0.scale.setScalar(r0);
  s1.scale.setScalar(r1);
  for (const o of [cyl, s0, s1]) { o.castShadow = true; g.add(o); }
  figure.add(g);
  const d = new THREE.Vector3();
  return (a, b) => {
    d.copy(b).sub(a);
    const len = d.length() || 1e-4;
    cyl.position.copy(a).addScaledVector(d, 0.5);
    cyl.quaternion.setFromUnitVectors(UP, d.divideScalar(len));
    cyl.scale.set(1, len, 1);
    s0.position.copy(a);
    s1.position.copy(b);
  };
}
function ball(m, r) {
  const s = new THREE.Mesh(sphere, m);
  s.scale.setScalar(r);
  s.castShadow = true;
  figure.add(s);
  return s;
}

const k = 1.3;
const B = {
  shoulders: bone(M.limb, 0.05 * k, 0.05 * k), spine: bone(M.limb, 0.045 * k, 0.045 * k),
  hips: bone(M.limb, 0.05 * k, 0.05 * k), neck: bone(M.limb, 0.05 * k, 0.045 * k),
  upperL: bone(M.limb, 0.062 * k, 0.052 * k), foreL: bone(M.limb, 0.052 * k, 0.04 * k),
  upperR: bone(M.limb, 0.062 * k, 0.052 * k), foreR: bone(M.limb, 0.052 * k, 0.04 * k),
};
const J = {
  shoulderL: ball(M.joint, 0.078 * k), shoulderR: ball(M.joint, 0.078 * k),
  elbowL: ball(M.joint, 0.066 * k), elbowR: ball(M.joint, 0.066 * k),
  hipL: ball(M.joint, 0.06 * k), hipR: ball(M.joint, 0.06 * k),
};
const head = new THREE.Mesh(new THREE.SphereGeometry(1, 40, 28), M.head);
head.castShadow = true;
figure.add(head);

// ---- head and face ("line" face: grey eye dots, a short nose line and a mouth line, on the head surface)
const F = settings.face;
const eyeL = ball(M.face, F.eye_radius), eyeR = ball(M.face, F.eye_radius);
// nose and mouth lines only when face.show_nose_mouth is true: otherwise no meshes are made at all
const lineParts = () => Array.from({ length: F.line_segments }, () => bone(M.face, F.line_radius, F.line_radius));
const noseLine = F.show_nose_mouth ? lineParts() : null;
const mouthLine = F.show_nose_mouth ? lineParts() : null;

// head size and centre from the ears, eyes and mouth corners, at least 0.42 shoulder widths wide
function headGeom(p) {
  const earL = p[7], earR = p[8], eyeY = (p[2].y + p[5].y) / 2, mouthY = (p[9].y + p[10].y) / 2;
  const sw = Math.abs(p[11].x - p[12].x);
  let hw = Math.max(Math.abs(earL.x - earR.x) * 1.05, sw * 0.42);
  const hh = Math.max(hw * 1.3, (eyeY - mouthY) / 0.36);
  hw = Math.max(hw, hh / 1.4);
  return { c: new THREE.Vector3((earL.x + earR.x) / 2, eyeY - hh * 0.05, 0.06), rx: hw / 2, ry: hh / 2,
    rz: (hw / 2) * 1.05, eyeY, mouthY };
}
// z of the front of the head at (x, y); far behind the head outside its outline
function surfZ(g, x, y) {
  const q = 1 - ((x - g.c.x) / g.rx) ** 2 - ((y - g.c.y) / g.ry) ** 2;
  return q > 0 ? g.c.z + g.rz * Math.sqrt(q) : -99;
}
// same, but at the outline instead of far behind, for face marks near the edge
function faceZ(g, x, y) {
  const q = 1 - ((x - g.c.x) / g.rx) ** 2 - ((y - g.c.y) / g.ry) ** 2;
  return g.c.z + g.rz * Math.sqrt(Math.max(0, q));
}
// points from (x0, y0) to (x1, y1) that follow the curve of the head, so a line between them never sinks into it
function curve(g, x0, y0, x1, y1, n) {
  return Array.from({ length: n + 1 }, (_, i) => {
    const t = i / n, x = x0 + (x1 - x0) * t, y = y0 + (y1 - y0) * t;
    return new THREE.Vector3(x, y, faceZ(g, x, y) + F.surface_offset);
  });
}
function drawLine(parts, pts) { parts.forEach((b, i) => b(pts[i], pts[i + 1])); }

// keep each hand in front of the face: when a hand would sink into the head, the whole hand (and the wrist) moves
// forward. The shift is widened and smoothed over time so the hand does not jump.
function pushHands(frames) {
  const n = frames.length, R = settings.hand_clearance_frames, clear = settings.hand_clearance;
  for (const [a, b, wrist] of [[25, 46, 15], [46, 67, 16]]) {
    const need = new Float32Array(n);
    frames.forEach((f, i) => {
      const g = headGeom(f.map(V));
      let m = 0;
      for (let j = a; j < b; j++) m = Math.max(m, surfZ(g, f[j][0], f[j][1]) + clear - f[j][2]);
      need[i] = m;
    });
    const wide = need.map((_, i) => {
      let m = 0;
      for (let j = Math.max(0, i - R); j <= Math.min(n - 1, i + R); j++) m = Math.max(m, need[j]);
      return m;
    });
    const shift = wide.map((_, i) => {
      let t = 0, c = 0;
      for (let j = Math.max(0, i - R); j <= Math.min(n - 1, i + R); j++) { t += wide[j]; c++; }
      return t / c;
    });
    frames.forEach((f, i) => {
      for (let j = a; j < b; j++) f[j][2] += shift[i];
      f[wrist][2] += shift[i];
    });
  }
}
const torso = new THREE.Mesh(new THREE.CylinderGeometry(1, 0.74, 1, 48, 1), M.torso);
torso.castShadow = true;
figure.add(torso);

// standing legs: ankles and feet stay where they are, only the hips follow the body
const legs = {};
for (const s of ['L', 'R']) {
  legs[s] = {
    thigh: bone(M.limb, 0.085 * k, 0.068 * k), shin: bone(M.limb, 0.068 * k, 0.05 * k),
    foot: bone(M.limb, 0.055 * k, 0.05 * k), knee: ball(M.joint, 0.075 * k), ankle: ball(M.joint, 0.06 * k),
  };
}
function stand(hipL, hipR) {
  for (const [s, hip, x] of [['L', hipL, 0.27], ['R', hipR, -0.27]]) {
    const L = legs[s];
    const ankle = new THREE.Vector3(x, FLOOR + 0.09, -0.02);
    const knee = hip.clone().lerp(ankle, 0.5);
    knee.z = 0.05;
    L.thigh(hip, knee);
    L.shin(knee, ankle);
    L.foot(ankle, new THREE.Vector3(x * 1.15, FLOOR + 0.06, 0.26));
    L.knee.position.copy(knee);
    L.ankle.position.copy(ankle);
  }
}

// hands: 21 MediaPipe hand points
const FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
const PALM = [[0, 1], [0, 5], [0, 9], [0, 13], [0, 17], [5, 9], [9, 13], [13, 17], [1, 5], [0, -1]]; // -1: palm centre
function hand(m) {
  const fingers = FINGERS.map((_, i) => {
    const r = i === 0 ? 0.03 : i === 4 ? 0.021 : 0.025;
    return [bone(m, r, r * 0.93), bone(m, r * 0.93, r * 0.85), bone(m, r * 0.85, r * 0.72)];
  });
  const palm = PALM.map(([, b]) => bone(m, b < 0 ? 0.05 : 0.034, b < 0 ? 0.058 : 0.03));
  const centre = new THREE.Vector3();
  return (h) => {
    FINGERS.forEach((F, i) => { for (let j = 0; j < 3; j++) fingers[i][j](h[F[j]], h[F[j + 1]]); });
    PALM.forEach(([a, b], i) => {
      if (b < 0) {
        centre.copy(h[5]).add(h[9]).add(h[13]).add(h[17]).multiplyScalar(0.25);
        palm[i](h[0].clone().lerp(centre, 0.15), h[0].clone().lerp(centre, 0.8));
      } else {
        palm[i](h[a], h[b]);
      }
    });
  };
}
const handL = hand(M.left), handR = hand(M.right);

// ---- one frame
let data = null, spans = [];
const V = (a) => new THREE.Vector3(a[0], a[1], a[2]);

function pose(i) {
  const p = data.frames[i].map(V);
  const ls = p[11], rs = p[12], mid = ls.clone().add(rs).multiplyScalar(0.5);
  const hipL = p[23].clone().setZ(0).lerp(ls, 0.12), hipR = p[24].clone().setZ(0).lerp(rs, 0.12);
  hipL.x *= 0.8;
  hipR.x *= 0.8;
  const hipMid = hipL.clone().add(hipR).multiplyScalar(0.5);
  B.shoulders(ls, rs);
  B.spine(mid, hipMid);
  B.hips(hipL, hipR);
  J.shoulderL.position.copy(ls);
  J.shoulderR.position.copy(rs);
  J.elbowL.position.copy(p[13]);
  J.elbowR.position.copy(p[14]);
  J.hipL.position.copy(hipL);
  J.hipR.position.copy(hipR);
  B.upperL(ls, p[13]);
  B.foreL(p[13], p[25]); // the forearm ends at the hand's own wrist point
  B.upperR(rs, p[14]);
  B.foreR(p[14], p[46]);

  // head and face at the real face points: eyes 2 and 5, nose 0, mouth corners 9 and 10
  const g = headGeom(p), hc = g.c, hh = g.ry * 2, sw = Math.abs(ls.x - rs.x);
  head.position.copy(hc);
  head.scale.set(g.rx, g.ry, g.rz);
  eyeL.position.set(p[2].x, g.eyeY, faceZ(g, p[2].x, g.eyeY) - 0.004);
  eyeR.position.set(p[5].x, g.eyeY, faceZ(g, p[5].x, g.eyeY) - 0.004);
  if (noseLine) drawLine(noseLine, curve(g, p[0].x, p[0].y + hh * F.nose_length, p[0].x, p[0].y, F.line_segments));
  if (mouthLine) drawLine(mouthLine, curve(g, p[9].x, g.mouthY, p[10].x, g.mouthY, F.line_segments));
  B.neck(mid, new THREE.Vector3(hc.x, hc.y - hh / 2 + 0.05, 0.02));

  torso.position.copy(mid).lerp(hipMid, 0.5);
  torso.position.z = -0.04;
  torso.scale.set(sw * 0.5, mid.y - hipMid.y, sw * 0.2);
  stand(hipL, hipR);
  handL(p.slice(25, 46));
  handR(p.slice(46, 67));

  const w = data.word[i];
  spans.forEach((s, j) => s.classList.toggle('on', j === w));
}

// ---- size and camera: fixed in front, pulled back on narrow windows so the arms stay in view
function resize() {
  const width = window.innerWidth, height = Math.max(1, window.innerHeight - caption.offsetHeight);
  stage.style.height = height + 'px';
  renderer.setSize(width, height);
  camera.aspect = width / height;
  const halfTan = Math.tan(THREE.MathUtils.degToRad(cam.fov / 2));
  const distance = Math.max(cam.distance, cam.min_half_width / (halfTan * camera.aspect));
  camera.position.set(target.x, target.y + cam.height, target.z + distance);
  camera.lookAt(target);
  camera.updateProjectionMatrix();
  renderer.render(scene, camera);
}
new ResizeObserver(resize).observe(caption);
window.addEventListener('resize', resize);

// ---- playback in real time at the sequence's fps
let start = 0, shown = -1;
function play() { start = performance.now(); shown = -1; }
function tick(now) {
  requestAnimationFrame(tick);
  if (!data || !data.frames.length) return;
  const i = Math.min(data.frames.length - 1, Math.floor(((now - start) / 1000) * data.fps));
  if (i === shown) return;
  shown = i;
  pose(i);
  renderer.render(scene, camera);
}

function load(next) {
  pushHands(next.frames);
  data = next;
  figure.visible = true;
  caption.replaceChildren();
  spans = next.tokens.map((t) => {
    const s = document.createElement('span');
    s.textContent = t.text;
    if (t.color) s.style.color = t.color;
    caption.appendChild(s);
    return s;
  });
  resize();
  play();
}

replay.addEventListener('click', play);
window.addEventListener('keydown', (e) => { if (e.key === 'r' || e.key === 'R') play(); });

// ---- new sentences arrive by themselves
let loaded = 0;
async function poll() {
  try {
    const { version } = await (await fetch('/version', { cache: 'no-store' })).json();
    if (version > 0 && version !== loaded) {
      const next = await (await fetch('/sequence.json', { cache: 'no-store' })).json();
      loaded = next.version;
      load(next);
    }
  } catch (e) {
    // the demo has stopped: keep showing the last sentence
  }
  setTimeout(poll, settings.poll_ms);
}

await document.fonts.load(`${settings.caption_font_size}px "SignBangla Bengali"`, 'বাবা').catch(() => {});
resize();
poll();
requestAnimationFrame(tick);
