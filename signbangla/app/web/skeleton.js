// 2D skeleton view: draws the same frames as the 3D mannequin (x right, y up, shoulder widths; z is not used).
// Body lines, both hands (right orange, left blue as in 3D) and a simple head with eyes and nose, so the head shake
// and tilt of questions and negation show here too. Settings: app.skeleton in configs/config.yaml.
'use strict';

const HAND_LINES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
  [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
const LH = 25, RH = 46;

export class SkeletonView {
  constructor(canvas, settings) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.s = settings;
    this.frames = null;
    this.index = 0;
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    this.resize();
  }

  resize() {
    const box = this.canvas.parentElement.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.max(1, Math.round(box.width * ratio));
    this.canvas.height = Math.max(1, Math.round(box.height * ratio));
    this.draw(this.index);
  }

  setData(data) {
    this.frames = data ? data.frames : null;
    this.index = 0;
    this.draw(0);
  }

  // shoulder-width coordinates -> canvas pixels
  map(p) {
    const { width, height } = this.canvas;
    const scale = height / (2 * this.s.half_height);
    return [width / 2 + p[0] * scale, height / 2 - (p[1] - this.s.center_y) * scale];
  }

  draw(i) {
    const ctx = this.ctx, C = this.s.colors, { width, height } = this.canvas;
    ctx.fillStyle = C.background;
    ctx.fillRect(0, 0, width, height);
    if (!this.frames || !this.frames.length) return;
    this.index = Math.max(0, Math.min(this.frames.length - 1, i | 0));
    const f = this.frames[this.index];
    const scale = height / (2 * this.s.half_height);
    const lw = Math.max(2, this.s.line_width * scale);
    const P = (k) => this.map(f[k]);
    const line = (a, b, w, color) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = w;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
      ctx.stroke();
    };
    const dot = (p, r, color) => {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
      ctx.fill();
    };

    // body: as the 3D figure, the hips are pulled up a little and in, and each forearm ends at its hand's wrist
    const ls = f[11], rs = f[12];
    const hip = (h, s) => [(h[0] + (s[0] - h[0]) * 0.12) * 0.8, h[1] + (s[1] - h[1]) * 0.12];
    const hipL = this.map(hip(f[23], ls)), hipR = this.map(hip(f[24], rs));
    ctx.fillStyle = C.head;
    ctx.beginPath();
    for (const p of [P(11), P(12), hipR, hipL]) ctx.lineTo(p[0], p[1]);
    ctx.closePath();
    ctx.fill();
    for (const [a, b] of [[P(11), P(12)], [P(11), hipL], [P(12), hipR], [hipL, hipR]]) line(a, b, lw, C.body);
    line(P(11), P(13), lw, C.body);
    line(P(13), P(LH), lw, C.body);
    line(P(12), P(14), lw, C.body);
    line(P(14), P(RH), lw, C.body);
    for (const k of [11, 12, 13, 14]) dot(P(k), lw * 0.7, C.body);

    // head: outline from the ears, eyes and mouth (as the 3D head); eyes and nose move with the face points
    const earL = f[7], earR = f[8];
    const eyeY = (f[2][1] + f[5][1]) / 2, mouthY = (f[9][1] + f[10][1]) / 2;
    const hw = Math.max(Math.abs(earL[0] - earR[0]) * 1.1, 0.42);
    const hh = Math.max(hw * 1.3, (eyeY - mouthY) / 0.36);
    const c = this.map([(earL[0] + earR[0]) / 2, eyeY - hh * 0.05]);
    const neck = this.map([(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2]);
    line(neck, [c[0], c[1] + (hh / 2) * scale * 0.9], lw, C.body);
    ctx.fillStyle = C.head;
    ctx.strokeStyle = C.body;
    ctx.lineWidth = lw * 0.6;
    ctx.beginPath();
    ctx.ellipse(c[0], c[1], (hw / 2) * scale, (hh / 2) * scale, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    dot(this.map([f[2][0], eyeY]), lw * 0.55, C.face);
    dot(this.map([f[5][0], eyeY]), lw * 0.55, C.face);
    dot(P(0), lw * 0.35, C.face);

    // hands, back one first
    const hands = [[LH, C.left_hand], [RH, C.right_hand]];
    for (const [start, color] of hands) {
      const pts = f.slice(start, start + 21);
      if (pts.every((p) => p[0] === 0 && p[1] === 0)) continue;
      const m = pts.map((p) => this.map(p));
      for (const [a, b] of HAND_LINES) line(m[a], m[b], lw * 0.45, color);
      for (const p of m) dot(p, lw * 0.3, color);
    }
  }
}
