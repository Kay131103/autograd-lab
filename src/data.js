/* 2-D toy datasets for binary classification (-1 / +1 labels).
   Each generator takes n and a "noise" factor and returns [{x:[x,y], label}]. */

function jitter(v, noise) { return v + (Math.random() * 2 - 1) * noise; }

export const DATASETS = {
  moons(n = 100, noise = 0.12) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const t = (i / n) * Math.PI;
      pts.push({ x: [jitter(Math.cos(t), noise), jitter(Math.sin(t) - 0.25, noise)], y: 1 });
      pts.push({ x: [jitter(1 - Math.cos(t), noise), jitter(Math.sin(-t) + 0.25, noise)], y: -1 });
    }
    return pts;
  },

  circles(n = 100, noise = 0.1) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const a = Math.random() * Math.PI * 2;
      pts.push({ x: [jitter(Math.cos(a) * 0.4, noise), jitter(Math.sin(a) * 0.4, noise)], y: 1 });
      pts.push({ x: [jitter(Math.cos(a) * (0.9 + Math.random() * 0.2), noise * 1.4), jitter(Math.sin(a) * (0.9 + Math.random() * 0.2), noise * 1.4)], y: -1 });
    }
    return pts;
  },

  spiral(n = 100, noise = 0.15) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const r = i / n * 4;
      for (const [sign, label] of [[1, 1], [-1, -1]]) {
        const a = (i / n) * 3.5 * Math.PI + (sign === 1 ? 0 : Math.PI);
        pts.push({ x: [jitter(Math.cos(a) * r * 0.25, noise), jitter(Math.sin(a) * r * 0.25, noise)], y: label });
      }
    }
    return pts;
  },

  xor(n = 100, noise = 0.25) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const x = jitter(Math.random() > 0.5 ? 1 : -1, noise);
      const y = jitter(Math.random() > 0.5 ? 1 : -1, noise);
      pts.push({ x: [x, y], y: x * y > 0 ? 1 : -1 });
    }
    return pts;
  },
};

export function normalize(pts) {
  const xs = pts.flatMap((p) => p.x);
  const mean = [0, 1].map((i) => xs.filter((_, j) => j % 2 === i).reduce((a, b) => a + b, 0) / (xs.length / 2));
  const std = [0, 1].map((i) => {
    const col = xs.filter((_, j) => j % 2 === i).map((v) => (v - mean[i]) ** 2);
    return Math.sqrt(col.reduce((a, b) => a + b, 0) / col.length) || 1;
  });
  return pts.map((p) => ({ x: p.x.map((v, i) => (v - mean[i]) / std[i]), y: p.y }));
}
