import { Value } from "./engine.js";
import { MLP } from "./nn.js";
import { DATASETS, normalize } from "./data.js";

/* ---------------- state ---------------- */
let dataset = "moons";
let data = [];
let net = null;
let training = false;
let steps = 0;
let lossHistory = [];
let lr = 0.05, l2 = 0, hiddenCount = 2, width = 8, noise = 0.12;

const $ = (id) => document.getElementById(id);
const boundaryCv = $("boundary"), lossCv = $("lossChart"), netCv = $("netGraph");

/* ---------------- setup / reset ---------------- */
function buildNet() {
  const hidden = Array.from({ length: hiddenCount }, () => width);
  net = new MLP(2, hidden);
  steps = 0;
  lossHistory = [];
}

function regenerate() {
  data = normalize(DATASETS[dataset](100, noise));
}

function reset() {
  training = false;
  $("playBtn").textContent = "▶ train";
  buildNet();
  renderGradTable();
}

/* ---------------- training loop ---------------- */
function tick() {
  if (!training) return;
  const batchSize = 24;
  const batch = Array.from({ length: batchSize }, () => data[Math.floor(Math.random() * data.length)]);
  const { loss, gradNorm } = net.step(batch, lr, l2);
  steps++;
  lossHistory.push(loss);
  if (lossHistory.length > 600) lossHistory.shift();
  updateStats(loss, gradNorm);
  drawAll();
  requestAnimationFrame(tick);
}

function updateStats(loss, gradNorm) {
  $("lossVal").textContent = loss.toFixed(4);
  $("accVal").textContent = (net.accuracy(data) * 100).toFixed(1) + "%";
  $("epochVal").textContent = steps;
  $("gradVal").textContent = gradNorm.toFixed(3);
}

/* ---------------- drawing ---------------- */
function fitCanvas(cv) {
  const w = cv.clientWidth;
  const dpr = window.devicePixelRatio || 1;
  cv.width = w * dpr;
  cv.height = +cv.getAttribute("height") * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  return { ctx, w, h: +cv.getAttribute("height") };
}

const C = {
  blue: "#2563eb", red: "#f87171", blueSoft: "rgba(37,99,235,", redSoft: "rgba(248,113,113,",
  ink: "#1a1a1a", paper: "#fffdf9", sage: "#86efac",
};

function drawBoundary() {
  const { ctx, w, h } = fitCanvas(boundaryCv);
  ctx.clearRect(0, 0, w, h);

  // input domain: [-3, 3]
  const S = 3;
  const toPx = ([x, y]) => [((x + S) / (2 * S)) * w, ((S - y) / (2 * S)) * h];

  // decision surface: evaluate the net on a coarse grid (throttle resolution for speed)
  if (net) {
    const G = 46;
    const gw = w / G, gh = h / G;
    for (let i = 0; i < G; i++) {
      for (let j = 0; j < G; j++) {
        const x = -S + (i / (G - 1)) * 2 * S;
        const y = S - (j / (G - 1)) * 2 * S;
        const out = net.forward([x, y]).data;
        const t = Math.max(-1, Math.min(1, out));      // tanh output ∈ [-1, 1]
        const a = Math.abs(t) * 0.85;
        ctx.fillStyle = t >= 0
          ? `${C.blueSoft}${a})`
          : `${C.redSoft}${a})`;
        ctx.fillRect(i * gw, j * gh, gw + 1, gh + 1);
      }
    }
  }

  // data points
  for (const p of data) {
    const [px, py] = toPx(p.x);
    ctx.beginPath();
    ctx.arc(px, py, 4.2, 0, Math.PI * 2);
    ctx.fillStyle = p.y === 1 ? C.blue : C.red;
    ctx.fill();
    ctx.lineWidth = 1.4;
    ctx.strokeStyle = "#fff";
    ctx.stroke();
  }

  // frame
  ctx.lineWidth = 2;
  ctx.strokeStyle = C.ink;
  ctx.strokeRect(1, 1, w - 2, h - 2);
}

function drawLoss() {
  const { ctx, w, h } = fitCanvas(lossCv);
  ctx.clearRect(0, 0, w, h);
  if (lossHistory.length < 2) {
    ctx.fillStyle = C.ink + "55";
    ctx.font = "12px 'JetBrains Mono'";
    ctx.fillText("training…", 12, 24);
    return;
  }
  const maxL = Math.max(...lossHistory, 0.1);
  ctx.beginPath();
  lossHistory.forEach((l, i) => {
    const x = (i / (lossHistory.length - 1)) * (w - 20) + 10;
    const y = h - 12 - (l / maxL) * (h - 26);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.lineWidth = 2.2;
  ctx.strokeStyle = C.blue;
  ctx.stroke();

  // current loss dot
  const last = lossHistory[lossHistory.length - 1];
  const lx = w - 10, ly = h - 12 - (last / maxL) * (h - 26);
  ctx.beginPath(); ctx.arc(lx, ly, 4, 0, Math.PI * 2); ctx.fillStyle = C.blue; ctx.fill();

  ctx.fillStyle = "#8a8a8a"; ctx.font = "10px 'JetBrains Mono'";
  ctx.fillText(maxL.toFixed(2), 8, 14);
  ctx.fillText(steps + " steps", w - 70, 14);
  ctx.strokeStyle = "#d8d2c8"; ctx.lineWidth = 1;
  ctx.strokeRect(1, 1, w - 2, h - 2);
}

function drawNetGraph() {
  const { ctx, w, h } = fitCanvas(netCv);
  ctx.clearRect(0, 0, w, h);
  if (!net) return;
  const layers = net.layers;
  const colW = (w - 80) / Math.max(1, layers.length - 1);
  const pos = layers.map((L, li) =>
    L.neurons.map((_, ni) => ({
      x: 40 + li * colW,
      y: h / 2 + (ni - (L.neurons.length - 1) / 2) * Math.min(34, (h - 40) / Math.max(1, L.neurons.length)),
    })));

  // edges with width = |weight|, colour = sign
  layers.forEach((L, li) => {
    if (li === layers.length - 1) return;
    L.neurons.forEach((n, ni) => {
      n.w.forEach((wt, pi) => {
        const a = pos[li][pi], b = pos[li + 1][ni];
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
        ctx.lineWidth = Math.min(5, 0.4 + Math.abs(wt.data) * 1.6);
        ctx.strokeStyle = wt.data >= 0 ? C.blue : C.red;
        ctx.globalAlpha = Math.min(0.9, 0.15 + Math.abs(wt.data));
        ctx.stroke();
        ctx.globalAlpha = 1;
      });
    });
  });

  // nodes
  pos.forEach((col, li) => col.forEach((p, ni) => {
    ctx.beginPath(); ctx.arc(p.x, p.y, 9, 0, Math.PI * 2);
    const n = layers[li].neurons[ni];
    const act = n.forward ? null : null;
    ctx.fillStyle = "#fff";
    ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = C.ink; ctx.stroke();
    if (li === 0) {
      ctx.fillStyle = "#8a8a8a"; ctx.font = "10px 'JetBrains Mono'";
      ctx.fillText(ni === 0 ? "x" : "y", p.x - 24, p.y + 3);
    }
    if (li === pos.length - 1) {
      ctx.fillStyle = "#8a8a8a"; ctx.font = "10px 'JetBrains Mono'";
      ctx.fillText("ŷ", p.x + 14, p.y + 3);
    }
  }));
}

function renderGradTable() {
  const tbody = $("gradTable").querySelector("tbody");
  tbody.innerHTML = "";
  if (!net || !net.layers[0]) return;
  net.layers[0].neurons[0].w.forEach((wt, i) => {
    const tr = document.createElement("tr");
    const cls = wt.grad >= 0 ? "pos" : "neg";
    tr.innerHTML = `<td>w${i} → n0</td><td>w = ${wt.data.toFixed(3)}</td><td class="${cls}">∂L/∂w = ${wt.grad.toFixed(4)}</td>`;
    tbody.appendChild(tr);
  });
}

function drawAll() {
  drawBoundary();
  drawLoss();
  drawNetGraph();
  renderGradTable();
}

/* ---------------- wiring ---------------- */
$("lr").addEventListener("input", (e) => { lr = e.target.value / 100; $("lrVal").textContent = lr.toFixed(2); });
$("l2").addEventListener("input", (e) => { l2 = e.target.value / 10000; $("l2Val").textContent = l2.toFixed(3); });
$("noise").addEventListener("input", (e) => { noise = e.target.value / 100; $("noiseVal").textContent = noise.toFixed(2); });
$("width").addEventListener("input", (e) => { width = +e.target.value; $("widthVal").textContent = width; });
$("arch").addEventListener("input", (e) => {
  hiddenCount = +e.target.value;
  $("archVal").textContent = Array.from({ length: hiddenCount }, () => width).join(",") || "0";
});

$("dsGrid").addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (!btn) return;
  for (const b of $("dsGrid").children) b.classList.toggle("active", b === btn);
  dataset = btn.dataset.ds;
  regenerate(); reset(); drawAll();
});

$("playBtn").addEventListener("click", () => {
  training = !training;
  $("playBtn").textContent = training ? "⏸ pause" : "▶ train";
  if (training) tick();
});
$("stepBtn").addEventListener("click", () => {
  const batch = Array.from({ length: 24 }, () => data[Math.floor(Math.random() * data.length)]);
  const { loss, gradNorm } = net.step(batch, lr, l2);
  steps++; lossHistory.push(loss);
  updateStats(loss, gradNorm); drawAll();
});
$("resetBtn").addEventListener("click", () => { regenerate(); reset(); drawAll(); });

window.addEventListener("resize", drawAll);

/* ---------------- boot ---------------- */
regenerate();
reset();
drawAll();
