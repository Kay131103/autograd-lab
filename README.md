# ∂ Autograd Lab

**Backpropagation from scratch, training a live neural network in your browser.**

A reverse-mode automatic differentiation engine written in ~120 lines of dependency-free
JavaScript (`src/engine.js`), a tanh MLP built on top of it (`src/nn.js`), and an interactive
lab where you watch the gradients flow: decision-boundary heatmap, live loss curve, network
graph with edge weights, and a per-weight gradient inspector.

No PyTorch. No libraries. View source — it's all here.

## What to try
- Set **learning rate to 1.0** on `spiral` — watch the loss explode
- Drop **hidden layers to 0** — the net can't bend, moons stay unsolved
- Crank **noise** and add **L2** — watch it resist memorising
- Pause mid-training and read the **gradient table** (every value hand-computed by the chain rule)

## The engine, in one paragraph
Every scalar in the network is a `Value` that remembers how it was created. Arithmetic
(`+ * ** exp tanh log relu`) builds a computation graph; `loss.backward()` walks it in
reverse topological order applying the chain rule at every node. SGD then subtracts
`lr * grad` (plus optional L2) from every parameter. That's all backpropagation is.

## Files
| File | What it is |
|---|---|
| `src/engine.js` | the autodiff engine (~120 lines) |
| `src/nn.js` | Neuron / Layer / MLP + SGD step + L2 |
| `src/data.js` | moons, circles, spiral, xor generators |
| `src/main.js` | canvas rendering + training loop |

## Dev / build / deploy
```bash
npm install
npm run dev      # → http://localhost:5173
npm run build    # → dist/
```
GitHub Actions deploys to Cloudflare Pages on every push to `main`.
Live: **https://autograd.krithiuk.space**

Part of the [krithiuk.space](https://portfolio.krithiuk.space) lab.
