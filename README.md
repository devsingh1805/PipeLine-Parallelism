# micropp — Pipeline Parallelism from Scratch

A beginner-friendly implementation of pipeline parallelism in PyTorch.

No fancy frameworks. No abstractions you have to dig through. Just plain Python you can read top to bottom.

---

## What is pipeline parallelism?

Normally, a neural network lives on one GPU. But what if the model is too big to fit?

**Pipeline parallelism** splits the model across multiple GPUs like an assembly line:

```
GPU 0  →  layers  1-4   →  sends activations →
GPU 1  →  layers  5-8   →  sends activations →
GPU 2  →  layers  9-12  →  sends activations →
GPU 3  →  layers 13-16  →  computes loss
```

Each GPU only owns a *slice* of the model. They pass data to each other to complete the full forward and backward pass.

---

## Project structure

```
micropp/
├── src/
│   ├── comms.py      ← How GPUs send data to each other
│   ├── model.py      ← The neural network, split into slices
│   ├── schedule.py   ← Three different training schedules
│   ├── main.py       ← Training loop (start here!)
│   ├── monolith.py   ← Same model on one GPU (for comparison)
│   └── ping_pong.py  ← Quick test: can the GPUs talk?
├── pyproject.toml
└── README.md
```

Read the files in this order: `comms.py` → `model.py` → `schedule.py` → `main.py`

---

## Quick start

**No GPU? No problem.** Everything runs on CPU too (just slower).

### Step 1 — Install

```bash
pip install torch
```

### Step 2 — Verify communication works

```bash
torchrun --nproc-per-node=2 src/ping_pong.py
```

You should see:
```
[Rank 0] Sending: [1.0, 2.0, 3.0]
[Rank 1] Received: [1.0, 2.0, 3.0]
Communication works!
```

### Step 3 — Run the single-GPU baseline

```bash
python src/monolith.py
```

This trains the full model on one process. Note the final loss — we'll compare it to the distributed version.

### Step 4 — Run pipeline parallelism

```bash
torchrun --nproc-per-node=4 src/main.py
```

This spawns 4 processes, each taking one slice of the model. The final loss should be similar to the monolith — proof that the distributed version is mathematically equivalent.

---

## The three schedules

Open `src/schedule.py` to see all three. You can swap them in `main.py` by changing the import.

### Schedule 1 — Naive (simplest)

```python
from schedule import naive_pipeline_step
```

One micro-batch, one step at a time. Easy to understand, but most GPUs sit idle most of the time.

```
GPU 0: [==FWD==]·········[==BWD==]
GPU 1:          [==FWD==][==BWD==]
GPU 2:                   [FWD][BWD]
GPU 3:                       [F][B]
         ↑ lots of idle time (the "bubble")
```

### Schedule 2 — GPipe (micro-batching)

```python
from schedule import gpipe_pipeline_step
```

Split the batch into smaller pieces (chunks). Run all forwards, then all backwards. Smaller bubble.

```
GPU 0: [F0][F1][F2][F3]────────[B0][B1][B2][B3]
GPU 1:      [F0][F1][F2][F3][B0][B1][B2][B3]
GPU 2:           [F0][F1][F2][F3][B0][B1][B2][B3]
GPU 3:                [F0][F1][F2][F3][B0][B1][B2][B3]
```

### Schedule 3 — 1F1B (most efficient)

```python
from schedule import onef_oneb_pipeline_step  # already selected in main.py
```

After a warmup phase, each GPU does one forward then one backward — alternating. Very little idle time, and memory-efficient because activations don't pile up.

```
Warmup → Steady (1F1B) → Cooldown
GPU 0: [F0][F1][F2][F3]────[B0][B1][B2][B3]
GPU 1:      [F0][F1][B0][F2][B1][F3][B2][B3]
GPU 2:           [F0][B0][F1][B1][F2][B2][F3][B3]  ← almost no bubble
GPU 3:                [F0][B0][F1][B1][F2][B2][F3][B3]
```

---

## Key concepts explained

### What is a "rank"?

When you run `torchrun --nproc-per-node=4`, it launches 4 copies of your script. Each copy is called a **process**, and each gets a unique ID called its **rank** (0, 1, 2, or 3).

```python
rank = int(os.environ["RANK"])   # which worker am I?
```

### Why does only Rank 0 have the data?

In a real pipeline, data enters at one end and flows through to the other. Rank 0 is the "input stage" — it receives the batch from the dataloader and passes activations forward.

### What is `requires_grad = True`?

When a tensor crosses a process boundary via `recv_forward()`, PyTorch doesn't know it's part of a computation graph. Setting `requires_grad = True` tells the autograd engine: "treat this as an input that needs gradients." Without it, gradients won't flow back through this GPU's layers.

### What is `.detach()` when sending?

`send_forward(output.detach())` cuts the computation graph at the boundary. The *next* GPU builds its own fresh graph from the received tensor. This is intentional — each GPU is responsible for its own backward pass.

### What is a "micro-batch"?

In GPipe and 1F1B, we split the batch (e.g. 32 samples) into smaller chunks (e.g. 4 × 8 samples). This lets multiple GPUs work simultaneously on different chunks. Smaller chunks = less memory per GPU = more pipeline overlap.

---

## Experiment ideas

1. **Compare schedules**: change the import in `main.py` and re-run. Which converges faster? Which is more stable?

2. **Change number of chunks**: try `CHUNKS = 1` (same as naive), `CHUNKS = 4`, `CHUNKS = 8`. How does loss curve change?

3. **Vary model depth**: change `TOTAL_LAYERS`. What happens when it's not evenly divisible by the number of GPUs?

4. **Add timing**: wrap the training loop with `time.time()` per step and print it. Which schedule is fastest per step?

---

## Common errors

| Error | Cause | Fix |
|---|---|---|
| `KeyError: 'RANK'` | Not launched with `torchrun` | Use `torchrun --nproc-per-node=N src/main.py` |
| Hangs forever | One rank is waiting to recv but nothing is sending | All ranks must call the schedule function — no `if rank == 0` guards around it |
| `RuntimeError: Expected all tensors on same device` | Data and model on different devices | Ensure you `.to(device)` both data and model |
| Loss diverges compared to monolith | Gradient scaling off | Make sure you divide loss by `chunks` in GPipe/1F1B |

---

## Further reading

- [GPipe paper](https://arxiv.org/abs/1811.06965) — the micro-batching idea
- [PipeDream paper](https://arxiv.org/abs/1806.03377) — the 1F1B schedule
- [PyTorch distributed docs](https://pytorch.org/docs/stable/distributed.html)
