# Pipeline Parallelism from Scratch

A hands-on PyTorch project that implements pipeline parallel training from the ground up.

No DeepSpeed. No Megatron. No heavy abstractions.

Just plain PyTorch code to understand how large models are actually split across workers and trained in practice. 

---

## Why this project?

When a model becomes too large to fit on a single GPU, one solution is **pipeline parallelism**.

Instead of placing the whole network on one device, the model is divided into stages and distributed across multiple workers.

Each worker owns only a portion of the layers and passes tensors to the next stage.

```text
GPU 0 → layers 1–4   → sends activations →
GPU 1 → layers 5–8   → sends activations →
GPU 2 → layers 9–12  → sends activations →
GPU 3 → layers 13–16 → computes loss
```

During backpropagation, gradients flow in the opposite direction.

This project recreates that full training flow manually using `torch.distributed`.

---

## What’s inside

```text
micropp/
├── src/
│   ├── comms.py       # Distributed communication helpers
│   ├── model.py       # Sharded model definition
│   ├── schedule.py    # Naive / GPipe / 1F1B schedules
│   ├── main.py        # Main distributed training entry point
│   ├── monolith.py    # Single-process baseline
│   ├── ping_pong.py   # Communication sanity check
│   └── tracker.py     # Optional dashboard logging
├── pyproject.toml
└── README.md
```

Recommended reading order:

```text
comms.py → model.py → schedule.py → main.py
```

---

## Quick start

### 1. Install PyTorch

```bash
pip install torch
```

---

### 2. Test communication

```bash
torchrun --nproc-per-node=2 src/ping_pong.py
```

Expected output:

```text
[Rank 0] Sending: [1.0, 2.0, 3.0]
[Rank 1] Received: [1.0, 2.0, 3.0]
Communication works!
```

---

### 3. Run single-process baseline

```bash
python src/monolith.py
```

This trains the same model normally on one process.

Use it to compare final loss with the distributed version.

---

### 4. Run pipeline parallel training

```bash
torchrun --nproc-per-node=4 src/main.py
```

This launches four workers, each owning one slice of the model.

If everything is working correctly, the final loss should be close to the monolithic baseline.

---

## Core concepts implemented

## 1. Model Sharding

The full MLP is split across workers.

Example with 16 layers and 4 workers:

| Rank | Layers                    |
| ---- | ------------------------- |
| 0    | 1–4                       |
| 1    | 5–8                       |
| 2    | 9–12                      |
| 3    | 13–16 + classifier + loss |

Only:

* Rank 0 receives raw input
* Final rank receives labels
* Final rank computes loss

---

## 2. Communication Between Workers

Built directly using `torch.distributed`.

### Forward pass

* `send_forward()` → send activations to next rank
* `recv_forward()` → receive activations from previous rank
* `isend_forward()` → async non-blocking send

### Backward pass

* `send_backward()` → send gradients to previous rank
* `recv_backward()` → receive gradients from next rank

---

## 3. Pipeline Schedules

The project includes three scheduling strategies.

---

### Naive Pipeline

Forward the full batch first, then backward.

```text
F1 → B1
```

Simple to understand, but inefficient because many workers stay idle.

---

### GPipe

Split the batch into microbatches.

Run all forward passes first, then all backward passes.

```text
F1 F2 F3 F4 B1 B2 B3 B4
```

Better hardware utilization than naive scheduling.

---

### 1F1B (One Forward One Backward)

Most efficient schedule in this project.

After warmup, each worker alternates:

```text
1 forward
1 backward
1 forward
1 backward
```

This reduces pipeline bubbles and activation memory usage.

Used in real production systems.

---

## Microbatching

Example configuration:

```python
BATCH_SIZE = 32
CHUNKS = 4
```

The batch becomes:

```text
4 microbatches of size 8
```

Different workers can process different chunks simultaneously.

---

## Why this project matters

Most people know how to call a training framework.

Far fewer understand what happens underneath:

* how pipeline stages communicate
* why bubbles reduce utilization
* why 1F1B is faster at scale
* how gradients move across workers
* why barriers are needed
* why async communication is tricky
* how to validate correctness

This project focuses on those fundamentals.

---

## Validation

A separate `monolith.py` script trains the same model without distribution.

The goal is simple:

```text
Pipeline loss ≈ Monolith loss
```

That confirms the distributed version is mathematically correct.

---

## Real-world observation

On a local CPU + Gloo + small model setup, naive scheduling may sometimes appear faster than GPipe or 1F1B.

That happens because communication overhead dominates compute.

In real multi-GPU training with large models:

```text
1F1B > GPipe > Naive
```

---

## Common issues

| Problem         | Likely Cause                          |
| --------------- | ------------------------------------- |
| Script hangs    | One worker waiting on recv            |
| `RANK` missing  | Script not launched with `torchrun`   |
| Device mismatch | Model and tensors on different device |
| Wrong loss      | Gradient scaling / chunk logic issue  |

---

## What I learned building this

* PyTorch distributed internals
* Pipeline parallelism mechanics
* Model sharding
* Async communication
* Synchronization barriers
* Debugging distributed deadlocks
* Throughput vs overhead tradeoffs
* Why large-scale training systems are hard

---

## Future improvements

* NCCL backend support
* Multi-GPU benchmarking
* Mixed precision training
* Activation checkpointing
* Uneven layer partitioning
* Tensor parallel integration
* Better dashboard visualizations

---

## References

* GPipe paper
* PipeDream paper
* PyTorch Distributed Docs

---

## Final Note

This project was built to understand distributed training deeply rather than hide behind frameworks.

If you want to learn how large models are *actually* trained, this is a good place to start.
