# micropp Dashboard — Setup Guide

## What's in this folder

| File                   | What it does                                              |
|------------------------|-----------------------------------------------------------|
| `tracker.py`           | Copy this into your `src/` folder                        |
| `dashboard_server.py`  | The server — run this before training                    |
| `dashboard.html`       | The browser UI — served automatically by the server      |
| `main_with_tracker.py` | Your updated `main.py` — copy this over the original     |
| `requirements_viz.txt` | One extra dependency (Flask)                             |

---

## Setup (3 steps)

### Step 1 — Copy the files into your project

```
micropp/
├── src/
│   ├── tracker.py           ← copy from this folder
│   ├── main.py              ← replace with main_with_tracker.py
│   ├── comms.py             (unchanged)
│   ├── model.py             (unchanged)
│   ├── schedule.py          (unchanged)
│   ├── monolith.py          (unchanged)
│   └── ping_pong.py         (unchanged)
├── dashboard_server.py      ← copy here (project root)
├── dashboard.html           ← copy here (project root)
└── ...
```

Concretely:
```bash
cp tracker.py         ../micropp/src/tracker.py
cp main_with_tracker.py ../micropp/src/main.py
cp dashboard_server.py  ../micropp/dashboard_server.py
cp dashboard.html        ../micropp/dashboard.html
```

### Step 2 — Install Flask

```bash
pip install flask
```

### Step 3 — Run

Open **two terminals**:

**Terminal 1 — start the dashboard server:**
```bash
cd micropp
python dashboard_server.py
# You should see:
#   Dashboard: http://localhost:5005
```

Open your browser at **http://localhost:5005**

**Terminal 2 — start training:**
```bash
cd micropp
torchrun --nproc-per-node=4 src/main.py
```

The dashboard updates in real time as training progresses.

---

## What you'll see on the dashboard

- **Current step** — which step the training is on
- **Latest loss** — most recent loss value from the last worker
- **Elapsed time** — how long training has been running + ms per step
- **Progress ring** — percentage complete
- **Loss chart** — loss curve over all steps
- **Pipeline timeline** — last 16 events per worker (F=forward, B=backward, S=step)
- **Worker activity** — what each worker is currently doing + a progress bar
- **Step duration chart** — how long each step is taking in milliseconds
- **Event log** — live stream of all events

---

## Changing the schedule

In `main.py`, change line 17:

```python
# Naive (simplest):
from schedule import naive_pipeline_step
# then call: naive_pipeline_step(model, comms, fixed_input, fixed_target, HIDDEN_DIM, device)

# GPipe:
from schedule import gpipe_pipeline_step

# 1F1B (default):
from schedule import onef_oneb_pipeline_step
```

Also update the `schedule_name` in the `tracker.send_schedule(...)` call so
the dashboard shows the right name.

---

## Changes made to main.py (summary)

Only 8 lines were added — nothing was changed or removed:

```python
from tracker import Tracker                     # line 17: import

tracker = Tracker(rank, world_size)             # after init_distributed()

tracker.send_schedule(...)                      # once, before the loop, rank 0 only

tracker.send(step=step, phase="forward")        # top of loop
tracker.send(step=step, phase="backward")       # after backward
tracker.send_timing(step=step, elapsed_ms=...) # timing
tracker.send(step=step, loss=..., phase="step") # loss (last rank only)
tracker.send(step=..., loss=..., phase="done")  # after loop (last rank only)
```

The tracker is **fire-and-forget**: if the server is not running, it silently
does nothing. Training will never crash because of it.

---

## Troubleshooting

**Dashboard shows "waiting for training to start"**
→ Make sure `tracker.py` is in `src/` and you've replaced `main.py`

**No loss values appearing**
→ Loss is only sent by the last rank (`rank == world_size - 1`). This is normal.

**"Connection refused" in the terminal**
→ The dashboard server isn't running. Start `python dashboard_server.py` first.

**Port 5005 already in use**
→ Change the port in both `dashboard_server.py` (line at the bottom: `port=5005`)
  and `tracker.py` (default `server_url` in `__init__`).
