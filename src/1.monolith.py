import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

# ── Try to import the tracker (optional — works without it) ──────────────────
# The tracker lives in the same src/ folder. We add it to the path so Python
# can find it whether you run from the project root or from inside src/.
sys.path.insert(0, os.path.dirname(__file__))                      # <<< ADDED
try:                                                                # <<< ADDED
    from tracker import Tracker                                     # <<< ADDED
    tracker = Tracker(rank=0, world_size=1)                        # <<< ADDED
    # world_size=1 tells the dashboard this is a single-process run
except ImportError:                                                 # <<< ADDED
    tracker = None                                                  # <<< ADDED
    print("(tracker.py not found — running without dashboard)")    # <<< ADDED

# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE   = 32
HIDDEN_DIM   = 128
TOTAL_LAYERS = 16
STEPS        = 50


class MonolithicMLP(nn.Module):
    """The full model in one class — no splitting across workers."""

    def __init__(self, dim, depth):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dim, 2))   # 2-class classifier
        self.net     = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, targets):
        return self.loss_fn(self.net(x), targets)


torch.manual_seed(42)
model     = MonolithicMLP(HIDDEN_DIM, TOTAL_LAYERS)
optimizer = optim.Adam(model.parameters(), lr=0.001)

fixed_input  = torch.randn(BATCH_SIZE, HIDDEN_DIM)
fixed_target = torch.randint(0, 2, (BATCH_SIZE,))

print("Training monolith (single process, ground truth)")
print("Compare the final loss to main.py — they should be similar!\n")

# Tell the dashboard what we're about to run                        # <<< ADDED
if tracker:                                                         # <<< ADDED
    tracker.send_schedule(                                          # <<< ADDED
        schedule_name="monolith",                                   # <<< ADDED
        chunks=1,                                                   # <<< ADDED
        total_steps=STEPS,                                          # <<< ADDED
        hidden_dim=HIDDEN_DIM,                                      # <<< ADDED
        total_layers=TOTAL_LAYERS,                                  # <<< ADDED
    )                                                               # <<< ADDED

start = time.time()
model.train()

for step in range(STEPS):
    optimizer.zero_grad()

    step_start = time.time()                                        # <<< ADDED

    tracker and tracker.send(step=step, phase="forward")           # <<< ADDED

    loss = model(fixed_input, fixed_target)
    loss.backward()

    tracker and tracker.send(step=step, phase="backward")          # <<< ADDED

    optimizer.step()

    # Timing for this step                                          # <<< ADDED
    elapsed_ms = (time.time() - step_start) * 1000                 # <<< ADDED
    tracker and tracker.send_timing(step=step, elapsed_ms=elapsed_ms)  # <<< ADDED

    loss_val = loss.item()

    if step % 5 == 0:
        print(f"Step {step:02d} | Loss: {loss_val:.4f}")

    # Send loss to dashboard every step                             # <<< ADDED
    tracker and tracker.send(step=step, loss=loss_val, phase="step")  # <<< ADDED

# ── Done ─────────────────────────────────────────────────────────────────────
duration   = time.time() - start
final_loss = loss.item()
print(f"\nDone! Final loss: {final_loss:.4f} | Time: {duration:.2f}s")

tracker and tracker.send(                                           # <<< ADDED
    step=STEPS - 1, loss=final_loss, phase="done"                  # <<< ADDED
)                                                                   # <<< ADDED
