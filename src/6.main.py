
import time
import torch
import torch.optim as optim

from comms   import PipelineComms, init_distributed
from model   import ShardedMLP
from tracker import Tracker

# ── Pick ONE schedule ─────────────────────────────────────────────────────────
from schedule import naive_pipeline_step   as pipeline_step
#from schedule import gpipe_pipeline_step    as pipeline_step
#from schedule import onef_oneb_pipeline_step as pipeline_step
SCHEDULE_NAME = pipeline_step.__name__.replace("_pipeline_step", "")

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE   = 32
HIDDEN_DIM   = 128
TOTAL_LAYERS = 16
STEPS        = 50    
CHUNKS       = 4

# 1. Distributed setup 
rank, world_size, device = init_distributed()
comms   = PipelineComms(rank, world_size)
tracker = Tracker(rank, world_size)

torch.manual_seed(42)
for _ in range(rank * (TOTAL_LAYERS // world_size) * 2):
    torch.randn(1)

if rank == 0:
    print(f"Starting pipeline parallelism — {world_size} workers ({device})")
    print(f"Model: {TOTAL_LAYERS} layers, hidden_dim={HIDDEN_DIM}, batch={BATCH_SIZE}")
    print(f"Schedule: {SCHEDULE_NAME}  |  Chunks: {CHUNKS}  |  Steps: {STEPS}\n")
    tracker.send_schedule(
        schedule_name=SCHEDULE_NAME,
        chunks=CHUNKS,
        total_steps=STEPS,
        hidden_dim=HIDDEN_DIM,
        total_layers=TOTAL_LAYERS,
    )

# 2. Model 
model     = ShardedMLP(HIDDEN_DIM, TOTAL_LAYERS, rank, world_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Data
if rank == 0:
    fixed_input = torch.randn(BATCH_SIZE, HIDDEN_DIM, device=device)
else:
    fixed_input = BATCH_SIZE

if rank == world_size - 1:
    fixed_target = torch.randint(0, 2, (BATCH_SIZE,), device=device)
else:
    fixed_target = None

# 4. Training loop 
start_time = time.time()
model.train()

for step in range(STEPS):
    optimizer.zero_grad()
    step_start = time.time()

    loss = pipeline_step(
        model, comms, fixed_input, fixed_target,
        HIDDEN_DIM, CHUNKS, device,
        tracker=tracker, step=step
    )

    optimizer.step()

    elapsed_ms = (time.time() - step_start) * 1000
    tracker.send_timing(step=step, elapsed_ms=elapsed_ms)

    if model.is_last:
        loss_val = loss.item()
        if step % 1 == 0:
            print(f"Step {step:02d} | Loss: {loss_val:.4f}")
        tracker.send(step=step, loss=loss_val, phase="step")

# 5. Done — barriers ensure ALL workers finish before process exits 
# Without these, fast workers (rank 3) exit before slow workers (rank 0)

torch.distributed.barrier()   # wait for every worker to reach this line

if model.is_last:
    duration   = time.time() - start_time
    final_loss = loss.item()
    print(f"\nDone! Final loss: {final_loss:.4f} | Total time: {duration:.2f}s")
    tracker.send(step=STEPS - 1, loss=final_loss, phase="done")

torch.distributed.barrier()   
torch.distributed.destroy_process_group()