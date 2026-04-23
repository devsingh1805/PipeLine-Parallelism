import os
import torch
import torch.distributed as dist


def init_distributed():
    
    rank       = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Safe defaults in case torchrun didn't set them
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")

    os.environ.pop("GLOO_SOCKET_IFNAME", None)


    device = torch.device("cpu")    
    # if torch.cuda.is_available():
    #     num_gpus = torch.cuda.device_count()
    #     if num_gpus >= world_size:
    #         # Ideal case: one GPU per worker
    #         device = torch.device(f"cuda:{local_rank}")
    #     else:
    #         # Single GPU (or fewer GPUs than workers): all share cuda:0
    #         device = torch.device("cuda:0")
    # else:
    #     device = torch.device("cpu")

    # NCCL requires one GPU per process → can't use it with 1 GPU + 4 workers.
    # Gloo works on CPU and shared memory → always safe on Windows.
    dist.init_process_group(
        backend    = "gloo",   # Always gloo on Windows (NCCL is Linux-only)
        rank       = rank,
        world_size = world_size,
    )

    return rank, world_size, device


class PipelineComms:    

    def __init__(self, rank, world_size):
        self.rank       = rank
        self.world_size = world_size
        self.prev_rank  = rank - 1 if rank > 0 else None
        self.next_rank  = rank + 1 if rank < world_size - 1 else None

    def send_forward(self, tensor):
        """Send activations to the next worker (blocking)."""
        dist.send(tensor.contiguous(), dst=self.next_rank)

    def recv_forward(self, shape, device, dtype=torch.float32):
        """Receive activations from the previous worker (blocking)."""
        buf = torch.zeros(shape, dtype=dtype, device=device)
        dist.recv(buf, src=self.prev_rank)
        return buf

    def isend_forward(self, tensor):
        """Send activations to the next worker (non-blocking)."""
        return dist.isend(tensor.contiguous(), dst=self.next_rank)

    def send_backward(self, tensor):
        """Send gradients back to the previous worker (blocking)."""
        dist.send(tensor.contiguous(), dst=self.prev_rank)

    def recv_backward(self, shape, device, dtype=torch.float32):
        """Receive gradients from the next worker (blocking)."""
        buf = torch.zeros(shape, dtype=dtype, device=device)
        dist.recv(buf, src=self.next_rank)
        return buf