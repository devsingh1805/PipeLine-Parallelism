
import torch
from II_comms import PipelineComms, init_distributed

def ping_pong():
    rank, world_size, device = init_distributed()
    comms = PipelineComms(rank, world_size)
    # Sync all processes before starting
    torch.distributed.barrier()
    if rank == 0:
        tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)
        print(f"[Rank 0] Sending: {tensor.tolist()}")
        comms.send_forward(tensor)
    elif rank == 1:
        received = comms.recv_forward((3,), device)
        print(f"[Rank 1] Received: {received.tolist()}")
        print("Communication works!" if received[0] == 1.0 else "Something went wrong.")
    torch.distributed.destroy_process_group()
if __name__ == "__main__":
    ping_pong()
