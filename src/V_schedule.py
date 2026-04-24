import time
import torch
from II_comms import PipelineComms
from IV_model import ShardedMLP

SLEEP = 0.0   

def _begin(tracker, step, phase):
    if tracker:
        tracker.send(step=step, phase=f"{phase}_start")

def _end(tracker, step, phase):
    if tracker:
        tracker.send(step=step, phase=f"{phase}_end")


def naive_pipeline_step(model: ShardedMLP, comms: PipelineComms,
                        batch, targets, hidden_dim, chunks, device,
                        tracker=None, step=0):

    # FORWARD
    if model.is_first:
        input_data = batch
    else:
        shape = (batch, hidden_dim)
        input_data = comms.recv_forward(shape, device)
        input_data.requires_grad = True

    _begin(tracker, step, "F1")
    time.sleep(SLEEP)
    output = model(input_data, targets if model.is_last else None)
    _end(tracker, step, "F1")

    if not model.is_last:
        comms.send_forward(output.detach())

    # BACKWARD
    if model.is_last:
        _begin(tracker, step, "B1")
        time.sleep(SLEEP)
        output.backward()
        _end(tracker, step, "B1")
    else:
        grad_from_next = comms.recv_backward(output.shape, device)
        _begin(tracker, step, "B1")
        time.sleep(SLEEP)
        output.backward(grad_from_next)
        _end(tracker, step, "B1")

    if not model.is_first:
        comms.send_backward(input_data.grad)

    if model.is_last:
        return output


def gpipe_pipeline_step(model: ShardedMLP, comms: PipelineComms,
                        batch, targets, hidden_dim, chunks, device,
                        tracker=None, step=0):
   

    if model.is_first:
        micro_batches = torch.chunk(batch, chunks)
    if model.is_last:
        micro_targets = targets.chunk(chunks)

    input_buffers  = []
    output_buffers = []

    # ALL FORWARDS
    for i in range(chunks):
        label = f"F{i+1}"

        if model.is_first:
            input_data = micro_batches[i]
        else:
            input_data = comms.recv_forward((batch // chunks, hidden_dim), device)
            input_data.requires_grad = True

        _begin(tracker, step, label)
        time.sleep(SLEEP)

        if model.is_last:
            output = model(input_data, micro_targets[i])
        else:
            output = model(input_data)

        _end(tracker, step, label)

        if not model.is_last:
            comms.send_forward(output.detach())

        input_buffers.append(input_data)
        output_buffers.append(output)

    # ALL BACKWARDS
    if model.is_last:
        total_loss = torch.tensor(0.0, device=device)

    for i in range(chunks):
        label = f"B{i+1}"
        input_data = input_buffers[i]
        output     = output_buffers[i]

        if model.is_last:
            total_loss += output.detach()
            _begin(tracker, step, label)
            time.sleep(SLEEP)
            output.backward()
            _end(tracker, step, label)
        else:
            grad_from_next = comms.recv_backward(output.shape, device)
            _begin(tracker, step, label)
            time.sleep(SLEEP)
            output.backward(grad_from_next)
            _end(tracker, step, label)

        if not model.is_first:
            comms.send_backward(input_data.grad)

    if model.is_last:
        return total_loss / chunks


def onef_oneb_pipeline_step(model: ShardedMLP, comms: PipelineComms,
                             batch, targets, hidden_dim, chunks, device,
                             tracker=None, step=0):

    if model.is_first:
        micro_batches = torch.chunk(batch, chunks)
    if model.is_last:
        micro_targets = targets.chunk(chunks)

    input_buffers  = [None] * chunks
    output_buffers = [None] * chunks
    async_requests = []

    num_warmup = comms.world_size - comms.rank - 1
    num_1f1b   = chunks - num_warmup

    def run_forward(idx):
        label = f"F{idx+1}"

        if model.is_first:
            input_data = micro_batches[idx]
        else:
            input_data = comms.recv_forward(
                (batch // chunks, hidden_dim), device)
            input_data.requires_grad = True

        _begin(tracker, step, label)
        time.sleep(SLEEP)

        if model.is_last:
            output = model(input_data, micro_targets[idx])
        else:
            output = model(input_data)

        _end(tracker, step, label)

        if not model.is_last:
            req = comms.isend_forward(output.detach())
            async_requests.append(req)

        input_buffers[idx]  = input_data
        output_buffers[idx] = output

    def run_backward(idx):
        label = f"B{idx+1}"
        input_data = input_buffers[idx]
        output     = output_buffers[idx]

        if model.is_last:
            loss_val = output.detach().clone()
            _begin(tracker, step, label)
            time.sleep(SLEEP)
            output.backward()
            _end(tracker, step, label)
        else:
            grad_from_next = comms.recv_backward(output.shape, device)
            _begin(tracker, step, label)
            time.sleep(SLEEP)
            output.backward(grad_from_next)
            _end(tracker, step, label)

        # KEY FIX: send_backward must be outside the if/else above
        # so it runs for ALL non-first workers after backward completes
        if not model.is_first:
            comms.send_backward(input_data.grad)

        if model.is_last:
            return loss_val

    if model.is_last:
        total_loss = torch.zeros(1, device=device)

    # Phase 1: Warmup — pure forwards (worker 0 does 3, worker 3 does 0)
    for i in range(num_warmup):
        run_forward(i)

    # Phase 2: Steady state — 1F then 1B interleaved
    for i in range(num_1f1b):
        run_forward(i + num_warmup)
        res = run_backward(i)
        if model.is_last:
            total_loss += res

    # Phase 3: Cooldown — pure backwards to drain
    for i in range(num_warmup):
        res = run_backward(i + num_1f1b)
        if model.is_last:
            total_loss += res

    # Wait for all async isend_forward to complete
    for req in async_requests:
        req.wait()

    if model.is_last:
        return total_loss / chunks
    return None