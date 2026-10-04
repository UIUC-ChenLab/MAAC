"""GPU side of the pipeline: GraphSAGE training on the sub-graphs sampled by the FPGA.

The FPGA writes every sampled sub-graph straight into GPU memory (P2P). For
each SLR it fills four int32 buffers:

    samples   global IDs of the sampled nodes; the 1,024 seed nodes come first
    rows      source of every sampled edge, as an index into `samples`
    cols      destination of every sampled edge, as an index into `samples`
    edges     position of every sampled edge in the input graph (not used for training)

The native pipeline (`_fpga_pipeline`) owns these buffers. Python sees them as
PyTorch tensors through DLPack, without a copy, and is told how many elements
of each buffer are valid.
"""
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

SEEDS = 1024   # seed nodes per mini-batch
SLRS = 3       # sampling kernels on the FPGA = mini-batches per iteration


@dataclass(frozen=True)
class TrainingConfig:
    layers: int
    learning_rate: float
    hidden_channels: int = 256
    dropout: float = 0.5


def configuration(dataset):
    """Model settings of the paper: 2 layers for Reddit, 3 for Products and Arxiv."""
    return TrainingConfig(2, 0.01) if dataset == "reddit" else TrainingConfig(3, 0.003)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, config):
        super().__init__()
        widths = [in_channels] + [config.hidden_channels] * (config.layers - 1) + [out_channels]
        self.convs = torch.nn.ModuleList(SAGEConv(a, b) for a, b in zip(widths, widths[1:]))
        self.dropout = config.dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i + 1 != len(self.convs):
                x = F.dropout(x.relu(), p=self.dropout, training=self.training)
        return x


def load_features(data_root, dataset):
    """Node features, labels and number of classes from <data_root>/<dataset>/features.pt."""
    data = torch.load(data_root / dataset / "features.pt", map_location="cpu", mmap=True, weights_only=True)
    return data["x"].to(torch.float32), data["y"].reshape(-1).to(torch.int64), int(data["num_classes"])


def import_views(pipeline, slots):
    """views[slot][slr] = (samples, rows, cols, edges): the GPU buffers as int32 tensors."""
    return [[tuple(torch.utils.dlpack.from_dlpack(pipeline.view(slot, slr, kind)) for kind in range(4))
             for slr in range(SLRS)] for slot in range(slots)]


def prepare_subgraph(buffers, counts, features, labels):
    """Turn one SLR's buffers into (x, y, edge_index) for GraphSAGE, entirely on the GPU.

    counts: number of valid elements in (samples, rows, cols, edges).
    """
    samples = buffers[0][:counts[0]].long()
    edge_index = torch.stack((buffers[1][:counts[1]].long(), buffers[2][:counts[2]].long()))
    return features[samples], labels[samples[:SEEDS]], edge_index


def train_phase(pipeline, views, features, labels, model, optimizer, stream, groups):
    """Run `groups` iterations: the FPGA samples three mini-batches, the GPU trains on them.

    The native producer thread keeps sampling and transferring into the second
    set of GPU buffers while the GPU trains here. Nothing is copied back to the
    host inside the loop; losses and timings are read after the last iteration.
    """
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(groups)]
    losses = torch.empty((groups, SLRS), device=features.device)
    rows = []
    wall_start = time.perf_counter_ns()
    pipeline.start(groups)
    with torch.cuda.stream(stream):
        for group in range(groups):
            item = pipeline.acquire()                    # waits until one iteration is in GPU memory
            slot, generation = item["slot"], item["generation"]
            gpu_begin, gpu_end = events[group]
            gpu_begin.record(stream)
            pipeline.begin_consume(slot, generation, int(stream.cuda_stream))
            for slr in range(SLRS):
                x, y, edge_index = prepare_subgraph(views[slot][slr], item["counts"][slr], features, labels)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x, edge_index)[:SEEDS], y)
                loss.backward()
                optimizer.step()
                losses[group, slr].copy_(loss.detach())
            gpu_end.record(stream)
            pipeline.release(slot, generation, int(stream.cuda_stream))   # buffers may be refilled after this
            rows.append(item)
    pipeline.finish()
    stream.synchronize()
    wall_ms = (time.perf_counter_ns() - wall_start) / 1e6

    loss_values = losses.cpu()
    if not bool(torch.isfinite(loss_values).all()):
        raise RuntimeError("training produced a nonfinite loss")
    for item, (gpu_begin, gpu_end), group_losses in zip(rows, events, loss_values):
        item["gpu_ms"] = gpu_begin.elapsed_time(gpu_end)   # feature gathering + three training steps
        item["losses"] = group_losses.tolist()
    return {"groups": groups, "wall_ms": wall_ms, "raw_groups": rows}


def verify_phase(pipeline, views, stream, groups):
    """Compare what arrived in GPU memory with the CPU golden model, element by element.

    In iteration g, SLR s samples mini-batch (s + g) % 3, so every buffer holds
    different data in consecutive iterations and a transfer that silently did
    nothing would be detected. The comparison runs on the GPU; one boolean is
    copied back at the end.
    """
    device = views[0][0][0].device
    references = [tuple(torch.as_tensor(array).to(device) for array in pipeline.reference(minibatch))
                  for minibatch in range(SLRS)]
    matches = torch.empty((groups, SLRS, 4), dtype=torch.bool, device=device)
    stream.synchronize()
    pipeline.start(groups)
    with torch.cuda.stream(stream):
        for group in range(groups):
            item = pipeline.acquire()
            slot, generation = item["slot"], item["generation"]
            pipeline.begin_consume(slot, generation, int(stream.cuda_stream))
            for slr in range(SLRS):
                expected = references[(slr + item["rotation"]) % SLRS]
                for kind in range(4):
                    count = item["counts"][slr][kind]
                    if count != expected[kind].numel():
                        raise RuntimeError(f"iteration {group}, SLR {slr}: element count differs from the golden model")
                    matches[group, slr, kind] = torch.all(views[slot][slr][kind][:count] == expected[kind])
            pipeline.release(slot, generation, int(stream.cuda_stream))
    pipeline.finish()
    stream.synchronize()
    if not bool(matches.all().item()):
        bad = [tuple(index) for index in (~matches).nonzero().tolist()]
        raise RuntimeError(f"GPU data differs from the golden model at (iteration, SLR, buffer): {bad[:8]}")
    return {"groups": groups, "buffers_compared": groups * SLRS * 4, "matched": True}
