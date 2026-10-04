#!/usr/bin/env python3
"""Download a dataset and write the two inputs of the pipeline.

    python prepare_data.py --dataset arxiv        # also: reddit, products

    fpga_graph_input/<name>/colptr_<name>_csc.bin   graph topology in CSC form,
    fpga_graph_input/<name>/row_<name>_csc.bin      little-endian int32, no header
                                                    (loaded into the FPGA through the SSD)
    data/<dataset>/features.pt                      node features and labels (go to the GPU)
    data/<dataset>/raw_dataset/                     the PyG / OGB download

CSC form: the neighbors that node v samples from are row[colptr[v] : colptr[v + 1]].
"""
import argparse
from pathlib import Path

import torch
from torch_geometric.sampler import NeighborSampler

ROOT = Path(__file__).resolve().parent
# Folder and file names the pipeline expects (src/workload.cpp).
GRAPH_NAME = {"reddit": "reddit", "products": "ogb_product", "arxiv": "ogb_arxiv"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(GRAPH_NAME), required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--graph-root", type=Path, default=ROOT / "fpga_graph_input")
    args = parser.parse_args()

    out = args.data_root / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    if args.dataset == "reddit":
        from torch_geometric.datasets import Reddit
        dataset = Reddit(str(out / "raw_dataset"))
    else:
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset("ogbn-" + args.dataset, root=str(out / "raw_dataset"))
    data = dataset[0]

    torch.save({"x": data.x.float().contiguous(), "y": data.y.view(-1).long().contiguous(),
                "num_classes": dataset.num_classes}, out / "features.pt")

    # Let PyG convert the edge list to CSC, so the graph is the one a PyG
    # NeighborLoader would sample from (original edge direction).
    sampler = NeighborSampler(data, num_neighbors=[-1])
    colptr, row = sampler.colptr, sampler.row
    assert colptr.numel() == data.num_nodes + 1 and int(colptr[-1]) == row.numel()
    assert max(colptr.numel(), row.numel()) < 2**31, "graph does not fit 32-bit indices"
    name = GRAPH_NAME[args.dataset]
    folder = args.graph_root / name
    folder.mkdir(parents=True, exist_ok=True)
    colptr.numpy().astype("<i4").tofile(folder / f"colptr_{name}_csc.bin")
    row.numpy().astype("<i4").tofile(folder / f"row_{name}_csc.bin")

    print(f"{args.dataset}: {data.num_nodes:,} nodes, {row.numel():,} edges, {data.x.shape[1]} features, "
          f"{dataset.num_classes} classes")
    print(f"  graph    -> {folder}")
    print(f"  features -> {out / 'features.pt'}")


if __name__ == "__main__":
    main()
