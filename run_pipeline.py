#!/usr/bin/env python3
"""MAAC pipeline: NVMe SSD -> FPGA neighbor sampling -> GPU GraphSAGE training.

    NVMe SSD --P2P--> FPGA DDR        graph topology, loaded once
                      FPGA            three SLRs sample three mini-batches at a time
    FPGA HBM --P2P--> GPU memory      the sampled sub-graphs
                      GPU             feature gathering + GraphSAGE training

    sudo $PYTHON run_pipeline.py --dataset products              # train and report throughput
    sudo $PYTHON run_pipeline.py --dataset products --verify     # check the sampled data in GPU memory

WARNING: this program WRITES the graph to the beginning of the NVMe drive
named in src/nvme_loader.hpp. Read README.md first.

One iteration ("group") = the three SLRs sample one mini-batch of 1,024 seed
nodes each, and the GPU performs three training steps. The three mini-batches
are different and fixed; every iteration shifts them by one SLR.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("products", "arxiv", "reddit"), required=True)
    parser.add_argument("--verify", action="store_true",
                        help="no training: compare the sub-graphs in GPU memory with the CPU golden model")
    parser.add_argument("--groups", type=int, default=100, help="measured iterations (3 mini-batches each)")
    parser.add_argument("--warmup-groups", type=int, default=10, help="iterations before the measurement")
    parser.add_argument("--graph-root", type=Path, default=ROOT / "fpga_graph_input",
                        help="int32 CSC graph files written by prepare_data.py")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data",
                        help="folder with <dataset>/features.pt written by prepare_data.py")
    parser.add_argument("--diagnostic", action="store_true",
                        help="print every stage as it happens (for troubleshooting; slows the run down)")
    parser.add_argument("--output", type=Path, help="write the complete report as JSON to this file")
    args = parser.parse_args()
    if args.groups < 1 or args.warmup_groups < 0:
        parser.error("--groups must be positive and --warmup-groups non-negative")
    return args


def print_minibatches(info):
    print(f"Graph: {info['num_nodes']:,} nodes, {info['num_edges']:,} edges; fan-outs {info['fanouts']}")
    for index, (seed, counts) in enumerate(zip(info["minibatch_seeds"], info["minibatch_counts"])):
        print(f"  mini-batch {index} (random seed {seed}): 1,024 seed nodes -> "
              f"{counts[0]:,} sampled nodes, {counts[3]:,} sampled edges")
    if info["skipped_seeds"]:
        print(f"  random seeds skipped because the sub-graph exceeds the FPGA kernel's buffers: {info['skipped_seeds']}")


def summarize(measurement):
    """Throughput and mean stage times of the measured iterations (warm-up excluded)."""
    rows = measurement["raw_groups"]
    minibatches = 3 * len(rows)
    sampling_ms = [row["sampling_all_ms"] for row in rows]
    return {
        "measured_groups": len(rows),
        "measured_minibatches": minibatches,
        "wall_ms": measurement["wall_ms"],
        "end_to_end_minibatches_per_second": minibatches * 1000 / measurement["wall_ms"],
        # The three SLRs run concurrently: one group time covers three mini-batches.
        "sampling_ms_per_group": statistics.mean(sampling_ms),
        "sampling_ms_per_slr": [statistics.mean(row["sampling_slr_ms"][slr] for row in rows) for slr in range(3)],
        "sampling_minibatches_per_second": minibatches * 1000 / sum(sampling_ms),
        "fpga_to_gpu_ms_per_group": statistics.mean(row["p2p_user_ms"] for row in rows),
        "gpu_training_ms_per_group": statistics.mean(row["gpu_ms"] for row in rows),
        "first_group_loss": statistics.mean(rows[0]["losses"]),
        "last_group_loss": statistics.mean(rows[-1]["losses"]),
    }


def print_summary(summary, info, warmup_groups):
    print(f"\nSUMMARY: {summary['measured_groups']} measured iterations x 3 SLRs = "
          f"{summary['measured_minibatches']} mini-batches of 1,024 seed nodes "
          f"({warmup_groups} warm-up iterations excluded)")
    print(f"  NVMe -> FPGA graph loading (once, before the iterations): {info['nvme_load_ms']:.3f} ms")
    slrs = ", ".join(f"SLR{i} {value:.3f} ms" for i, value in enumerate(summary["sampling_ms_per_slr"]))
    print(f"  FPGA sampling, each SLR one mini-batch: {slrs}")
    print(f"  FPGA sampling, all three SLRs together: {summary['sampling_ms_per_group']:.3f} ms per iteration "
          f"= {summary['sampling_minibatches_per_second']:.1f} mini-batches/s")
    print(f"  FPGA -> GPU transfer (12 buffers at once): {summary['fpga_to_gpu_ms_per_group']:.3f} ms per iteration")
    print(f"  GPU feature gathering + 3 training steps: {summary['gpu_training_ms_per_group']:.3f} ms per iteration")
    print(f"  Training loss: {summary['first_group_loss']:.4f} (first iteration) -> "
          f"{summary['last_group_loss']:.4f} (last iteration)")
    print(f"  END-TO-END THROUGHPUT: {summary['end_to_end_minibatches_per_second']:.3f} mini-batches/s "
          f"({summary['measured_minibatches']} mini-batches in {summary['wall_ms']:.1f} ms)")
    print("  Sampling and transfer of one iteration overlap with GPU training of the previous one,")
    print("  so the stage times must not be added up.")


def main():
    args = parse_arguments()
    sys.path.insert(0, str(ROOT / "build"))
    import torch
    import _fpga_pipeline
    from training import GraphSAGE, configuration, import_views, load_features, train_phase, verify_phase
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("no ROCm GPU: start this script with the Python that has ROCm PyTorch and PyG")
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)

    # Read the features before the pipeline is created: a missing file must be
    # noticed before anything is written to the SSD.
    host_features = None if args.verify else load_features(args.data_root, args.dataset)

    report = {"dataset": args.dataset, "mode": "verify" if args.verify else "train",
              "gpu": torch.cuda.get_device_name(0)}
    # Creating the pipeline does all one-time work: CPU golden model, GPU buffers and
    # their registration with the QDMA driver, NVMe -> FPGA graph loading, seed upload.
    pipeline = _fpga_pipeline.Pipeline(str(args.graph_root.resolve()), args.dataset, args.diagnostic)
    views = None
    try:
        info = report["pipeline"] = pipeline.info()
        print_minibatches(info)
        stream = torch.cuda.Stream(device=0)
        views = import_views(pipeline, info["slot_count"])

        if args.verify:
            report["verification"] = verify_phase(pipeline, views, stream, args.groups)
            print(f"VERIFIED: {args.groups} iterations x 3 SLRs x 4 buffers in GPU memory are identical "
                  "to the CPU golden model.")
        else:
            x, y, classes = host_features
            if x.shape[0] != info["num_nodes"]:
                raise ValueError("features.pt and the graph files have different numbers of nodes")
            device = torch.device("cuda", 0)
            features, labels = x.to(device), y.to(device)
            config = configuration(args.dataset)
            model = GraphSAGE(features.shape[1], classes, config).to(device).train()
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            if args.warmup_groups:
                train_phase(pipeline, views, features, labels, model, optimizer, stream, args.warmup_groups)
            report["measurement"] = train_phase(pipeline, views, features, labels, model, optimizer, stream,
                                                args.groups)
            report["summary"] = summarize(report["measurement"])
            print_summary(report["summary"], info, args.warmup_groups)
        report["status"] = "complete"
    finally:
        # Waits for outstanding transfers and GPU work, then removes the
        # registrations before any GPU buffer is freed.
        views = None
        pipeline.close()
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    start = time.perf_counter()
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print(f"run_pipeline.py failed: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"(total run time {time.perf_counter() - start:.1f} s)")
