# MAAC

Code of the paper *MAAC: Multi-Accelerator Architecture and Communication
Co-design for Accelerating Graph Neural Networks* (IEEE Transactions on
Computers).

## System

| | |
|---|---|
| FPGA | AMD Alveo V80 |
| GPU | AMD Instinct MI210 |
| SSD | KIOXIA CD8P NVMe (KCD8XPUG7T68) |
| CPU | AMD EPYC 9654 |
| OS | Ubuntu 24.04, Linux 6.8 |
| Software | ROCm 6.2; Python 3.12 with PyTorch 2.5.0 (ROCm), PyTorch Geometric 2.5.3, `ogb`; CMake ≥ 3.19 |

System set-up:

* Set `iommu=pt` in the GRUB file (`/etc/default/grub`).
* PCIe ACS off.
* The FPGA, the GPU and the NVMe SSD are on the same NUMA node.

## Set-up

### 1. QDMA driver

FPGA P2P needs our modified AMD/Xilinx QDMA Linux driver. Apply the patch to
the upstream driver, then build and install it:

```bash
git clone https://github.com/Xilinx/dma_ip_drivers && cd dma_ip_drivers
git checkout 0ff1e9d3ad0179fdaa246a105deade131e2b951b
git apply /path/to/MAAC/qdma_driver/qdma_p2p_dmabuf.patch
cd QDMA/linux-kernel && make && sudo make install
```

The patch, like the QDMA driver it modifies, is licensed under the GNU GPL v2.

### 2. FPGA

Program the V80 with `bitstream/v80DFXplatform_wrapper.pdi`, then **reboot the
machine**.

After every reboot, create the QDMA queues:

```bash
sudo bash create_qdma_queue.sh
```

### 3. Data

Download the dataset and write the two inputs of the pipeline:

```bash
python prepare_data.py --dataset arxiv        # also: reddit, products
```

* graph files → `fpga_graph_input/` (change with `--graph-root DIR`)
* node features and labels → `data/<dataset>/features.pt` (change with `--data-root DIR`)

### 4. Build

```bash
PYTHON=/path/to/python          # the Python with ROCm PyTorch and PyG
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=$PYTHON \
      -DQDMA_INCLUDE_DIR=/path/to/dma_ip_drivers/QDMA/linux-kernel/driver/include
cmake --build build -j
```

## Run

The SSD must not be the boot drive. Make sure the SSD is empty: the run
overwrites its first 505 MB with the graph.

```bash
sudo $PYTHON run_pipeline.py --dataset products          # also: arxiv, reddit
```

If the data is not in the default folders, point the pipeline to it:

* `--graph-root DIR`: the folder with the graph files (default `fpga_graph_input/`)
* `--data-root DIR`: the folder with the node features (default `data/`)

Add `--output FILE` to save the full report as JSON.

## License

The code in this repository is released under the Clear BSD License
([`LICENSE`](LICENSE)). The QDMA driver patch in `qdma_driver/` is licensed
under the GNU GPL v2.
