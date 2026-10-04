#!/usr/bin/env bash
# Run once before run_pipeline.py: reloads the QDMA driver and creates the twelve queues.
set -euo pipefail

qdma_setup_dry_run=0
if [[ $# -eq 1 && $1 == --dry-run ]]; then
    qdma_setup_dry_run=1
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash create_qdma_queue.sh [--dry-run]" >&2
    exit 2
fi
if (( ! qdma_setup_dry_run && EUID != 0 )); then
    exec sudo bash "$0" "$@"
fi

run() {
    if (( qdma_setup_dry_run )); then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

# Reset old queue configuration using the existing installed driver.
if (( qdma_setup_dry_run )) || [[ -d /sys/module/qdma_pf ]]; then
    run modprobe -r qdma-pf
fi
run modprobe qdma-pf
qdma_setup_qmax=/sys/bus/pci/devices/0000:21:00.0/qdma/qmax
if (( qdma_setup_dry_run )); then
    printf 'printf "1024\\n" > %q\n' "$qdma_setup_qmax"
else
    printf '1024\n' > "$qdma_setup_qmax"
fi

# Twelve bidirectional MM queues: one per SLR output buffer.
# Queue 0 also handles seed/count/DDR host I/O during separate stages.
for qdma_setup_queue in {0..11}; do
    run dma-ctl qdma21000 q add idx "$qdma_setup_queue" mode mm dir bi
    run dma-ctl qdma21000 q start idx "$qdma_setup_queue" dir bi
done
if (( ! qdma_setup_dry_run )); then
    for qdma_setup_queue in {0..11}; do
        qdma_setup_device="/dev/qdma21000-MM-${qdma_setup_queue}"
        if [[ ! -c "$qdma_setup_device" ]]; then
            echo "Missing QDMA character device: $qdma_setup_device" >&2
            exit 1
        fi
    done
fi
run dma-ctl qdma21000 q list 0 12
printf 'GPU P2P mapping (one buffer per queue):\n'
qdma_setup_kinds=(samples rows cols edges)
for qdma_setup_slr in 0 1 2; do
    for qdma_setup_kind in 0 1 2 3; do
        qdma_setup_queue=$((4 * qdma_setup_slr + qdma_setup_kind))
        printf '  MM-%s=SLR%s/%s\n' "$qdma_setup_queue" "$qdma_setup_slr" "${qdma_setup_kinds[$qdma_setup_kind]}"
    done
done
