#!/usr/bin/env bash
# run_all.sh — run rfdetr_bench across all available engine files and collect JSON reports.
#
# Usage:
#   ./benchmarks/scripts/run_all.sh \
#       --engine-dir  trt-files/onnx \
#       --image       test_img.jpg \
#       --device      rtx-5070ti \
#       --out-dir     benchmarks/results \
#       [--bench      ./build/rfdetr_bench] \
#       [--warmup     50] \
#       [--iters      500] \
#       [--batch      1]
#
# Finds all *.engine files under --engine-dir, runs rfdetr_bench for each,
# writes one JSON per engine to --out-dir/<device>_<stem>.json.
# After all runs, calls compare_devices.py to print a summary table.
#
# Requirements: rfdetr_bench binary on PATH or via --bench, jq (optional).

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
ENGINE_DIR="trt-files/onnx"
IMAGE="test_img.jpg"
DEVICE="unknown"
OUT_DIR="benchmarks/results"
BENCH="./build/rfdetr_bench"
WARMUP=50
ITERS=500

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --engine-dir) ENGINE_DIR="$2"; shift 2 ;;
        --image)      IMAGE="$2";      shift 2 ;;
        --device)     DEVICE="$2";     shift 2 ;;
        --out-dir)    OUT_DIR="$2";    shift 2 ;;
        --bench)      BENCH="$2";      shift 2 ;;
        --warmup)     WARMUP="$2";     shift 2 ;;
        --iters)      ITERS="$2";      shift 2 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Validate ──────────────────────────────────────────────────────────────────
if [[ ! -f "$BENCH" ]]; then
    echo "error: rfdetr_bench not found at '$BENCH'. Build with -DRFDETR_BUILD_BENCHMARKS=ON." >&2
    exit 1
fi
if [[ ! -f "$IMAGE" ]]; then
    echo "error: image not found: $IMAGE" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

# ── Discover engines ──────────────────────────────────────────────────────────
mapfile -t ENGINES < <(find "$ENGINE_DIR" -name "*.engine" | sort)
if [[ ${#ENGINES[@]} -eq 0 ]]; then
    echo "No *.engine files found under $ENGINE_DIR" >&2
    exit 1
fi

echo "=== rfdetr benchmark run ==="
echo "  device:     $DEVICE"
echo "  image:      $IMAGE"
echo "  warmup:     $WARMUP  iters: $ITERS"
echo "  engines:    ${#ENGINES[@]}"
echo "  output dir: $OUT_DIR"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
PASS=0
FAIL=0
for ENGINE in "${ENGINES[@]}"; do
    STEM=$(basename "$ENGINE" .engine)
    REPORT="${OUT_DIR}/${DEVICE}_${STEM}.json"

    echo "--- $STEM ---"
    if "$BENCH" image "$ENGINE" "$IMAGE" \
        --device     "$DEVICE" \
        --warmup     "$WARMUP" \
        --iters      "$ITERS" \
        --output-dir "$OUT_DIR" \
        --json; then
        echo "    -> $OUT_DIR"
        PASS=$((PASS + 1))
    else
        echo "    FAILED (exit $?)" >&2
        FAIL=$((FAIL + 1))
    fi
    echo ""
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo "=== Done: $PASS passed, $FAIL failed ==="
echo ""

if command -v python3 &>/dev/null && [[ -f "benchmarks/scripts/compare_devices.py" ]]; then
    python3 benchmarks/scripts/compare_devices.py "$OUT_DIR"
fi
