#!/usr/bin/env bash
#
# run_exp.sh — robustness experiment runner for the Apptainer LLM runner.
#
# Runs a single (or multiple) model(s) across several seeds with sampling
# enabled, so each seed produces a different output for measuring robustness.
#
# Defaults can be overridden with environment variables or CLI flags.
#
# Examples:
#   bash run_exp.sh                          # run all models, seeds 1..5
#   bash run_exp.sh --model medgemma-4b      # single model
#   bash run_exp.sh --seeds "10 20 30"       # custom seeds
#   bash run_exp.sh --instructions task.txt --input patient.txt --outdir results
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RUN_SCRIPT="$SCRIPT_DIR/run.py"

# ---- Options (env-overridable) -----------------------------------

MODEL="${MODEL:-}"
SEEDS="${SEEDS:-1 2 3 4 5}"
TEMPERATURE="${TEMPERATURE:-0.7}"
NVIDIA="${NVIDIA:---nv}"
BIND="${BIND:-}"                     # extra bind mounts: "src:dst,src2:dst2"
INSTRUCTIONS="${INSTRUCTIONS:-$PROJECT_DIR/example/single-txt/patient_01.txt}"
INPUT="${INPUT:-}"
INPUT_DIR="${INPUT_DIR:-}"
IMAGE="${IMAGE:-}"
IMAGE_DIR="${IMAGE_DIR:-}"
OUTDIR="${OUTDIR:-$PROJECT_DIR/results}"
CONTAINER_DIR="${CONTAINER_DIR:-$PROJECT_DIR/containers}"
SIF_EXT="${SIF_EXT:-.sif}"

# ---- CLI flag parsing (override env) -----------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --instructions) INSTRUCTIONS="$2"; shift 2 ;;
        --input) INPUT="$2"; shift 2 ;;
        --input-dir) INPUT_DIR="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --image-dir) IMAGE_DIR="$2"; shift 2 ;;
        --outdir) OUTDIR="$2"; shift 2 ;;
        --container-dir) CONTAINER_DIR="$2"; shift 2 ;;
        --sif-ext) SIF_EXT="$2"; shift 2 ;;
        --no-gpu) NVIDIA=""; shift ;;
        --bind) BIND="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash run_exp.sh [options]"
            echo "  --model NAME          Model name (default: ALL)"
            echo "  --seeds 'S1 S2 ...'    Seeds to run (default: '1 2 3 4 5')"
            echo "  --temperature FLOAT    Sampling temperature (default: 0.7)"
            echo "  --instructions FILE    Task instructions file"
            echo "  --input FILE           Single input file (xor --input-dir)"
            echo "  --input-dir DIR        Batch input directory"
            echo "  --image FILE           Image (multimodal models, single)"
            echo "  --image-dir DIR        Image directory (multimodal, batch)"
            echo "  --outdir DIR           Output directory (default: ./results)"
            echo "  --container-dir DIR    Path to containers/ (default: ../containers)"
            echo "  --sif-ext EXT          SIF file extension (default: .sif)"
            echo "  --no-gpu               Do not add --nv"
            echo "  --bind 'src:dst,...'   Extra apptainer bind mounts"
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---- Validation ---------------------------------------------------

if [[ -z "$MODEL" ]]; then
    # Default: all models that have a built .sif
    MODELS=( "$CONTAINER_DIR"/*/ )
else
    MODELS=( "$CONTAINER_DIR/$MODEL/" )
fi

[[ -n "$INPUT" && -n "$INPUT_DIR" ]] && { echo "ERROR: use only one of --input / --input-dir" >&2; exit 1; }
[[ -z "$INPUT" && -z "$INPUT_DIR" ]] && { echo "ERROR: set --input or --input-dir" >&2; exit 1; }

mkdir -p "$OUTDIR"

SAMPLING_ARGS="--do-sample --temperature $TEMPERATURE"

for model_dir in "${MODELS[@]}"; do
    model_name="$(basename "$model_dir")"
    [[ "$model_name" == ".*" ]] && continue

    sif_path="$model_dir/${model_name}${SIF_EXT}"
    if [[ ! -f "$sif_path" ]]; then
        echo "WARN: skipping $model_name — no container at $sif_path" >&2
        continue
    fi

    for seed in $SEEDS; do
        echo ""
        echo "=========================================================="
        echo "  MODEL: $model_name  |  SEED: $seed  |  TEMP: $TEMPERATURE"
        echo "=========================================================="

        run_out="$OUTDIR/$model_name/seed_$seed"
        mkdir -p "$run_out"

        # Build input/output file args --------------------------------------------------
        if [[ -n "$INPUT" ]]; then
            in_args=(--input "$INPUT")
            out_args=(--output "$run_out/$(basename "$INPUT" .txt).json")
        else
            in_args=(--input-dir "$INPUT_DIR")
            out_args=(--output-dir "$run_out")
        fi

        if [[ -n "$IMAGE" ]]; then
            in_args+=(--image "$IMAGE")
        fi
        if [[ -n "$IMAGE_DIR" ]]; then
            in_args+=(--image-dir "$IMAGE_DIR")
        fi

        bind_args=""
        if [[ -n "$BIND" ]]; then
            bind_args="--bind $BIND"
        fi

        # shellcheck disable=SC2086
        apptainer run $NVIDIA $bind_args \
            --bind "$RUN_SCRIPT:/tmp/run.py" \
            "$sif_path" \
            /tmp/run.py \
            --model "$model_name" \
            --instructions "$INSTRUCTIONS" \
            "${in_args[@]}" \
            "${out_args[@]}" \
            $SAMPLING_ARGS \
            --seed "$seed" \
            --debug
    done
done

echo ""
echo "DONE. Results written to: $OUTDIR"
