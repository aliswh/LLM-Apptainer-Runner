#!/usr/bin/env bash
#
# Run one question with one model and one seed.
# Place this script in the same folder as your containers (.sif files),
# or set CONTAINER_DIR below, then run:   bash run_exp.sh
#
# Edit the lines below, then run the script.
#
set -euo pipefail

# ============================================================
#  EDIT THESE LINES:
# ============================================================
# Available models (must match a container in CONTAINER_DIR):
#   medgemma-4b    medgemma-27b    gemma4-31b    llama3-70b    gpt-oss-120b
MODEL="medgemma-4b"
SEED="42"
PROMPT="example/single-txt/patient_01.txt"
OUTDIR="results"
# Where the .sif containers are. Leave empty to use this script's own folder.
CONTAINER_DIR="/student/marccummings/MedGemma-Apptainer-Runner"
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

[[ -n "$CONTAINER_DIR" ]] && CONTAINER_DIR="$(cd "$CONTAINER_DIR" && pwd)" || CONTAINER_DIR="$PROJECT_DIR"

# Find the .sif file for this model anywhere under CONTAINER_DIR
# (its folder name may not match the model, e.g. Medgemma/ holds both medgemma sifs)
SIF_PATH="$(find "$CONTAINER_DIR" -name "$MODEL.sif" -print -quit 2>/dev/null || true)"
[[ -n "$SIF_PATH" && -f "$SIF_PATH" ]] || { echo "ERROR: no container for model '$MODEL' ($MODEL.sif) under $CONTAINER_DIR" >&2; exit 1; }

INPUT="$PROJECT_DIR/$PROMPT"
INSTRUCTIONS="$PROJECT_DIR/$PROMPT"

run_out="$PROJECT_DIR/$OUTDIR/$MODEL/seed_$SEED"
mkdir -p "$run_out"

# Bind our run.py over the container's built-in /opt/run.py so the
# container runs OUR script (works even on older containers whose runscript
# always executes /opt/run.py and doesn't support a custom .py argument).
# Also bind the host data dirs into the container at the same absolute paths
# so the input/instructions/output files are visible inside.
apptainer run --nv \
    --bind "$SCRIPT_DIR/run.py:/opt/run.py" \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --bind "$CONTAINER_DIR:$CONTAINER_DIR" \
    "$SIF_PATH" \
    --model "$MODEL" \
    --instructions "$INSTRUCTIONS" \
    --input "$INPUT" \
    --output "$run_out/result.json" \
    --do-sample --temperature 0.7 \
    --seed "$SEED" \
    --debug

echo "DONE. Result in: $run_out/result.json"
