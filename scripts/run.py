#!/usr/bin/env python3
"""
Consolidated Apptainer LLM runner.

Runs any of the supported models by selecting it with --model.
Supports single-file and batch processing, deterministic greedy decoding
(default) or seeded sampling (--do-sample --seed --temperature) for
robustness runs.

Run from the host WITHOUT rebuilding containers:

    apptainer run --nv \
      --bind $PWD/scripts/run.py:/tmp/run.py \
      containers/<model>/<model>.sif \
      /tmp/run.py \
        --model medgemma-4b \
        --instructions task.txt \
        --input patient.txt --output out.json \
        --do-sample --seed 42 --temperature 0.7
"""

import argparse
import json
import sys
import os
from pathlib import Path
import torch
import re

from transformers import (
    AutoModelForCausalLM,
    set_seed,
)

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

try:
    from transformers import AutoProcessor
except ImportError:
    AutoProcessor = None

try:
    from PIL import Image
except ImportError:
    Image = None


# ---- MODEL REGISTRY -------------------------------------------------

MODELS = {
    "medgemma-4b": {
        "display": "MedGemma 4B (multimodal)",
        "path": "/opt/models/medgemma-4b-it",
        "modality": "image",
        "dtype": "bfloat16",
        "attn": None,
        "max_new_tokens": 300,
        "max_context": None,
        "overlap": 0,
    },
    "medgemma-27b": {
        "display": "MedGemma 27B (multimodal)",
        "path": "/opt/models/medgemma-27b-it",
        "modality": "image",
        "dtype": "bfloat16",
        "attn": "flash_attention_2",
        "max_new_tokens": 500,
        "max_context": 6000,
        "overlap": 200,
        "image_placeholder": 256,
    },
    "llama3-70b": {
        "display": "LLaMA 3.3 70B Instruct",
        "path": "/opt/models/llama-3-70b",
        "modality": "text",
        "dtype": "bfloat16",
        "attn": "flash_attention_2",
        "max_new_tokens": 500,
        "max_context": 10000,
        "overlap": 500,
    },
    "gpt-oss-120b": {
        "display": "GPT-OSS 120B",
        "path": "/opt/models/gpt-oss-120b",
        "modality": "text",
        "dtype": "auto",
        "attn": "kernels-community/vllm-flash-attn3",
        "max_new_tokens": 500,
        "max_context": 3000,
        "overlap": 200,
    },
    "gemma4-31b": {
        "display": "Gemma4 31B (multimodal)",
        "path": "/opt/models/gemma4-31b",
        "modality": "image",
        "dtype": "bfloat16",
        "attn": "sdpa",
        "max_new_tokens": 500,
        "max_context": 100000,
        "overlap": 500,
        "image_placeholder": 1120,
    },
}


def log(msg):
    print(f"[runner] {msg}", file=sys.stderr)


def debug(msg):
    if args.debug:
        print(f"[runner][debug] {msg}", file=sys.stderr)


# ---- ARGUMENTS ------------------------------------------------------

def build_help_text():
    listed = "".join(f"    {k:16s} {v['display']}\n" for k, v in MODELS.items())
    return f"""
Consolidated Apptainer LLM Runner
=================================

Select a model with --model, then process single files or batches.
By default decoding is greedy (deterministic). For robustness runs add
"--do-sample" together with a distinct "--seed" per run.

AVAILABLE MODELS
----------------
{listed}

USAGE
-----
Single-file mode:
  apptainer run --nv --bind $PWD/scripts/run.py:/tmp/run.py <MODEL>.sif \\
    /tmp/run.py --model NAME --instructions FILE --input FILE --output FILE [options]

Batch mode:
  apptainer run --nv --bind $PWD/scripts/run.py:/tmp/run.py <MODEL>.sif \\
    /tmp/run.py --model NAME --instructions FILE --input-dir DIR --output-dir DIR [options]

SAMPLING / ROBUSTNESS
---------------------
--do-sample          Enable sampling (seeds now produce varied outputs)
--temperature FLOAT  Sampling temperature (default 0.7, only with --do-sample)
--seed INT           Seed the RNG (only meaningful with --do-sample)

DEBUG OPTIONS
-------------
--dry-run  Preview actions without loading the model
--debug    Print raw model output before JSON parsing
"""


parser = argparse.ArgumentParser(
    description=build_help_text(),
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument("--model", required=True,
                    help="Model name, one of: " + ", ".join(MODELS.keys()))
parser.add_argument("--instructions", required=False)
parser.add_argument("--input")
parser.add_argument("--input-dir")
parser.add_argument("--output")
parser.add_argument("--output-dir")
parser.add_argument("--image")
parser.add_argument("--image-dir")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--seed")
parser.add_argument("--do-sample", action="store_true",
                    help="Enable sampling (required for seeds to vary output)")
parser.add_argument("--temperature", type=float, default=0.7,
                    help="Sampling temperature (only used with --do-sample)")

args = parser.parse_args()

if args.model not in MODELS:
    sys.exit(f"ERROR: unknown model '{args.model}'. Choose one of: {', '.join(MODELS)}")
CFG = MODELS[args.model]

# ---- Validation ----------------------------------------------------

if bool(args.input) == bool(args.input_dir):
    sys.exit("ERROR: Use exactly one of --input or --input-dir")
if args.input and not args.output:
    sys.exit("ERROR: --output is required in single-file mode")
if args.input_dir and not args.output_dir:
    sys.exit("ERROR: --output-dir is required in batch mode")
if CFG["modality"] == "image" and Image is None:
    sys.exit("ERROR: PIL (Pillow) is not available in this container")

# ---- Seed ----------------------------------------------------------

if args.seed is not None:
    try:
        seed = int(args.seed)
        set_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        log(f"Seed set to: {seed}")
    except (ValueError, TypeError):
        sys.exit("ERROR: 'seed' should be a valid integer number")

# ---- Dry run -------------------------------------------------------

if args.dry_run:
    log("DRY RUN (no model loaded)\n")
    log(f"Model             : {CFG['display']} ({args.model})")
    log(f"Instructions file : {args.instructions}")
    log(f"Do sample         : {args.do_sample}")
    if args.do_sample:
        log(f"Temperature       : {args.temperature}")
    if args.seed is not None:
        log(f"Seed              : {args.seed}")
    if args.input:
        log("Mode              : single-file")
        log(f"Input file        : {args.input}")
        log(f"Output file       : {args.output}")
        log(f"Image file        : {args.image or 'none'}")
    else:
        log("Mode              : batch")
        log(f"Input directory   : {args.input_dir}")
        log(f"Output directory  : {args.output_dir}")
        log(f"Image directory   : {args.image_dir or 'none'}")
    log("\nOutput format     : strict JSON")
    log("\nDry run complete.")
    sys.exit(0)

# ---- System prompt ------------------------------------------------

if args.instructions:
    instructions_text = f"""
You must respond with VALID JSON ONLY.
No prose, no markdown.

JSON SCHEMA:
{{
  "task": "string",
  "input_file": "string",
  "result": "string",
  "confidence": "low | medium | high"
}}

INSTRUCTIONS:
{Path(args.instructions).read_text()}
"""
else:
    instructions_text = """
You must respond with VALID JSON ONLY.
No prose, no markdown.

JSON SCHEMA:
{{
  "task": "string",
  "input_file": "string",
  "result": "string",
  "confidence": "low | medium | high"
}}

INSTRUCTIONS:
Summarize the input.
"""

# ---- Helper Functions ---------------------------------------------

def read_input_content(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".json":
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return json.dumps(data, indent=2)
        else:
            return file_path.read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Error reading input file '{file_path}': {e}")


def parse_json_strict(text: str):
    text = text.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "task": "unknown",
        "result": text,
        "confidence": "low",
        "error": "Failed to parse JSON",
    }


def make_gen_kwargs():
    kwargs = {"max_new_tokens": CFG["max_new_tokens"], "do_sample": args.do_sample}
    if args.do_sample:
        kwargs["temperature"] = args.temperature
    return kwargs


def run_inference(messages, max_new_tokens=None):
    """Raw inference wrapper."""
    kwargs = dict(make_gen_kwargs())
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = max_new_tokens

    if CFG["modality"] == "image":
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)
    else:
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **kwargs)

    generated = output_ids[0][input_len:]
    if CFG["modality"] == "image":
        decoded = processor.decode(generated, skip_special_tokens=True).strip()
    else:
        decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()

    if args.debug:
        debug("\n===== RAW MODEL OUTPUT =====")
        debug(repr(decoded))
        debug("===== END RAW OUTPUT =====\n")

    return decoded


def get_chunks(text, image_placeholder=0):
    if not CFG.get("max_context"):
        return [text]

    max_context = CFG["max_context"]
    overlap = CFG["overlap"]
    effective_limit = max_context - 500 - image_placeholder

    tokens = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(tokens)
    if total_tokens <= effective_limit:
        return [text]

    log(f"Input too large ({total_tokens} tokens). Splitting into chunks...")
    chunks = []
    start = 0
    while start < total_tokens:
        end = min(start + effective_limit, total_tokens)
        chunk_ids = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end == total_tokens:
            break
        start += effective_limit - overlap

    log(f"Split into {len(chunks)} chunks.")
    return chunks


def generate_single_pass(instructions, context, filename, image=None):
    if CFG["modality"] == "image":
        messages = [
            {"role": "system", "content": [{"type": "text", "text": instructions}]},
            {
                "role": "user",
                "content": (
                    [{"type": "text", "text": context}]
                    + ([{"type": "image", "image": image}] if image is not None else [])
                ),
            },
        ]
    else:
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": context},
        ]

    raw_output = run_inference(messages)
    data = parse_json_strict(raw_output)

    if isinstance(data, dict):
        data["input_file"] = filename
    return data


def generate_smart(*, instructions_text, context_text, input_filename, image=None):
    placeholder = 0
    if CFG["modality"] == "image" and image is not None:
        placeholder = CFG.get("image_placeholder", 256)

    chunks = get_chunks(context_text, image_placeholder=placeholder)

    if len(chunks) == 1:
        return generate_single_pass(instructions_text, chunks[0], input_filename, image)

    log(f"Processing {len(chunks)} chunks for {input_filename}...")
    intermediate_results = []
    for i, chunk in enumerate(chunks):
        debug(f"--- Chunk {i+1}/{len(chunks)} ---")
        chunk_inst = (
            f"{instructions_text}\n\n"
            f"NOTE: This is PART {i+1} of a larger file. "
            "Extract any relevant information found in this segment. "
            "If the information is not present, return an empty result."
        )
        current_image = image if i == 0 else None
        response_json = generate_single_pass(chunk_inst, chunk, input_filename, current_image)

        if response_json and "result" in response_json:
            val = response_json["result"]
            if val and str(val).lower() not in ["none", "null", "not found", ""]:
                intermediate_results.append(str(val))

    if not intermediate_results:
        return {
            "task": "processing",
            "input_file": input_filename,
            "result": "No relevant information found in any text chunk.",
            "confidence": "low",
        }

    log("Aggregating results from chunks...")
    combined_context = "\n---\n".join(intermediate_results)
    final_instruction = (
        f"Original Task: {instructions_text}\n\n"
        "Below are extracted findings from different parts of the file. "
        "Consolidate them into a single coherent final answer in the requested JSON format."
    )
    return generate_single_pass(final_instruction, combined_context, input_filename)


# ---- Model Loading ------------------------------------------------

log(f"Model loaded: {CFG['display']}")
log("Loading model into GPU memory...")

load_kwargs = {}
if CFG.get("attn"):
    load_kwargs["attn_implementation"] = CFG["attn"]

if CFG["modality"] == "image":
    if AutoProcessor is None or AutoModelForImageTextToText is None:
        sys.exit("ERROR: multimodal dependencies not available in this container")
    processor = AutoProcessor.from_pretrained(CFG["path"], use_fast=False)
    model = AutoModelForImageTextToText.from_pretrained(
        CFG["path"], dtype=torch.bfloat16, device_map="auto", **load_kwargs
    )
else:
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        sys.exit("ERROR: text model dependencies not available in this container")
    tokenizer = AutoTokenizer.from_pretrained(CFG["path"], use_fast=False)
    if CFG["dtype"] == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            CFG["path"], dtype="auto", device_map="auto", **load_kwargs
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            CFG["path"], dtype=torch.bfloat16, device_map="auto", **load_kwargs
        )

model.eval()
log("Model ready")


def write_output(path, data):
    with Path(path).open("w") as f:
        json.dump(data, f, indent=2)


# =========================
# SINGLE FILE MODE
# =========================

if args.input:
    input_path = Path(args.input)
    output_path = Path(args.output)

    image = None
    if args.image and Image is not None:
        image = Image.open(args.image).convert("RGB")

    context_text = read_input_content(input_path)

    data = generate_smart(
        instructions_text=instructions_text,
        context_text=context_text,
        input_filename=input_path.name,
        image=image,
    )

    try:
        write_output(output_path, data)
    except OSError as e:
        raise RuntimeError(
            f"Cannot write output file '{output_path}'. "
            f"Make sure the directory is bind-mounted and writable."
        ) from e

# =========================
# BATCH MODE
# =========================

else:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir) if args.image_dir else None

    valid_extensions = {".txt", ".csv", ".json"}
    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_extensions
    )

    log(f"Processing {len(files)} files...")
    for idx, fpath in enumerate(files, 1):
        log(f"→ {fpath.name} ({idx}/{len(files)})")

        image = None
        if image_dir:
            for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                candidate = image_dir / f"{fpath.stem}{ext}"
                if candidate.exists():
                    image = Image.open(candidate).convert("RGB")
                    break

        try:
            context = read_input_content(fpath)
            data = generate_smart(
                instructions_text=instructions_text,
                context_text=context,
                input_filename=fpath.name,
                image=image,
            )
            write_output(output_dir / f"{fpath.stem}.json", data)
        except Exception as e:
            log(f"FAILED {fpath.name}: {e}")
