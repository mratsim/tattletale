#!/usr/bin/env python3
"""Re-record the chain decision frames of bf16-03-full-forward-to-logits
from the 002 probe schema (ttt-tf-002-logit-decisions-probe-h2) to the 005 argmax schema (ttt-tf-005-argmax-decisions).

The harness's 32-wide truncated-KL instrument applies to chains too, chain
suites read the same ArgmaxRecord records the greedy suites read.

Contract of the re-record:

  - the input ids come verbatim from the existing 002 frame key `input_ids`, falling back to `input_tokens`, never re-tokenized
  - Qwen3-0.6B runs the full forward with use_cache=False (gen_bf16_qwen3_03_full_forward_to_logits.py)
  - Qwen3.5-0.8B runs the sequential-replay wrapper forward of gen_bf16_qwen35dense_03_full_forward_to_logits.py
  - the GDN chunked rule is patched to the recurrent rule and SEED_SEQUENTIAL drives the replay
  - the 6 recorded positions stay the 6 recorded positions, final-logit rows 0..5, one argmax_record_from_row per row
  - no instrument constant serializes, the allowances derive at check time
  - the derivation reads the assert's error kind, the reduction class of tests/harness/harness.nim
  - that reduction class is uniform for every model
  - the pre-write guard refuses any argmax id or margin divergence from the 002 frame
  - a divergence means this machine does not reproduce the committed recording, the swap would corrupt the chain truth

Writes over fixtures/bf16-03-full-forward-to-logits/<Model>/final_logits.decisions.json.zst
via write_argmax_decisions (level 19, content size and checksum recorded).

invocation:

  - cd tattletale
  - .venv/bin/python workspace/transformers/tests/testgen/gen_rerecord_chains_005.py
"""
import json
import os
import struct
import sys

import torch

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TESTS_DIR, "testgen"))

from fixture_stats import (  # noqa, the path insert precedes the import
    argmax_record_from_row,
    read_text_zst,
    write_argmax_decisions,
)

DECISION_PROBE_SCHEMA = "ttt-tf-002-logit-decisions-probe-h2"

# Frame serialization stays free of instrument constants, allowances derive:
#
# - at check they come from the assert's error kind, the reduction class
#   defined in tests/harness/harness.nim

DECISION_ULP_DATATYPE = "bf16"
# The ulp datatype of the chains, written as the 005 decisions frame key
# "ulp_datatype", the unit of every decision allowance.

NUM_POSITIONS = 6


def load_002_frame(model):
    """Returns the existing 002 frame of one model, decompressed and parsed."""
    path = os.path.join(
        TESTS_DIR, "fixtures", "bf16-03-full-forward-to-logits", model,
        "final_logits.decisions.json.zst")
    obj = json.loads(read_text_zst(path))
    if obj["schema"] != DECISION_PROBE_SCHEMA:
        raise SystemExit(f"{model}: frame schema {obj['schema']!r} is not 002")
    ids = obj.get("input_ids", obj.get("input_tokens"))
    if not ids:
        raise SystemExit(f"{model}: 002 frame carries no input ids")
    return path, obj, list(ids)


def load_qwen3(input_text):
    """Returns the Qwen3-0.6B logits on CPU bf16, loaded in the original generator's order."""
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    model_path = os.path.join(TESTS_DIR, "hf_models", "Qwen3-0.6B")
    hf_model = Qwen3ForCausalLM.from_pretrained(model_path)
    hf_model.eval().to("cpu")
    # Preserve inv_freq buffers in float32, model.to(bfloat16) would corrupt them.
    inv_freq = hf_model.model.rotary_emb.inv_freq.float()
    original_inv_freq = hf_model.model.rotary_emb.original_inv_freq.float()
    hf_model = hf_model.to(torch.bfloat16)
    hf_model.model.rotary_emb.inv_freq = inv_freq
    hf_model.model.rotary_emb.original_inv_freq = original_inv_freq
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    with torch.no_grad():
        logits = hf_model(
            **tokenizer(input_text, return_tensors="pt"), use_cache=False
        ).logits
    return logits


def load_qwen35(input_ids):
    """Returns sequential-replay logits from the Qwen3.5-0.8B wrapper on CPU bf16.

    The committed 002 frame records the sequential replay, the 0.00 reference.
    So the re-record runs the same patched forward with the same seed, never
    the chunked run.
    """
    import importlib.util

    gen_path = os.path.join(
        TESTS_DIR, "testgen", "gen_bf16_qwen35dense_03_full_forward_to_logits.py")
    spec = importlib.util.spec_from_file_location("gen_qwen35_03", gen_path)
    gen = importlib.util.module_from_spec(spec)
    sys.modules["gen_qwen35_03"] = gen
    spec.loader.exec_module(gen)

    cfg = gen.load_wrapper_config()
    model = gen.build_model(cfg)
    gen.patch_recurrent()
    ids_tensor = torch.tensor([input_ids], dtype=torch.long)
    _, logits_seq = gen.run_forward(model, ids_tensor, gen.SEED_SEQUENTIAL)
    return logits_seq


def rerecord(model):
    """Re-records one model's decision frame to 005, gated on the 002 truth."""
    path, old, input_ids = load_002_frame(model)
    print(f"== {model}")
    print(f"   input_ids (verbatim from 002): {input_ids}")

    if model == "Qwen3-0.6B":
        logits = load_qwen3(old["input_text"])
    elif model == "Qwen3.5-0.8B":
        logits = load_qwen35(input_ids)
    else:
        raise SystemExit(f"no re-record loader for {model}")

    if logits.shape[1] < NUM_POSITIONS:
        raise SystemExit(
            f"{model}: forward produced {logits.shape[1]} positions, "
            f"the 002 frame records {NUM_POSITIONS}")

    records = []
    print("   pos  argmax_002  argmax_new  match  margin_002  margin_new")
    for pos in range(NUM_POSITIONS):
        row = logits[0, pos].detach().cpu().to(torch.float32)
        rec = argmax_record_from_row(row)
        old_step = old["steps"][pos]
        ok = rec["argmax_id"] == old_step["argmax_id"]
        margin_new = struct.unpack("<d", struct.pack("<Q", int(rec["margin"], 16)))[0]
        print(f"   {pos:>3}  {old_step['argmax_id']:>10}  {rec['argmax_id']:>10}"
              f"  {str(ok):>5}  {old_step['argmax_margin']:>10.6g}  {margin_new:>10.6g}")
        if not ok:
            raise SystemExit(
                f"{model}: argmax divergence at position {pos}: computed "
                f"{rec['argmax_id']} vs recorded {old_step['argmax_id']}. "
                "The HF forward on this machine does not reproduce the "
                "committed recording, refusing to corrupt the chain truth.")
        margin_old = old_step["argmax_margin"]
        # The margins agree to the bf16 grid step, the 002 margins are
        # the f32-rounded top1 - top2 of the same rows.
        if abs(margin_new - margin_old) > 0.5:
            raise SystemExit(
                f"{model}: margin divergence at position {pos}: computed "
                f"{margin_new} vs recorded {margin_old}")
        records.append(rec)

    write_argmax_decisions(path, "final_logits.decisions", records,
                           DECISION_ULP_DATATYPE)
    print(f"   wrote 005 frame over {path} ({len(records)} records)")
    return path


def main():
    """Re-records the decision frames of both chain models to the 005 schema."""
    for model in ["Qwen3-0.6B", "Qwen3.5-0.8B"]:
        rerecord(model)


if __name__ == "__main__":
    main()
