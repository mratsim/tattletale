#!/usr/bin/env python3
"""Bootstrap the fingerprint stats and decision sidecars of the exl3 fixture
families over the committed payload bytes.

The exl3 generators run on the CUDA box (exllamav3 kernels) and emit the
sidecars at record time. This tool produces the identical frames from the
committed safetensor payloads alone, so the mac battery stays green between
the record waves. The entry spec per payload mirrors the generators exactly:
one stats frame per payload, histograms only on the margin-critical tensors,
the final-logits decision projection with its strided probe, and the
fingerprint entry of the retired raw logits tensor.

Payload and metadata bytes are never written; only the sidecar files are.
Run from tests/ inside the uv env:

    uv run python testgen/fixture_sidecar_bootstrap.py
"""

import json
import os
import sys

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from safetensors import safe_open  # noqa: E402

from fixture_stats import (  # noqa: E402
    STATS_SCHEMA,
    decision_steps_probed,
    read_text_zst,
    write_stats_file,
    write_text_zst,
)

DECISION_PROBE_SCHEMA = "ttt-tf-002-logit-decisions-probe-h2"
# Format registry id of the final_logits.decisions.json.zst frames this
# bootstrap writes and verifies.

TESTS_DIR = os.path.dirname(os.path.abspath(__file__)) + "/.."
EXL3_DIR = os.path.join(TESTS_DIR, "fixtures")


def load_tensors(path):
    """Tensor map of one committed safetensor payload."""
    tensors = {}
    with safe_open(path, framework="pt") as f:
        for name in f.keys():
            tensors[name] = f.get_tensor(name)
    return tensors


def linear_spec(name):
    """Entry spec of one 01-layer-internals payload."""
    base = os.path.basename(name)
    if base.startswith("linear-"):
        return [("output", False)]
    if base.startswith("attn-"):
        return [("output", True)]
    if base.startswith("transformer-block-"):
        return [("output", True), ("output_residual", True)]
    raise SystemExit("unknown 01-layer-internals payload: " + base)


def trace_spec(tensors):
    """Entry spec of the block-02 trace payload: every stage tensor, the raw
    input stays out of the sidecar."""
    return [(k, False) for k in sorted(tensors) if k != "input_hidden_states"]


def main():
    written = 0

    # 01-layer-internals: linear (quantile-only), attn and block outputs
    # (histograms, the margin-critical class of the family).
    layer_dir = os.path.join(EXL3_DIR, "exl3-01-layer-internals",
                             "Qwen3-0.6B-EXL3-5bpw-layer-0")
    for name in sorted(os.listdir(layer_dir)):
        if not name.endswith(".safetensor"):
            continue
        path = os.path.join(layer_dir, name)
        tensors = load_tensors(path)
        entries = [(n, tensors[n], h, False) for n, h in linear_spec(name)]
        write_stats_file(path + ".stats.json.zst", name, entries)
        written += 1
    print(f"01-layer-internals: {written} stats frames")

    # 01-block-02-trace: one quantile-only entry per stage tensor.
    trace_path = os.path.join(EXL3_DIR, "exl3-01-block-02-trace",
                              "layer02_trace.safetensor")
    tensors = load_tensors(trace_path)
    entries = [(n, tensors[n], h, False) for n, h in trace_spec(tensors)]
    write_stats_file(trace_path + ".stats.json.zst",
                     os.path.basename(trace_path), entries)
    print(f"01-block-02-trace: 1 stats frame, {len(entries)} entries")

    # 03-full-forward-to-logits: per-layer quantile-only entries, the final
    # logits decision projection plus the fingerprint of the retired raw
    # logits tensor.
    dir03 = os.path.join(EXL3_DIR, "exl3-03-full-forward-to-logits",
                         "Qwen3-0.6B-EXL3-5bpw")
    layers = 0
    for name in sorted(os.listdir(dir03)):
        if not (name.startswith("layer-") and name.endswith(".safetensor")):
            continue
        path = os.path.join(dir03, name)
        tensors = load_tensors(path)
        entries = [("layer_output", tensors["layer_output"], False, False)]
        write_stats_file(path + ".stats.json.zst", name, entries)
        layers += 1
    print(f"03-full-forward: {layers} layer stats frames")

    # Final logits: the raw tensor is retired from the tree once the
    # decision projection and the stats entry carry the recorded surface.
    # With the payload present the frames regenerate from it; after the
    # retirement the committed frames stay frozen data and the tool
    # verifies them instead of failing on the missing source.
    logits_path = os.path.join(dir03, "final_logits.safetensor")
    if os.path.exists(logits_path):
        logits = load_tensors(logits_path)["logits"]
        # Quantile-only entry: the margin-critical surface of the family is
        # the decision projection, and the depth-28 chain drift moves real
        # histogram mass, so the logits fingerprint carries order statistics
        # only.
        write_stats_file(logits_path + ".stats.json.zst",
                         "final_logits.safetensor",
                         [("logits", logits, False, False)])
        write_text_zst(os.path.join(dir03, "final_logits.decisions.json.zst"),
                       json.dumps({
                           "schema": DECISION_PROBE_SCHEMA,
                           "model": "Qwen3-0.6B-EXL3-5bpw",
                           "input_text": "Hello, how are you?",
                           "input_tokens": [9707, 11, 1246, 525, 498, 30],
                           "vocab_size": int(logits.shape[-1]),
                           "steps": decision_steps_probed(
                               logits.to(torch.float32)),
                       }, sort_keys=True, indent=2).encode("utf-8") + b"\n")
        print("03-full-forward: final_logits decisions frame + stats entry")
    else:
        print("03-full-forward: final_logits raw tensor retired, "
              "the committed decision and stats frames stay frozen data")

    # Read every written frame back: a frame that fails to inflate or parse
    # fails the bootstrap instead of shipping silently.
    count = 0
    for family in ("exl3-01-layer-internals", "exl3-01-block-02-trace",
                   "exl3-03-full-forward-to-logits"):
        for root, _dirs, files in os.walk(os.path.join(EXL3_DIR, family)):
            for name in files:
                if not name.endswith((".stats.json.zst", ".decisions.json.zst")):
                    continue
                payload = read_text_zst(os.path.join(root, name))
                parsed = json.loads(payload.decode("utf-8"))
                assert parsed.get("schema") in (DECISION_PROBE_SCHEMA, STATS_SCHEMA), \
                    os.path.join(root, name)
                count += 1
    print(f"verified {count} sidecar frames inflate and parse")
    print("bootstrap complete")


if __name__ == "__main__":
    main()
