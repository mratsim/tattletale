#!/usr/bin/env python3
"""
Qwen3.6-35B-A3B bf16-03 full-forward-to-logits fixtures, the complete
embed -> 40 hybrid MoE decoder layers -> norm -> lm_head chain, recorded from the installed transformers modeling on CPU torch bf16.

Generated under tests/fixtures/bf16-03-full-forward-to-logits/Qwen3.6-35B-A3B/:

  - layer-{i:02d}.safetensor.metadata.json.zst, one per decoder layer i
  - no layer payload safetensor leaves the tree, the 004 stats frames carry the recorded surface
  - routing_weights records hold the sequential-run router renormalized routing weights ([seq_len, top_k], cast to the hidden dtype)
  - the boundary input is the embedding output at layer 0, then the prior sequential layer output for layers 1+
  - final_logits.decisions.json.zst carries the per-position decision records, the schema id lives in the `schema` key
  - each decision record holds the argmax id, the top-32 set ids with f32 logits, the margin, the softmax tail probability

Run shape:

  - the chunked run is the installed forward
  - the sequential run replaces the GDN chunked rule with the installed recurrent rule at the modeling-module level
  - so the whole chain runs the exact op sequence the Nim implementation mirrors
  - the expert dispatch is locked to `eager`
  - the default `from_pretrained` backend resolution picks `grouped_mm`, a different accumulation formulation
  - the statistics-only emission keeps the metadata frames and the decisions frame
  - every layer payload tensor is left under the FIXTURE_GENERATION.md tiering rules
  - the Nim full-forward-to-logits test asserts the routing_weights records and the projection decisions
  - the recorded seq-vs-chunked bands stay as metadata documentation
  - the 0.8B full-forward fixtures of the same family keep the earlier shape
  - run the command below twice, both run checksums must match before the fixtures are installed

Tolerances asserted here:

  - the sequential run reproduces its records on a second execution
  - every GDN layer recomputes value-identically from its recorded sequential input
  - the recompute path is the installed norms, the recurrent core and routed block
  - exact fp32 ties at the router top-k boundary are structural in this checkpoint, the tie order of `torch.topk` and sort disagrees
  - `topk_margin_min` and the boundary-tie token count are recorded metadata, never asserted positive
  - the sequential run records its `torch.topk` indices and routing weights as the recorded values
  - the seq-vs-chunked bands stay inside the bounds below

Weights flow through model.safetensors.index.json:

  - the out-of-scope prefixes `mtp.*` (19 keys) and `model.visual.*` (333 keys) are never read
  - `lm_head.weight` (file 26) loads as an independent untied parameter
  - the assert suite below proves embed_tokens and lm_head share no storage and no values

  cd <worktree root> && .venv/bin/python workspace/transformers/tests/testgen/gen_bf16_qwen36moe_03_full_forward_to_logits.py

One model-resident process globally:

  - the full checkpoint carries about 70.2 GB of text-stack weights in bf16, plus the random-initialized draft and vision tower under 2 GB
  - the script verifies a free+inactive+speculative pool above 32 GiB and that no other python/torch process runs before it loads anything
  - weights materialize through the `from_pretrained` streaming loader, so the text stack never exists twice
  - the floor formula (free + inactive + speculative) is the op RAM rule
  - the constant is sized to the measured anonymous peak of this process
  - the loader assigns mmap-backed storages, the 70 GB weight stack uses file-backed pages the OS evicts under pressure
"""

import json
import compression.zstd

ZSTD_WRITE_OPTIONS = {
    compression.zstd.CompressionParameter.compression_level: 19,
    compression.zstd.CompressionParameter.content_size_flag: 1,
    compression.zstd.CompressionParameter.checksum_flag: 1,
}



def write_json_zst(path, obj, ensure_ascii=True):
    """Writes obj as one zstd frame.

    - level 19, content size and checksum recorded in the frame header
    - JSON fixtures stay reviewable via the generator, the frame stays
      out of text diffs
    """
    payload = json.dumps(
        obj, sort_keys=True, indent=2, ensure_ascii=ensure_ascii
    ).encode("utf-8") + b"\n"
    with open(path, "wb") as f:
        f.write(compression.zstd.compress(payload, options=ZSTD_WRITE_OPTIONS))

import os
import subprocess
import sys


import torch  # noqa, the E402 import follows the path insert
from safetensors import safe_open

# The script directory first keeps the sibling generator imports working.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixture_stats import (  # noqa, the E402 import follows the path insert
    argmax_record_from_row,
    grid_of,
    write_argmax_decisions,
    write_json_zst,
    assert_path_equivalent,
)

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (  # noqa, the E402 import follows the path insert
    Qwen3_5MoeConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa, the E402 import follows the path insert
    Qwen3_5MoeForConditionalGeneration,
)
# The GDN replay and router-margin helpers are shared with the layer
# generator and load from its file path, the naming-rule filename
# carries characters import syntax rejects.
def _load_sibling(filename: str):
    """Loads and returns a testgen generator module by filename.

    - the naming rule allows dots and hyphens that import syntax rejects,
      so the module loads from its file path under a derived name
    - a second call returns the cached module, never a second execution
      of its top level
    """
    name = filename[:-3] if filename.endswith(".py") else filename
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    import importlib.util

    path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load sibling generator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_layer_fixtures = _load_sibling("gen_bf16_qwen36moe_01_layer_internals.py")
gdn_forward_replay = _layer_fixtures.gdn_forward_replay
router_margins = _layer_fixtures.router_margins

import transformers  # noqa, the E402 import follows the path insert
TRANSFORMERS_VERSION = transformers.__version__

# Determinism, one intra-op thread and deterministic kernels.
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Identity and fixture-location constants.

NUM_POSITIONS = 6
    # Decision records written, one per input position.
MODEL_NAME = "Qwen3.6-35B-A3B"
INPUT_TEXT = "Hello, how are you?"
GRANDPARENT_DIR = os.path.dirname(os.path.dirname(__file__))


FIXTURE_DIR = os.path.join(
    GRANDPARENT_DIR, "fixtures", "bf16-03-full-forward-to-logits", "Qwen3.6-35B-A3B"
)
MODEL_DIR = os.path.join(
    os.path.dirname(GRANDPARENT_DIR), "tests/hf_models", MODEL_NAME
)
INDEX_PATH = os.path.join(MODEL_DIR, "model.safetensors.index.json")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

NUM_THREADS = 1

# Checkpoint key counts, verbatim from model.safetensors.index.json:
#
# - LANGUAGE_MODEL_KEYS the text stack
# - LM_HEAD_KEYS the untied lm_head
# - VISUAL_KEYS and MTP_KEYS, the two out-of-scope prefixes a text-only
#   load never requests
LANGUAGE_MODEL_KEYS = 692
LM_HEAD_KEYS = 1
VISUAL_KEYS = 333
MTP_KEYS = 19

# GDN chunked-vs-recurrent band bounds:
#
# - on identical inputs the chunked and recurrent GDN cores agree within
#   ~1e-8 f32, but through the bf16 layer boundaries the sub-ULP core
#   differences flip bf16 rounding boundaries, then accumulate through the chain
# - the accumulated band grows with the layer count and the hidden width, since the absolute bf16 ulp tracks residual stream magnitude
# - the 24-layer, 1024-wide Qwen3.5-0.8B chain measured layer-level bands up to ~3.1e-2 and final logits up to ~0.17
# - this 40-layer, 2048-wide chain runs about an order of magnitude higher
# - first generation measured layer-level bands past 0.09 at layer 23, still growing
# - the bounds below are guards sized several times above the expected scale,
#   catching gross generator or checkpoint drift rather than the band itself,
#   the true bands stay measured per run, recorded per layer in the metadata,
#   with the Nim suite asserting against the recorded bands
# - guard scale, measured first hand on the eager-experts evaluation
#   run with the guards neutralized (stubbed saves), the seq-vs-chunked
#   input band peaked 0.500000, the output band peaked 0.500000,
#   the final logits band 0.8828125, both branches still on the bf16 grid
# - the guards sit four to five times above the measured scale
#   and the recorded per-layer bands stay the Nim-side contract
INPUT_BAND_GUARD = 2.00
OUTPUT_BAND_GUARD = 2.00
LOGITS_BAND_GUARD = 4.00

# Pool floor:
#
# - the floor formula stays the op RAM rule, free + inactive + speculative
# - the constant is sized to the measured single-process footprint,
#   this generator runs `from_pretrained` with mmap-backed storages, so
#   the 70 GB weight stack sits on file-backed pages and the anonymous
#   peak stays at a few GiB, measured at 2.2 GiB anon after load, before any forward
# - 32 GiB keeps ample headroom beyond that measured peak
MIN_POOL_BYTES = 32 * 1024 ** 3


def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit("[gen_bf16_qwen36moe_03_full_forward_to_logits] vm_stat gave no page size line")


def pool_bytes() -> int:
    """Free+inactive+speculative physical memory in bytes from vm_stat."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    page = vm_page_size()
    wanted = ("Pages free:", "Pages inactive:", "Pages speculative:")
    pool = 0
    for line in out.stdout.splitlines():
        for label in wanted:
            if line.startswith(label):
                pool += int(line.split()[2].rstrip(".")) * page
    if pool == 0:
        raise SystemExit("[gen_bf16_qwen36moe_03_full_forward_to_logits] vm_stat gave no pool lines")
    return pool


def ancestor_pids() -> set:
    """PIDs of this process and its ancestors, up to init."""
    chain = set()
    pid = os.getpid()
    for _ in range(16):
        if pid <= 1:
            break
        chain.add(pid)
        out = subprocess.run(
            ["ps", "-o", "ppid=", "-p", str(pid)],
            capture_output=True, text=True)
        try:
            pid = int(out.stdout.strip())
        except ValueError:
            break
    return chain


def check_ram() -> None:
    """Refuses to load weights when the memory pool is low or another
    python/torch process holds RAM.

    - the pgrep match excludes this process chain
    - the match command line spells the torch dependency of this run
    """
    pool = pool_bytes()
    if pool < MIN_POOL_BYTES:
        raise SystemExit(
            f"[gen_bf16_qwen36moe_03_full_forward_to_logits] free+inactive+speculative pool "
            f"{pool / 1024 ** 3:.1f} GiB below the "
            f"{MIN_POOL_BYTES / 1024 ** 3:.0f} GiB floor; stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_bf16_qwen36moe_03_full_forward_to_logits] other python/torch processes hold RAM: {stray}; "
            "stop and retry when idle")


def load_wrapper_config() -> Qwen3_5MoeConfig:
    """Load the wrapper Qwen3_5MoeConfig from the checkpoint config.json."""
    with open(CONFIG_PATH) as f:
        wrapper = json.load(f)
    cfg = Qwen3_5MoeConfig.from_dict(wrapper)
    cfg._attn_implementation = "sdpa"
    cfg.text_config._attn_implementation = "sdpa"
    return cfg


def build_model(cfg, weight_map: dict) -> Qwen3_5MoeForConditionalGeneration:
    """Wrapper model from the real checkpoint of 26 safetensors files, bf16 eval on CPU, through the installed `from_pretrained` loader.

    - the loader streams per file with mmap-backed storages, weights
      materialize per file, never the whole 70 GB twice
    - the reference wrapper construction keeps its own lm_head at the outer
      dtype and `from_pretrained` casts every checkpoint tensor
      (all 1045 ship as bf16), so the untied head ends bf16 like the text stack
    - the expert dispatch is locked to `eager`
    - the rotary inv_freq buffers are restored in f32 after the blanket cast,
      the reference rotary forward computes cos and sin in f32, bf16 storage would round the frequency values

    Raises SystemExit when the head shares storage or values with the embedding,
    or when the loaded head disagrees with the raw file-26 `lm_head.weight` tensor.
    """
    assert cfg.tie_word_embeddings is False, "this checkpoint must be untied"
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        MODEL_DIR, config=cfg, dtype=torch.bfloat16,
        experts_implementation="eager")
    resolved = model.config.text_config._experts_implementation
    assert resolved == "eager", (
        f"experts_implementation setting did not take effect: resolved {resolved!r}; "
        "the recorded chain must run the eager expert loop the Nim "
        "implementation mirrors, not the grouped_mm accumulation")
    for layer in model.model.language_model.layers:
        bound = layer.mlp.experts.config._experts_implementation
        assert bound == "eager", (
            f"experts module forward dispatch is {bound!r}, not the locked eager path")
    rotary = model.model.language_model.rotary_emb
    inv_freq = rotary.inv_freq.float()
    original_inv_freq = rotary.original_inv_freq.float()
    model.eval().to(torch.bfloat16)
    rotary.inv_freq = inv_freq
    rotary.original_inv_freq = original_inv_freq

    embed = model.model.language_model.embed_tokens.weight
    head = model.lm_head.weight
    if head.data_ptr() == embed.data_ptr():
        raise SystemExit(
            "[gen_bf16_qwen36moe_03_full_forward_to_logits] lm_head shares storage with embed_tokens; "
            "the untied head was silently tied")
    if bool((head == embed).all()):
        raise SystemExit(
            "[gen_bf16_qwen36moe_03_full_forward_to_logits] lm_head equals embed_tokens elementwise; "
            "the untied head was silently tied")

    # Head tensor identity against the file-26 checkpoint tensor:
    #
    # - a silently tied head would pass the shape check and produce garbage logits
    with safe_open(os.path.join(
            MODEL_DIR, weight_map["lm_head.weight"]), framework="pt") as f:
        raw_head = f.get_tensor("lm_head.weight")
    assert torch.equal(head, raw_head),         "loaded lm_head.weight disagrees with the checkpoint tensor"  # noqa: torch-equal, the same-interpreter construction self-check
    del raw_head
    return model


def install_capture_hooks(layers):
    """Wraps every decoder layer forward to record input and output tensors.

    - returns the capture list (one dict per layer) and a restore closure
    - each install restores the pristine class forward after the run
      (a second install never chains onto a previous wrapper)
    """
    captured = [None] * len(layers)
    originals = []

    def make_wrapper(layer_idx, layer, original):
        def wrapper(hidden_states, *args, **kwargs):
            entry = {"layer_input": hidden_states.clone()}
            output = original(hidden_states, *args, **kwargs)
            entry["layer_output"] = output.clone()
            captured[layer_idx] = entry
            return output

        return wrapper

    for i, layer in enumerate(layers):
        originals.append(layer.forward)
        layer.forward = make_wrapper(i, layer, originals[-1])

    def restore():
        for layer, original in zip(layers, originals):
            layer.forward = original

    return captured, restore


def install_router_hooks(layers):
    """Wraps every router (layer.mlp.gate, the gating mechanism) forward
    to record its logits.

    - the logits are the source of the recorded per-layer topk_indices
      and routing_weights fixtures and of the boundary-tie metadata
    - the checkpoint produces exact fp32 ties at the top-k boundary, so
      the indices record precisely the choice torch.topk made
    """
    routed = [None] * len(layers)
    originals = []

    def make_wrapper(layer_idx, router, original):
        def wrapper(hidden_states):
            logits, *rest = original(hidden_states)
            routed[layer_idx] = logits.clone()
            return (logits, *rest)

        return wrapper

    for i, layer in enumerate(layers):
        originals.append(layer.mlp.gate.forward)
        layer.mlp.gate.forward = make_wrapper(i, layer.mlp.gate, originals[-1])

    def restore():
        for layer, original in zip(layers, originals):
            layer.mlp.gate.forward = original

    return routed, restore


def run_forward(model, input_ids: torch.Tensor, capture_routers: bool):
    """Runs the wrapper forward with per-layer capture, per-router
    capture included when `capture_routers` is set.

    - returns the capture list, the router logits (None when routers were not captured), and the wrapper logits
    - only the sequential run captures routers, its tensors are the ones the Nim implementation asserts 0.00 against
    """
    layers = model.model.language_model.layers
    captured, restore_layer = install_capture_hooks(layers)
    if capture_routers:
        router_captured, restore_router = install_router_hooks(layers)
        with torch.no_grad():
            output = model(input_ids)
        restore_router()
    else:
        router_captured = None
        with torch.no_grad():
            output = model(input_ids)
    restore_layer()
    return captured, router_captured, output.logits


def replay_gdn_layer(layer, layer_input):
    """Manual sequential replay of one GDN decoder layer from its input.

    - recomputes input_layernorm, the GDN block on the recurrent rule,
      the post-attention norm and the routed block
    - the caller verifies the result against the hooked sequential
      forward output through the ulp-band instrument, the manual
      sequential replay and the patched real forward agree within it
    """
    normed = layer.input_layernorm(layer_input)
    gdn_out = gdn_forward_replay(layer.linear_attn, normed, use_recurrent=True)["output"]
    h1 = layer_input + gdn_out
    h2 = layer.post_attention_layernorm(h1)
    moe_out = layer.mlp(h2)
    if isinstance(moe_out, tuple):
        moe_out = moe_out[0]
    return h1 + moe_out


def recorded_router_outputs(router_logits: torch.Tensor, top_k: int) -> dict:
    """Top-k expert ids and routing weights exactly as the reference
    router computes them, from the router logits.

    - softmax over all experts in f32, torch.topk, renormalized
      in f32, cast to the logits dtype
    - the computation is deterministic under the locked thread count,
      so replaying it from the captured logits reproduces the module
      result value-identically, verified through the instrument
    """
    probs = torch.nn.functional.softmax(router_logits, dtype=torch.float32, dim=-1)
    topk_values, topk_indices = torch.topk(probs, top_k, dim=-1)
    renorm = topk_values / topk_values.sum(dim=-1, keepdim=True)
    return {
        "topk_indices": topk_indices,
        "routing_weights": renorm.to(router_logits.dtype),
    }


def boundary_tie_count(router_logits: torch.Tensor, top_k: int) -> int:
    """Token count whose top-k boundary probabilities (descending positions top_k-1 and top_k) tie exactly at fp32.

    - such a tie makes the selected last expert order-sensitive
    - the recorded topk_indices locks the choice torch.topk made
    """
    probs = torch.nn.functional.softmax(router_logits, dtype=torch.float32, dim=-1)
    sorted_probs = torch.sort(probs, dim=-1, descending=True).values
    return int(
        (sorted_probs[:, top_k - 1] == sorted_probs[:, top_k]).sum().item())


def save_layer_metadata(name: str, metadata: dict) -> str:
    """Writes one layer's deterministic metadata frame.

    No layer payload safetensor leaves the tree, the 004 stats frames
    carry the recorded boundary and router surface.

    Returns:
    - the metadata frame path
    """
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    metadata_path = os.path.join(FIXTURE_DIR, f"{name}.safetensor.metadata.json.zst")
    write_json_zst(metadata_path, metadata)
    return metadata_path


def load_tokenizer():
    """Tokenizes prompts through the checkpoint's own tokenizer files with no special tokens added (no bos)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_DIR)


def main() -> None:
    """Runs the full fixture generation end to end."""
    print(f"Generating {MODEL_NAME} full-model full-forward-to-logits fixtures")
    print("=" * 60)
    check_ram()
    print(f"transformers {TRANSFORMERS_VERSION}")

    cfg = load_wrapper_config()
    with open(INDEX_PATH) as f:
        weight_map = json.load(f)["weight_map"]
    assert len(weight_map) == (
        LANGUAGE_MODEL_KEYS + LM_HEAD_KEYS + VISUAL_KEYS + MTP_KEYS), (
        "checkpoint index key count disagrees with the fixed counts")
    model = build_model(cfg, weight_map)
    # The eager setting comes out of build_model and goes into every
    # fixture metadata entry, the consumers see the numeric path the chain ran.
    resolved_impl = model.config.text_config._experts_implementation
    tokenizer = load_tokenizer()

    tokenizer_ids = tokenizer(INPUT_TEXT, add_special_tokens=False)["input_ids"]
    assert tokenizer_ids and len(tokenizer_ids) < 64, (
        "the bf16-03-full-forward-to-logits fixture prompt must stay inside one GDN chunk (64 tokens)")
    input_ids = torch.tensor([tokenizer_ids])
    seq_len = input_ids.shape[1]
    layers = model.model.language_model.layers
    num_layers = len(layers)
    assert num_layers == cfg.text_config.num_hidden_layers == 40, (
        "the checkpoint must carry 40 decoder layers")

    # Chunked forward, the reference ground truth. Router outputs go
    # uncaptured here, the sequential run outputs stay the fixture truth.
    chunked_captured, _, logits_chunked = run_forward(
        model, input_ids, capture_routers=False)

    # Sequential replay through the patched model, twice for determinism.
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as moe_modeling
    original_chunk_rule = moe_modeling.torch_chunk_gated_delta_rule
    moe_modeling.torch_chunk_gated_delta_rule = moe_modeling.torch_recurrent_gated_delta_rule
    seq_captured, seq_router, logits_seq = run_forward(
        model, input_ids, capture_routers=True)
    seq_captured_again, _, logits_seq_again = run_forward(
        model, input_ids, capture_routers=False)
    moe_modeling.torch_chunk_gated_delta_rule = original_chunk_rule
    for i in range(num_layers):
        assert torch.equal(  # noqa: torch-equal, the same-interpreter determinism self-check
            seq_captured[i]["layer_output"], seq_captured_again[i]["layer_output"]
        ), f"sequential replay layer {i} is not deterministic"
    assert torch.equal(logits_seq, logits_seq_again), "sequential replay logits are not deterministic"  # noqa: torch-equal, the same-interpreter determinism self-check

    top_k = cfg.text_config.num_experts_per_tok
    assert top_k == 8, (
        "the bf16-03-full-forward-to-logits fixtures lock the reference top-k of 8 experts")
    max_input_band = 0.0
    max_output_band = 0.0
    for i in range(num_layers):
        layer_obj = layers[i]
        kind = layer_obj.block_type if hasattr(layer_obj, "block_type") else \
            ("linear_attention" if hasattr(layer_obj, "linear_attn") else "full_attention")
        layer_input = chunked_captured[i]["layer_input"]
        layer_input_seq = seq_captured[i]["layer_input"]
        layer_output = chunked_captured[i]["layer_output"]
        layer_output_seq = seq_captured[i]["layer_output"]

        # The two runs see identical inputs at layer 0 and drift only
        # after the first GDN layer. Any nonzero input band at layer 0
        # means the patch changed something outside the GDN core.
        if i == 0:
            assert torch.equal(  # noqa: torch-equal, the same-interpreter determinism self-check
                layer_input, layer_input_seq), \
                "layer 0 inputs must be identical across the chunked and sequential runs"
        input_diff = (layer_input_seq.float() - layer_input.float()).abs().max().item()
        output_diff = (layer_output_seq.float() - layer_output.float()).abs().max().item()

        # The manual sequential replay is the 0.00 reference.
        #
        # Every GDN layer recomputes value-identically from its recorded
        # sequential input through the shared norms, the recurrent GDN
        # core and the routed block, verified through the instrument.
        if kind == "linear_attention":
            replay = replay_gdn_layer(layer_obj, layer_input_seq)
            assert_path_equivalent(replay, layer_output_seq,
                f"manual sequential replay of GDN layer {i} vs the patched forward")

        # Router outputs of the sequential run, the run the Nim
        # implementation asserts 0.00 against:
        #
        # - the recorded indices and weights verify the router through the instrument
        # - the boundary-tie counts and margin go to the metadata
        # - the top-k boundary of this checkpoint carries exact fp32 ties,
        #   the margin is recorded information, never a positive-assert failure
        router_top_margin, _inner = router_margins(seq_router[i])
        router_outputs = recorded_router_outputs(seq_router[i], top_k)
        ties = boundary_tie_count(seq_router[i], top_k)

        # Regime guards keep the chunked-vs-recurrent ladder in the sub-ULP
        # core scale through the full 40-layer chain.
        assert input_diff < INPUT_BAND_GUARD, \
            f"sequential vs chunked layer {i} input diff too large: {input_diff}"
        assert output_diff < OUTPUT_BAND_GUARD, \
            f"sequential vs chunked layer {i} output diff too large: {output_diff}"
        max_input_band = max(max_input_band, input_diff)
        max_output_band = max(max_output_band, output_diff)

        metadata = {
            "model": MODEL_NAME,
            "layer": f"model.language_model.layers.{i}",
            "layer_type": kind,
            "input_text": INPUT_TEXT,
            "input_tokens": tokenizer_ids,
            "batch_size": 1,
            "seq_len": seq_len,
            "num_threads": NUM_THREADS,
            "dtype": "bfloat16",
            "device": "cpu",
            "experts_implementation": resolved_impl,
            "torch_version": torch.__version__,
            "transformers_version": TRANSFORMERS_VERSION,
            "bands": {
                "input_band": input_diff,
                "output_band": output_diff,
            },
            "margins": {
                "topk_margin_min": router_top_margin,
                "boundary_tie_tokens": ties,
            },
            "note": "layer_input_seq is the sequential replay (0.00 for Nim). "
                    "The chunked copies and the sequential outputs left the "
                    "payload under the fixture contract v2; the recorded "
                    "bands keep documenting the divergence they measured.",
        }
        # Statistics-only fixture contract, no layer payload safetensor:
        #
        # - the 004 stats frame carries the router records
        # - the metadata frame carries the recorded bands
        save_layer_metadata(f"layer-{i:02d}", metadata)
        print(f"  layer {i:02d} ({kind}): input_band {input_diff:.3e}, "
              f"output_band {output_diff:.3e}, topk_margin {router_top_margin:.3e}, "
              f"boundary_tie_tokens {ties}")

    logits_diff = (logits_seq.float() - logits_chunked.float()).abs().max().item()
    assert logits_diff < LOGITS_BAND_GUARD, \
        f"sequential vs chunked logits diff too large: {logits_diff}"
    # Decision projection of the sequential reference, the 0.00 Nim
    # comparison:
    #
    # - the raw logits tensors leave the tree
    # - the consumers carry argmax, the top-2 competing pair and the tail probability
    # - the sequential vs chunked band stays as the recorded metadata band
    if logits_seq.shape[1] < NUM_POSITIONS:
        raise SystemExit(
            f"{MODEL_NAME}: the forward produced {logits_seq.shape[1]} "
            f"positions, the script records {NUM_POSITIONS}")
    records = [argmax_record_from_row(logits_seq[0, pos].to(torch.float32))
               for pos in range(NUM_POSITIONS)]
    write_argmax_decisions(
        os.path.join(FIXTURE_DIR, "final_logits.decisions.json.zst"),
        "final_logits.decisions", records, grid_of(logits_seq))
    write_json_zst(os.path.join(FIXTURE_DIR, "final_logits.safetensor.metadata.json.zst"), {
        "model": MODEL_NAME,
        "input_text": INPUT_TEXT,
        "input_tokens": tokenizer_ids,
        "seq_len": seq_len,
        "num_threads": NUM_THREADS,
        "dtype": "bfloat16",
        "device": "cpu",
        "experts_implementation": resolved_impl,
        "torch_version": torch.__version__,
        "transformers_version": TRANSFORMERS_VERSION,
        "bands": {
            "logits_band": logits_diff,
        },
        "note": "the projection is the sequential replay (0.00 for Nim), "
                "logits_band is the sequential vs chunked band.",
    })
    # A summary of the recorded bands. The recorded bands remain
    # the only tolerance source on the Nim side, and these guards catch
    # generator or checkpoint drift at generation time.
    print(f"  max input band {max_input_band:.3e} (guard {INPUT_BAND_GUARD})")
    print(f"  max output band {max_output_band:.3e} (guard {OUTPUT_BAND_GUARD})")
    print(f"  logits band {logits_diff:.3e} (guard {LOGITS_BAND_GUARD})")
    import resource
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)
    print(f"  peak rss of this process: {maxrss:.2f} GiB (anonymous peak)")
    print(f"Generated {num_layers} layer fixtures + final_logits under {FIXTURE_DIR}")
    print("=" * 60)
    print(f"Fixture generation complete: {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
