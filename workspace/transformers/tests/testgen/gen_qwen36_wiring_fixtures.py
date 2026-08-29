#!/usr/bin/env python3
"""Generate the Qwen3.6-35B-A3B full-model wiring fixtures: the complete
embed -> 40 hybrid MoE decoder layers -> norm -> lm_head chain of the real
checkpoint, recorded from the vendored transformers modeling on CPU torch
bf16.

What is generated (under tests/fixtures/qwen36-wiring/):

  layer-{i:02d}.safetensor (+ .metadata.json), one per decoder layer i
    layer_input       the chunked-run layer input (embedding output for
                      layer 0, previous chunked layer output for layers 1+)
    layer_input_seq   the sequential-run layer input at the same boundary
    layer_output      the chunked-forward layer output
    layer_output_seq  the sequential-replay layer output
    topk_indices      the sequential-run router's torch.topk expert ids
                      [seq_len, top_k] int64
    routing_weights   the sequential-run router's renormalized routing
                      weights [seq_len, top_k], cast to the hidden dtype
  final_logits.safetensor (+ .metadata.json)
    logits      the wrapper logits of the chunked forward, all positions
    logits_seq  the sequential-replay logits

The chunked run is the vendored forward. The sequential run replaces the
GDN chunked rule with the vendored recurrent rule at the modeling-module
level, so the whole chain runs the exact op sequence the Nim
implementation mirrors. The expert dispatch stays pinned to `eager`
(the unpinned `from_pretrained` default is `grouped_mm`, a different
accumulation formulation). The Nim wiring test asserts 0.00 against the
*_seq tensors and the recorded seq-vs-chunked bands against the chunked
tensors, mirroring the Qwen3.5-0.8B ids-to-logits fixtures of the same
shape.

Tolerances asserted here:
  - the sequential run is bit-identical on a second execution
  - every GDN layer recomputes bitwise from its recorded sequential
    input through the vendored norms, the recurrent GDN core and the
    routed block
  - exact fp32 ties at the router top-k boundary are structural in this
    checkpoint (`torch.topk` vs sort disagree on the tie order), so
    `topk_margin_min` and the boundary-tie token count are recorded
    metadata, never asserted positive, and the sequential run's
    `torch.topk` indices and routing weights are recorded bitwise
  - the seq-vs-chunked bands stay inside the regime bounds below

Weights flow through model.safetensors.index.json. The out-of-scope
prefixes `mtp.*` (19 keys) and `model.visual.*` (333 keys) are never
read. `lm_head.weight` (shard 26) loads as an independent untied
parameter; the assert battery below proves embed_tokens and lm_head
share no storage and no values.

Run under the pinned ephemeral environment from the worktree root, twice;
the checksums of both runs must be bit-identical before the fixtures are
installed:
  uv run --no-project --python 3.14 --with torch==2.11.0 \
    --with 'tokenizers>=0.23.1,<0.24.0' \
    --with 'transformers @ file://<repo>/_references_prod/transformers' \
    workspace/transformers/tests/testgen/gen_qwen36_wiring_fixtures.py

The python must be 3.14.7 with transformers 5.16.0.dev0, tokenizers
0.23.1, and `uv`'s torch wheel libtorch_cpu.dylib byte-identical to the
one in tattletale/.venv (the exact build the Nim tests link). Verify
with `cmp` before generating.

RAM: one model-resident process globally. The full checkpoint is about
70.2 GB of text-stack weights in bf16 (plus the random-initialized draft
and vision tower under 2 GB), so the script verifies a
free+inactive+speculative pool above 32 GiB and that no other
python/torch process is running before it loads anything. Weights
materialize through the vendored `from_pretrained` streaming loader, so
the text stack never exists twice. The floor formula (free +
inactive + speculative) is the op RAM rule; the constant is sized to the
measured anonymous peak of this process (a few GiB) because the vendored
loader assigns mmap-backed storages: the 70 GB weight stack rides
file-backed pages the OS evicts under pressure.
"""

import json
from collections import OrderedDict
import os
import subprocess
import sys
import torch
from safetensors import safe_open
from safetensors import torch as st

# The reference transformers checkout is the source of truth: the lookup
# walks up from this script until a _references_prod/transformers/src
# directory appears (a worktree sits one level deeper than the repo root).
# QWEN36_VENDORED_SRC overrides the walk.
_here = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = _here
VENDORED_SRC = None
for _ in range(8):
    _candidate = os.path.join(_REPO_ROOT, "_references_prod", "transformers", "src")
    if os.path.isdir(_candidate):
        VENDORED_SRC = _candidate
        break
    _REPO_ROOT = os.path.dirname(_REPO_ROOT)
if VENDORED_SRC is None:
    VENDORED_SRC = os.environ.get("QWEN36_VENDORED_SRC")
if not VENDORED_SRC or not os.path.isdir(VENDORED_SRC):
    raise SystemExit(
        f"[gen_qwen36_wiring_fixtures] vendored modeling not found above {_here}. "
        "Set QWEN36_VENDORED_SRC to the _references_prod/transformers/src directory")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeForConditionalGeneration,
)
from gen_qwen36_layer_fixtures import (  # noqa: E402
    gdn_forward_replay,
    router_margins,
)

import transformers  # noqa: E402
TRANSFORMERS_VERSION = transformers.__version__

# Determinism: single intra-op thread, deterministic kernels.
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Config.
MODEL_NAME = "Qwen3.6-35B-A3B"
INPUT_TEXT = "Hello, how are you?"
GRANDPARENT_DIR = os.path.dirname(os.path.dirname(__file__))
FIXTURE_DIR = os.path.join(GRANDPARENT_DIR, "fixtures", "qwen36-wiring")
MODEL_DIR = os.path.join(
    os.path.dirname(GRANDPARENT_DIR), "tests/hf_models", MODEL_NAME
)
INDEX_PATH = os.path.join(MODEL_DIR, "model.safetensors.index.json")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# The vendored tree sha this fixture was generated against.
VENDORED_SHA = "36deb0b53ed0863f4b4dfdea23dcaec7f3df3701"
NUM_THREADS = 1

# Counts of the real checkpoint (model.safetensors.index.json): the text
# stack, the untied head, and the two out-of-scope prefixes a text-only
# load never requests.
LANGUAGE_MODEL_KEYS = 692
LM_HEAD_KEYS = 1
VISUAL_KEYS = 333
MTP_KEYS = 19

# The GDN chunked-vs-recurrent band regime. The chunked and recurrent GDN
# cores agree to ~1e-8 f32 on identical inputs, but through the bf16
# layer boundaries the sub-ULP core differences flip bf16 rounding
# boundaries and accumulate. The accumulated band grows with the layer
# count and the hidden width, since the absolute bf16 ulp tracks
# residual stream magnitude. The 24-layer, 1024-wide Qwen3.5-0.8B chain
# measured layer-level bands up to ~3.1e-2 and final logits up to ~0.17.
# This 40-layer, 2048-wide chain runs about an order of magnitude higher:
# first generation measured layer-level bands past 0.09 at layer 23,
# still growing. The bounds below are therefore tripwires, several times
# the expected scale in size, catching gross generator or checkpoint
# drift, not the band itself: the true bands are measured per run,
# recorded per layer in the metadata, with the Nim suite asserting
# against the recorded bands.
# Tripwire scale, measured first hand on the eager-experts regime, run
# with these guards neutralized (stubbed saves): seq-vs-chunked input
# band peaked 0.500000, output band peaked 0.500000, final logits band
# 0.8828125, both branches still riding the bf16 grid. The tripwires sit
# four to five times above the measured scale, with the recorded
# per-layer bands staying the Nim-side contract.
INPUT_BAND_GUARD = 2.00
OUTPUT_BAND_GUARD = 2.00
LOGITS_BAND_GUARD = 4.00

# Pool floor. The floor formula stays the op RAM-rulebook: free +
# inactive + speculative, with the constant sized to the measured
# single-process footprint of this generator: `from_pretrained` assigns
# mmap-backed storages, so the 70 GB weight stack rides file-backed
# pages and the anonymous peak stays at a few GiB, measured at 2.2 GiB
# anon after load, before any forward. 32 GiB keeps ample headroom
# beyond that peak.
MIN_POOL_BYTES = 32 * 1024 ** 3


def vm_page_size() -> int:
    """macOS VM page size from the vm_stat header line."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if "page size of" in line:
            return int(line.split("page size of")[1].split()[0])
    raise SystemExit("[gen_qwen36_wiring_fixtures] vm_stat gave no page size line")


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
        raise SystemExit("[gen_qwen36_wiring_fixtures] vm_stat gave no pool lines")
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
    """Refuse to load weights when the memory pool is low or another
    python/torch process holds RAM (this process chain is excluded from
    the pgrep match, whose command line spells the torch dependency of
    this run)."""
    pool = pool_bytes()
    if pool < MIN_POOL_BYTES:
        raise SystemExit(
            f"[gen_qwen36_wiring_fixtures] free+inactive+speculative pool "
            f"{pool / 1024 ** 3:.1f} GiB below the "
            f"{MIN_POOL_BYTES / 1024 ** 3:.0f} GiB floor; stop and retry when idle")
    out = subprocess.run(
        ["pgrep", "-f", r"python.*(torch|hf)"], capture_output=True, text=True)
    found = {int(p) for p in out.stdout.split() if p.strip().isdigit()}
    stray = sorted(found - ancestor_pids())
    if stray:
        raise SystemExit(
            f"[gen_qwen36_wiring_fixtures] other python/torch processes hold RAM: {stray}; "
            "stop and retry when idle")


def check_vendored_sha() -> str:
    """Record the vendored tree sha and fail loudly on drift."""
    out = subprocess.run(
        ["git", "-C", VENDORED_SRC, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True)
    sha = out.stdout.strip()
    if sha != VENDORED_SHA:
        raise SystemExit(
            f"[gen_qwen36_wiring_fixtures] vendored tree sha {sha} differs from "
            f"{VENDORED_SHA}; the fixture truth moved")
    return sha


def load_wrapper_config() -> Qwen3_5MoeConfig:
    """Load the wrapper Qwen3_5MoeConfig from the checkpoint config.json."""
    with open(CONFIG_PATH) as f:
        wrapper = json.load(f)
    cfg = Qwen3_5MoeConfig.from_dict(wrapper)
    cfg._attn_implementation = "sdpa"
    cfg.text_config._attn_implementation = "sdpa"
    return cfg


def index_counts(weight_map: dict) -> None:
    """Pin the checkpoint index against the verbatim key counts of the real
    checkpoint: the text stack, the untied head, and the two out-of-scope
    prefixes."""
    total = len(weight_map)
    expected = LANGUAGE_MODEL_KEYS + LM_HEAD_KEYS + VISUAL_KEYS + MTP_KEYS
    assert total == expected, (
        f"[gen_qwen36_wiring_fixtures] checkpoint index holds {total} keys, "
        f"expected {expected}")
    visual = sum(1 for key in weight_map if key.startswith("model.visual."))
    mtp = sum(1 for key in weight_map if key.startswith("mtp."))
    lm_head = sum(1 for key in weight_map if key == "lm_head.weight")
    assert visual == VISUAL_KEYS, f"visual tensors: {visual}, expected {VISUAL_KEYS}"
    assert mtp == MTP_KEYS, f"mtp tensors: {mtp}, expected {MTP_KEYS}"
    assert lm_head == LM_HEAD_KEYS, "exactly one lm_head.weight entry expected"


def build_model(cfg, weight_map: dict) -> Qwen3_5MoeForConditionalGeneration:
    """Wrapper model from the real 26-shard checkpoint, bf16, eval, CPU,
    through the vendored `from_pretrained` (shard-streaming, mmap-backed:
    weights materialize per shard, never the whole 70 GB twice).

    The reference wrapper construction keeps its own lm_head at the outer
    dtype; from_pretrained casts every checkpoint tensor (this checkpoint
    ships all 1045 as bf16), so the untied head ends bf16 like the text
    stack. The expert dispatch is pinned to `eager`: the unpinned
    `from_pretrained` default resolves to `grouped_mm`, whose reshape+sum
    accumulation is a different formulation from the eager index_add_
    loop the Nim implementation mirrors, so the fixture truth is the
    eager loop. The rotary inv_freq buffers are restored in f32 after the
    blanket cast: the reference rotary forward computes cos/sin in f32 and
    bf16 storage would round the frequency values.

    Raises SystemExit when the head shares storage or values with the
    embedding, or when the loaded head disagrees with the raw shard-26
    `lm_head.weight` tensor."""
    assert cfg.tie_word_embeddings is False, "this checkpoint must be untied"
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        MODEL_DIR, config=cfg, dtype=torch.bfloat16,
        experts_implementation="eager")
    resolved = model.config.text_config._experts_implementation
    assert resolved == "eager", (
        f"experts_implementation pin did not take effect: resolved {resolved!r}; "
        "the recorded chain must run the eager expert loop the Nim "
        "implementation mirrors, not the grouped_mm accumulation")
    for layer in model.model.language_model.layers:
        bound = layer.mlp.experts.config._experts_implementation
        assert bound == "eager", (
            f"experts module forward dispatch is {bound!r}, not the pinned eager path")
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
            "[gen_qwen36_wiring_fixtures] lm_head shares storage with embed_tokens; "
            "the untied head was silently tied")
    if torch.equal(head, embed):
        raise SystemExit(
            "[gen_qwen36_wiring_fixtures] lm_head equals embed_tokens elementwise; "
            "the untied head was silently tied")

    # The head must be the shard-26 tensor itself, bitwise: a silently
    # tied head would pass the shape gate (both tables are [248320, 2048])
    # and would produce garbage logits.
    with safe_open(os.path.join(
            MODEL_DIR, weight_map["lm_head.weight"]), framework="pt") as f:
        raw_head = f.get_tensor("lm_head.weight")
    assert torch.equal(head, raw_head),         "loaded lm_head.weight disagrees bitwise with the checkpoint tensor"
    del raw_head
    return model


def install_capture_hooks(layers):
    """Wrap every decoder layer forward to record input and output tensors.

    Returns the capture list (one dict per layer) and a restore closure.
    Each install restores the pristine class forward after the run, so a
    second install never chains onto a previous wrapper."""
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
    """Wrap every router (layer.mlp.gate) forward to record its logits.

    The logits are the source of the recorded per-layer topk_indices and
    routing_weights fixtures and of the boundary-tie metadata: the
    checkpoint produces exact fp32 ties at the top-k boundary, so the
    indices record precisely the choice torch.topk made."""
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
    """Run the wrapper forward with per-layer capture and, when
    `capture_routers` is set, per-router capture. Returns captures,
    router logits (None when routers were not captured) and the wrapper
    logits. Only the sequential run captures routers: its tensors are
    the ones the Nim implementation asserts 0.00 against."""
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

    Recomputes input_layernorm, the GDN block on the recurrent rule, the
    post-attention norm and the routed block. The caller asserts the
    result is bit-identical to the hooked sequential forward output,
    proving the manual sequential replay and the patched real forward
    agree exactly."""
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
    router computes them, from the router logits: softmax over the fp32
    spelling of all experts, torch.topk, renormalized in fp32, cast to
    the logits dtype. The computation is deterministic under the pinned
    thread count, so replaying it from the captured logits is bitwise
    the module's own result."""
    probs = torch.nn.functional.softmax(router_logits, dtype=torch.float32, dim=-1)
    topk_values, topk_indices = torch.topk(probs, top_k, dim=-1)
    renorm = topk_values / topk_values.sum(dim=-1, keepdim=True)
    return {
        "topk_indices": topk_indices,
        "routing_weights": renorm.to(router_logits.dtype),
    }


def boundary_tie_count(router_logits: torch.Tensor, top_k: int) -> int:
    """Token count whose top-k boundary probabilities (descending
    positions top_k-1 and top_k) tie exactly at fp32. Such a tie makes
    the selected last expert order-sensitive; the recorded topk_indices
    pins the choice torch.topk made."""
    probs = torch.nn.functional.softmax(router_logits, dtype=torch.float32, dim=-1)
    sorted_probs = torch.sort(probs, dim=-1, descending=True).values
    return int(
        (sorted_probs[:, top_k - 1] == sorted_probs[:, top_k]).sum().item())


def save_fixture(name: str, metadata: dict, tensors: dict) -> str:
    """Save a fixture to safetensors with a separate deterministic metadata
    file."""
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    filepath = os.path.join(FIXTURE_DIR, f"{name}.safetensor")
    sorted_tensors = OrderedDict(
        (key, tensor.detach().cpu().contiguous())
        for key, tensor in sorted(tensors.items())
        if tensor is not None
    )
    serialized = st.save(sorted_tensors, metadata=None)
    with open(filepath, "wb") as f:
        f.write(serialized)
    metadata_path = filepath + ".metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, sort_keys=True, indent=2)
        f.write("\n")
    return filepath


def load_tokenizer():
    """Tokenize prompts through the checkpoint's own tokenizer files, the
    no-special-tokens spelling (no bos) the decode contract fixes."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_DIR)


def main() -> None:
    print(f"Generating {MODEL_NAME} full-model wiring fixtures")
    print("=" * 60)
    check_ram()
    sha = check_vendored_sha()

    cfg = load_wrapper_config()
    with open(INDEX_PATH) as f:
        weight_map = json.load(f)["weight_map"]
    assert len(weight_map) == (
        LANGUAGE_MODEL_KEYS + LM_HEAD_KEYS + VISUAL_KEYS + MTP_KEYS), (
        "checkpoint index key count disagrees with the pinned counts")
    model = build_model(cfg, weight_map)
    # The eager pin resolved in build_model and recorded in every fixture
    # metadata so the consumers see the numeric path the chain ran.
    resolved_impl = model.config.text_config._experts_implementation
    tokenizer = load_tokenizer()

    tokenizer_ids = tokenizer(INPUT_TEXT, add_special_tokens=False)["input_ids"]
    assert tokenizer_ids and len(tokenizer_ids) < 64, (
        "the wiring fixture prompt must stay inside one GDN chunk (64 tokens)")
    input_ids = torch.tensor([tokenizer_ids])
    seq_len = input_ids.shape[1]
    layers = model.model.language_model.layers
    num_layers = len(layers)
    assert num_layers == cfg.text_config.num_hidden_layers == 40, (
        "the checkpoint must carry 40 decoder layers")

    # Chunked forward: the reference ground truth. Router outputs go
    # uncaptured here, the sequential run's outputs stay the fixture truth.
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
        assert torch.equal(
            seq_captured[i]["layer_output"], seq_captured_again[i]["layer_output"]
        ), f"sequential replay layer {i} is not deterministic"
    assert torch.equal(logits_seq, logits_seq_again), \
        "sequential replay logits are not deterministic"

    top_k = cfg.text_config.num_experts_per_tok
    assert top_k == 8, (
        "the wiring fixtures pin the reference top-k of 8 experts")
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
            assert torch.equal(layer_input, layer_input_seq), \
                "layer 0 inputs must be identical across the chunked and sequential runs"
        input_diff = (layer_input_seq.float() - layer_input.float()).abs().max().item()
        output_diff = (layer_output_seq.float() - layer_output.float()).abs().max().item()

        # The manual sequential replay is the 0.00 reference. Every GDN
        # layer recomputes bitwise from its recorded sequential input
        # through the shared norms, the recurrent GDN core and the routed
        # block of the module.
        if kind == "linear_attention":
            replay = replay_gdn_layer(layer_obj, layer_input_seq)
            assert torch.equal(replay, layer_output_seq), (
                f"manual sequential replay of GDN layer {i} diverged from the patched forward")

        # Router outputs of the sequential run, the run the Nim
        # implementation asserts 0.00 against: recorded indices and weights
        # assert the router bitwise, the boundary-tie census
        # and margin go to the metadata. Exact fp32 ties at the top-k
        # boundary are structural in this checkpoint, making the margin
        # recorded information rather than a positive-assert failure.
        router_top_margin, _inner = router_margins(seq_router[i])
        router_outputs = recorded_router_outputs(seq_router[i], top_k)
        ties = boundary_tie_count(seq_router[i], top_k)

        # Regime guards keep the chunked-vs-recurrent ladder in the sub-ULP
        # core regime through the full 40-layer chain.
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
            "vendored_sha": sha,
            "bands": {
                "input_band": input_diff,
                "output_band": output_diff,
            },
            "margins": {
                "topk_margin_min": router_top_margin,
                "boundary_tie_tokens": ties,
            },
            "note": "layer_output is the vendored chunked forward. "
                    "layer_output_seq is the sequential replay (0.00 for Nim).",
        }
        save_fixture(f"layer-{i:02d}", metadata, {
            "layer_input": layer_input,
            "layer_input_seq": layer_input_seq,
            "layer_output": layer_output,
            "layer_output_seq": layer_output_seq,
            "topk_indices": router_outputs["topk_indices"],
            "routing_weights": router_outputs["routing_weights"],
        })
        print(f"  layer {i:02d} ({kind}): input_band {input_diff:.3e}, "
              f"output_band {output_diff:.3e}, topk_margin {router_top_margin:.3e}, "
              f"boundary_tie_tokens {ties}")

    logits_diff = (logits_seq.float() - logits_chunked.float()).abs().max().item()
    assert logits_diff < LOGITS_BAND_GUARD, \
        f"sequential vs chunked logits diff too large: {logits_diff}"
    save_fixture("final_logits", {
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
        "vendored_sha": sha,
        "bands": {
            "logits_band": logits_diff,
        },
        "note": "logits is the vendored chunked forward. "
                "logits_seq is the sequential replay (0.00 for Nim).",
    }, {
        "logits": logits_chunked,
        "logits_seq": logits_seq,
    })

    # A summary of the recorded band regime. The recorded bands remain
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
