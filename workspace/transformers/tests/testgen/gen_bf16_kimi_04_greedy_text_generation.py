#!/usr/bin/env python3
"""Kimi-Linear-48B-A3B-Instruct fixture generator over the real checkpoint shards, resolved through the gitignored tests/hf_models symlink.

bf16-04 greedy-text-generation records, token chains argmax-decoded
through the reference modeling on torch bf16:
- fixture dir tests/fixtures/bf16-04-greedy-text-generation/Kimi-Linear-48B-A3B-Instruct/
- consumer tests/q_bf16/t_bf16_kimi_04_greedy_text_generation.nim

- <prompt>_<horizon>_steps.json.zst, the ttt-tf-001-greedy-steps-h2 chain with the env frame
- <prompt>_<horizon>_steps.decisions.json.zst, the ttt-tf-005-argmax-decisions frame over the same steps

Chain contract:
- hand-rolled single-token decode over the whole-prompt prefill, single unpadded chains, no batched pass backs the values
- every KDA length runs the recurrent kernel class, the prefill via the forced recurrent spelling
- the router e_score_correction_bias and the KDA A_log/dt_bias buffers assert bitwise against the checkpoint
- a divergence is a near-tie iff the diverging pick equals the recorded runner-up id and the top-2 gap sits within a few bf16 ulps

Recording environment:
- recorded_from defaults to m4max-cpu, the TTT_RECORD_FROM environment variable overrides it
- the --device argument, cpu or mps, fills the device rows of the metadata and env frames

Run from the worktree root, PYTHONPATH pointing at the kimi-linear reference
worktree src carrying the torch fallback kernels:
  PYTHONPATH=<kimi-linear-reference-worktree>/src uv run python workspace/transformers/tests/testgen/gen_bf16_kimi_04_greedy_text_generation.py
"""
import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixture_stats import (  # noqa: E402, the path insert precedes the import
    argmax_record_from_step,
    recording_env,
    write_argmax_decisions,
)


def _load_sibling(filename: str):
    """Execute and return a testgen generator module by file path under a derived module name, a second call returns the cached module."""
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


# This generator is self-contained. It carries the shared kimi model build,
# config load, recurrent shim and dtype checks directly.
# The layer-internals sibling supplies the zstd writer and the head check.
# - determinism locks at import time through one intra-op torch thread
_kimi01 = _load_sibling("gen_bf16_kimi_01_layer_internals.py")
NUM_THREADS = _kimi01.NUM_THREADS
GREEDY_STEPS_SCHEMA = "ttt-tf-001-greedy-steps-h2"
MODEL_NAME = "Kimi-Linear-48B-A3B-Instruct"
MODEL_DIR = _kimi01.MODEL_DIR
INDEX_PATH = os.path.join(MODEL_DIR, "model.safetensors.index.json")
FIXTURE_DIR = os.path.join(
    _kimi01.GRANDPARENT_DIR, "fixtures", "bf16-04-greedy-text-generation", MODEL_NAME
)
write_json_zst = _kimi01.write_json_zst

import transformers  # noqa: E402
import transformers.models.kimi_linear.modeling_kimi_linear as kimi_ref  # noqa: E402
from transformers.models.kimi_linear.configuration_kimi_linear import (  # noqa: E402
    KimiLinearConfig,
)
import torch  # noqa: E402

TRANSFORMERS_VERSION = transformers.__version__

# Checkpoint counts and geometry. The stack has 27 layers, top-8 experts,
# 256 experts, and single-group wiring.
with open(os.path.join(MODEL_DIR, "config.json")) as f:
    _CKPT_CONFIG = json.load(f)
TOTAL_KEYS = 20493
NUM_LAYERS = _CKPT_CONFIG["num_hidden_layers"]
NUM_MOE_LAYERS = 26
FIRST_K_DENSE_REPLACE = 1
TOP_K = _CKPT_CONFIG["num_experts_per_token"]
NUM_EXPERTS = _CKPT_CONFIG["num_experts"]
N_GROUP = 1
TOPK_GROUP = _CKPT_CONFIG["topk_group"]
ROUTED_SCALING_FACTOR = 2.446
MLA_LAYERS_0BASED = [3, 7, 11, 15, 19, 23, 26]
# Prompt one reuses the exact token ids the bf16-03 full-forward fixture
# recorded. Both fixture families share one deciding row.
PROMPT_IDS = [19180, 11, 1632, 554, 398, 30]


def force_recurrent_shim() -> None:
    """Route every chunked-KDA call through the recurrent kernel, one recorded spelling:
    - the port spells the recurrent kernel at q_len <= 64, the reference
      prefill dispatch selects the chunked kernel at every non-decode
      length against an empty cache
    - forcing the recurrent spelling onto the recording keeps the chain
      budget free of a cross-spelling term
    - the chunked spelling stays covered by the committed KDA
      kernel-boundary fixtures, both kernels recorded per sequence length
      on the layer-0 geometry"""
    recurrent = kimi_ref.torch_recurrent_kda

    def shim(*args, **kwargs):
        kwargs.pop("cu_seqlens", None)
        return recurrent(*args, **kwargs)

    kimi_ref.torch_chunk_kda = shim


def index_census(weight_map: dict) -> None:
    """Check the checkpoint index against the recorded counts, 20493 tensors, all under model.* except the untied lm_head.weight."""
    total = len(weight_map)
    assert total == TOTAL_KEYS, (
        f"[gen_bf16_kimi_04] checkpoint index holds {total} keys, "
        f"expected {TOTAL_KEYS}")
    lm_head = sum(1 for key in weight_map if key == "lm_head.weight")
    assert lm_head == 1, "exactly one lm_head.weight entry expected"
    model_prefixed = sum(1 for key in weight_map if key.startswith("model."))
    assert model_prefixed == TOTAL_KEYS - 1, (
        "every non-head tensor must sit under model.*, found " + str(model_prefixed))
    embed_tokens = sum(
        1 for key in weight_map if key == "model.embed_tokens.weight")
    assert embed_tokens == 1, "exactly one embed_tokens entry expected"
    for key in weight_map:
        if key.startswith("model.layers."):
            rest = key[len("model.layers."):]
            layer_idx = int(rest.split(".", 1)[0])
            assert 0 <= layer_idx < NUM_LAYERS, (
                f"layer index {layer_idx} outside 0..{NUM_LAYERS - 1} in key {key}")


def load_config() -> KimiLinearConfig:
    """KimiLinearConfig built over the checkpoint config block:
    - the checkpoint spells the router fields under different key names
      than the dataclass, num_experts, num_experts_per_token, num_expert_group, num_shared_experts and moe_renormalize, the from_dict maps
      them onto the dataclass field names
    - first_k_dense_replace reaches the post-init derivation, mlp_layer_types
      derives dense below it, both travel the from_dict kwargs path"""
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        raw = json.load(f)
    cfg = KimiLinearConfig.from_dict({
        "model_type": raw["model_type"],
        "vocab_size": raw["vocab_size"],
        "hidden_size": raw["hidden_size"],
        "num_hidden_layers": raw["num_hidden_layers"],
        "num_attention_heads": raw["num_attention_heads"],
        "num_key_value_heads": raw["num_key_value_heads"],
        "kv_lora_rank": raw["kv_lora_rank"],
        "q_lora_rank": raw["q_lora_rank"],
        "qk_rope_head_dim": raw["qk_rope_head_dim"],
        "qk_nope_head_dim": raw["qk_nope_head_dim"],
        "v_head_dim": raw["v_head_dim"],
        "rms_norm_eps": raw["rms_norm_eps"],
        "max_position_embeddings": raw["model_max_length"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "n_routed_experts": raw["num_experts"],
        "num_experts_per_tok": raw["num_experts_per_token"],
        "n_group": raw["num_expert_group"],
        "topk_group": raw["topk_group"],
        "norm_topk_prob": raw["moe_renormalize"],
        "n_shared_experts": raw["num_shared_experts"],
        "routed_scaling_factor": raw["routed_scaling_factor"],
        "moe_intermediate_size": raw["moe_intermediate_size"],
        "intermediate_size": raw["intermediate_size"],
        "first_k_dense_replace": raw["first_k_dense_replace"],
        "hidden_act": raw.get("hidden_act", "silu"),
        "tie_word_embeddings": raw["tie_word_embeddings"],
        "pad_token_id": raw["pad_token_id"],
        "bos_token_id": raw["bos_token_id"],
        "eos_token_id": raw["eos_token_id"],
        "linear_attn_config": raw["linear_attn_config"],
    })
    cfg._attn_implementation = "eager"
    return cfg


def build_model(cfg: KimiLinearConfig):
    """Full reference model from the real checkpoint through the installed
    `from_pretrained`, per-file streaming, mmap-backed, bf16, eval, CPU:
    - raises SystemExit when a dtype-plan row moves:
      the f32 A_log and dt_bias the reference creates with explicit
      dtypes or the bf16 e_score_correction_bias buffer the reference
      deploys without a round trip"""
    model = kimi_ref.KimiLinearForCausalLM.from_pretrained(
        MODEL_DIR, config=cfg, dtype=torch.bfloat16, device_map=None)
    model.eval()

    with open(INDEX_PATH) as f:
        weight_map = json.load(f)["weight_map"]
    head = model.lm_head.weight.detach()
    embed = model.model.embed_tokens.weight.detach()
    assert head.data_ptr() != embed.data_ptr(), (
        "lm_head shares storage with the embedding, the untied reading moved")
    with safe_open(os.path.join(MODEL_DIR, weight_map["lm_head.weight"]),
                   framework="pt") as f:
        raw_head = f.get_tensor("lm_head.weight")
    assert torch.equal(head, raw_head), (
        "loaded lm_head.weight disagrees bitwise with the checkpoint tensor")
    del raw_head
    for i in range(NUM_LAYERS):
        attn = model.model.layers[i].self_attn
        if i in MLA_LAYERS_0BASED:
            continue
        assert attn.A_log.dtype == torch.float32, (
            f"layer {i} A_log drifted off the checkpoint f32 grid")
        assert attn.dt_bias.dtype == torch.float32, (
            f"layer {i} dt_bias drifted off the checkpoint f32 grid")
    for i in range(FIRST_K_DENSE_REPLACE, NUM_LAYERS):
        bias = model.model.layers[i].mlp.gate.e_score_correction_bias
        assert bias.dtype == torch.bfloat16, (
            f"layer {i} e_score_correction_bias drifted off the reference "
            f"bf16 buffer dtype, dtype {bias.dtype!r} breaks the recorded chain")
    return model

import tiktoken  # noqa: E402
import tiktoken.load  # noqa: E402
from safetensors import safe_open  # noqa: E402

# 3 distinct simple English prompts, all 12 tokens or fewer under this checkpoint's tokenizer through the tiktoken 0.14.0 reference
# engine:
# - encode_ordinary, no special tokens
# - horizons stay inside 32 greedy steps
# - prompt one reuses the exact token ids the bf16-03 full-forward fixture
#   already recorded, both fixture families share one deciding row
PROMPT_SPEC = [
    ("Hello, how are you?", 32),
    ("The capital of France is", 32),
    ("Big blue whales eat krill.", 32),
]
RECORDED_PROMPT_IDS = {
    "Hello, how are you?": [19180, 11, 1632, 554, 398, 30],
    "The capital of France is": [1008, 10484, 318, 15383, 387],
    "Big blue whales eat krill.": [21707, 9447, 94597, 9985, 44685, 534, 13],
}
MAX_PROMPT_TOKENS = 12
MAX_HORIZON = 32
STOP_IDS_RECORDED = [163586]


def load_reference_engine():
    """Tiktoken 0.14.0 engine over the checkpoint tokenizer sources,
    the exact measurement pattern of the tokenizer unit suite spot rows:
    - pat_str parts extracted from the checkpoint tokenization_kimi.py raw
      string literals joined with |, mergeable ranks from tiktoken.model
    - the 258-slot special map over added_tokens_decoder with the reserved-token fillers"""
    with open(os.path.join(MODEL_DIR, "tokenization_kimi.py"), encoding="utf-8") as f:
        src = f.read()
    start = src.find("pat_str")
    end = src.find("def __init__", start)
    parts = re.findall(r'r"""(.*?)"""', src[start:end], flags=re.S)
    assert len(parts) == 8, (
        f"expected the 8 recorded pat_str alternatives, found {len(parts)}")
    pat = "|".join(parts)
    ranks = tiktoken.load.load_tiktoken_bpe(os.path.join(MODEL_DIR, "tiktoken.model"))
    with open(os.path.join(MODEL_DIR, "tokenizer_config.json"), encoding="utf-8") as f:
        decoder = json.load(f)["added_tokens_decoder"]
    num_base = len(ranks)
    assert num_base == 163584, (
        f"mergeable rank count {num_base} moved off the recorded 163584")
    specials = {}
    for k in range(num_base, num_base + 258):
        if str(k) in decoder:
            specials[decoder[str(k)]["content"]] = k
        else:
            specials["<|reserved_token_%d|>" % k] = k
    assert len(specials) == 258, (
        f"special map holds {len(specials)} slots, expected 258")
    return tiktoken.Encoding(
        name="kimi", pat_str=pat, mergeable_ranks=ranks, special_tokens=specials)


def prompt_token_ids(engine, text: str) -> tuple[int, ...]:
    """Prompt ids under the reference engine, no special tokens added:
    - hard-asserted against the recorded corpus ids
    - prompt one additionally against the bf16-03 full-forward recorded ids"""
    ids = tuple(int(t) for t in engine.encode_ordinary(text))
    expected = RECORDED_PROMPT_IDS[text]
    assert list(ids) == expected, (
        f"prompt {text!r} tokenized to {list(ids)}, recorded corpus ids are {expected}")
    assert 1 <= len(ids) <= MAX_PROMPT_TOKENS, (
        f"prompt {text!r} holds {len(ids)} tokens, expected 1..{MAX_PROMPT_TOKENS}")
    if text == "Hello, how are you?":
        assert list(ids) == list(PROMPT_IDS), (
            "prompt one moved off the bf16-03 recorded prompt ids, the "
            "cross-fixture deciding row would break")
    return ids


def generation_stop_ids() -> list[int]:
    """Stop ids from config.json, the single-int eos value asserted equal
    to generation_config.json:
    - the greedy chains assert none of these ids
    - every step stays a live argmax pick"""
    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        raw = json.load(f)["eos_token_id"]
    assert raw is not None, "config.json eos_token_id missing"
    ids = [int(raw)] if isinstance(raw, int) else [int(t) for t in raw]
    with open(os.path.join(MODEL_DIR, "generation_config.json")) as f:
        gen = json.load(f)["eos_token_id"]
    assert gen is not None, "generation_config.json eos_token_id missing"
    gen_ids = [int(gen)] if isinstance(gen, int) else [int(t) for t in gen]
    assert ids == gen_ids, (
        f"config stop set {ids} disagrees with generation_config {gen_ids}")
    return ids


def assert_dtype_plan(model, weight_map: dict) -> None:
    """Value-level dtype-plan asserts on the loaded model, enforced before any chain runs:
    - every routed e_score_correction_bias buffer sits on the checkpoint
      bf16 grid, bitwise equal to the checkpoint tensor, the reference deploys the bias without a round trip, the f32 upcast happens
      at the score add
    - every KDA A_log and dt_bias keeps its checkpoint f32 values
      bit-exactly, the reference module creates both with an explicit f32 dtype, the meta dtype wins the load
    - module-side shapes differ from the checkpoint rows and the reshape is
      value-preserving, the comparison flattens both sides"""
    layers = model.model.layers
    for i in range(NUM_LAYERS):
        if i in MLA_LAYERS_0BASED:
            continue
        prefix = f"model.layers.{i}.self_attn"
        with safe_open(os.path.join(MODEL_DIR, weight_map[f"{prefix}.A_log"]),
                       framework="pt") as f:
            raw_a_log = f.get_tensor(f"{prefix}.A_log")
        assert layers[i].self_attn.A_log.dtype == torch.float32, (
            f"layer {i} A_log drifted off the checkpoint f32 grid")
        live_a_log = layers[i].self_attn.A_log.detach().reshape(raw_a_log.shape)
        assert torch.equal(live_a_log, raw_a_log), (
            f"layer {i} A_log values drifted off the checkpoint f32 values")
        with safe_open(os.path.join(MODEL_DIR, weight_map[f"{prefix}.dt_bias"]),
                       framework="pt") as f:
            raw_dt_bias = f.get_tensor(f"{prefix}.dt_bias")
        assert layers[i].self_attn.dt_bias.dtype == torch.float32, (
            f"layer {i} dt_bias drifted off the checkpoint f32 grid")
        live_dt_bias = layers[i].self_attn.dt_bias.detach().reshape(raw_dt_bias.shape)
        assert torch.equal(live_dt_bias, raw_dt_bias), (
            f"layer {i} dt_bias values drifted off the checkpoint f32 values")
    for i in range(FIRST_K_DENSE_REPLACE, NUM_LAYERS):
        # The checkpoint key prefix carries the block_sparse_moe MoE block name,
        # the module gate weights live at the mlp.gate attribute under the branch
        # module naming, the load maps the checkpoint key onto the weights.
        prefix = f"model.layers.{i}.block_sparse_moe.gate.e_score_correction_bias"
        bias = layers[i].mlp.gate.e_score_correction_bias
        assert bias.dtype == torch.bfloat16, (
            f"layer {i} e_score_correction_bias drifted off the checkpoint "
            f"bf16 grid, dtype {bias.dtype!r} breaks the recorded chain")
        with safe_open(os.path.join(MODEL_DIR, weight_map[prefix]),
                       framework="pt") as f:
            raw_bias = f.get_tensor(prefix)
        assert torch.equal(bias.detach(), raw_bias), (
            f"layer {i} e_score_correction_bias values drifted off the "
            "checkpoint bf16 values, the bias buffer must ride the "
            "checkpoint grid directly")
        # On-grid proof by the idempotence form:
        #   re-rounding a bf16 value
        # through bf16 is the identity, the grid never widens under it.
        assert torch.equal(raw_bias, raw_bias.to(torch.bfloat16)), (
            f"layer {i} e_score_correction_bias round trip is not idempotent")


class ConvDecodeRecorder:
    """Conv-decode validation recorder, module-global patching only:
    - wraps every KDA attention module forward to tag the active layer
      and capture its input row
    - wraps the branch conv fallbacks, causal_conv1d_update at decode
      and causal_conv1d_fn at prefill and replay, capturing the post-conv
      mixed_qkv rows beside the decode rows
    - the kernels stay absent on this stack, the decorators resolve
      to the fallback identities, the forward bodies look the conv
      functions up in the module globals at call time, the same patch
      surface the forced-recurrent shim uses"""

    def __init__(self, model):
        self.model = model
        self.kda_indices = [i for i in range(NUM_LAYERS) if i not in set(MLA_LAYERS_0BASED)]
        self.phase = "idle"
        self.current_layer = None
        self.prefill_inputs = {}
        self.decode_inputs = {}
        self.decode_conv = {}
        self.replay_conv = None
        self._originals = {}
        self._orig_update = None
        self._orig_fn = None

    def __enter__(self):
        for i in self.kda_indices:
            attn = self.model.model.layers[i].self_attn
            self._originals[i] = attn.forward

            def make_wrapper(layer_idx, original, rec):
                def wrapper(hidden_states, *args, **kwargs):
                    rec.current_layer = layer_idx
                    if rec.phase == "prefill":
                        rec.prefill_inputs[layer_idx] = hidden_states.detach().clone()
                    elif rec.phase.startswith("decode:"):
                        rec.decode_inputs[layer_idx].append(hidden_states.detach().clone())
                    out = original(hidden_states, *args, **kwargs)
                    rec.current_layer = None
                    return out

                return wrapper

            attn.forward = make_wrapper(i, self._originals[i], self)
            self.decode_inputs[i] = []
            self.decode_conv[i] = []

        self._orig_update = kimi_ref.causal_conv1d_update
        self._orig_fn = kimi_ref.causal_conv1d_fn

        def update_recorder(hidden_states, conv_state, weight, bias=None,
                            activation=None):
            out = self._orig_update(hidden_states, conv_state, weight, bias,
                                    activation=activation)
            layer = self.current_layer
            assert layer is not None, (
                "causal_conv1d_update fired outside a tagged KDA layer forward")
            self.decode_conv[layer].append(out.detach().clone())
            return out

        def fn_recorder(hidden_states, weight, bias=None, activation=None, **kwargs):
            out = self._orig_fn(hidden_states, weight, bias, activation=activation, **kwargs)
            if self.phase == "replay":
                self.replay_conv = out.detach().clone()
            return out

        kimi_ref.causal_conv1d_update = update_recorder
        kimi_ref.causal_conv1d_fn = fn_recorder
        return self

    def __exit__(self, *exc):
        for i in self.kda_indices:
            self.model.model.layers[i].self_attn.forward = self._originals[i]
        kimi_ref.causal_conv1d_update = self._orig_update
        kimi_ref.causal_conv1d_fn = self._orig_fn
        return False

    def begin_prefill(self):
        self.phase = "prefill"
        self.prefill_inputs = {}

    def end_prefill(self):
        self.phase = "idle"
        assert len(self.prefill_inputs) == len(self.kda_indices), (
            "prefill capture missed a KDA layer input row")

    def begin_decode_step(self):
        self.phase = "decode:"
        for i in self.kda_indices:
            assert len(self.decode_conv[i]) == len(self.decode_inputs[i]), (
                f"layer {i} conv capture fell out of step with the input capture")

    def end_decode_step(self):
        self.phase = "idle"
        for i in self.kda_indices:
            assert len(self.decode_conv[i]) == len(self.decode_inputs[i]), (
                f"layer {i} conv capture fell out of step with the input capture")

    def replay_conv_last(self, attn, prefix_inputs):
        """Replay one KDA attention module over a fresh cache-free prefill of the prefix inputs, returning the last-position post-conv row."""
        self.phase = "replay"
        self.replay_conv = None
        try:
            with torch.no_grad():
                attn.forward(prefix_inputs, past_key_values=None, attention_mask=None)
        finally:
            self.phase = "idle"
        assert self.replay_conv is not None, (
            "replay never reached the causal_conv1d_fn fallback capture")
        return self.replay_conv[:, :, -1]

    def validate_conv_decode(self, horizon: int) -> float:
        """Every recorded decode step of every KDA layer replays through a fresh prefill of the same tokens, the last-position post-conv row
        compared channel-by-channel against the recorded decode row. Returns the worst channel drift, 0.0 on an exact replay.
        Any mismatch raises before fixtures are written."""
        worst = 0.0
        checked = 0
        # Decode rows, the chain runs horizon - 1 decode forwards, decode
        # forward j consumes the step-j argmax pick and produces
        # the step-j+1 deciding row, its conv window ending on that pick.
        for t in range(horizon - 1):
            for i in self.kda_indices:
                attn = self.model.model.layers[i].self_attn
                prefix = torch.cat(
                    [self.prefill_inputs[i]] + self.decode_inputs[i][: t + 1], dim=1)
                replay_row = self.replay_conv_last(attn, prefix)
                recorded = self.decode_conv[i][t][:, :, 0]
                drift = float((replay_row.float() - recorded.float()).abs().max().item())
                worst = max(worst, drift)
                checked += 1
                if drift != 0.0:
                    raise AssertionError(
                        f"conv-decode replay drifted on layer {i} step {t}: "
                        f"worst channel drift {drift}, the recorded decode "
                        "window does not match the fresh prefill of the same "
                        "tokens, refusing to write fixtures")
            if (t + 1) % 8 == 0:
                print(f"    r9 replay: {t + 1}/{horizon - 1} decode steps validated, "
                      f"worst drift {worst}")
        assert checked == (horizon - 1) * len(self.kda_indices), (
            "conv-decode replay validated fewer rows than the chain recorded")
        print(f"  conv-decode replay: {checked} rows validated, worst drift {worst}")
        return worst


def greedy_chain(model, token_ids: tuple[int, ...], max_new_tokens: int,
                 recorder: ConvDecodeRecorder) -> dict:
    """Greedy-decode one unpadded chain through the installed forward:
    - the prefill covers the whole prompt over an explicit DynamicCache
    - each step runs one single-token forward, the previous argmax id feeds back as the next input
    - every decode step runs the natural recurrent dispatch at the cached
      one-token state, the recorder captures the per-layer decode rows for the conv-decode validation
    - returns the generated ids and the per-step records of the deciding
      last-position logits row, the top-32 ids with f32 logits, the argmax
      margin and the softmax tail probability beyond the top-32 support
    - exactly max_new_tokens argmax picks are recorded, the cache makes
      each step one one-token forward"""
    from transformers.cache_utils import DynamicCache
    input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=DEVICE)
    generated = []
    steps = []
    stop = generation_stop_ids()
    with torch.no_grad():
        # Cache built over the model config, the DynamicCache infers the hybrid layer structure (MLA and KDA cache layers) from it, a bare
        # DynamicCache() carries no layer entries and the KDA update_conv_state indexing would run off the empty layer list.
        cache = DynamicCache(config=model.config)
        recorder.begin_prefill()
        out = model(input_ids, past_key_values=cache, use_cache=True)
        recorder.end_prefill()
        for step in range(max_new_tokens):
            last_f32 = out.logits[0, -1].float()
            # torch.topk over the f32 logits records the top-32 support, the argmax
            # margin uses the top-2 values:
            # - torch.topk orders equal values by a selection-internal detail
            #   that flips with k, torch.argmax returns the first-max index
            # - the recorded chosen_token stays the argmax pick, an exact tie
            #   records margin 0.0, the consumer checks classify the divergence
            #   at that step
            argmax_id = int(last_f32.argmax().item())
            generated.append(argmax_id)
            assert argmax_id not in stop, (
                f"chain hit a configured stop id {argmax_id} at step {step}, "
                "prompt or horizon must be respecified")
            top_vals, top_idxs = torch.topk(last_f32, 32)
            probs = torch.softmax(last_f32, dim=-1)
            tail = float(1.0 - probs[top_idxs].sum().item())
            margin = float(top_vals[0].item() - top_vals[1].item())
            steps.append({
                "step": step,
                "chosen_token": argmax_id,
                "top32_ids": top_idxs.tolist(),
                "top32_logits": [float(v) for v in top_vals.tolist()],
                "argmax_margin": margin,
                "tail_probability": tail,
            })
            if step + 1 < max_new_tokens:
                nxt = torch.tensor([[argmax_id]], dtype=torch.long, device=DEVICE)
                recorder.begin_decode_step()
                out = model(nxt, past_key_values=cache, use_cache=True)
                recorder.end_decode_step()
    return {
        "generated_ids": generated,
        "steps": steps,
    }


def prompt_fixture_name(text: str, max_new_tokens: int) -> str:
    """File name for one prompt fixture, alphanumerics only, horizon suffix, one file per prompt."""
    safe = "".join(c if c.isalnum() else "_" for c in text).strip("_")
    safe = "_".join(part for part in safe.split("_") if part)
    return f"{safe}_{max_new_tokens}_steps.json.zst"


def main() -> None:
    """Records the greedy chain fixtures and the decision frames."""
    global DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu",
                        help="recording device, the metadata device rows record it")
    args = parser.parse_args()
    global DEVICE
    DEVICE = args.device
    print(f"Generating {MODEL_NAME} bf16-04-greedy-text-generation fixtures")
    print("=" * 60)

    with open(INDEX_PATH) as f:
        weight_map = json.load(f)["weight_map"]
    index_census(weight_map)
    cfg = load_config()
    assert list(cfg.layer_types) == [
        "full_attention" if i in MLA_LAYERS_0BASED else "linear_attention"
        for i in range(NUM_LAYERS)], cfg.layer_types
    force_recurrent_shim()
    model = build_model(cfg)
    if DEVICE != "cpu":
        model = model.to(DEVICE)
    assert_dtype_plan(model, weight_map)
    engine = load_reference_engine()
    stop = generation_stop_ids()
    assert stop == STOP_IDS_RECORDED, (
        f"stop set {stop} moved off the recorded single-int config value "
        f"{STOP_IDS_RECORDED}")

    texts = [spec[0] for spec in PROMPT_SPEC]
    assert len(set(texts)) == len(texts), "prompts must be distinct"

    os.makedirs(FIXTURE_DIR, exist_ok=True)
    for prompt_text, max_new_tokens in PROMPT_SPEC:
        assert 0 < max_new_tokens <= MAX_HORIZON, (
            f"horizon for {prompt_text!r} must stay inside {MAX_HORIZON} steps")
        token_ids = prompt_token_ids(engine, prompt_text)
        recorder = ConvDecodeRecorder(model)
        with recorder:
            chain = greedy_chain(model, token_ids, max_new_tokens, recorder)
            conv_decode_maxdiff = recorder.validate_conv_decode(max_new_tokens)
        assert conv_decode_maxdiff == 0.0, (
            f"conv-decode replay drifted {conv_decode_maxdiff}, refusing to write fixtures")
        assert len(chain["generated_ids"]) == max_new_tokens
        assert len(chain["steps"]) == max_new_tokens
        full_ids = list(token_ids) + chain["generated_ids"]
        full_text = engine.decode(full_ids)
        generated_text = engine.decode(chain["generated_ids"])
        fixture = {
            "schema": GREEDY_STEPS_SCHEMA,
            "model": MODEL_NAME,
            "env": recording_env(
                model=MODEL_NAME,
                generator="testgen/gen_bf16_kimi_04_greedy_text_generation.py",
                extra={"dtype": "bfloat16", "device": DEVICE,
                       "num_threads": NUM_THREADS,
                       "recorded_from": os.environ.get("TTT_RECORD_FROM", "m4max-cpu")}),
            "prompt": prompt_text,
            "prompt_ids": list(token_ids),
            "num_prompt_tokens": len(token_ids),
            "generated_ids": chain["generated_ids"],
            "num_generated_tokens": max_new_tokens,
            "full_ids": full_ids,
            "full_text": full_text,
            "generated_text": generated_text,
            "steps": chain["steps"],
            "conv_decode_replay_maxdiff": conv_decode_maxdiff,
            "moe_dispatch": "eager",
            "attn_implementation": "eager",
            "kda_dispatch": "recurrent",
            "stop_ids": stop,
            "note": "single unpadded chains; the serialized values are the "
                    "single-chain values, no batched pass backs them. The "
                    "reference runs the eager token-gather MoE dispatch "
                    "(per-expert loop, index_add_ accumulation, no grouped_mm "
                    "alternative) and the eager MLA spelling, both hardcoded "
                    "on the template.",
            "kda_dispatch_note": "the prefill rides the forced branch recurrent "
                                 "spelling (the shim the bf16-03 recording "
                                 "installs), every decode step rides the natural "
                                 "branch recurrent dispatch at the cached "
                                 "one-token state, the same kernel class the "
                                 "port spells at q_len <= 64",
            "conv_decode_note": "every recorded decode step of every KDA layer replayed "
                       "through a fresh cache-free prefill of the same tokens, "
                       "the post-conv rows compared channel-by-channel exact, "
                       "the conv-decode window law",
        }
        out_path = os.path.join(
            FIXTURE_DIR, prompt_fixture_name(prompt_text, max_new_tokens))
        write_json_zst(out_path, fixture)
        decisions_path = str(out_path)[:-len(".json.zst")] + ".decisions.json.zst"
        write_argmax_decisions(
            decisions_path,
            os.path.basename(decisions_path)[:-len(".decisions.json.zst")],
            [argmax_record_from_step(step) for step in chain["steps"]],
            "bf16")
        digest = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
        print(f"  prompt ({len(token_ids)} tokens): {prompt_text!r}")
        print(f"    generated: {chain['generated_ids']}")
        print(f"    fixture: {out_path}")
        print(f"    sha256: {digest}")
        print(f"    step0 margin {chain['steps'][0]['argmax_margin']:.4f}")

    import resource
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)
    print(f"  peak rss of this process: {maxrss:.2f} GiB (anonymous peak)")
    print("=" * 60)
    print(f"Fixture generation complete: {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
