"""
Generate fixtures for the LFM2.5-230M op: conv layer, attention layer,
3-layer chain, full-model ids->logits, and greedy t2t trajectories, from the
REAL LFM2.5-230M checkpoint on CPU torch bf16 (eval mode).

Fixtures are the ground truth for the Nim q_bf16 tests
(test_lfm2_25_*.nim). The Nim implementation is fixed to match these
fixtures, never the other way around.

Regeneration, pinned to the revision set whose output is committed
(transformers 5.16.1, torch 2.13.0, safetensors 0.8.0, CPython 3.14.7:
rerunning that set reproduced every committed fixture byte for byte):

  LFM2_MODEL_DIR=/path/to/LFM2.5-230M \\
  uv run --python 3.14.7 --with "transformers==5.16.1" --with "torch==2.13.0" \
         --with "safetensors==0.8.0" python \\
    workspace/transformers/tests/testgen/gen_lfm2_25_fixtures.py

LFM2_MODEL_DIR defaults to the workspace MODELS/LFM2.5-230M directory.
The generator needs the checkpoint on disk (config.json, model.safetensors,
tokenizer.json, chat_template.jinja); it uses installed transformers' own
Lfm2ForCausalLM, not a vendored modeling module. Checkpoint identity:
model.safetensors is 459,401,112 bytes, sha256
f630da86651136c9aee893b04b7542007e90fdd718355358e57e7ecc31517cfd.
A later transformers or torch revision can shift bf16 reductions, so after
regenerating, re-run the Nim suites rather than assuming the committed
fixtures were reproduced.

Weights are NOT committed as fixtures: the Nim tests mmap the real
checkpoint through tests/hf_models/LFM2.5-230M (symlink, git-ignored), the
same convention as the Qwen3.5 suites. Committed fixtures carry only
seeded activations, input ids, and greedy token ids, so the total fixture
footprint stays well under a megabyte.

Real checkpoint shape: hidden 1024, 14 layers, vocab 65536, GQA 16 q / 8 kv
heads (head_dim 64), intermediate 2560, layer_types conv at
{0,1,3,5,7,9,11,13} and full_attention at {2,4,6,8,10,12}.

What is generated (all under tests/fixtures):

  layers/LFM2.5-230M-layer-0/           real conv layer (index 0)
    conv-prefill.safetensor             T=5 prefill: seeded x, conv-block
                                        intermediates (in_proj, B/C/x,
                                        mixed, conv out, y, out_proj),
                                        block output
    conv-decode.safetensor              3-token prefill + 2 decode steps
                                        through the HF module with a
                                        cache: per-step conv states and
                                        decode outputs. The generator
                                        asserts the decode path equals a
                                        one-shot 5-token forward bit for
                                        bit.

  layers/LFM2.5-230M-layer-2/           first real attention layer (index 2)
    attn-prefill.safetensor             T=5 prefill: q/k/v, per-head q/k
                                        norms, RoPE'd q/k, attn output,
                                        block output

  chain/LFM2.5-230M/
    chain.safetensor                    real layers 0..2 (conv, conv,
                                        full_attention), T=4: per-layer
                                        inputs/outputs, embedding_norm
                                        output

  ids-inference/LFM2.5-230M/
    ids-logits.safetensor               real-model input_ids (tokenized
                                        prose prompt) + last-position
                                        logits

  greedy-decoding/LFM2.5-230M/*.json    greedy t2t fixtures: 16 tokens,
                                        temp 0, two seeded prompts (ASCII
                                        prose, chat-template rendered).
                                        prompt_ids pin the reference
                                        tokenization (the Nim tokenizer
                                        differential is not wired for LFM);
                                        generated_ids lock the
                                        token-for-token decode
                                        differential. argmax_margins and
                                        argmax_top2_ids (per-step top-1 and
                                        runner-up ids) say how much room
                                        each decision had, so a Nim argmax
                                        that lands on the runner-up of a
                                        one-step tie is classified by
                                        measurement rather than assumed
                                        wrong. Base-model greedy on the
                                        prose prompt degenerates to 16
                                        eos(7) at margin 3.19; the chat
                                        prompt decodes 16 content tokens at
                                        margins 7.75 down to 0.125.
                                        Degenerate output is a property of
                                        the reference, not a defect: the
                                        gate is that Nim matches these ids.

Determinism: torch.manual_seed per section, CPU only. Every replay is
asserted bit-identical (torch.equal) to the module's own forward before
anything is saved. Greedy trajectories are generated twice and asserted
identical before saving.

Conventions follow gen_qwen3_5_gdn_fixtures.py (save_fixture layout,
sorted tensor keys) and gen_qwen3_5_greedy_fixtures.py (t2t JSON layout).
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from safetensors import torch as st

MODEL_DIR = os.environ.get(
    "LFM2_MODEL_DIR",
    "/Users/pi/Documents/Programming/workspace-tattletale/MODELS/LFM2.5-230M")
if not os.path.isfile(os.path.join(MODEL_DIR, "model.safetensors")):
    raise SystemExit(
        f"[gen_lfm2_25_fixtures] checkpoint not found at {MODEL_DIR}. "
        "Set LFM2_MODEL_DIR to the LFM2.5-230M HF-layout directory")

from transformers import AutoTokenizer, Lfm2Config, Lfm2ForCausalLM
from transformers.models.lfm2.modeling_lfm2 import (
    Lfm2RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

FIXTURES_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fixtures"))
LAYER0_DIR = os.path.join(FIXTURES_ROOT, "layers", "LFM2.5-230M-layer-0")
LAYER2_DIR = os.path.join(FIXTURES_ROOT, "layers", "LFM2.5-230M-layer-2")
CHAIN_DIR = os.path.join(FIXTURES_ROOT, "chain", "LFM2.5-230M")
IDS_DIR = os.path.join(FIXTURES_ROOT, "ids-inference", "LFM2.5-230M")
GREEDY_DIR = os.path.join(FIXTURES_ROOT, "greedy-decoding", "LFM2.5-230M")

SEED_LAYER0 = 0x5EED
SEED_LAYER2 = 0xABCD
SEED_CHAIN = 0x1234
SEED_FULL = 0xCAFE

SECTIONS = ("conv", "attn", "chain", "ids", "greedy")

SECTION_DIRS = {
    "conv": (LAYER0_DIR,),
    "attn": (LAYER2_DIR,),
    "chain": (CHAIN_DIR,),
    "ids": (IDS_DIR,),
    "greedy": (GREEDY_DIR,),
}

# Greedy t2t prompts. Prose is ASCII-only
# (no NFC surface for a future Nim tokenizer differential), 60-120 chars.
# The chat prompt is rendered through the checkpoint's chat_template.jinja
# before tokenization.
PROSE_PROMPT = ("The old lighthouse keeper climbed the spiral stairs every "
                "evening to light the lamp above the harbor.")
CHAT_USER_MSG = "Explain in one sentence why the sky appears blue."
GREEDY_TOKENS = 16

_MODEL = None
_CONFIG = None
_TOKENIZER = None


def load_checkpoint():
    """Real checkpoint, CPU bf16 eval, loaded once per generator run."""
    global _MODEL, _CONFIG, _TOKENIZER
    if _MODEL is None:
        _CONFIG = Lfm2Config.from_pretrained(MODEL_DIR)
        _MODEL = Lfm2ForCausalLM.from_pretrained(
            MODEL_DIR, dtype=torch.bfloat16, device_map=None).to("cpu").eval()
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR)
    return _MODEL, _CONFIG, _TOKENIZER


def ensure_fixture_dirs(sections) -> None:
    for section in sections:
        for fixture_dir in SECTION_DIRS[section]:
            os.makedirs(fixture_dir, exist_ok=True)


def save_fixture(fixture_dir: str, name: str, tensors: dict) -> str:
    """Save tensors to a safetensor (sorted keys, bf16 preserved)."""
    sorted_tensors = {}
    for k, v in sorted(tensors.items()):
        if v is None:
            continue
        sorted_tensors[k] = v.detach().cpu().contiguous().clone()
    filepath = os.path.join(fixture_dir, name)
    with open(filepath, "wb") as f:
        f.write(st.save(sorted_tensors, metadata=None))
    return filepath


def seeded_input(seed: int, batch, seq, hidden) -> torch.Tensor:
    """Seeded bf16 input in [-1, 1]."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(batch, seq, hidden, generator=g) * 2.0 - 1.0
    return x.to(torch.bfloat16)


def get_conv_state(cache, layer_idx: int) -> torch.Tensor:
    """Conv cache window across the DynamicCache layouts of the supported
    transformers range (4.57 layer-object vs later flat lists)."""
    layers = getattr(cache, "layers", None)
    if layers is not None and getattr(layers[layer_idx], "conv_states", None):
        return layers[layer_idx].conv_states[0]
    flat = getattr(cache, "conv_states", None)
    if flat is not None:
        return flat[layer_idx][0]
    raise SystemExit("[gen_lfm2_25_fixtures] unrecognized DynamicCache layout")


# =============================================================================
# Section A - conv layer (real layer 0) fixture
# =============================================================================

def generate_conv_layer_fixture() -> None:
    model, cfg, _ = load_checkpoint()
    torch.manual_seed(SEED_LAYER0)
    layer = model.model.layers[0]
    assert cfg.layer_types[0] == "conv"

    # ---- Prefill T=5 with intermediates (submodule replay, bit-identical)
    x = seeded_input(0x1111, 1, 5, cfg.hidden_size)
    with torch.no_grad():
        block_out = layer(x)  # module forward (one-shot, no cache)

    with torch.no_grad():
        h_norm = layer.operator_norm(x)
        bcx = layer.conv.in_proj(h_norm).transpose(-1, -2)   # (1, 3H, 5)
        b_b, c_b, x_b = bcx.chunk(3, dim=-2)
        mixed = b_b * x_b
        conv_full = layer.conv.conv(mixed)                   # (1, H, 5+2)
        conv_out = conv_full[:, :, :5]
        y = c_b * conv_out
        out_proj_out = layer.conv.out_proj(y.transpose(-1, -2))  # (1, 5, H)
        ffn_out = layer.feed_forward(layer.ffn_norm(x + out_proj_out))
        replay_out = x + out_proj_out + ffn_out
    assert torch.equal(replay_out, block_out), \
        "conv-layer replay diverged from the module forward"

    save_fixture(LAYER0_DIR, "conv-prefill.safetensor", {
        "x": x,
        "operator_norm_out": h_norm,
        "in_proj_out": bcx,
        "branch_b": b_b,
        "branch_c": c_b,
        "branch_x": x_b,
        "mixed": mixed,
        "conv_out": conv_out,
        "post_conv_y": y,
        "out_proj_out": out_proj_out,
        "block_out": block_out,
    })

    # ---- Decode trajectory: 3-token prefill + 2 decode steps, with cache
    from transformers.cache_utils import DynamicCache
    x3 = seeded_input(0x2222, 1, 3, cfg.hidden_size)
    xd1 = seeded_input(0x3333, 1, 1, cfg.hidden_size)
    xd2 = seeded_input(0x4444, 1, 1, cfg.hidden_size)
    cache = DynamicCache(config=cfg)
    with torch.no_grad():
        out3 = layer(x3, past_key_values=cache)
        state_after_prefill = get_conv_state(cache, 0).clone()
        out_d1 = layer(xd1, past_key_values=cache)
        state_after_d1 = get_conv_state(cache, 0).clone()
        out_d2 = layer(xd2, past_key_values=cache)
        state_after_d2 = get_conv_state(cache, 0).clone()

    # One-shot 5-token forward must match the prefill+decode outputs.
    x5 = torch.cat([x3, xd1, xd2], dim=1)
    with torch.no_grad():
        out5 = layer(x5)
    assert torch.equal(out5[:, :3], out3), "decode path diverged from one-shot (prefix)"
    assert torch.equal(out5[:, 3:4], out_d1), "decode step 1 diverged from one-shot"
    assert torch.equal(out5[:, 4:5], out_d2), "decode step 2 diverged from one-shot"

    save_fixture(LAYER0_DIR, "conv-decode.safetensor", {
        "x_prefill3": x3,
        "x_decode1": xd1,
        "x_decode2": xd2,
        "out_prefill3": out3,
        "out_decode1": out_d1,
        "out_decode2": out_d2,
        "conv_state_prefill": state_after_prefill,
        "conv_state_d1": state_after_d1,
        "conv_state_d2": state_after_d2,
        "out_oneshot5": out5,
    })
    print("[conv] real layer-0 fixture: prefill + decode saved")


# =============================================================================
# Section B - attention layer (real layer 2, first full_attention) fixture
# =============================================================================

def generate_attention_layer_fixture() -> None:
    model, cfg, _ = load_checkpoint()
    torch.manual_seed(SEED_LAYER2)
    layer = model.model.layers[2]
    assert cfg.layer_types[2] == "full_attention"

    x = seeded_input(0x5555, 1, 5, cfg.hidden_size)
    pos_ids = torch.arange(5).unsqueeze(0)
    rotary = Lfm2RotaryEmbedding(config=cfg)
    with torch.no_grad():
        cos, sin = rotary(x, position_ids=pos_ids)
        block_out = layer(
            x,
            position_embeddings=(cos, sin),
            attention_mask=None,
            position_ids=pos_ids,
        )

    with torch.no_grad():
        h_norm = layer.operator_norm(x)
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        q = layer.self_attn.q_proj(h_norm).view(1, 5, cfg.num_attention_heads, head_dim).transpose(1, 2)
        k = layer.self_attn.k_proj(h_norm).view(1, 5, cfg.num_key_value_heads, head_dim).transpose(1, 2)
        v = layer.self_attn.v_proj(h_norm).view(1, 5, cfg.num_key_value_heads, head_dim).transpose(1, 2)
        qn = layer.self_attn.q_layernorm(q)
        kn = layer.self_attn.k_layernorm(k)
        qr, kr = apply_rotary_pos_emb(qn, kn, cos, sin)
        attn_out = F.scaled_dot_product_attention(
            qr, repeat_kv(kr, cfg.num_attention_heads // cfg.num_key_value_heads),
            repeat_kv(v, cfg.num_attention_heads // cfg.num_key_value_heads),
            is_causal=True, scale=head_dim ** -0.5)
        attn_out = attn_out.transpose(1, 2).reshape(1, 5, -1)
        o_proj_out = layer.self_attn.out_proj(attn_out)
        ffn_out = layer.feed_forward(layer.ffn_norm(x + o_proj_out))
        replay_out = x + o_proj_out + ffn_out
    assert torch.equal(replay_out, block_out), \
        "attention-layer replay diverged from the module forward"

    save_fixture(LAYER2_DIR, "attn-prefill.safetensor", {
        "x": x,
        "cos": cos,
        "sin": sin,
        "operator_norm_out": h_norm,
        "q_proj": q,
        "k_proj": k,
        "v_proj": v,
        "q_normed": qn,
        "k_normed": kn,
        "q_rot": qr,
        "k_rot": kr,
        "attn_out": attn_out,
        "out_proj_out": o_proj_out,
        "block_out": block_out,
    })
    print("[attn] real layer-2 fixture saved")


# =============================================================================
# Section C - 3-layer chain fixture (real layers 0..2: conv, conv, attention)
# =============================================================================

def generate_chain_fixture() -> None:
    model, cfg, _ = load_checkpoint()
    torch.manual_seed(SEED_CHAIN)

    x = seeded_input(0x6666, 1, 4, cfg.hidden_size)
    pos_ids = torch.arange(4).unsqueeze(0)
    rotary = Lfm2RotaryEmbedding(config=cfg)
    with torch.no_grad():
        cos, sin = rotary(x, position_ids=pos_ids)
        h = x
        tensors = {"layer0_in": x}
        for i, layer in enumerate(model.model.layers[:3]):
            if cfg.layer_types[i] == "full_attention":
                h = layer(h, position_embeddings=(cos, sin),
                          attention_mask=None, position_ids=pos_ids)
            else:
                h = layer(h)
            tensors[f"layer{i}_out"] = h
            if i < 2:
                tensors[f"layer{i + 1}_in"] = h
        tensors["embedding_norm_out"] = model.model.embedding_norm(h)
    save_fixture(CHAIN_DIR, "chain.safetensor", tensors)
    print("[chain] real layers 0..2 chain fixture saved")


# =============================================================================
# Section D - full-model ids -> logits
# =============================================================================

def generate_ids_logits_fixture() -> None:
    model, cfg, tok = load_checkpoint()
    torch.manual_seed(SEED_FULL)

    ids = tok(PROSE_PROMPT, return_tensors="pt", add_special_tokens=False).input_ids
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits
    save_fixture(IDS_DIR, "ids-logits.safetensor", {
        "input_ids": ids,
        "logits_last": logits[:, -1:, :],
    })
    print(f"[ids] real-model ids->logits fixture saved (T={ids.shape[1]})")


# =============================================================================
# Section E - greedy t2t fixtures (temp 0, 16 new tokens, differential gate)
# =============================================================================

def greedy_run(model, ids: torch.Tensor):
    """Greedy (temp 0) continuation of GREEDY_TOKENS tokens, no eos stop.
    Returns the generated ids, per-step top1-top2 logit margins, and the
    per-step (top1, runner-up) token id pairs. The pairs are what let the Nim
    test tell a near-tie from a wrong token: a margin of one or two bf16 steps
    at these logit magnitudes is inside the measured bf16 noise band, so the
    runner-up is an equally admissible argmax."""
    new_ids, margins, top2_ids = [], [], []
    past = None
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        step = out.logits[:, -1, :]
        for _ in range(GREEDY_TOKENS):
            top2 = torch.topk(step.float(), k=2, dim=-1)
            values, indices = top2.values[0], top2.indices[0]
            margins.append(round(float(values[0] - values[1]), 6))
            top2_ids.append([int(indices[0]), int(indices[1])])
            nxt = step.argmax(dim=-1, keepdim=True)
            new_ids.append(int(nxt))
            out = model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            step = out.logits[:, -1, :]
    return torch.tensor([new_ids]), margins, top2_ids


def save_greedy_fixture(name: str, prompt: str, ids: torch.Tensor) -> None:
    tok = _TOKENIZER
    first = greedy_run(_MODEL, ids)
    second = greedy_run(_MODEL, ids)
    assert torch.equal(first[0], second[0]), "greedy trajectory not deterministic"
    generated_ids = first[0][0].tolist()
    payload = {
        "prompt": prompt,
        "prompt_ids": ids[0].tolist(),
        "generated_ids": generated_ids,
        "generated_text": tok.decode(generated_ids),
        "num_prompt_tokens": int(ids.shape[1]),
        "num_generated_tokens": len(generated_ids),
        "eos_token_id": _CONFIG.eos_token_id,
        "argmax_margins": first[1],
        "argmax_top2_ids": first[2],
        "note": "temp 0 greedy, no eos stop, 16 tokens; margins are debug-only",
    }
    with open(os.path.join(GREEDY_DIR, name), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[t2t] {name}: {len(generated_ids)} tokens, "
          f"margins[{first[1][0]:.2f}..{min(first[1]):.2f}]")


def generate_greedy_fixtures() -> None:
    _, _, tok = load_checkpoint()

    prose_ids = tok(PROSE_PROMPT, return_tensors="pt", add_special_tokens=False).input_ids
    save_greedy_fixture("Long_prose_prompt.json", PROSE_PROMPT, prose_ids)

    chat_text = tok.apply_chat_template(
        [{"role": "user", "content": CHAT_USER_MSG}],
        tokenize=False, add_generation_prompt=True)
    chat_ids = tok(chat_text, return_tensors="pt", add_special_tokens=False).input_ids
    assert chat_ids.shape[1] >= 10, "chat-template render failed to include specials"
    save_greedy_fixture("Chat_template_prompt.json", chat_text, chat_ids)


RUNNERS = {
    "conv": generate_conv_layer_fixture,
    "attn": generate_attention_layer_fixture,
    "chain": generate_chain_fixture,
    "ids": generate_ids_logits_fixture,
    "greedy": generate_greedy_fixtures,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the LFM2.5-230M Nim q_bf16 fixtures.")
    parser.add_argument(
        "--only", choices=SECTIONS, action="append", metavar="SECTION",
        help="generate only this section (repeatable); default is all of them")
    args = parser.parse_args()
    sections = args.only or list(SECTIONS)

    load_checkpoint()
    ensure_fixture_dirs(sections)
    for section in sections:
        RUNNERS[section]()
    print("LFM2.5-230M fixtures generated from the real checkpoint: "
          + ", ".join(sections))


if __name__ == "__main__":
    main()
