#!/usr/bin/env python3
"""Fixture-recording helper (fixture_ prefix, never a gen_ generator) emitting
sanctioned re-record payload classes:

- uniform stats frames, argmax decision frames, value-bearing slices
- decision projections

The record byte format mirrors tests/harness/harness.nim procs writeUniformStats and writeArgmaxDecisions. Shared container rules:

- one single-line JSON object plus a trailing newline inside one zstd frame, compact JSON separators, ensure_ascii=False
- hex bit patterns carry 8 uppercase digits (f32) or 16 (f64)
- the zstd frame runs level 19 with content size and checksum
- python and Nim must produce the same record on the same tensor bytes

Uniform stats record (ttt-tf-004-uniform-stats), one record per tensor:

- n, grid ("bf16" / "fp16" / "f32", the storage grid of the recorded dtype)
- quantiles cover min, the 9 fixed probabilities, and max
- quantiles are exact order statistics of the ascending sort, index floor(p * (n - 1)) computed in f64, no interpolation
- quantile values promote to f32 from the native dtype
- the histogram is a binade-log soft histogram over the bf16 pattern of the promoted values, integer counts scaled x2
- each histogram element contributes (2 - low) toward its bucket, low toward the neighbor (low is the dropped 8th mantissa bit)
- dedicated bins cover exact zero (65534) and subnormal patterns (65535)
- NaN and +Inf raise, -Inf stays unbucketed
- mean_abs / signed_mean / tail_probability / max_magnitude are hex f64 values
- the means accumulate in f64 over the finite elements in index order
- the tail fraction divides by the full element count
- the tail threshold is max_magnitude / 2^4
- tail_edge counts the finite elements within TailEdgeSteps grid steps (the grid step on the record's storage grid)

Argmax decision frame (ttt-tf-005-argmax-decisions), one record per greedy step:

- the argmax id and the top-32 set ids with f32 logits as hex bit patterns
- the margin = top1 - top2 and the softmax mass beyond the top-32 set
- statistical descriptors of the recorded step only, the bands derive at check
- the assert's error kind feeds the bands, no serialized instrument constant
- the frame carries {"schema", "source", "ulp_datatype", "steps"} and nothing else
- "ulp_datatype" is the recorded activation dtype of the whole frame, "bf16" / "fp16" / "f32", one model per frame
- one unit = the activation ulp of that dtype, never a hardcoded weight grid
"""

import json
import math
import os
import struct

import compression.zstd
import numpy as np

ZSTD_WRITE_OPTIONS = {
    compression.zstd.CompressionParameter.compression_level: 19,
    compression.zstd.CompressionParameter.content_size_flag: 1,
    compression.zstd.CompressionParameter.checksum_flag: 1,
}

import torch

QUANTILE_PROBS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    # Fixed quantile probabilities of the uniform record. Min and max are
    # recorded alongside the nine.

QUANTILE_NAMES = ["p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95",
                  "p99"]
    # Stats-frame keys of the fixed quantiles, in QUANTILE_PROBS order.

HIST_KEY_ZERO = 0xFFFE
    # Histogram key of the exact zero bin.

HIST_KEY_SUBNORMAL = 0xFFFF
    # Histogram key of the subnormal bin.

HIST_BINADE_MIN = -64
    # Lowest binade a histogram bucket covers.

HIST_BINADE_MAX = 63
    # Highest binade a histogram bucket covers.

STATS_SCHEMA = "ttt-tf-004-uniform-stats"
    # Format registry id of the uniform stats record frame.

ARGMAX_SCHEMA = "ttt-tf-005-argmax-decisions"
    # Format registry id of the argmax decision frame.

    # Strided sample target in f32 words, about 2 kB of values

TAIL_BINADES_BELOW_MAX = 4.0
    # Tail threshold placement, threshold = recorded max / 2^4
    # (TailBinadesBelowMax in tests/ulp_utils.nim).

TAIL_EDGE_STEPS = 4
    # Edge-band width of the tail instrument (TailEdgeSteps), counted
    # in grid steps of the threshold binade.

GRID_MANTISSA_BITS = {"bf16": 7, "fp16": 10, "f32": 23}
    # Mantissa bits per storage grid, the grid-step exponent offset
    # (gridStep in tests/ulp_utils.nim).


def f32_bits(v):
    """Bit pattern of one f32 value, little-endian layout."""
    return struct.unpack("<I", struct.pack("<f", v))[0]


def bf16_bits_from_f32(u):
    """Round-to-nearest-even f32 bit pattern to a bf16 bit pattern,
    matching torch `.to(bfloat16)` and tests/ulp_utils.nim bf16BitsFromF32."""
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000) >> 16


def hex_f32(v):
    """One f32 value as a hex bit pattern, 8 uppercase digits, -Inf included, matching hexF32 in tests/harness/harness.nim."""
    return "0x%08X" % f32_bits(v)


def hex_f64(v):
    """One f64 value as a hex bit pattern, 16 uppercase digits,
    matches hexF64 in tests/harness/harness.nim."""
    return "0x%016X" % struct.unpack("<Q", struct.pack("<d", v))[0]


def grid_of(x):
    """Storage-grid key of one float tensor ("bf16", "fp16" or "f32"),
    anything else raises (integer payloads carry no stats record)."""
    if x.dtype == torch.bfloat16:
        return "bf16"
    if x.dtype == torch.float16:
        return "fp16"
    if x.dtype == torch.float32:
        return "f32"
    raise ValueError("stats record of unsupported dtype: %s" % x.dtype)


def binade_of(v):
    """Binade index of a normal nonzero magnitude, |v| in [2^e, 2^(e+1))
    reading binade e, matching binadeOf in tests/ulp_utils.nim.

    Raises:
    - zero and non-finite input (no grid reference exists)
    """
    if v == 0.0 or not math.isfinite(v):
        raise ValueError("grid binade of a zero or non-finite magnitude "
                         "has no reference: %r" % v)
    return math.frexp(abs(v))[1] - 1


def grid_step(grid, binade):
    """Width of one grid step in binade [2^e, 2^(e+1)), i.e.
    2^(binade - mantissa bits) on the record's storage grid (gridStep in tests/ulp_utils.nim)."""
    return 2.0 ** (binade - GRID_MANTISSA_BITS[grid])


def _promoted(x):
    """Flat f64-promoted value list plus per-element bf16 pattern list,
    both in index order.

    - bf16 payloads promote by pattern shift, the exact f32 promotion
    - fp16 and f32 payloads promote through torch `.to(float32)`, exact
      for fp16 including the 2^-24 subnormal grid
    - the histogram patterns read the bf16 rounding of the promoted
      value (the same words the Nim tensorStats reads for every dtype)
    """
    flat = x.reshape(-1).contiguous()
    if flat.dtype == torch.bfloat16:
        bits = flat.view(torch.int16).numpy().astype(np.uint32) & np.uint32(0xFFFF)
        vals = ((bits << np.uint32(16))).view(np.float32).tolist()
        return vals, bits.tolist()
    vals = flat.to(torch.float32)
    words = vals.numpy().view(np.uint32)
    return vals.tolist(), [bf16_bits_from_f32(int(u)) for u in words]


def _check_finite(vals, name):
    """NaN and +Inf raise, -Inf self-declares (unbucketed, absent from the means and the tail)."""
    for v in vals:
        u = f32_bits(v)
        exp, man = (u >> 23) & 0xFF, u & 0x7FFFFF
        if exp != 0xFF:
            continue
        if man != 0:
            raise ValueError("stats record input holds NaN: " + name)
        if not (u & 0x80000000):
            raise ValueError("stats record input holds +Inf: " + name)


def _soft_histogram(bits, name):
    """Soft binade histogram over the bf16 patterns, integer math only.

    - one element contributes (2 - low) toward its bucket, low toward
      the neighbor bucket (low is the dropped 8th mantissa bit)
    - counts scale x2 and one bucket spans 4 ulp
    - exact zero and subnormal patterns land in the dedicated bins
    - the 0xFF exponent field (Inf and NaN families) never buckets
    - a binade outside [-64, 63] raises

    Returns:
    - the sorted "key:count" bucket string plus the exact total
    """
    counts = {}
    for b in bits:
        sign = b & 0x8000
        exp = (b >> 7) & 0xFF
        mant = b & 0x7F
        if exp == 0 and mant == 0:
            counts[HIST_KEY_ZERO] = counts.get(HIST_KEY_ZERO, 0) + 2
        elif exp == 0:
            counts[HIST_KEY_SUBNORMAL] = counts.get(HIST_KEY_SUBNORMAL, 0) + 2
        elif exp == 0xFF:
            continue
        else:
            e = exp - 127
            if e < HIST_BINADE_MIN or e > HIST_BINADE_MAX:
                raise ValueError(
                    "stats record binade %d outside [%d, %d]: %s"
                    % (e, HIST_BINADE_MIN, HIST_BINADE_MAX, name)
                )
            mtop, low = mant >> 1, mant & 1
            key = (sign >> 2) | ((e - HIST_BINADE_MIN) << 6) | mtop
            counts[key] = counts.get(key, 0) + (2 - low)
            if low:
                if mtop == 63:
                    if e + 1 <= HIST_BINADE_MAX:
                        nk = (sign >> 2) | ((e + 1 - HIST_BINADE_MIN) << 6)
                        counts[nk] = counts.get(nk, 0) + 1
                else:
                    counts[key + 1] = counts.get(key + 1, 0) + 1
    packed = ",".join("%d:%d" % (k, counts[k]) for k in sorted(counts))
    return packed, sum(counts.values())


def tensor_stats(x, name):
    """One uniform stats record of one tensor, the encodeStatsBody
    byte shape in tests/harness/harness.nim.

    Args:
    - x, one non-empty float tensor (bf16, fp16 or f32), host memory
    - name, the recorded tensor name, the frame lookup key

    Returns:
    - the record dict in the stored key order, first holding n / grid
      / quantiles / histogram
    - then mean_abs / signed_mean / tail_probability / tail_edge / max_magnitude
    """
    grid = grid_of(x)
    vals, bits = _promoted(x)
    n = len(vals)
    if n == 0:
        raise ValueError("stats record of an empty tensor " + name)
    _check_finite(vals, name)

    # Means and max magnitude accumulate in f64 over the INDEX order
    # over the finite elements, the same arithmetic the Nim tensorStats
    # runs. Sorted accumulation would break the byte contract.
    n_finite = 0
    abs_sum = 0.0
    signed_sum = 0.0
    max_magnitude = 0.0
    for v in vals:
        if v == -math.inf:
            continue
        n_finite += 1
        abs_sum += abs(v)
        signed_sum += v
        if abs(v) > max_magnitude:
            max_magnitude = abs(v)
    if n_finite == 0:
        raise ValueError(
            "stats record of an all -Inf tensor has no finite mean: " + name
        )

    svals = sorted(vals)
    quantiles = {"min": hex_f32(svals[0]), "max": hex_f32(svals[-1])}
    for qn, p in zip(QUANTILE_NAMES, QUANTILE_PROBS):
        quantiles[qn] = hex_f32(svals[math.floor(p * (n - 1))])

    packed, total = _soft_histogram(bits, name)

    record = {
        "n": n,
        "grid": grid,
        "quantiles": quantiles,
        "histogram": {"total": total, "buckets": packed},
        "mean_abs": hex_f64(abs_sum / n_finite),
        "signed_mean": hex_f64(signed_sum / n_finite),
        "tail_probability": hex_f64(0.0),
        "tail_edge": 0,
        "max_magnitude": hex_f64(max_magnitude),
    }
    if max_magnitude > 0.0:
        threshold = max_magnitude / 2.0 ** TAIL_BINADES_BELOW_MAX
        edge_band = TAIL_EDGE_STEPS * grid_step(grid, binade_of(threshold))
        tail_count = 0
        edge_count = 0
        for v in vals:
            if v == -math.inf:
                continue
            a = abs(v)
            if a > threshold:
                tail_count += 1
            if abs(a - threshold) <= edge_band:
                edge_count += 1
        record["tail_probability"] = hex_f64(tail_count / n)
        record["tail_edge"] = edge_count
    return record


def assert_path_equivalent(got, want, what):
    """Path-equivalence guard between two compute orders of one step.

    Args:
    - got, want, the two compute orders of the same step
    - what, the label carried in the rejection message

    Every element agrees within two ulps of its own dtype, the ulp width measured
    at that element's magnitude. Verifies path equivalence, never bit identity.
    """
    bits = {torch.bfloat16: 7.0, torch.float16: 10.0, torch.float32: 23.0}[got.dtype]
    g = got.float()
    w = want.float()
    mag = torch.maximum(g.abs(), w.abs()).clamp_min(1e-30)
    step = torch.pow(2.0, torch.floor(torch.log2(mag)) - bits)
    drift = (g - w).abs()
    assert bool((drift <= 2.0 * step).all()), (
        f"{what}: path drift past two ulps of the dtype, max {drift.max().item()}")


def stats_file_bytes(source, entries):
    """Deterministic single-line stats frame text, the writeUniformStats
    byte shape plus a trailing newline.

    Args:
    - source, the recorded payload file name embedded in the frame
    - entries, list of (name, tensor) pairs with one uniform record
      per pair, the frame insertion order following the list order

    Returns:
    - the exact utf-8 frame bytes
    """
    tensors = {}
    for name, tensor in entries:
        if name in tensors:
            # A duplicate name would silently keep only the last entry
            # (dict last-wins) and the earlier record would never be
            # written or read, so the mismatch raises instead.
            raise ValueError("duplicate stats entry name: " + name)
        tensors[name] = tensor_stats(tensor, name)
    obj = {"schema": STATS_SCHEMA, "source": source, "tensors": tensors}
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    ) + b"\n"


def write_stats_file(path, source, entries):
    """Write the stats sidecar `<fixture>.stats.json.zst` beside the recorded payload."""
    write_text_zst(path, stats_file_bytes(source, entries))


def write_text_zst(path, payload):
    """One zstd frame around exact utf-8 payload bytes.

    - level 19, content size and checksum in the frame header
    - the JSON format stays the caller's own, the container carries
      the bytes and defines nothing
    """
    with open(path, "wb") as f:
        f.write(compression.zstd.compress(payload, options=ZSTD_WRITE_OPTIONS))


def write_json_zst(path, obj):
    """One zstd frame around the compact single-line JSON of obj, the 001 fixture container."""
    with open(path, "wb") as f:
        f.write(compression.zstd.compress(
            json.dumps(obj, separators=(",", ":")).encode("utf-8"),
            options=ZSTD_WRITE_OPTIONS))


def read_text_zst(path):
    """Exact payload bytes of one zstd frame, the reader paired with write_text_zst.

    - a corrupt or content-size-unknown frame fails loudly, never silently
    """
    with open(path, "rb") as f:
        return compression.zstd.decompress(f.read())


# -------------------------------------------------------------------- Argmax decision frames

def argmax_record(argmax_id, top_k, top_k_logits, margin, tail_probability):
    """One argmax decision record, the encodeArgmaxBody byte shape (tests/harness/harness.nim).

    Args:
    - argmax_id, the recorded argmax token id (top_k[0] equals it)
    - top_k, the top-32 token ids in descending logit order
    - top_k_logits, the top-32 f32 logits, index-aligned
    - margin, the recorded top1 - top2 gap (f64)
    - tail_probability, the softmax mass beyond the top-32 set (f64)

    Returns:
    - the record dict in the stored key order, holding argmax_id /
      margin / tail_probability / top_k / top_k_logits, statistical
      descriptors of the recorded step, the bands derive at check
      from the assert's error kind
    """
    if len(top_k) != len(top_k_logits):
        raise ValueError("argmax record top-32 ids and logits misaligned")
    if top_k[0] != argmax_id:
        raise ValueError(
            "argmax record top-32 head %d is not the argmax id %d"
            % (top_k[0], argmax_id)
        )
    return {
        "argmax_id": int(argmax_id),
        "margin": hex_f64(margin),
        "tail_probability": hex_f64(tail_probability),
        "top_k": [int(i) for i in top_k],
        "top_k_logits": " ".join(hex_f32(v) for v in top_k_logits),
    }


def argmax_record_from_row(row):
    """argmax_record of one flat f32 logits row.

    - the top-32 set comes from torch.topk, the margin = top1 - top2
    - the tail probability is the softmax mass beyond the top-32 set
    - row is one f32 logits row of any single-token shape, flattened
      to the [V] view
    """
    vals = row.reshape(-1).contiguous().to(torch.float32)
    top_vals, top_idxs = torch.topk(vals, 32)
    probs = torch.softmax(vals, dim=-1)
    tail = 1.0 - probs[top_idxs].sum().item()
    margin = float(top_vals[0].item()) - float(top_vals[1].item())
    return argmax_record(
        int(top_idxs[0].item()), top_idxs.tolist(),
        [float(v) for v in top_vals.tolist()],
        margin, tail)


def argmax_record_from_step(step):
    """argmax_record transcribed from one recorded greedy step node.

    - chosen_token, top32_ids, top32_logits, argmax_margin,
      tail_probability are the recorded values
    - the transcription tie-normalizes the head on the equal-value top tie
    - the chain pick can sit inside the tie group off the topk head
    - the pick swaps into the head slot carrying its own recorded logit
    - an off-set or off-value pick raises
    """
    ids = list(step["top32_ids"])
    vals = list(step["top32_logits"])
    pick = step["chosen_token"]
    if ids[0] != pick:
        if pick not in ids:
            raise ValueError(
                "recorded pick %d is outside the top-32 support" % pick)
        j = ids.index(pick)
        if vals[j] != vals[0]:
            raise ValueError(
                "recorded pick %d is not inside the equal-value tie group"
                % pick)
        ids[0], ids[j] = ids[j], ids[0]
        vals[0], vals[j] = vals[j], vals[0]
    return argmax_record(
        pick, ids, vals,
        step["argmax_margin"], step["tail_probability"])


def argmax_decisions_bytes(source, records, ulp_datatype):
    """Deterministic single-line argmax decision frame text carrying
    the writeArgmaxDecisions byte shape plus a trailing newline.

    Args:
    - source, the recorded chain file name embedded in the frame
    - records, the per-step record dicts in frame order
    - ulp_datatype, the recorded activation dtype of the frame, "bf16" /
      "fp16" / "f32", the unit every decision band measures in, one key
      per frame (the whole frame = one model)

    Returns:
    - the exact utf-8 frame bytes
    """
    if ulp_datatype not in GRID_MANTISSA_BITS:
        raise ValueError(
            "argmax decisions frame ulp_datatype %r is not one of %s"
            % (ulp_datatype, sorted(GRID_MANTISSA_BITS)))
    obj = {"schema": ARGMAX_SCHEMA, "source": source,
           "ulp_datatype": ulp_datatype, "steps": records}
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    ) + b"\n"


def write_argmax_decisions(path, source, records, ulp_datatype):
    """Write the argmax decision sidecar (argmax_decisions_bytes content, container defines nothing)."""
    write_text_zst(path, argmax_decisions_bytes(source, records, ulp_datatype))


# ------------------------------------------------------------- Slices (safetensors)

def save_slices(path, tensors, metadata=None):
    """Write a value-bearing slice fixture as safetensors.

    Args:
    - tensors, a list of (name, tensor) pairs
    - metadata, a JSON-serializable dict (optional)
    """
    from safetensors.torch import save_file

    payload = {name: t.contiguous() for name, t in tensors}
    if metadata is not None:
        meta = {k: json.dumps(v) for k, v in metadata.items()}
        save_file(payload, path, metadata=meta)
    else:
        save_file(payload, path)


def recording_env(model=None, generator=None, seed=None, extra=None):
    """Standard recording-environment dict for one recording run.

    recorded_from names box plus device.

    - the TTT_RECORD_FROM environment variable overrides the default recording box
    - a recording on a non-default box must set it, the value landing
      in the greedy env frame
    """
    import os
    import platform

    import transformers

    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "recorded_from": os.environ.get("TTT_RECORD_FROM", "m4max-cpu"),
        "platform": platform.platform(),
        "date": __import__("datetime").date.today().isoformat(),
    }
    if model is not None:
        env["model"] = model
    if generator is not None:
        env["generator"] = generator
    if seed is not None:
        env["seed"] = str(seed)
    if extra:
        env.update(extra)
    return env
