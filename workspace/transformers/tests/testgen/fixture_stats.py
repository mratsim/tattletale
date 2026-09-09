#!/usr/bin/env python3
"""Shared fixture-recording helper (fixture_ prefix, never a gen_
generator). Emits the four payload classes of a sanctioned re-record:
fingerprint stats files, value-bearing slices, decision projections, and
PROVENANCE.md rows.

The stats byte format mirrors harness/gen_stats.nim (writeFingerprintStats):
one jsony single-line JSON object plus a trailing newline, hex f32 bit
patterns for quantiles, packed sorted "key:count" bucket strings for
histograms, field order fixed by declaration order on the Nim side. The
python twin and gen_stats.nim must agree byte for byte on the same tensor
bytes, verified on the validation corpus and the cross-implementation corpus.

Fixed quantile method: exact order statistic of the ascending sort, index
floor(p * (n - 1)) computed in f64, no interpolation, values promoted from
the native dtype to f32 for storage (harness/SPEC.md signatures).
Fixed probabilities: 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99,
plus min and max. Order statistics are bit-exact cross-implementation.

Fixed bucket spec: binade-log soft histogram over the bf16 bit pattern.
Bucket = (sign, binade -64..63, mantissa top 6 bits), 14-bit keys in sparse
sorted storage, plus dedicated zero (65534) and subnormal (65535) bins.
Soft bucketing: each element contributes (2 - low) toward its bucket and
low toward the neighbor bucket, low being the dropped 8th mantissa bit,
integer math, bit-exact cross-implementation. Bucket width is 4 ulp
(6 of 8 mantissa bits kept). NaN and +Inf raise. -Inf is
whitelisted only through allow_minus_inf and stays unbucketed.
"""

import json
import math
import os
import struct
import compression.zstd

ZSTD_WRITE_OPTIONS = {
    compression.zstd.CompressionParameter.compression_level: 19,
    compression.zstd.CompressionParameter.content_size_flag: 1,
    compression.zstd.CompressionParameter.checksum_flag: 1,
}

import torch

QUANTILE_PROBS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
HIST_KEY_ZERO = 0xFFFE
HIST_KEY_SUBNORMAL = 0xFFFF
HIST_BINADE_MIN = -64
HIST_BINADE_MAX = 63
STATS_SCHEMA = "ttt-tf-003-tensor-stats-h2"

# Probe subset size target, in f32 words: about 2 kB of values, with the stride spreading the probe
# over the whole tensor. Mirrors harness/tolerance.nim DescriptorProbeWords.
DESCRIPTOR_PROBE_WORDS = 512
# The tail threshold sits four binades under the recorded max: threshold = max / 2^4. Mirrors
# TailBinadesBelowMax.
TAIL_BINADES_BELOW_MAX = 4.0
# Mode-pair and descriptor drift cap for f32 state-like tensors: four fp32 ulps at the state max
# magnitude. Mirrors SsmUlpMargin.
SSM_ULP_MARGIN = 4.0

QuantileNames = ["p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95",
                 "p99"]
  ## Stats-file keys of the fixed quantiles, in QuantileNames order.

RECORDING_SCHEMA = 1
  ## Version of the recording schema read by harness/recording.nim


def f32_bits(v):
    """Bit pattern of one f32 value, little-endian layout."""
    return struct.unpack("<I", struct.pack("<f", v))[0]


def bf16_bits_from_f32(u):
    """Round-to-nearest-even f32 bit pattern to a bf16 bit pattern, matching
    torch `.to(bfloat16)` and harness/tolerance.nim bf16BitsFromF32."""
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000) >> 16


def quantile_hex(v):
    """Store one quantile as a hex f32 bit pattern, uppercase, or "-inf"."""
    if math.isinf(v) and v < 0:
        return "-inf"
    return "0x%08X" % f32_bits(v)


def _flat_values(x):
    """Flat list of f64-promoted values plus bf16 bits when the tensor is
    bf16. Promotion of a bf16 pattern into the f32 mantissa top is exact,
    matching tolerance.nim bf16ToF32."""
    flat = x.reshape(-1).contiguous()
    if flat.dtype == torch.bfloat16:
        bits = flat.view(torch.int16).numpy().astype("int64") & 0xFFFF
        vals = [
            struct.unpack("<f", struct.pack("<I", int(b) << 16))[0] for b in bits
        ]
        return vals, [int(b) for b in bits]
    vals = flat.to(torch.float32).numpy().tolist()
    return vals, None


def _check_finite(vals, name, allow_minus_inf):
    """NaN and +Inf raise, -Inf only through the whitelist."""
    for v in vals:
        u = f32_bits(v)
        exp, man = (u >> 23) & 0xFF, u & 0x7FFFFF
        if exp != 0xFF:
            continue
        if man != 0:
            raise ValueError("fingerprint input holds NaN: " + name)
        if not (u & 0x80000000):
            raise ValueError("fingerprint input holds +Inf: " + name)
        if not allow_minus_inf:
            raise ValueError(
                "fingerprint input holds -Inf without the mask whitelist: " + name
            )


def tensor_stats(x, name, with_hist, allow_minus_inf=False, desc_mode=None):
    """One tensor entry of a stats file, byte-shape identical to the
    TensorStatsJson encoding of tolerance.nim encodeTensorStatsBody.
    desc_mode None writes the fingerprint-only historical layout; the
    strings "exact" and "drift" append the descriptor keys."""
    vals, bits = _flat_values(x)
    n = len(vals)
    if n == 0:
        raise ValueError("fingerprint of an empty tensor " + name)
    _check_finite(vals, name, allow_minus_inf)

    # The descriptor means accumulate in f64 over the INDEX order, the same arithmetic
    # tolerance.nim tensorDescriptors runs. The sort below must not reorder the accumulation: f64
    # sums are order-sensitive and the byte law breaks on sorted accumulation.
    desc = None
    if desc_mode is not None:
        desc = descriptor_fields(vals, max(vals), desc_mode)

    vals.sort()
    pinned = [vals[math.floor(p * (n - 1))] for p in QUANTILE_PROBS]
    # Stored field order follows the Nim declaration order: min and max first, then the fixed
    # quantiles in QuantileNames order (p01 to p99).
    quantiles = {"min": quantile_hex(vals[0]), "max": quantile_hex(vals[-1])}
    for k, v in zip(QuantileNames, pinned):
        quantiles[k] = quantile_hex(v)

    entry = {
        "n": n,
        "allow_minus_inf": allow_minus_inf,
        "quantiles": quantiles,
        "histogram": None,
    }
    if desc is not None:
        entry.update(desc)
    if not with_hist:
        return entry

    if bits is None:
        # _raw_f32 yields the f32 bit patterns already. Repacking them through f32_bits would reinterpret
        # the pattern as a float value.
        bits = [bf16_bits_from_f32(u) for u in _raw_f32(x)]
    counts = {}
    for b in bits:
        sign = b & 0x8000
        exp = (b >> 7) & 0xFF
        mant = b & 0x7F
        if exp == 0 and mant == 0:
            counts[HIST_KEY_ZERO] = counts.get(HIST_KEY_ZERO, 0) + 2
        elif exp == 0:
            counts[HIST_KEY_SUBNORMAL] = counts.get(HIST_KEY_SUBNORMAL, 0) + 2
        else:
            e = exp - 127
            if e < HIST_BINADE_MIN or e > HIST_BINADE_MAX:
                raise ValueError(
                    "fingerprint binade %d outside [%d, %d]: %s"
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
    total = sum(counts.values())
    packed = ",".join("%d:%d" % (k, counts[k]) for k in sorted(counts))
    entry["histogram"] = {"total": total, "buckets": packed}
    return entry


def ulp_fp32_at(m):
    """One fp32 ulp at magnitude m, mirroring tolerance.nim ulpFp32At.
    fp32 has 23 significand bits, so for m in [2**e, 2**(e+1)) the ulp
    is 2**(e-23). Zero maps to 0."""
    if m <= 0:
        return 0.0
    return 2.0 ** (math.floor(math.log2(m)) - 23.0)


def f64_hex(v):
    """Store one f64 descriptor as a hex bit pattern, uppercase, 16
    digits, byte-identical to tolerance.nim encodeF64."""
    return "0x%016X" % struct.unpack("<Q", struct.pack("<d", v))[0]


def descriptor_fields(vals, max_val, desc_mode):
    """Descriptor keys of one tensor entry, in the stored key order.
    vals is the f64-promoted value list in INDEX order, the same list
    order the Nim tensorDescriptors accumulates over (f64 sums are
    order-sensitive), max_val the promoted f32 max. The bulk and signed
    means accumulate in f64 over that index order, the same arithmetic
    harness/tolerance.nim tensorDescriptors runs, so the two writers
    agree bit-exactly on the same tensor bytes."""
    n = len(vals)
    abs_sum = 0.0
    sum_ = 0.0
    for v in vals:
        abs_sum += abs(v)
        sum_ += v
    mean_abs = abs_sum / n
    signed_mean = sum_ / n
    threshold = abs(max_val) / 2.0 ** TAIL_BINADES_BELOW_MAX
    max_drift = (
        SSM_ULP_MARGIN * ulp_fp32_at(abs(max_val))
        if desc_mode == "drift" else 0.0
    )
    tail_count = 0
    edge_count = 0
    for v in vals:
        a = abs(v)
        if a > threshold:
            tail_count += 1
        if abs(a - threshold) <= max_drift:
            edge_count += 1
    stride = 1 if n <= DESCRIPTOR_PROBE_WORDS else -(-n // DESCRIPTOR_PROBE_WORDS)
    count = -(-n // stride)
    bits = "".join(
        "%08X" % struct.unpack("<I", struct.pack("<f", vals[i * stride]))[0]
        for i in range(count))
    return {
        "mean_abs": f64_hex(mean_abs),
        "signed_mean": f64_hex(signed_mean),
        "tail_probability": f64_hex(tail_count / n),
        "tail_edge": edge_count,
        "probe_stride": stride,
        "probe_mode": desc_mode,
        "probe_bits": bits,
    }


def _raw_f32(x):
    """Flat list of raw f32 bit patterns, the promoted f32 view of any float
    tensor. The promotion is a no-op copy for f32 and the exact f32 promotion
    for fp16, so the histogram bf16 bits read the same patterns the Nim
    tensorStats reads for both dtypes."""
    flat = x.reshape(-1).contiguous().to(torch.float32)
    return flat.numpy().view("uint32").tolist()


def stats_file_bytes(source, entries):
    """Deterministic single-line stats file text plus newline, byte-shape
    identical to writeFingerprintStats. entries is a list of
    (name, tensor, with_hist, allow_minus_inf) tuples, optionally
    (name, tensor, with_hist, allow_minus_inf, desc_mode) for
    descriptor-carried entries."""
    tensors = {}
    for entry in entries:
        name, tensor, with_hist, allow_minus_inf = entry[:4]
        desc_mode = entry[4] if len(entry) > 4 else None
        if name in tensors:
            # A duplicate name would silently keep only the last
            # entry under dict last-wins behavior and the earlier
            # format would never be written or verified. The mismatch raises
            # instead.
            raise ValueError("duplicate stats entry name: " + name)
        tensors[name] = tensor_stats(
            tensor, name, with_hist, allow_minus_inf=allow_minus_inf,
            desc_mode=desc_mode)
    obj = {"schema": STATS_SCHEMA, "source": source, "tensors": tensors}
    return json.dumps(obj, separators=(",", ":")) + "\n"


def write_text_zst(path, payload):
    """One zstd frame, level 19, content size and checksum in the frame
    header, around exact utf-8 payload bytes.
    Container twin of write_json_zst: the JSON format stays the
    caller's own, the frame carries the bytes and defines nothing."""
    with open(path, "wb") as f:
        f.write(compression.zstd.compress(payload, options=ZSTD_WRITE_OPTIONS))


def read_text_zst(path):
    """Exact payload bytes of one zstd frame, the reader twin of
    write_text_zst. A corrupt or content-size-unknown frame fails
    loudly, never a silent empty result."""
    with open(path, "rb") as f:
        return compression.zstd.decompress(f.read())


def write_stats_file(path, source, entries):
    """Write the stats sidecar beside a committed fixture: the
    stats_file_bytes payload inside one zstd frame, the path spelled
    `<fixture>.stats.json.zst`. The byte law is content-exact, the
    container carries the bytes and defines nothing."""
    write_text_zst(path, stats_file_bytes(source, entries).encode("utf-8"))


def write_json_zst(path, obj):
    """Single zstd frame, level 19, content size and checksum in the
    frame header, matching the recording convention of recording.nim."""
    with open(path, "wb") as f:
        f.write(compression.zstd.compress(
            json.dumps(obj, separators=(",", ":")).encode("utf-8"),
            options=ZSTD_WRITE_OPTIONS))


# ########################################################################## Decision projections
# ##########################################################################

def decision_steps(logits):
    """Decision projection of final logits [1, seq, vocab]: per position the
    argmax id, the top-2 competing pair with f32 logits, and the softmax
    tail probability beyond the pair. Raw [1, seq, vocab] tensors of the
    13.4 MB class shrink to tens of KB."""
    assert logits.dim() == 3 and logits.size(0) == 1, "logits must be [1, seq, vocab]"
    steps = []
    row_all = logits[0]
    for pos in range(row_all.size(0)):
        row = row_all[pos].to(torch.float32)
        top2 = torch.topk(row, 2)
        ids = top2.indices.tolist()
        lv = top2.values.tolist()
        probs = torch.softmax(row, dim=-1)
        tail = 1.0 - probs[ids[0]].item() - probs[ids[1]].item()
        steps.append({
            "position": pos,
            "argmax_id": ids[0],
            "top2_ids": ids,
            "top2_logits": lv,
            "argmax_margin": lv[0] - lv[1],
            "tail_probability": tail,
        })
    return steps


LOGITS_PROBE_WORDS = 512
    ## Strided probe word count of the final-logits decision rows: one f32 word every ceil(vocab/512)
    ## positions, bit-exact over the reference device.


def decision_steps_probed(logits):
    """decision_steps plus the strided probe of every deciding row: the
    external surface of the raw logits grows from the top-2 pair to a
    512-word bit-exact sample per position, without any full row."""
    steps = decision_steps(logits)
    row_all = logits[0]
    for pos, step in enumerate(steps):
        flat = row_all[pos].to(torch.float32).flatten().tolist()
        n = len(flat)
        stride = -(-n // LOGITS_PROBE_WORDS)
        # The Nim reader (assertProjection) requires the stride to cover
        # exactly 512 words: (n - 1) div stride + 1 == 512. Not an identity
        # for every vocab, so the writer refuses at record time a fixture
        # no suite could read, naming the offending vocab.
        words = -(-n // stride)
        if words != LOGITS_PROBE_WORDS:
            raise ValueError(
                "probe geometry: vocab %d yields %d words at stride %d, "
                "the reader requires exactly %d (fix the vocab, not the "
                "stride)" % (n, words, stride, LOGITS_PROBE_WORDS))
        step["probe_stride"] = stride
        step["probe_mode"] = "exact"
        step["probe_bits"] = "".join(
            "%08X" % struct.unpack("<I", struct.pack("<f", flat[i * stride]))[0]
            for i in range(LOGITS_PROBE_WORDS))
    return steps


# ########################################################################## Slices
# ##########################################################################

def save_slices(path, tensors, metadata=None):
    """Write a value-bearing slice fixture as safetensors. tensors is a
    list of (name, tensor) pairs, metadata a JSON-serializable dict."""
    from safetensors.torch import save_file

    payload = {name: t.contiguous() for name, t in tensors}
    if metadata is not None:
        meta = {k: json.dumps(v) for k, v in metadata.items()}
        save_file(payload, path, metadata=meta)
    else:
        save_file(payload, path)


# ########################################################################## PROVENANCE
# ##########################################################################

PROVENANCE_TITLE = "# PROVENANCE\n\n"
PROVENANCE_NOTE = ("Recording environment rows. Generated by "
                   "harness/provenance.nim, never hand-edited.\n\n")


def render_provenance(entries):
    """Deterministic markdown rendering, byte-identical to
    harness/provenance.nim renderProvenance: title, contract note, then one
    | key | value | row per entry in sorted key order. entries is a list of
    (key, value) pairs."""
    rows = [PROVENANCE_TITLE, PROVENANCE_NOTE, "| key | value |\n",
            "|-----|-------|\n"]
    for key, value in sorted(entries, key=lambda kv: kv[0]):
        rows.append("| %s | %s |\n" % (key, value))
    return "".join(rows)


def write_provenance(path, entries):
    """Render the rows and write them to path."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_provenance(entries))


def provenance_entries(env):
    """Recording stamps: date, versions, platform, generator, seed,
    kept light enough to recreate the recording run. Required
    keys python, torch, transformers, recorded_from are carried by
    env."""
    return list(env.items())


def recording_env(model=None, generator=None, seed=None, extra=None):
    """Standard recording-environment dict for one recording run. recorded_from
    names box plus device; the TTT_RECORD_FROM environment variable overrides
    the default recording box (a recording on a non-default box must set it,
    the value lands in PROVENANCE.md and in the env frame of the greedy
    fixtures)."""
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
