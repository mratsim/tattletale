# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Bootstrap tool for fingerprint stats files: one recorded stats sidecar
## per fixture family, one entry per recorded tensor.
##
## Dataflow:
##
##   fixture safetensors -> summary computation -> sidecar beside the fixture
##   (committed bytes)     (tensorStats,            `<fixture>.stats.json.zst`
##                          tensorDescriptors)      `<fixture>.descriptors.json.zst`
##
## Run manually, not through the test suite:
##   nim cpp -r --hints:off --warnings:off --outdir:build/tools \
##     --nimcache:nimcache/tools \
##     workspace/transformers/tests/harness/gen_stats.nim
##
## Stats files are committed data. Regenerate only after a sanctioned
## re-record (see SPEC.md and PLAYBOOK.md).
##
## Two sidecar classes per fixture:
## - `<fixture>.stats.json.zst`, the fingerprint sidecar frame: entries for
##   the fingerprint-only tensors (dmNone), regenerable from the committed
##   payload bytes, byte-exact format enforced by fixture_stats_selfcheck.py
## - `<fixture>.descriptors.json.zst`, the descriptor sidecar frame: entries
##   for the descriptor-carried tensors (dmExact, dmDrift)
##
## Frozen descriptor entries: the entries written when the fixtures moved
## from inline raw tensors to the descriptor summaries and zstd frames
## describe tensors absent from the payload, their source bytes are archived
## outside the repo with the re-record records (checksums before and after
## plus the re-run results, see FIXTURE_GENERATION.md section 10), so the
## sidecar is frozen committed data. Each recorded entry states its frozenSource flag
## explicitly, absence is never interpreted:
## - a non-frozen missing tensor is a hard tooling error and raises
## - a frozen entry prints a line, the file stays committed
## - the regenerable entries of a frozen file are byte-verified against
##   the committed sidecar
## - a sanctioned re-record regenerates the file from the fresh recording
##
## The python twin writers (the descriptor_fields writer of
## fixture_stats.py) emit the same bytes. The harness stats-corpus keeps
## descriptor entries whose source tensors stay committed, so the byte
## agreement is verifiable in both directions through the selfcheck and
## the selftest bit-exact agreement.

import
  std/os,
  std/strutils,
  workspace/libtorch,
  workspace/safetensors,
  workspace/transformers/tests/harness

type EntrySpec = tuple[name: string, withHist: bool, mode: DescMode, frozenSource: bool]
  ## One recorded entry: (tensor name, withHistogram, descriptor mode, frozenSource).
  ## frozenSource true marks an entry whose source tensor retired from the payload
  ## when the fixtures moved from inline raw tensors to the descriptor summaries
  ## and zstd frames: the entry states that explicitly, the tool
  ## never derives frozenness from an exception.

const RecordedFixtures: seq[(string, seq[EntrySpec])] = @[
  # (fixture path relative to tests/, recorded entries). Histograms are stored only on the
  # margin-critical tensor of each family. The rest carry quantile stats only.
  ("fixtures/bf16-01-layer-internals/Qwen3.5-0.8B-layer-3/rope-Qwen3.5-0.8B-00.safetensor",
    @[("q_rot", true, dmNone, false), ("k_rot", false, dmNone, false)]),
  ("fixtures/bf16-01-layer-internals/Qwen3.5-0.8B-layer-3/attn-Qwen3.5-0.8B-00.safetensor",
    @[("output", true, dmNone, false)]),
  ("fixtures/bf16-02-first-8-layers-plus-final/Qwen3-0.6B/block-00.safetensor",
    @[("hf_layer_output", false, dmNone, false)]),
  ("fixtures/bf16-02-first-8-layers-plus-final/Qwen3-0.6B/block-01.safetensor",
    @[("hf_layer_output", false, dmNone, false)]),
  ("fixtures/bf16-02-first-8-layers-plus-final/Qwen3-0.6B/block-02.safetensor",
    @[("hf_layer_output", false, dmNone, false)]),
  ("fixtures/bf16-02-first-8-layers-plus-final/Qwen3-0.6B/block-03.safetensor",
    @[("hf_layer_output", false, dmNone, false)]),
  ("fixtures/bf16-02-first-8-layers-plus-final/Qwen3-0.6B/block-04.safetensor",
    @[("hf_layer_output", false, dmNone, false)]),
  ("fixtures/bf16-02-first-8-layers-plus-final/Qwen3-0.6B/block-05.safetensor",
    @[("hf_layer_output", false, dmNone, false)]),
  ("fixtures/bf16-02-first-8-layers-plus-final/Qwen3-0.6B/block-06.safetensor",
    @[("hf_layer_output", false, dmNone, false)]),
  ("fixtures/bf16-02-first-8-layers-plus-final/Qwen3-0.6B/block-07.safetensor",
    @[("hf_layer_output", false, dmNone, false)]),
  ("fixtures/bf16-02-first-8-layers-plus-final/Qwen3-0.6B/tail.safetensor",
    @[("pre_final_norm", true, dmNone, false)]),
  ("fixtures/bf16-03-full-forward-to-logits/Qwen3.6-35B-A3B/layer-39.safetensor",
    @[("layer_output_seq", true, dmExact, true)]),
  ("fixtures/bf16-03-full-forward-to-logits/Qwen3-0.6B/layer-27.safetensor",
    @[("layer_output", true, dmExact, true)]),
  ("fixtures/bf16-03-full-forward-to-logits/Qwen3.5-0.8B/layer-23.safetensor",
    @[("layer_output_seq", true, dmExact, true)]),
  ("fixtures/bf16-02-first-8-layers-plus-final/Qwen3.5-0.8B/block-00.safetensor",
    @[("layer_output_seq", false, dmNone, false)]),
  ("fixtures/bf16-02-first-8-layers-plus-final/Qwen3.5-0.8B/block-01.safetensor",
    @[("layer_output_seq", false, dmNone, false)]),
  ("fixtures/bf16-02-first-8-layers-plus-final/Qwen3.5-0.8B/block-02.safetensor",
    @[("layer_output_seq", false, dmNone, false)]),
  ("fixtures/bf16-02-first-8-layers-plus-final/Qwen3.5-0.8B/block-03.safetensor",
    @[("layer_output_seq", false, dmNone, false)]),
  ("fixtures/bf16-02-first-8-layers-plus-final/Qwen3.5-0.8B/block-04.safetensor",
    @[("layer_output_seq", false, dmNone, false)]),
  ("fixtures/bf16-02-first-8-layers-plus-final/Qwen3.5-0.8B/block-05.safetensor",
    @[("layer_output_seq", false, dmNone, false)]),
  ("fixtures/bf16-02-first-8-layers-plus-final/Qwen3.5-0.8B/block-06.safetensor",
    @[("layer_output_seq", false, dmNone, false)]),
  ("fixtures/bf16-02-first-8-layers-plus-final/Qwen3.5-0.8B/block-07.safetensor",
    @[("layer_output_seq", false, dmNone, false)]),
  ("fixtures/bf16-02-first-8-layers-plus-final/Qwen3.5-0.8B/tail.safetensor",
    @[("pre_final_norm", true, dmNone, false)]),
  # 35B GDN layer-0 prefill: the two f32 final states and the sublayer intermediates that the suite
  # does not replay left the payload. Their descriptor entries are frozen committed data
  # bootstrapped from the bytes of the original inline raw-tensor payload. The output pair stays full and keeps its fingerprint
  # in the stats sidecar.
  ("fixtures/bf16-01-layer-internals/Qwen3.6-35B-A3B-layer-0/gdn-Qwen3.6-35B-A3B-00.safetensor",
    @[("output_seq", true, dmNone, false),
      ("ssm_state_seq", false, dmDrift, true),
      ("ssm_state_chunked", false, dmDrift, true),
      ("z", false, dmExact, true),
      ("rmsnorm_gated_output", false, dmExact, true),
      ("core_attn_out_seq", false, dmExact, true),
      ("core_attn_out_chunked", false, dmExact, true)]),
  ("fixtures/bf16-01-layer-internals/Qwen3.6-35B-A3B-layer-0/layer-Qwen3.6-35B-A3B-00.safetensor",
    @[("gdn_block_output_seq", false, dmNone, false), ("moe_output", false, dmNone, false),
      ("layer_output_seq", true, dmNone, false)]),
  # 0.8B GDN state trajectory: the per-step f32 states and the one-shot intermediates that the suite
  # does not replay left the payload. Steps 3 to 5 are the trajectory the suite replays: after the 3-token
  # prefill and the two decode steps, the minimum step count that exercises the prefill-then-decode
  # property. Steps 0 to 2 retire with the payload slice. The bracket suffix selects one step of the recorded
  # [6, heads, Dk, Dv] sequence.
  ("fixtures/bf16-01-layer-internals/Qwen3.5-0.8B-layer-0/gdn-Qwen3.5-0.8B-01.safetensor",
    @[("one_shot_ssm_states[3]", false, dmDrift, true),
      ("one_shot_ssm_states[4]", false, dmDrift, true),
      ("one_shot_ssm_states[5]", false, dmDrift, true),
      ("one_shot_conv_output", false, dmExact, true),
      ("one_shot_core_attn_out", false, dmExact, true)]),
]

const TestsDir = currentSourcePath().parentDir() / ".."

proc recordedTensor(st: Safetensor, name: string): Tensor =
  ## One recorded tensor, with a "[k]" suffix selecting step k of a recorded per-step state
  ## sequence: the entry narrows dim 0 to that step, so the element count matches the state
  ## tensor computed at test time that the suite compares against. A missing source raises: absence is never
  ## interpreted, the caller states frozenness explicitly.
  let bracket = name.find('[')
  if bracket < 0:
    return st.getTensorOwned(name)
  doAssert name.endsWith("]"), "bad step selector: " & name
  let base = name[0 ..< bracket]
  let step = parseBiggestInt(name[bracket + 1 ..^ 2])
  let full = st.getTensorOwned(base)
  doAssert step < full.size(0),
    "step " & $step & " outside the recorded sequence of " & base
  full.narrow(0, step.int, 1)

proc main() =
  for (rel, names) in RecordedFixtures:
    let path = TestsDir / rel
    let st = Safetensor.open(path)
    var statsFile, descFile: FingerprintStatsFile
    statsFile.schema = FingerprintStatsFileSchema
    descFile.schema = FingerprintStatsFileSchema
    statsFile.source = splitFile(path).name & splitFile(path).ext
    descFile.source = statsFile.source
    for (name, withHist, mode, frozenSource) in names:
      if mode != dmNone:
        continue
      if frozenSource:
        raise newException(ValueError,
          "fingerprint entry marked frozenSource but fingerprint entries" &
          " regenerate from the payload always: " & rel & " " & name)
      # Fingerprint entries must be regenerable from the committed payload bytes: a missing source
      # is a tooling error and raises.
      let t = recordedTensor(st, name)
      var ts = tensorStats(t, withHistogram = withHist, name = name)
      ts.name = name
      statsFile.tensors.add ts
    if statsFile.tensors.len > 0:
      let outPath = path & ".stats.json.zst"
      writeFingerprintStats(outPath, statsFile)
      var total = 0
      for ts in statsFile.tensors: total += ts.n
      echo "wrote " & outPath & " (" & $total & " elements)"

    var hasFrozen = false
    for (name, withHist, mode, frozenSource) in names:
      if mode == dmNone:
        continue
      if frozenSource:
        # The source retired from the payload when the fixtures moved from inline
        # raw tensors to the descriptor summaries and zstd frames:
        # the entry states that explicitly and the whole descriptor file stays
        # committed data.
        echo "frozen descriptor entry, source retired from the payload: " &
          rel & " " & name
        hasFrozen = true
        continue
      let t = recordedTensor(st, name)
      var ts = tensorDescriptors(t, name, mode, withHistogram = withHist)
      ts.name = name
      descFile.tensors.add ts
    if descFile.tensors.len > 0:
      let outPath = path & ".descriptors.json.zst"
      if hasFrozen:
        # Frozen file: the regenerable entries must still byte-match the committed
        # sidecar. A wrong-but-plausible frozen value or a drifted twin fails here
        # instead of shipping silently.
        let committed = loadFingerprintStats(outPath)
        for ts in descFile.tensors:
          let want = committed.statsTensor(ts.name)
          if encodeTensorStatsBody(ts) != encodeTensorStatsBody(want):
            raise newException(ValueError,
              "regenerable descriptor entry " & ts.name & " of " & outPath &
              " byte-diverges from the committed sidecar")
        echo "frozen descriptor file verified entry-wise against the committed sidecar: " &
          outPath
      else:
        writeFingerprintStats(outPath, descFile)
        var total = 0
        for ts in descFile.tensors: total += ts.n
        echo "wrote " & outPath & " (" & $total & " elements)"

main()
