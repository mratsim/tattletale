# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Orchestrator suite (Metal): sequence lifecycle over the pool and trie.
##
## Test that the Orchestrator API drives the full sequence lifecycle over
## the pool and trie:
##
## - a fresh startSequence, an LPM-matched restart, and a COW partial
##   page verified device-side against the cached page slab
## - page-boundary borrowing in appendToken
## - an endSequence graft with a re-LPM round trip
##
## Run from the worktree root:
##   nim c -r -d:release --nimcache:nimcache/wipM08orc workspace/tattleformers/tests/stateful/test_orchestrator.nim

import workspace/crucible
import workspace/ceramic/src/tensors
import workspace/ceramic/src/layout_algebra
import workspace/tensorcery/src/device_view
import workspace/tensorcery/src/tensor_datatypes
from workspace/tattleformers/src/stateful/kvcache import TokensPerPage
import workspace/tattleformers/src/stateful/page_pool
import workspace/tattleformers/src/stateful/inference_context
import workspace/tattleformers/src/stateful/orchestrator
import workspace/tattleformers/src/ops_Ni1o/pool_write

{.experimental: "callOperator".}

proc k_primer*(z, x: distinct DeviceView, shift: static int) =
  ## z = x + shift, one element. Compile-time primer for the erased face.
  let
    zv = make_view(z.data +% z.offset, z.layout)
    xv = make_view(x.data +% x.offset, x.layout)
  let s = xv[0].to(float32) + float32(shift)
  zv[0] = s.to(bfloat16)

const
  NumLayers = 2
  NumPages = 8
  KvHeads = 2
  HeadDim = 4
  PrefillLen = 10
  DecodeSteps = 5

proc makeTokens(n: int, base: uint32 = 1): seq[uint32] =
  ## Deterministic distinct token ids base .. base+n-1 (0 is the trie's "no token").
  result = newSeq[uint32](n)
  for i in 0 ..< n:
    result[i] = base + uint32(i)

proc fillSequencePages[E: HwEngine](engine: E, orc: var Orchestrator, seqLen: int, seed: int): seq[bfloat16] =
  ## Simulates the attention layer's pool writes for the active sequence:
  ## per layer, one pool-write node over the context's page table, one eval.
  ## Returns the host copy of the K slab after the writes.
  let pool = orc.getInferenceContextMut().pool
  var kHost = newSeq[bfloat16](seqLen * KvHeads * HeadDim)
  var vHost = newSeq[bfloat16](seqLen * KvHeads * HeadDim)
  for i in 0 ..< kHost.len:
    kHost[i] = (float32((i * 7 + seed) mod 97) * 0.25'f32 - 4.0'f32).to(bfloat16)
    vHost[i] = (float32((i * 11 + seed) mod 89) * 0.25'f32 - 4.0'f32).to(bfloat16)
  let kSrc = newTensor(engine, @[1, seqLen, KvHeads, HeadDim], kBf16)
  let vSrc = newTensor(engine, @[1, seqLen, KvHeads, HeadDim], kBf16)
  kSrc.copyToDeviceFromHost(kHost)
  vSrc.copyToDeviceFromHost(vHost)
  var nodes: seq[TensorNode]
  for layer in 0 ..< NumLayers:
    nodes.add(tn_pool_write(engine, pool.kBuffer().node, pool.vBuffer().node,
      kSrc.node, vSrc.node, layer, 0, seqLen, 0,
      orc.getInferenceContextMut().pageSlots()))
  engine.eval(nodes.toSpan())
  result = newSeq[bfloat16](NumPages * NumLayers * TokensPerPage * KvHeads * HeadDim)
  copyToHostFromDevice(result, pool.kBuffer())

proc slabIdx(page, layer, pos, h, d: int): int =
  (((page * NumLayers + layer) * TokensPerPage + pos) * KvHeads + h) * HeadDim + d

proc runMain[E: HwEngine](engine: E) =
  block: # primer enqueue: textually precede the first erased-Tensor call
    let
      zDev = engine.allocDevice(bfloat16, 1)
      xDev = engine.allocDevice(bfloat16, 1)
      zb = TensorBridge[typeof(zDev), (int,), (int,)](
        data: zDev, offset: 0, layout: make_layout((int(1),), (int(1),)))
      xb = TensorBridge[typeof(xDev), (int,), (int,)](
        data: xDev, offset: 0, layout: make_layout((int(1),), (int(1),)))
    var batch = engine.recordBatch()
    batch.enqueue<<(1, 1)>> k_primer(zb, xb, 3)
    batch.submit()

  var orc = Orchestrator.init(engine, NumLayers, batch_size = 1,
    KvHeads, max_seq = 1024, HeadDim, NumPages)

  # ── fresh startSequence: LPM miss, one page borrowed, positions arange ──
  block:
    let tokens = makeTokens(PrefillLen)
    orc.startSequence(tokens)
    let ctx = orc.getInferenceContextMut()
    doAssert ctx.cached_tokens == 0, "no cached tokens on a fresh trie"
    doAssert ctx.pages.len == 1, "one page for " & $PrefillLen & " tokens"
    doAssert ctx.kv_position == 0, "kv_position is 0 after startSequence"
    doAssert ctx.position_ids == @[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "prefill arange"
    doAssert ctx.input_tokens == tokens, "input tokens tracked"

  # ── simulated attention writes, then decode steps with page-boundary lazy borrow ──
  block:
    discard fillSequencePages(engine, orc, PrefillLen, seed = 3)
    # kv_position reflects the prefill write total after the forward pass
    orc.setKvPosition(PrefillLen)
    for step in 0 ..< DecodeSteps:
      let pos = PrefillLen + step
      # Appended token: the sampled token continues the sequence.
      orc.appendToken(pos, uint32(1 + pos))
      # attention reads kv_position as the write offset (== position_ids[0])
      let ctx = orc.getInferenceContextMut()
      doAssert ctx.position_ids == @[pos], "decode position " & $pos
      doAssert ctx.kv_position == pos,
        "kv_position is the write offset during the step, got " & $ctx.kv_position
      orc.setKvPosition(pos + 1)
    let ctx = orc.getInferenceContextMut()
    doAssert ctx.input_tokens.len == PrefillLen + DecodeSteps, "decode tokens tracked"
    doAssert ctx.pages.len == 1,
      $(PrefillLen + DecodeSteps) & " tokens stay inside one " &
      $TokensPerPage & "-token page"

  # ── endSequence: graft into the trie, context resets ──
  block:
    orc.endSequence()
    let ctx = orc.getInferenceContextMut()
    doAssert ctx.pages.len == 0, "pages dropped after endSequence"
    doAssert ctx.input_tokens.len == 0, "input_tokens dropped"
    doAssert ctx.kv_position == 0, "kv_position reset"

    let rematched = orc.lpm(makeTokens(PrefillLen + DecodeSteps))
    doAssert rematched.totalTokenMatched == PrefillLen + DecodeSteps,
      "the grafted sequence LPM-matches fully, got " & $rematched.totalTokenMatched

  # ── restart on the same prefix: LPM match + COW partial page ──
  block:
    # Prompt: the grafted 15-token prefix. appendToken adds the 3 new
    # tokens after the prefill forward.
    let tokens2 = makeTokens(PrefillLen + DecodeSteps + 3)
    orc.startSequence(makeTokens(PrefillLen + DecodeSteps))
    let ctx = orc.getInferenceContextMut()
    doAssert ctx.cached_tokens == PrefillLen + DecodeSteps,
      "cached_tokens from LPM, got " & $ctx.cached_tokens
    # partialTokens = 15 mod 256 = 15 > 0 and the match ends mid-page → COW
    doAssert ctx.pages.len == 1, "the COW page replaces the partial cached page"
    # COW borrows a fresh page: a different slot than the trie's cached page
    let matched = orc.lpm(makeTokens(PrefillLen + DecodeSteps))
    doAssert int(ctx.pages[0].pageIndex()) != int(matched.pages[0].pageIndex()),
      "COW borrows a private page"

    # Content check: the COW page copies the cached page's first 15 positions.
    # Reference: the host copy of the K slab taken right after the simulated
    # attention writes of the first sequence, downloaded again now.
    let pool = orc.getInferenceContextMut().pool
    var kNow = newSeq[bfloat16](NumPages * NumLayers * TokensPerPage * KvHeads * HeadDim)
    copyToHostFromDevice(kNow, pool.kBuffer())
    let cachedSlot = int(matched.pages[0].pageIndex())
    let cowSlot = int(ctx.pages[0].pageIndex())
    for layer in 0 ..< NumLayers:
      for pos in 0 ..< PrefillLen + DecodeSteps:
        for h in 0 ..< KvHeads:
          for d in 0 ..< HeadDim:
            let ia = slabIdx(cachedSlot, layer, pos, h, d)
            let ic = slabIdx(cowSlot, layer, pos, h, d)
            doAssert kNow[ic].to(float32) == kNow[ia].to(float32),
              "COW copy mismatch (layer " & $layer & ", pos " & $pos & ")"

    # appendToken continues on the COW page, then endSequence re-grafts
    orc.setKvPosition(PrefillLen + DecodeSteps)
    for step in 0 ..< 3:
      let pos = PrefillLen + DecodeSteps + step
      orc.appendToken(pos, tokens2[pos])
      orc.setKvPosition(pos + 1)
    doAssert orc.getInferenceContextMut().input_tokens == tokens2, "full token chain tracked"
    orc.endSequence()

    # round trip: the extended sequence now LPM-matches through the COW page
    let final = orc.lpm(tokens2)
    doAssert final.totalTokenMatched == tokens2.len,
      "re-grafted sequence matches, got " & $final.totalTokenMatched
    orc.endSequence()

  stderr.writeLine "orchestrator (pool + trie + COW): all checks passed"

when isMainModule:
  runMain(gMetalEngine())
