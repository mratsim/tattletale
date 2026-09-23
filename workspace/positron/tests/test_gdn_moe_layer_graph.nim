# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

##
## $ nim cpp -r -d:release --outdir:build/wip --nimcache:nimcache/wip workspace/positron/tests/test_gdn_moe_layer_graph.nim
##
## Host tests for the GDN+MoE decoder layer graph derivation
## (decode_layers/gdn_moe_layer_graph.nim), no Metal run needed:
##
## - Qwen3.6-35B-A3B delta-zero, every derived section offset, arena extent and stage table entry reproducing the baked consts
## - the GLM-4.7-Flash shape, the GLM MoE dims instantiating the graph with the nine GDN-mixer stages absent, hand-derived expected values
## - graph invariants at both configs, contiguous intervals and the offsets accumulating over the widths

import std/strutils
import workspace/positron/src/mega_kernels/decode_layers/gdn_moe_layer_graph
import workspace/positron/src/mega_kernels/decode_layers/qwen35_moe/qwen35_moe_decode_gdn_bf16

template test(label: string; body: untyped) =
  block:
    body
  echo "  [OK] ", label

func qwen36Cfg(): GdnMoeCfg =
  ## Qwen3.6-35B-A3B geometry, the megakernel's geometry consts transcribed
  ## into a runtime config record.
  GdnMoeCfg(hidden: 2048, numVHeads: 32, numKHeads: 16, headKDim: 128,
    headVDim: 128, convKernel: 4, topK: 8, numExperts: 256, inter: 512)

func glm47Cfg(): GdnMoeCfg =
  ## GLM-4.7-Flash MoE dims at the verified GLM suite shape, 2048 hidden,
  ## 64 routed experts, top-4, 1536 intermediate and 1 shared, GLM's decoder
  ## being an MLA mixer with no GDN head geometry to declare.
  GdnMoeCfg(hidden: 2048, numVHeads: 0, numKHeads: 0, headKDim: 0,
    headVDim: 0, convKernel: 4, topK: 4, numExperts: 64, inter: 1536)

# ═════════════════════════════════════════════════════════════════════════
#  Qwen3.6-35B-A3B delta-zero vs the baked consts
# ═════════════════════════════════════════════════════════════════════════

proc checkBfSectionsDeltaZero(g: LayerGraph) =
  ## Every derived bf16 section offset equals the baked `s*` const chain.
  let baked: array[BfSectionKind, int32] = [
      sQkvCol.int32, sZ.int32, sA.int32, sB.int32, sQN.int32, sKN.int32,
      sBeta.int32, sY.int32, sNormed.int32, sConv.int32, sH.int32, sHs.int32,
      sMoeOut.int32, sH1.int32, sNormed2.int32, sStream.int32, sNorm1.int32,
      sBlockOut.int32]
  for s in BfSectionKind:
    doAssert g.bfOffsets[s] == baked[s],
      "derived " & BfSectionNames[s] & " offset " & $g.bfOffsets[s] &
      " must equal the baked offset " & $baked[s]

test "qwen36 delta-zero: the derived bf16 section offsets reproduce the baked s* consts":
  let g = deriveGdnLayerGraph(qwen36Cfg())
  checkBfSectionsDeltaZero(g)

test "qwen36 delta-zero: the derived arena lengths reproduce the baked extents":
  let g = deriveGdnLayerGraph(qwen36Cfg())
  doAssert g.bfArenaLen == BfArenaLen,
    "derived bf16 extent " & $g.bfArenaLen & " must equal the baked " & $BfArenaLen
  doAssert g.f32Offsets[f32G] == sG and g.f32Offsets[f32Partial] == sPartial
  doAssert g.f32ArenaLen == F32ArenaLen,
    "derived f32 extent " & $g.f32ArenaLen & " must equal the baked " & $F32ArenaLen

test "qwen36 delta-zero: the derived stage table reproduces the baked tables":
  let g = deriveGdnLayerGraph(qwen36Cfg())
  for i in 0 ..< 13:
    doAssert g.stages[i].blocks == StageBlocks[i],
      "stage " & StageNames[i] & ": derived blocks " & $g.stages[i].blocks &
      " must equal the baked " & $StageBlocks[i]
    doAssert g.stages[i].stop == StageEnds[i],
      "stage " & StageNames[i] & ": derived boundary must equal the baked"
    doAssert g.stages[i].present
    doAssert g.stages[i].blocks > 0
    doAssert stageNames[i] == StageNames[i]
  doAssert g.stages[0].start == 0'u32
  doAssert g.waveTotal == 950'u32

test "qwen36 delta-zero: the derived section widths reproduce the baked shape expressions":
  let g = deriveGdnLayerGraph(qwen36Cfg())
  doAssert g.bfWidths[bfQkvCol] == ConvDim and g.bfWidths[bfConv] == ConvDim
  doAssert g.bfWidths[bfZ] == NumVHeads * HeadVDim
  doAssert g.bfWidths[bfH] == TopK * Inter and g.bfWidths[bfHs] == Inter
  doAssert g.bfWidths[bfQn] == NumKHeads * HeadKDim
  doAssert g.f32Widths[f32Partial] == (TopK + 1) * Hidden

# ═════════════════════════════════════════════════════════════════════════
#  GLM-4.7-Flash instantiation, the MoE dims with no GDN mixer
# ═════════════════════════════════════════════════════════════════════════

test "glm47 shape: the graph instantiates with the mixer stages absent":
  let g = deriveGdnLayerGraph(glm47Cfg())
  const wantBlocks: array[13, uint32] = [1'u32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 64]
  for i in 0 ..< 13:
    doAssert g.stages[i].blocks == wantBlocks[i]
  for i in 0 ..< 13:
    let want = i in [0, 10, 11, 12]
    doAssert g.stages[i].present == want,
      "stage " & stageNames[i] & " presence"
  doAssert g.waveTotal == 71'u32

test "glm47 shape: the derived sections carry only the bookend and MoE shapes":
  let g = deriveGdnLayerGraph(glm47Cfg())
  # Hand-derived from the shapes, the nine mixer sections collapsing to 0,
  # h = topK·inter = 4·1536, hs = 1536, six hidden rows of 2048.
  for s in [bfQkvCol, bfZ, bfA, bfB, bfQn, bfKn, bfBeta, bfY, bfNormed,
            bfConv]:
    doAssert g.bfWidths[s] == 0, BfSectionNames[s] & " must be absent"
  doAssert g.bfWidths[bfH] == 4 * 1536 and g.bfWidths[bfHs] == 1536
  for s in [bfMoeOut, bfH1, bfNormed2, bfStream, bfNorm1, bfBlockOut]:
    doAssert g.bfWidths[s] == 2048, BfSectionNames[s] & " must be a hidden row"
  let wantOffsets: array[BfSectionKind, int32] = [
      0'i32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6144'i32, 7680,
      9728, 11776, 13824, 15872, 17920]
  doAssert g.bfOffsets == wantOffsets
  doAssert g.bfArenaLen == 19968
  doAssert g.f32Widths[f32G] == 0 and g.f32Widths[f32Partial] == 5 * 2048
  doAssert g.f32ArenaLen == 10240

# ═════════════════════════════════════════════════════════════════════════
#  Graph invariants, both configs
# ═════════════════════════════════════════════════════════════════════════

proc checkGraphInvariants(g: LayerGraph) =
  ## Offsets accumulate over the widths, the stage intervals are contiguous,
  ## the blocks sum to the grid total.
  var off = 0'i32
  for s in BfSectionKind:
    doAssert g.bfOffsets[s] == off, BfSectionNames[s] & " offset"
    off += g.bfWidths[s]
  doAssert g.bfArenaLen == off
  off = 0'i32
  for s in F32SectionKind:
    doAssert g.f32Offsets[s] == off, F32SectionNames[s] & " offset"
    off += g.f32Widths[s]
  doAssert g.f32ArenaLen == off

  var sum = 0'u32
  for i in 0 ..< 13:
    let row = g.stages[i]
    doAssert row.stop == row.start + row.blocks, stageNames[i] & " interval"
    if i > 0:
      doAssert row.start == g.stages[i - 1].stop,
        stageNames[i] & " must start where " & stageNames[i - 1] & " stops"
    doAssert (row.blocks > 0'u32) == row.present, stageNames[i] & " presence"
    sum += row.blocks
  doAssert g.stages[0].start == 0'u32
  doAssert sum == g.waveTotal

test "invariants: offsets accumulate, intervals stay contiguous, blocks sum":
  checkGraphInvariants(deriveGdnLayerGraph(qwen36Cfg()))
  checkGraphInvariants(deriveGdnLayerGraph(glm47Cfg()))

test "guards: an out-of-maxima config is rejected with a naming message":
  var overTopk = qwen36Cfg()
  overTopk.topK = 10
  try:
    discard deriveGdnLayerGraph(overTopk)
    doAssert false, "topK 10 must be rejected"
  except AssertionDefect as e:
    doAssert "topK" in e.msg, e.msg

echo "test_gdn_moe_layer_graph: all checks passed"
