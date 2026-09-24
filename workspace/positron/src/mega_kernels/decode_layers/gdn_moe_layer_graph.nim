# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ─────────────────────────────────────────────────────────────────────
# ─── GDN+MoE decoder layer graph ─────────────────────────────────────

## One-launch GDN+MoE decoder layer graph on the ceramic Tile API, derived
## from a runtime-parsed model config (`GdnMoeCfg`), the parts delta-zero
## checked against the baked tables of gdn_moe_decode_megakernel.nim:
##
## - `bfWidths`/`bfOffsets` and `f32Widths`/`f32Offsets`, the scratch section shapes and the accumulated offsets
## - `stages` and `waveTotal`, one threadgroup interval per stage, contiguous over the 13 stages
## - tests/test_gdn_moe_layer_graph.nim, the delta-zero check suite
##
## Per-stage threadgroup counts, `qkvWidth = 2·hk·dk + hv·dv`, `hvDim = hv·dv`, each fixed share the compiled-in slice, ceil-div:
##
## | stage        | count             | fixed share per threadgroup     |
## | ------------ | ----------------- | ------------------------------- |
## | norm1        | 1                 | the whole hidden row            |
## | qkv gemv     | ceil(qkvWidth/64) | 64 projection columns           |
## | z gemv       | ceil(hvDim/64)    | 64 projection columns           |
## | a/b proj     | 2·ceil(hv/32)     | 32 decay/beta columns           |
## | conv         | ceil(qkvWidth/64) | 64 conv channels                |
## | qk l2norm    | ceil(hk/4)        | 4 q rows + 4 k rows             |
## | gate values  | ceil(hv/32)       | 32 heads, one per lane          |
## | gdn state    | hv·ceil(dv/8)     | one (head, 8-row) state tile    |
## | o-norm       | ceil(hv/8)        | 8 heads                         |
## | out-proj     | ceil(hidden/64)   | 64 projection columns           |
## | fold + norm2 | 1                 | the whole hidden row            |
## | moe decode   | topK + 1          | one (token, slot) pair          |
## | moe merge    | ceil(hidden/32)   | 32 output columns, one per lane |
##
## Compiled-in constraints of the baked device composition, guarded inside `deriveGdnLayerGraph`:
##
## - head dims of 128, the l2norm and GDN-state row widths
## - at most 32 value heads, one per lane in the g/beta stage
## - hidden, qkv and value widths multiples of 64, plus the MoE core's routing maxima (topK 8 over 512 experts)
##
## A no-mixer config drops the nine GDN mixer stages and their sections,
## (`hk = 0`, GLM-4.7-Flash's decoder is a different mixer family), keeping
## the norm bookends and the MoE tail.

type GdnMoeCfg* = object
  ## Layer geometry in the qwen35_moe parser's field spellings
  ## (`linear_num_*_heads`, `linear_*_head_dim`, `linear_conv_kernel_dim`, routed experts).
  hidden*: int32
    ## Layer width, the norm rows, the MoE hidden and the out_proj rows.
  numVHeads*, numKHeads*: int32
    ## GDN mixer value/key head counts. `numKHeads = 0` declares a layer
    ## with no GDN mixer, the mixer stages and sections dropping out.
  headKDim*, headVDim*: int32
    ## GDN mixer per-head key/value widths.
  convKernel*: int32
    ## Depthwise conv width, the decode ring carrying `convKernel - 1` history taps.
  topK*, numExperts*, inter*: int32
    ## Routing top-K, routed expert count and per-expert intermediate width.

func convDim*(cfg: GdnMoeCfg): int32 {.inline.} =
  ## Fused qkv projection width, the conv column order, key channels
  ## first then value channels.
  2 * cfg.numKHeads * cfg.headKDim + cfg.numVHeads * cfg.headVDim

func valueDim*(cfg: GdnMoeCfg): int32 {.inline.} =
  ## Value-channel width `hv·dv`, the z projection's rows and the mixer's value rows.
  cfg.numVHeads * cfg.headVDim

func keyDim*(cfg: GdnMoeCfg): int32 {.inline.} =
  ## Key-channel width `hk·dk`, the l2-normalized q/k rows.
  cfg.numKHeads * cfg.headKDim

type BfSectionKind* = enum
  ## Bf16 scratch sections in arena order, the baked `s*` consts' identities.
  bfQkvCol, bfZ, bfA, bfB, bfQn, bfKn, bfBeta, bfY, bfNormed, bfConv,
  bfH, bfHs, bfMoeOut, bfH1, bfNormed2, bfStream, bfNorm1, bfBlockOut

type F32SectionKind* = enum
  ## F32 scratch sections in arena order.
  f32G, f32Partial, f32Scores

type StageKind* = enum
  ## Stages 1..13 in counter-index order, one stage branch per threadgroup
  ## block in the dispatcher's role dispatch.
  stgNorm1, stgQkvGemv, stgZGemv, stgAbProj, stgConv, stgQkL2norm,
  stgGateValues, stgGdnState, stgONorm, stgOutProj, stgFoldNorm2,
  stgMoeDecode, stgMoeMerge

const stageKinds*: array[13, StageKind] = [
    stgNorm1, stgQkvGemv, stgZGemv, stgAbProj, stgConv, stgQkL2norm,
    stgGateValues, stgGdnState, stgONorm, stgOutProj, stgFoldNorm2,
    stgMoeDecode, stgMoeMerge]
  ## Stages by counter index, the dispatcher's stage order.

const stageNames*: array[13, string] = [
    "norm1+residual", "qkv-gemv", "z-gemv", "ab-proj", "conv", "qk-l2norm",
    "gate-values", "gdn-state", "o-norm", "out-proj", "fold+norm2",
    "moe-fwd", "moe-merge"]
  ## Stage labels by counter index, the bounded-wait expiry diagnostic's spellings.

const BfSectionNames*: array[BfSectionKind, string] = [
    "qkv-col", "z", "a", "b", "qn", "kn", "beta", "y", "normed", "conv",
    "h", "hs", "moe-out", "h1", "normed2", "stream", "norm1", "block-out"]
  ## Section labels, the diagnostics' spellings.

const F32SectionNames*: array[F32SectionKind, string] = ["g", "partial", "scores"]
  ## F32 section labels, the diagnostics' spellings.

type StageRow* = object
  ## One stage's derived slice of the threadgroup grid.
  present*: bool
    ## Whether the stage has work at this config, an absent stage's interval
    ## empty and its block count 0.
  blocks*: uint32
    ## Threadgroup block count.
  start*, stop*: uint32
    ## Stage grid.x interval `[start, stop)`, read by the dispatcher's role dispatch as the stage boundaries.

type LayerGraph* = object
  ## Layer graph derived at one config.
  bfWidths*: array[BfSectionKind, int32]
    ## Bf16 section shapes in elements.
  bfOffsets*: array[BfSectionKind, int32]
    ## Bf16 section start offsets, accumulating over the preceding widths.
    ## A width-0 section shares the previous offset.
  bfArenaLen*: int32
    ## Bf16 scratch extent in elements.
  f32Widths*: array[F32SectionKind, int32]
    ## F32 section shapes in elements.
  f32Offsets*: array[F32SectionKind, int32]
    ## F32 section start offsets, accumulating over the preceding widths.
  f32ArenaLen*: int32
    ## F32 scratch extent in elements.
  stages*: array[13, StageRow]
    ## One row per stage by counter index.
  waveTotal*: uint32
    ## Threadgroup grid extent, the sum of the stage block counts.

func divCeil(x, share: int32): int32 {.inline.} =
  ## Smallest stage-block count whose shares of `share` elements cover `x`.
  (x + share - 1) div share

func stageBlocks(cfg: GdnMoeCfg, s: StageKind): uint32 =
  ## Threadgroup count of one stage at `cfg`, ceil-div over each fixed
  ## share of the module doc's table.
  case s
  of stgNorm1, stgFoldNorm2: 1'u32
  of stgQkvGemv, stgConv: uint32 divCeil(cfg.convDim(), 64)
  of stgZGemv: uint32 divCeil(cfg.valueDim(), 64)
  of stgAbProj: 2 * uint32 divCeil(cfg.numVHeads, 32)
  of stgQkL2norm: uint32 divCeil(cfg.numKHeads, 4)
  of stgGateValues: uint32 divCeil(cfg.numVHeads, 32)
  of stgGdnState:
    uint32(cfg.numVHeads * divCeil(cfg.headVDim, 8))
  of stgONorm: uint32 divCeil(cfg.numVHeads, 8)
  of stgOutProj: uint32 divCeil(cfg.hidden, 64)
  of stgMoeDecode: uint32 cfg.topK + 1
  of stgMoeMerge: uint32 divCeil(cfg.hidden, 32)

func stagePresent(cfg: GdnMoeCfg, s: StageKind): bool =
  ## Whether a stage has work at `cfg`, mixer stages absent on no-mixer layers, bookends and MoE tail always present.
  if s in {stgQkvGemv, stgZGemv, stgAbProj, stgConv, stgQkL2norm,
           stgGateValues, stgGdnState, stgONorm, stgOutProj}:
    cfg.numKHeads > 0
  else:
    true

func mixerPresent*(cfg: GdnMoeCfg): bool {.inline.} =
  ## Whether the config declares a GDN mixer.
  cfg.numKHeads > 0

func bfSectionWidth(cfg: GdnMoeCfg, s: BfSectionKind): int32 =
  ## One bf16 section's shape at `cfg`:
  ##
  ## | section            | shape        | notes                    |
  ## | ------------------ | ------------ | ------------------------ |
  ## | qkv-col            | `convDim()`  | key channels then values |
  ## | z, y, normed       | `valueDim()` |                          |
  ## | a, b, beta         | `numVHeads`  |                          |
  ## | qn, kn             | `keyDim()`   |                          |
  ## | h                  | `topK·inter` | routed h scratch         |
  ## | hs                 | `inter`      | shared h scratch         |
  ## | moe-out..block-out | `hidden`     | one row each             |
  ##
  ## The mixer sections collapse to 0 on a no-mixer layer.
  case s
  of bfQkvCol, bfConv: cfg.convDim()
  of bfZ, bfY, bfNormed: cfg.valueDim()
  of bfA, bfB, bfBeta: cfg.numVHeads
  of bfQn, bfKn: cfg.keyDim()
  of bfH: cfg.topK * cfg.inter
  of bfHs: cfg.inter
  of bfMoeOut, bfH1, bfNormed2, bfStream, bfNorm1, bfBlockOut: cfg.hidden

func f32SectionWidth(cfg: GdnMoeCfg, s: F32SectionKind): int32 =
  ## One f32 section's shape at `cfg`:
  ## - f32G spans `numVHeads` log-decay rows, 0 on a no-mixer layer
  ## - f32Partial spans `(topK + 1)·hidden` MoE fp32 partials
  ## - f32Scores spans `(topK + 1)·numExperts` router score rows
  case s
  of f32G: cfg.numVHeads
  of f32Partial: (cfg.topK + 1) * cfg.hidden
  of f32Scores: (cfg.topK + 1) * cfg.numExperts

proc deriveGdnLayerGraph*(cfg: GdnMoeCfg): LayerGraph =
  ## Derives the layer graph at `cfg`.
  ##
  ## Contract:
  ##
  ## - section widths with the offsets accumulated over them
  ## - one threadgroup interval per stage, contiguous, summing to `waveTotal`
  ##
  ## Guards the compiled-in constraints of the module doc, each message
  ## naming its compiling stage, a rejected config being a structural fork.
  doAssert cfg.hidden > 0, "GdnMoeCfg: hidden must be positive"
  doAssert cfg.hidden mod 32 == 0,
    "GdnMoeCfg: hidden must be a multiple of 32, the norm walk's per-lane share " &
    "and the merge's lane tile compile it in"
  doAssert cfg.hidden mod 64 == 0,
    "GdnMoeCfg: hidden must be a multiple of 64, the out-proj GEMV's 64-column tile"
  doAssert cfg.convKernel >= 1, "GdnMoeCfg: convKernel must be at least 1"
  doAssert cfg.topK >= 1, "GdnMoeCfg: topK must be at least 1"
  doAssert cfg.topK <= 8,
    "GdnMoeCfg: topK exceeds the MoE decode core's compiled top-K maximum of 8"
  doAssert cfg.numExperts >= 1, "GdnMoeCfg: numExperts must be at least 1"
  doAssert cfg.numExperts <= 512,
    "GdnMoeCfg: numExperts exceeds the MoE core's compiled expert maximum of 512"
  doAssert cfg.inter > 0, "GdnMoeCfg: inter must be positive"
  if cfg.mixerPresent():
    doAssert cfg.numVHeads > 0 and cfg.headKDim > 0 and cfg.headVDim > 0,
      "GdnMoeCfg: a declared mixer needs positive head counts and head dims"
    doAssert cfg.headKDim == 128 and cfg.headVDim == 128,
      "GdnMoeCfg: head dims of 128 are compiled into the l2norm and GDN-state stages"
    doAssert cfg.numVHeads mod cfg.numKHeads == 0,
      "GdnMoeCfg: the value head count must be a multiple of the key head count"
    doAssert cfg.numVHeads <= 32,
      "GdnMoeCfg: numVHeads exceeds the gate-values stage's one-head-per-lane maximum of 32"
    doAssert cfg.valueDim() mod 64 == 0,
      "GdnMoeCfg: the value-channel width must be a multiple of 64, the z GEMV's tile"
    doAssert cfg.convDim() mod 64 == 0,
      "GdnMoeCfg: the qkv width must be a multiple of 64, the qkv GEMV's tile"

  for s in BfSectionKind:
    result.bfWidths[s] = bfSectionWidth(cfg, s)
  for s in F32SectionKind:
    result.f32Widths[s] = f32SectionWidth(cfg, s)

  var off = 0'i32
  for s in BfSectionKind:
    result.bfOffsets[s] = off
    off += result.bfWidths[s]
  result.bfArenaLen = off

  off = 0'i32
  for s in F32SectionKind:
    result.f32Offsets[s] = off
    off += result.f32Widths[s]
  result.f32ArenaLen = off

  var start = 0'u32
  var total = 0'u32
  for i in 0 ..< 13:
    let s = stageKinds[i]
    let present = stagePresent(cfg, s)
    let blocks = if present: stageBlocks(cfg, s) else: 0'u32
    result.stages[i] = StageRow(
      present: present, blocks: blocks, start: start, stop: start + blocks)
    start += blocks
    total += blocks
  result.waveTotal = total
