# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Tile passes: Tile IR ops to per-thread layout-algebra assignments.
## Nim port of the python reference implementation for the axpy subset,
## python/passes.py plus python/program.py, matched output for output.
##
## Gemm extension: rank-2 tiles, multi-dimensional views and the mmaf op.
## The abstract mmaf lowers to a per-thread FMA loop with no mma atom,
## an applied atom selection lowers through the hardware fragment program instead.
## Constructs outside the subset refuse loudly, naming the construct and the kernel:
## - unsupported ops, element types and dynamic tile dims
## - thread counts that do not divide a tile, broken token chains
## - view disagreements: a partition tile rank unlike its tensor_view's, or a loaded or stored tile unlike its view's partition tile
## - unequal-shaped or non-square mmaf operands
##
## Passes in TileLang-cutedsl order, selectMmaAtoms last:
## - resolveViews (legalization): per view SSA name, the defining view's shape, strides and tile shape
## - inferThreadValueLayouts (layout inference): the compact tile layout plus the coalesced thread-value layout per tile SSA value
## - lowerToProgram (lowering): executable per-thread program ops
## - selectMmaAtoms (atom selection): a hardware mma atom per abstract mmaf, none when no atom applies
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/tattletile \
##     --outdir:build/tattletile workspace/tattletile/tests/test_tile_passes.nim

import std/tables

import ./layout_algebra
import ./mma_atom
import ./tile_ir

type
  ViewInfo* = object
    ## Resolved global-memory view behind a load or store.
    ##
    ## Fields:
    ## - base: entry argument name holding the pointer
    ## - shapeEntries / strideEntries: the make_tensor_view operand lists, one
    ##   entry per dimension, a static int or the SSA name of the tile<i32>
    ##   scalar holding a dynamic value
    ## - tileShape: the statically shaped partition tile
    base*: string
    shapeEntries*: seq[ShapeEntry]
    strideEntries*: seq[ShapeEntry]
    tileShape*: seq[int]

  TvLayout* = object
    ## Thread-value layout: consecutive threads own consecutive tile elements.
    ##
    ## The tv call `tv(t, v) = t + v * threads` maps the pair of a thread
    ## index and a value slot onto one flat tile coordinate.
    ## - a warp of threads therefore reads a contiguous span, the coalesced
    ##   access pattern
    ## - same call form as the python reference implementation,
    ##   `Layout((T, V), (1, T))`, built and evaluated through the layout
    ##   algebra layer (`layout_algebra`)
    threads*: int
    values*: int  ## per-thread value count, size(threads*values) must divide the tile
    l*: Layout    ## the algebra layout `(threads, values):(1, threads)`

  TileValueInfo* = object
    ## Layout assignment for one SSA tile value:
    ## - the tile shape
    ## - the element type
    ## - the coalesced thread-value partition
    ##
    ## The tile layout is compact with implicit strides, the last dimension fastest.
    tileShape*: seq[int]
    elem*: ScalarElem
    tv*: TvLayout
    threads*: int

  ProgramOpKind* = enum
    ## Lowered program ops, one per Tile IR op that survives the pipeline.
    ## Same op set as python/program.py plus `pgMma` for the gemm fixture.
    pgGetBlockId
    pgLoadTile
    pgStoreTile
    pgConstTile
    pgConstScalar
    pgBinary
    pgMma
    pgYield

  BinaryKind* = enum
    bkAddf
    bkMulf

  ProgramOp* = object
    ## One lowered op. Only the payload fields of the active kind branch are
    ## set in place. View and layout info come frozen from the earlier passes.
    results*: seq[string]
    case kind*: ProgramOpKind
    of pgGetBlockId:
      discard
    of pgLoadTile:
      loadView*: string
      loadIndices*: seq[string]
      loadViewInfo*: ViewInfo
      loadInfo*: TileValueInfo
    of pgStoreTile:
      storeValue*: string
      storeView*: string
      storeIndices*: seq[string]
      storeViewInfo*: ViewInfo
      storeInfo*: TileValueInfo
    of pgConstTile:
      constValue*: ConstPayload
      constElem*: ScalarElem
      constInfo*: TileValueInfo
    of pgConstScalar:
      scalarValue*: ConstScalar
      scalarElem*: ScalarElem
    of pgBinary:
      binKind*: BinaryKind
      binLhs*: string
      binRhs*: string
      binInfo*: TileValueInfo
    of pgMma:
      mmaA*: string       ## SSA name of the A operand tile
      mmaB*: string       ## SSA name of the B operand tile
      mmaAcc*: string     ## SSA name of the accumulator operand tile
      mmaInfo*: TileValueInfo
    of pgYield:
      discard

  Program* = object
    ## Lowered per-thread executable form of one kernel entry.
    ## atomMmas: the atom selection per mmaf result name, empty if every mmaf
    ## lowers through the abstract per-thread FMA fallback.
    name*: string
    args*: seq[EntryArg]
    ops*: seq[ProgramOp]
    atomMmas*: Table[string, MmaAtomSel]

func tileSize*(shape: seq[int]): int =
  ## Number of elements of a statically shaped tile.
  result = 1
  for d in shape:
    result *= d

func `==`*(a, b: TvLayout): bool =
  ## Two tv layouts are equal when their thread-value partition matches.
  ## The algebra layout is a pure function of the two partition fields.
  ## Equality says nothing about the tile shape: the fused loop additionally
  ## requires shape agreement, checked in `loopTripCount`.
  a.threads == b.threads and a.values == b.values

func tvCall*(tv: TvLayout, t, v: int): int =
  ## Thread-value layout call: the flat tile coordinate owned by thread `t`
  ## at value slot `v`.
  ##
  ## Examples: thread 2 of the axpy tile owns coordinates {2, 6}.
  ## The algebra layout evaluation gives the same value for both coordinate forms.
  tv.l.callInt(it(t, v))

proc refuse*(kernel, construct, detail: string): ref SubsetError =
  ## Loud refusal for a Tile IR construct the passes cannot resolve.
  ## Message names the construct and the kernel, same contract as the parser.
  subsetError("kernel '" & kernel & "': construct '" & construct &
    "' is outside the supported tattletile subset: " & detail)

proc resolveViews*(entry: Entry): Table[string, ViewInfo] =
  ## Pass 1 (legalization): resolve each load/store view chain to a ViewInfo.
  ##
  ## Walks the entry in order and pairs every load_view_tko/store_view_tko
  ## view operand with its defining make_partition_view and make_tensor_view,
  ## recording shape/strides/tile shape per view SSA name.
  ## - the tensor view is looked up by its SSA name, so interleaved view
  ##   definitions resolve to their own operand lists
  ## - refusals: mismatched partition operands, a partition tile whose rank
  ##   differs from its tensor_view's rank, load/store of views with no
  ##   make_partition_view definition
  ##
  ## Raises:
  ## - SubsetError naming the construct and the kernel
  var tensorViews: Table[string, Op]
  for op in entry.ops:
    case op.kind
    of okMakeTensorView:
      tensorViews[op.results[0]] = op
    of okMakePartitionView:
      let viewName = op.operands[0]
      if viewName notin tensorViews:
        raise refuse(entry.name, "make_partition_view", "%" & op.results[0] & " reads %" & viewName & " with no make_tensor_view definition")
      let tv = tensorViews[viewName]
      let inner = op.resultTypes[0]
      let tvTy = TileIrType(kind: tkTensorView, viewShape: inner.pvShape,
                            viewElem: inner.pvElem, viewStrides: inner.pvStrides)
      if inner.kind != tkPartitionView or tv.resultTypes[0] != tvTy:
        raise refuse(entry.name, "make_partition_view", "%" & op.results[0] & " does not match its tensor_view operand %" & viewName)
      if inner.pvTile.len != tv.resultTypes[0].viewShape.len:
        raise refuse(entry.name, "make_partition_view", "%" & op.results[0] & " partitions a " & $tv.resultTypes[0].viewShape.len & "-d tensor_view with a " & $inner.pvTile.len & "-d tile")
      result[op.results[0]] = ViewInfo(
        base: tv.operands[0],
        shapeEntries: tv.dynShape,
        strideEntries: tv.dynStrides,
        tileShape: inner.pvTile)
    of okLoadViewTko, okStoreViewTko:
      let viewName = op.operands[op.operands.len - 1]
      if viewName notin result:
        raise refuse(entry.name, opSpelling(op.kind), "reads view %" & viewName & " with no make_partition_view definition")
    else:
      discard

func coalescedTv*(entry: Entry, tileShape: seq[int], threads: int): TvLayout =
  ## Coalesced thread-value layout of a tile: tv(t, v) = t + v * threads.
  ##
  ## Raises:
  ## - SubsetError when `threads` does not divide the tile size
  let size = tileSize(tileShape)
  if size mod threads != 0:
    raise refuse(entry.name, "thread-value partition", $threads & " threads do not divide a tile of " & $size & " elements")
  let values = size div threads
  TvLayout(threads: threads, values: values,
           l: makeLayout(@[threads, values], @[1, threads]))

proc inferThreadValueLayouts*(entry: Entry, threads: int): Table[string, TileValueInfo] =
  ## Pass 2 (layout inference): assign a thread-value partition to each tile
  ## SSA value. Every tile value gets TileValueInfo: a compact tile layout
  ## (strides implicit in the shape, the last dimension fastest) plus the coalesced tv layout.
  ## - the partition covers the tile exactly once: the tv call is a bijection
  ##   onto the tile coordinates, checked by the partition-identity test
  ##
  ## Before (kernel axpy, tile<8xf32> loaded tile, threads=4):
  ##   %tile_a : tile<8xf32>  (no ownership assignment)
  ## After:
  ## - assignment: TileValueInfo(tileShape: @[8], tv: (threads 4, values 2))
  ## - thread 2 owns tile coordinates {2, 6}
  ##
  ## Raises:
  ## - SubsetError for dynamically shaped tiles, unsupported element types
  ##   (the subset supports f32 and i32) and thread counts that do not
  ##   divide the tile
  for op in entry.ops:
    var tileTy: TileIrType
    case op.kind
    of okLoadViewTko:
      tileTy = op.resultTypes[0]
    of okStoreViewTko:
      tileTy = op.operandTypes[0]
    of okConstant:
      if op.resultTypes[0].kind == tkTile and op.resultTypes[0].tileShape.len > 0:
        tileTy = op.resultTypes[0]
    of okAddf, okMulf, okMmaf:
      tileTy = op.resultTypes[0]
    else:
      discard
    if tileTy.kind != tkTile:
      continue
    for d in tileTy.tileShape:
      if isDynamic(d):
        raise refuse(entry.name, opSpelling(op.kind), "has a dynamically shaped tile, tile shapes are static")
    if tileTy.tileElem != TypeElem(kind: teScalar, scalar: eF32) and
        tileTy.tileElem != TypeElem(kind: teScalar, scalar: eI32):
      raise refuse(entry.name, opSpelling(op.kind), "unsupported tile element type")
    let valueName = if op.kind == okStoreViewTko:
      op.operands[0]
    else:
      op.results[0]
    result[valueName] = TileValueInfo(
      tileShape: tileTy.tileShape,
      elem: tileTy.tileElem.scalar,
      tv: coalescedTv(entry, tileTy.tileShape, threads),
      threads: threads)

proc checkToken(entry: Entry, op: Op, lastToken: string) =
  ## Refusal for a broken token chain: a tko op must consume the previous
  ## tko op's token result.
  if op.token != lastToken:
    raise refuse(entry.name, opSpelling(op.kind), "token chain broken at %" & op.token & ", previous token is " & (if lastToken == "": "none" else: "%" & lastToken))

proc checkViewIndices(entry: Entry, op: Op, view: ViewInfo) =
  ## One view index operand per view dimension.
  ## The gemm fixture indexes each 2-d view with the block x and y coordinates.
  ##
  ## Raises:
  ## - SubsetError naming the op and the kernel
  if op.indices.len != view.tileShape.len:
    raise refuse(entry.name, opSpelling(op.kind), "on %" & (if op.kind == okLoadViewTko: op.operands[0] else: op.operands[1]) & " carries " & $op.indices.len & " view indices for a " & $view.tileShape.len & "-d view")

proc checkTileMatchesView(entry: Entry, op: Op, view: ViewInfo, tileShape: seq[int]) =
  ## A loaded or stored tile must match its view's partition tile:
  ## per-thread addresses decompose the value's tile coordinates against
  ## the view's per-dimension strides, so a shape mismatch mis-addresses
  ## every element after the first.
  ##
  ## Raises:
  ## - SubsetError naming the op and the kernel
  if tileShape != view.tileShape:
    let viewName = if op.kind == okLoadViewTko: op.operands[0] else: op.operands[1]
    raise refuse(entry.name, opSpelling(op.kind), "on %" & viewName & " carries a tile of shape " & $tileShape & " for a " & $view.tileShape & " partition tile")

proc checkMmaf(entry: Entry, op: Op, layouts: Table[string, TileValueInfo]) =
  ## Subset form of `mmaf`: every operand and the result is one square rank-2
  ## tile of equal shape, `tile<RxR>`, the gemm fixture's form.
  ## The result shape fixes the k loop bound, so unequal operand shapes refuse,
  ## and a rectangular tile has no coherent k convention for the B operand.
  ##
  ## Raises:
  ## - SubsetError naming `mmaf` and the kernel
  let shape = op.resultTypes[0].tileShape
  if shape.len != 2:
    raise refuse(entry.name, "mmaf", "lowers rank-2 tile<RowsxCols> values only, got a tile of shape " & $shape)
  if shape[0] != shape[1]:
    raise refuse(entry.name, "mmaf", "lowers square tile<RxR> values only, got tile<" & $shape[0] & "x" & $shape[1] & ">")
  for i, operand in op.operands:
    if operand notin layouts:
      raise refuse(entry.name, "mmaf", "operand %" & operand & " is not a tile value with a layout assignment")
    if layouts[operand].tileShape != shape:
      raise refuse(entry.name, "mmaf", "operand %" & operand & " has shape " & $layouts[operand].tileShape & ", the gemm subset requires tile<" & $shape[0] & "x" & $shape[1] & ">")

proc lowerToProgram*(entry: Entry, views: Table[string, ViewInfo], layouts: Table[string, TileValueInfo]): Program =
  ## Pass 3 (lowering): Tile IR ops to executable ProgramOps.
  ##
  ## Consumes the frozen layout assignments op by op, in program order.
  ## The TileLang contract: annotate every tile value first, then lower.
  ## - token chaining is validated here: every tko op consumes the previous
  ##   tko op's token result
  ##
  ## Before (kernel axpy fragment, after passes 1-2):
  ##   %scaled = mulf %alpha, %tile_a : tile<8xf32>
  ##   %sum    = addf %scaled, %tile_b : tile<8xf32>
  ## After (program ops, executed per block, per thread, per owned coordinate):
  ## - mulf -> ProgramOp(kind: pgBinary, binKind: bkMulf, binLhs: "alpha", binRhs: "tile_a")
  ## - addf -> ProgramOp(kind: pgBinary, binKind: bkAddf, binLhs: "scaled", binRhs: "tile_b")
  ##
  ## Raises:
  ## - SubsetError for a broken token chain, a tile shape that does not match
  ##   its view's partition tile, or an op outside the subset
  var lastToken = ""
  for op in entry.ops:
    case op.kind
    of okMakeToken:
      lastToken = op.results[0]
    of okGetTileBlockId:
      result.ops.add ProgramOp(kind: pgGetBlockId, results: op.results)
    of okLoadViewTko:
      entry.checkToken(op, lastToken)
      lastToken = op.results[1]
      entry.checkViewIndices(op, views[op.operands[0]])
      entry.checkTileMatchesView(op, views[op.operands[0]], layouts[op.results[0]].tileShape)
      result.ops.add ProgramOp(
        kind: pgLoadTile, results: op.results,
        loadView: op.operands[0], loadIndices: op.indices,
        loadViewInfo: views[op.operands[0]],
        loadInfo: layouts[op.results[0]])
    of okStoreViewTko:
      entry.checkToken(op, lastToken)
      entry.checkViewIndices(op, views[op.operands[1]])
      entry.checkTileMatchesView(op, views[op.operands[1]], layouts[op.operands[0]].tileShape)
      result.ops.add ProgramOp(
        kind: pgStoreTile, results: op.results,
        storeValue: op.operands[0], storeView: op.operands[1],
        storeIndices: op.indices,
        storeViewInfo: views[op.operands[1]],
        storeInfo: layouts[op.operands[0]])
    of okConstant:
      if op.resultTypes[0].kind == tkTile and op.resultTypes[0].tileShape.len == 0:
        result.ops.add ProgramOp(kind: pgConstScalar, results: op.results,
                                 scalarValue: op.constValue.scalar,
                                 scalarElem: op.constElem)
      else:
        result.ops.add ProgramOp(kind: pgConstTile, results: op.results,
                                 constValue: op.constValue, constElem: op.constElem,
                                 constInfo: layouts[op.results[0]])
    of okAddf:
      result.ops.add ProgramOp(kind: pgBinary, results: op.results,
                               binKind: bkAddf, binLhs: op.operands[0],
                               binRhs: op.operands[1], binInfo: layouts[op.results[0]])
    of okMulf:
      result.ops.add ProgramOp(kind: pgBinary, results: op.results,
                               binKind: bkMulf, binLhs: op.operands[0],
                               binRhs: op.operands[1], binInfo: layouts[op.results[0]])
    of okMmaf:
      entry.checkMmaf(op, layouts)
      result.ops.add ProgramOp(
        kind: pgMma, results: op.results,
        mmaA: op.operands[0], mmaB: op.operands[1], mmaAcc: op.operands[2],
        mmaInfo: layouts[op.results[0]])
    of okMakeTensorView, okMakePartitionView:
      # folded into ViewInfo by resolveViews, nothing to lower
      discard
    of okAssume, okReshape, okBroadcast, okFor, okContinue, okReturn:
      raise refuse(entry.name, opSpelling(op.kind), "is parsed but not lowered, the tile-passes subset ends at yield")
    of okYield:
      result.ops.add ProgramOp(kind: pgYield)
  result.name = entry.name
  result.args = entry.args

proc selectMmaAtoms*(entry: Entry, layouts: Table[string, TileValueInfo], threads: int): Table[string, MmaAtomSel] =
  ## Pass 4 (mma atom selection): pick a hardware atom per abstract mmaf, then derive the operand
  ## thread-value layouts through the layout algebra.
  ##
  ## For each mmaf op the pass asks the atom layer whether a registered atom applies:
  ## f32 values, the atom thread group size, and tile dims that are positive multiples of the atom cell.
  ## Matches record:
  ## - the full (T, V, reps) -> tile layouts of the A, B and accumulator operands (`mma_atom.operandTvLayout`)
  ## - the bit-linear per-thread fragment address forms, verified against all 32 lanes
  ## - per-operand fragment maps from cell offsets to view coordinates
  ##
  ## No match records no selection, the abstract per-thread FMA lowering of lowerToProgram stays in charge.
  ## That is the fallback for shapes the atom layer does not cover.
  ##
  ## Before (kernel gemm fragment, after passes 1-3, threads = 32):
  ##   %acc = mmaf %tile_a, %tile_b, %zero : tile<16x16xf32> x3
  ## After:
  ## - selection for "acc": the Apple 8x8x8 f32 simdgroup atom, reps (2, 2, 2)
  ## - accumulator tv and A tv: thread 0 owns tile flats {0, 8, 16, 24, 128, 136, 144, 152}, B tv flats {0, 1, 8, 9, 128, 129, 136, 137}
  ## - one 8x8 cell per repetition, column-major tile flats: C flat = m + 16*n, A flat = m + 16*k, B flat = n + 16*k
  ## - no selection at threads = 4, the abstract FMA lowering stays
  ##
  ## Raises:
  ## - LayoutError on a fragment layout not bit-linear in the thread id
  for op in entry.ops:
    if op.kind != okMmaf:
      continue
    let shape = layouts[op.results[0]].tileShape
    let sel = atomFor(layouts[op.results[0]].elem, shape[0], shape[1],
                      shape[1], threads)
    if sel.selected:
      result[op.results[0]] = sel

proc runTilePipeline*(entry: Entry, threads: int): Program =
  ## Whole tile-pass pipeline: resolve views, infer layouts, lower, select mma atoms.
  ##
  ## Raises:
  ## - SubsetError naming the construct and the kernel, per pass
  let views = entry.resolveViews()
  let layouts = entry.inferThreadValueLayouts(threads)
  result = entry.lowerToProgram(views, layouts)
  result.atomMmas = entry.selectMmaAtoms(layouts, threads)
