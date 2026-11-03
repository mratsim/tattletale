# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Bridge pass: resolved Tile IR program to crucible's CIRv2, then through crucible's shipped pass pipeline to Metal or SPIR-V output.
##
## Lowering contract:
## - register tiles become per-thread scalar statements, fused into one loop over the thread-value slots
## - views become pointer-index arithmetic through the frozen view layout, the thread id enters through the canonical MSL coordinate builtins
## - the abstract `mmaf` lowers to a per-thread FMA loop over the accumulator coordinates that each thread owns
## - an atom-selected mmaf lowers to the per-thread fragment program with one simdgroup mma per cell triple
##
## Bridge mechanics:
## - the CIRv2 module is built directly through crucible's `cirv2_build` constructors, one kernel proc, no Nim kernel body
## - this module runs crucible's pass pipeline itself, crucible is untouched
## - the `weak` memory-ordering qualifier drops, the subset has no async copies and program order is the only order
##
## Run:
##   nim c -r --hints:off --warnings:off --nimcache:nimcache/tattletile \
##     --outdir:build/tattletile workspace/tattletile/tests/test_axpy_msl.nim

import std/tables

import workspace/crucible/src/codegen/backends/backends_api
import workspace/crucible/src/codegen/ir/cirv2_build
import workspace/crucible/src/codegen/passes/pass_registry
import workspace/crucible/src/codegen/passes/passes_normalizations
import workspace/crucible/src/codegen/passes/passes_lowering
import workspace/crucible/src/codegen/passes/passes_lowering_msl
import workspace/crucible/src/codegen/passes/passes_lowering_cfamily
import workspace/crucible/src/codegen/passes/passes_low_reserved
import workspace/crucible/src/codegen/passes/passes_low_pointers
import workspace/crucible/src/codegen/passes/passes_low_array_assign
import workspace/crucible/src/codegen/passes/passes_low_address_spaces
import workspace/crucible/src/codegen/passes/passes_validations
import workspace/crucible/src/crucible_codegen

import ./tile_ir
import ./tile_passes
import ./mma_atom

type
  LoadView* = object
    ## Defining view of one loaded tile SSA value, for mma operand
    ## resolution:
    ## - viewName: the view SSA name, keys the block offset local
    ## - base: the entry argument holding the pointer
    ## - viewInfo: the resolved shape, strides and tile shape
    viewName*: string
    base*: string
    viewInfo*: ViewInfo

const Ivec3Name = "tt_ivec3"
  ## CIRv2 struct type carrying the coordinate builtins' x/y/z components,
  ## registered in `module.types` once per lowered kernel.

proc ivec3Type(): CirType =
  tStruct(Ivec3Name,
    fld("x", tInt(32)), fld("y", tInt(32)), fld("z", tInt(32)))

proc cirElemType(e: ScalarElem): CirType =
  case e
  of eI32: tInt(32)
  of eF32: tFloat(32)

proc constToExpr(v: ConstScalar, e: ScalarElem): CirExpr =
  if v.isInt:
    intLit(v.intVal, cirElemType(e))
  else:
    fltLit(v.floatVal, cirElemType(e))

proc binSpelling(k: BinaryKind): string =
  case k
  of bkAddf: "addf"
  of bkMulf: "mulf"

proc refuseBridge(kernel, construct, detail: string): ref SubsetError =
  subsetError("kernel '" & kernel & "': construct '" & construct &
    "' is outside the supported tattletile subset: " & detail)

proc coordBuiltinExpr(k: CirCoordBuiltinKind): CirExpr =
  ## Coordinate builtin symbol, downstream behavior:
  ##
  ## - the MSL printer appends the attributed `uint3` kernel parameter, the SPIR-V encoder declares the matching input variable
  ## - component access goes through `dotE`, the printer folds the int cast from the assigned local's declared type
  let e = sym(CoordBuiltinNames[k], ivec3Type(), builtin = true)
  e.coordBuiltin = k
  e

proc builtinComponent(k: CirCoordBuiltinKind, component: string): CirExpr =
  dotE(coordBuiltinExpr(k), component)

proc whileLoop(counter: string, bound: int, body: seq[CirStmt]): seq[CirStmt] =
  let c = sym(counter, tInt(32))
  result = @[varD(counter, tInt(32)), asgnS(c, intLit(0, tInt(32)))]
  result.add whileS(ltE(c, intLit(bound, tInt(32))),
    body & asgnS(c, addE(tInt(32), c, intLit(1, tInt(32)))))

proc storeTargetOrder(prog: Program): seq[string] =
  ## Entry argument names in launch order: the store target's pointer and shape
  ## argument names first, the remaining entry arguments in entry order.
  ## The launch binds the output buffer to slot 0, hence the store target first.
  for op in prog.ops:
    if op.kind == pgStoreTile:
      result.add op.storeViewInfo.base
      for e in op.storeViewInfo.shapeEntries:
        if e.isSsa:
          result.add e.name

proc entryArgCirType(prog: Program, arg: EntryArg): CirType =
  ## CIRv2 type of one entry argument's kernel parameter:
  ## - `tile<ptr<E>>` becomes a device array pointer to E
  ## - 0-d scalar tiles become int32/f32 value types
  ##
  ## Raises:
  ## - SubsetError naming the kernel, for a shaped tile argument or any
  ##   other argument type
  case arg.typ.kind
  of tkTile:
    if arg.typ.tileShape.len != 0:
      raise refuseBridge(prog.name, "kernel argument %" & arg.name, "shaped tile arguments are outside the axpy tracer, scalars only")
    case arg.typ.tileElem.kind
    of teScalar:
      result = cirElemType(arg.typ.tileElem.scalar)
    of tePointer:
      result = tAptr(caDevice, cirElemType(arg.typ.tileElem.pointee))
  else:
    raise refuseBridge(prog.name, "kernel argument %" & arg.name, "unsupported kernel argument type " & $arg.typ.kind)

proc lowerParams(prog: Program, env: Table[string, CirExpr]): seq[CirParam] =
  ## Entry arguments to kernel parameters in launch order, per `storeTargetOrder`.
  ## Store targets bound to 0-d constants resolve through the kernel bindings
  ## and take no parameter.
  ##
  ## Raises:
  ## - SubsetError via entryArgCirType for any other argument type
  ## - SubsetError for a store target name bound nowhere in the kernel
  let first = prog.storeTargetOrder()
  for name in first:
    if name notin env:
      raise refuseBridge(prog.name, "store_view_tko", "the store target %" & name & " is not bound in the kernel")
    for arg in prog.args:
      if arg.name == name:
        result.add param(arg.name, prog.entryArgCirType(arg))
  for arg in prog.args:
    if arg.name notin first:
      result.add param(arg.name, prog.entryArgCirType(arg))

proc blockEnv(prog: Program): Table[string, CirExpr] =
  ## Block-level name bindings, resolved before the per-thread loop:
  ## - entry arguments to their parameter symbols
  ## - get_tile_block_id results to the block-coordinate builtin components
  ## - 0-d tile constants to their literal (inlined, the MSL compiler folds)
  ##
  ## Raises:
  ## - SubsetError for a get_tile_block_id result count other than 3
  for arg in prog.args:
    result[arg.name] = sym(arg.name, prog.entryArgCirType(arg))
  for op in prog.ops:
    case op.kind
    of pgGetBlockId:
      if op.results.len != 3:
        raise refuseBridge(prog.name, "get_tile_block_id", "expects 3 block coordinates, got " & $op.results.len)
      const Components = ["x", "y", "z"]
      for i, name in op.results:
        result[name] = builtinComponent(cbkThreadgroupPositionInGrid, Components[i])
    of pgConstScalar:
      result[op.results[0]] = constToExpr(op.scalarValue, op.scalarElem)
    else:
      discard

proc entryToExpr(prog: Program, e: ShapeEntry, env: Table[string, CirExpr], what: string): CirExpr =
  ## Value expression of one view shape or stride entry: a static int
  ## becomes a literal, an SSA name resolves to its kernel binding.
  ##
  ## Raises:
  ## - SubsetError when the SSA name is not bound in the kernel
  if e.isSsa:
    if e.name notin env:
      raise refuseBridge(prog.name, what, "%" & e.name & " is not bound in the kernel")
    env[e.name]
  else:
    intLit(e.value, tInt(32))

proc opViewInfo(op: ProgramOp): ViewInfo =
  case op.kind
  of pgLoadTile: op.loadViewInfo
  of pgStoreTile: op.storeViewInfo
  else: raise subsetError("opViewInfo: not a view op")

proc opViewName(op: ProgramOp): string =
  case op.kind
  of pgLoadTile: op.loadView
  of pgStoreTile: op.storeView
  else: raise subsetError("opViewName: not a view op")

proc opViewIndices(op: ProgramOp): seq[string] =
  case op.kind
  of pgLoadTile: op.loadIndices
  of pgStoreTile: op.storeIndices
  else: raise subsetError("opViewIndices: not a view op")

proc lowerBlockOffsets(prog: Program, env: Table[string, CirExpr], statements: var seq[CirStmt]) =
  ## Block offset of every loaded or stored view, one offset local per view
  ## SSA name, emitted before the value loop. One term per view dimension:
  ## block index x tile size x view stride.
  ##
  ## - several ops on one view with the same block indices share the local,
  ##   an in-place load + store
  ## - one view addressed with two different block index lists refuses:
  ##   the shared local would bind the first list's offset
  ##
  ## Before, gemm (view pv_a, tile (16, 16), strides (16, 1)):
  ## - block indices: bid_x, bid_y
  ##   LoadTile(view: "pv_a", indices: @["bid_x", "bid_y"])
  ## After:
  ##   int tt_block_off_pv_a = ((bid_x * 16) * 16) + ((bid_y * 16) * 1)
  ##
  ## Before, axpy (view pv_a, tile (8), stride st):
  ## - block index: bid_x
  ##   LoadTile(view: "pv_a", indices: @["bid_x"])
  ## After:
  ##   int tt_block_off_pv_a = (bid_x * 8) * st
  ##
  ## Raises:
  ## - SubsetError for an unbound view index or stride name
  var addressed: Table[string, seq[string]]
  for op in prog.ops:
    case op.kind
    of pgLoadTile, pgStoreTile:
      let name = op.opViewName()
      let indices = op.opViewIndices()
      if name in addressed:
        if addressed[name] != indices:
          raise refuseBridge(prog.name, $op.kind & " on %" & name, "view %" & name & " is addressed with two different block index lists")
        continue
      addressed[name] = indices
      let view = op.opViewInfo()
      var off: CirExpr
      for d in 0 ..< view.tileShape.len:
        if indices[d] notin env:
          raise refuseBridge(prog.name, $op.kind & " on %" & name, "view index %" & indices[d] & " is not bound in the kernel")
        let stride = prog.entryToExpr(view.strideEntries[d], env,
          "stride of view %" & name)
        let blockOff = mulE(tInt(32), env[indices[d]], intLit(view.tileShape[d], tInt(32)))
        let term = mulE(tInt(32), blockOff, stride)
        off = if d == 0: term
              else: addE(tInt(32), off, term)
      statements.add varD("tt_block_off_" & name, tInt(32), off)
    else:
      discard

proc loopTripCount(prog: Program): TileValueInfo =
  ## Thread-value partition and tile shape shared by every tile value.
  ## All tiles fuse into one loop over one coordinate decomposition.
  ## The tile values must agree on shape and partition.
  ##
  ## Raises:
  ## - SubsetError when the kernel has no tile value or the tile values
  ##   disagree on the tile shape or the partition
  for op in prog.ops:
    let info = case op.kind
      of pgLoadTile: op.loadInfo
      of pgStoreTile: op.storeInfo
      of pgConstTile: op.constInfo
      of pgBinary: op.binInfo
      of pgMma: op.mmaInfo
      else: TileValueInfo(tileShape: @[], elem: eF32, tv: TvLayout(), threads: 0)
    if info.tileShape.len == 0:
      continue
    if result.tileShape.len == 0:
      result = info
    elif result.tv != info.tv or result.tileShape != info.tileShape:
      raise refuseBridge(prog.name, "thread-value partitions", "tile values disagree on the tile shape or the thread-value partition")
  if result.tileShape.len == 0:
    raise refuseBridge(prog.name, "kernel body", "no tile value to partition")

proc loadViews(prog: Program): Table[string, LoadView] =
  for op in prog.ops:
    if op.kind == pgLoadTile:
      result[op.results[0]] = LoadView(
        viewName: op.loadView, base: op.loadViewInfo.base,
        viewInfo: op.loadViewInfo)

proc cirTileOffset(prog: Program, name: string, view: ViewInfo, dimLocals: seq[string], env: Table[string, CirExpr]): CirExpr =
  result = sym("tt_block_off_" & name, tInt(32))
  for d in 0 ..< view.strideEntries.len:
    let stride = prog.entryToExpr(view.strideEntries[d], env,
      "stride of view %" & name)
    let term = mulE(tInt(32), sym(dimLocals[d], tInt(32)), stride)
    result = addE(tInt(32), result, term)

proc cirMmaOffsets(prog: Program, view: LoadView, dim0, dim1: string, env: Table[string, CirExpr]): CirExpr =
  let stride0 = prog.entryToExpr(view.viewInfo.strideEntries[0], env,
    "stride of view %" & view.viewName)
  let stride1 = prog.entryToExpr(view.viewInfo.strideEntries[1], env,
    "stride of view %" & view.viewName)
  let term0 = mulE(tInt(32), sym(dim0, tInt(32)), stride0)
  let term1 = mulE(tInt(32), sym(dim1, tInt(32)), stride1)
  addE(tInt(32), sym("tt_block_off_" & view.viewName, tInt(32)),
       addE(tInt(32), term0, term1))

proc refuseMmaOperand(kernel, result, operand: string) =
  raise refuseBridge(kernel, "mmaf %" & result, "%" & operand & " is not a loaded tile, read the operands through their load views")

proc refuseRedefinition(kernel, construct, name: string) =
  raise refuseBridge(kernel, construct, "%" & name & " redefines a kernel-level name")

proc refuseUnboundOperand(kernel, construct, name: string) =
  raise refuseBridge(kernel, construct, "operand %" & name & " is not bound in the kernel")

proc lowerMma(prog: Program, op: ProgramOp, loadMap: Table[string, LoadView], env: var Table[string, CirExpr]): seq[CirStmt] =
  ## `mmaf` statements for one thread-value slot: a per-thread FMA loop over
  ## the k dimension, reading the A and B operand tiles through their views
  ## and adding the accumulator.
  ##
  ## The abstract mmaf has no atom.
  ## This is the per-thread decomposition that the python reference executes.
  ## Hardware simdgroup mma goes through the atom fragment program.
  ##
  ## Before, gemm fragment (tile<16x16xf32>, strides (16, 1)):
  ## - accumulator operand: `%zero`, bound to the splat scalar local
  ##   Mma(results: @["acc"], a: "tile_a", b: "tile_b", acc: "zero")
  ## After, inside the value loop (k = the tile's column size):
  ## - float tt_mma_sum_acc = 0.0f
  ## - a k loop from 0 to k, each step:
  ##     int tt_mma_off_a = (tt_block_off_pv_a + ((tt_c0 * 16) + (tt_kk * 1)))
  ##     int tt_mma_off_b = (tt_block_off_pv_b + ((tt_kk * 16) + (tt_c1 * 1)))
  ##     tt_mma_sum_acc = (tt_mma_sum_acc + (ptr_a[tt_mma_off_a] * ptr_b[tt_mma_off_b]))
  ## - float acc = (tt_mma_sum_acc + zero)
  ##
  ## Raises:
  ## - SubsetError when an A or B operand is not a loaded tile
  ## - SubsetError when the accumulator is not bound in the kernel
  if op.mmaA notin loadMap:
    refuseMmaOperand(prog.name, op.results[0], op.mmaA)
  if op.mmaB notin loadMap:
    refuseMmaOperand(prog.name, op.results[0], op.mmaB)
  if op.mmaAcc notin env:
    raise refuseBridge(prog.name, "mmaf %" & op.results[0], "accumulator %" & op.mmaAcc & " is not bound in the kernel")
  let aView = loadMap[op.mmaA]
  let bView = loadMap[op.mmaB]
  let elem = cirElemType(op.mmaInfo.elem)
  let zero = if op.mmaInfo.elem == eF32: fltLit(0.0, tFloat(32)) else: intLit(0, tInt(32))
  let sumName = "tt_mma_sum_" & op.results[0]
  let k = op.mmaInfo.tileShape[1]
  let sumSym = sym(sumName, elem)
  result.add varD(sumName, elem, zero)
  var inner: seq[CirStmt] = @[]
  # A[i, kk]: the m coordinate is tt_c0, the k coordinate is the loop variable
  inner.add varD("tt_mma_off_a", tInt(32),
    prog.cirMmaOffsets(aView, "tt_c0", "tt_kk", env))
  # B[kk, j]: the k coordinate is the loop variable, the n coordinate is tt_c1
  inner.add varD("tt_mma_off_b", tInt(32),
    prog.cirMmaOffsets(bView, "tt_kk", "tt_c1", env))
  inner.add asgnS(sumSym,
    addE(elem, sumSym,
      mulE(elem,
        patE(env[aView.base], sym("tt_mma_off_a", tInt(32))),
        patE(env[bView.base], sym("tt_mma_off_b", tInt(32))))))
  result.add whileLoop("tt_kk", k, inner)
  result.add varD(op.results[0], elem, addE(elem, sumSym, env[op.mmaAcc]))
  env[op.results[0]] = sym(op.results[0], elem)

proc lowerValueStatements(prog: Program, env: var Table[string, CirExpr],
                          info: TileValueInfo,
                          loadMap: Table[string, LoadView]): seq[CirStmt] =
  ## Per-thread, per-slot statements of the program ops, in program order,
  ## inside the fused value loop. Tile values are SSA scalar locals named by their Tile IR names.
  ## One tt_off local per view SSA name, shared by every op that addresses the view.
  ##
  ## Before (kernel axpy fragment):
  ## - mulf(lhs: "alpha", rhs: "tile_a"), addf(lhs: "scaled", rhs: "tile_b")
  ## After (inside the value loop, coord is the owned flat tile coordinate):
  ## - mulf  ->  float scaled = (alpha * tile_a)
  ## - addf  ->  float sum = (scaled + tile_b)
  ##
  ## Before, kernel gemm fragment (rank-2 tile<16x16xf32>):
  ## - ops: the load of tile_a, then mmaf(tile_a, tile_b, zero)
  ## After, inside the value loop, the flat coordinate decomposes into one
  ## tile coordinate per dimension:
  ## - int tt_c0 = (tt_coord / 16), int tt_c1 = (tt_coord % 16)
  ## - float tile_a = ptr_a[tt_off_pv_a]
  ## - the lowerMma FMA loop, then float acc = (tt_mma_sum_acc + zero)
  ##
  ## Raises:
  ## - SubsetError for an unbound operand or an explicit constant list
  ## - SubsetError for a store of a value that is not bound
  result.add varD("tt_coord", tInt(32),
    addE(tInt(32),
      sym("tt_tid", tInt(32)),
      mulE(tInt(32), sym("tt_v", tInt(32)), intLit(info.tv.threads, tInt(32)))))
  # Rank >= 2 tiles decompose the flat owned coordinate into one local per dimension, most-significant first.
  # For tile<16x16> coordinate f: tt_c0 = f / 16, tt_c1 = f % 16.
  var dimLocals: seq[string] = @[]
  if info.tileShape.len == 1:
    dimLocals.add "tt_coord"
  else:
    var remExpr = sym("tt_coord", tInt(32))
    let n = info.tileShape.len
    for d in 0 ..< n - 1:
      var mult = 1
      for e in d + 1 ..< n:
        mult *= info.tileShape[e]
      result.add varD("tt_c" & $d, tInt(32),
        divE(tInt(32), remExpr, intLit(mult, tInt(32))))
      remExpr = modE(tInt(32), remExpr, intLit(mult, tInt(32)))
    result.add varD("tt_c" & $(n - 1), tInt(32), remExpr)
    for d in 0 ..< n:
      dimLocals.add "tt_c" & $d
  var offLocals: seq[string]
    ## views whose tt_off local is already declared in the loop body
  for op in prog.ops:
    case op.kind
    of pgLoadTile:
      let name = op.opViewName()
      if name notin offLocals:
        result.add varD("tt_off_" & name, tInt(32),
          prog.cirTileOffset(name, op.opViewInfo(), dimLocals, env))
        offLocals.add name
      if op.results[0] in env:
        refuseRedefinition(prog.name, "load_view_tko %" & op.results[0], op.results[0])
      result.add varD(op.results[0], cirElemType(op.loadInfo.elem),
        patE(env[op.loadViewInfo.base], sym("tt_off_" & name, tInt(32))))
      env[op.results[0]] = sym(op.results[0], cirElemType(op.loadInfo.elem))
    of pgConstTile:
      if op.results[0] in env:
        refuseRedefinition(prog.name, "constant %" & op.results[0], op.results[0])
      if op.constValue.isList:
        raise refuseBridge(prog.name, "constant %" & op.results[0], "explicit constant value lists are outside the axpy tracer, splats only")
      result.add varD(op.results[0], cirElemType(op.constElem),
        constToExpr(op.constValue.scalar, op.constElem))
      env[op.results[0]] = sym(op.results[0], cirElemType(op.constElem))
    of pgBinary:
      if op.results[0] in env:
        refuseRedefinition(prog.name, binSpelling(op.binKind) & " %" & op.results[0], op.results[0])
      if op.binLhs notin env:
        refuseUnboundOperand(prog.name, binSpelling(op.binKind), op.binLhs)
      if op.binRhs notin env:
        refuseUnboundOperand(prog.name, binSpelling(op.binKind), op.binRhs)
      let elem = cirElemType(op.binInfo.elem)
      let rhs = case op.binKind
        of bkAddf: addE(elem, env[op.binLhs], env[op.binRhs])
        of bkMulf: mulE(elem, env[op.binLhs], env[op.binRhs])
      result.add varD(op.results[0], elem, rhs)
      env[op.results[0]] = sym(op.results[0], elem)
    of pgMma:
      if op.results[0] in env:
        refuseRedefinition(prog.name, "mmaf %" & op.results[0], op.results[0])
      for stmt in prog.lowerMma(op, loadMap, env):
        result.add stmt
    of pgStoreTile:
      if op.storeValue notin env:
        raise refuseBridge(prog.name, "store_view_tko", "stored value %" & op.storeValue & " is not bound in the kernel")
      let name = op.opViewName()
      if name notin offLocals:
        result.add varD("tt_off_" & name, tInt(32),
          prog.cirTileOffset(name, op.opViewInfo(), dimLocals, env))
        offLocals.add name
      result.add asgnS(
        patE(env[op.storeViewInfo.base], sym("tt_off_" & name, tInt(32))),
        env[op.storeValue])
    else:
      discard

proc atomBitsExpr(bl: BitLinear, v: int): CirExpr =
  ## Per-thread fragment cell offset of value slot `v`, the bit-linear form
  ## over the thread id: o = base + sum_i bit_i(tt_tid) * s[i] + v * sV.
  ##
  ## Example, the A fragment of the Apple 8x8x8 atom, slot 0: the offset is
  ## (tt_tid % 2) * 16 + ((tt_tid / 2) % 2) * 1 + ... + ((tt_tid / 16) % 2) * 4.
  result = intLit(v * bl.sV, tInt(32))
  for i in 0 ..< 5:
    let bit = modE(tInt(32),
      divE(tInt(32), sym("tt_tid", tInt(32)), intLit(1 shl i, tInt(32))),
      intLit(2, tInt(32)))
    result = addE(tInt(32), result, mulE(tInt(32), bit, intLit(bl.s[i], tInt(32))))

proc repAxisExpr(axis: RepAxis, rm, rn, rk: CirExpr): CirExpr =
  case axis
  of raM: rm
  of raN: rn
  of raK: rk

proc atomFragOffset(prog: Program, bl: BitLinear, map: FragmentMap, v: int, rm, rn, rk: CirExpr, view: LoadView, env: Table[string, CirExpr]): CirExpr =
  ## Global address of one fragment element:
  ## - the cell offset splits col-major into the two cell coordinates
  ## - each repetition axis scales one cell coordinate
  ## - each cell coordinate indexes one view dimension through its stride, plus the view block offset local
  ##
  ## Worked example, A fragment of the gemm fixture at repetition (rm, rk), slot 0:
  ## - call: atomFragOffset(aBits, aMap, 0, tt_rm, tt_rn, tt_rk, loadMap["tile_a"], env)
  ## - emitted:
  ##   (tt_block_off_pv_a + ((tt_rm * 8 + cell % 8) * 16 + (tt_rk * 8 + cell / 8) * 1))
  ##   with cell the bit-linear offset form of thread tt_tid at slot 0
  let cell = atomBitsExpr(bl, v)
  let low = modE(tInt(32), cell, intLit(8, tInt(32)))
  let high = divE(tInt(32), cell, intLit(8, tInt(32)))
  let row = addE(tInt(32),
    mulE(tInt(32), repAxisExpr(map.repRowAxis, rm, rn, rk), intLit(8, tInt(32))),
    (if map.rowIsLow: low else: high))
  let col = addE(tInt(32),
    mulE(tInt(32), repAxisExpr(map.repColAxis, rm, rn, rk), intLit(8, tInt(32))),
    (if map.rowIsLow: high else: low))
  let strideRow = prog.entryToExpr(view.viewInfo.strideEntries[map.rowViewDim],
                                   env, "stride of view %" & view.viewName)
  let strideCol = prog.entryToExpr(view.viewInfo.strideEntries[map.colViewDim],
                                   env, "stride of view %" & view.viewName)
  addE(tInt(32), sym("tt_block_off_" & view.viewName, tInt(32)),
    addE(tInt(32),
      mulE(tInt(32), row, strideRow),
      mulE(tInt(32), col, strideCol)))

proc atomMmaAsm(dNames, aNames, bNames: seq[string]): CirStmt =
  ## One simdgroup_multiply_accumulate call: sd <- sa*sb + sd.
  ## The staged scalars sit in the operand slots the printer splices.
  ##
  ## Operand names are literals unique per fragment instance:
  ##
  ## - crucible's rename pass rewrites reserved words only, it never
  ##   uniquifies repeated declarations
  ## - the splice references operands by name, so a reused name would
  ##   silently rebind to the first declaration
  # operand splice: the printer splices operands in text order, so the two
  # accumulator write-back slots repeat d0 and d1 after the staged operand names
  let names = dNames & aNames & bNames & dNames
  let parts: seq[string] = @[
    "{\n" &
    "  simdgroup_float8x8 sd = make_filled_simdgroup_matrix<float, 8>(float(0.0f));\n" &
    "  sd.thread_elements()[0] = ",
    ";\n  sd.thread_elements()[1] = ",
    ";\n" &
    "  simdgroup_float8x8 sa = make_filled_simdgroup_matrix<float, 8>(float(0.0f));\n" &
    "  sa.thread_elements()[0] = ",
    ";\n  sa.thread_elements()[1] = ",
    ";\n" &
    "  simdgroup_float8x8 sb = make_filled_simdgroup_matrix<float, 8>(float(0.0f));\n" &
    "  sb.thread_elements()[0] = ",
    ";\n  sb.thread_elements()[1] = ",
    ";\n  simdgroup_multiply_accumulate(sd, sa, sb, sd);\n  ",
    " = sd.thread_elements()[0];\n  ",
    " = sd.thread_elements()[1];\n}"
  ]
  var operands: seq[CirExpr] = @[]
  for n in names:
    operands.add sym(n, tFloat(32))
  asmS(parts, operands)

proc atomShapeCheck(prog: Program, mma, store: ProgramOp, loadMap: Table[string, LoadView]) =
  ## Subset check for the atom fragment program: the kernel tile dataflow must
  ## be exactly one mmaf, its two loaded operands, one splat accumulator
  ## constant and one store of the mma result.
  ##
  ## The store view's partition tile matches the mmaf tile,
  ## the invariant checkTileMatchesView enforces. Fragment addressing derives
  ## from the mmaf tile and reads the store view's strides.
  ##
  ## Raises:
  ## - SubsetError naming the construct and the kernel
  if mma.mmaA notin loadMap:
    refuseMmaOperand(prog.name, mma.results[0], mma.mmaA)
  if mma.mmaB notin loadMap:
    refuseMmaOperand(prog.name, mma.results[0], mma.mmaB)
  for op in prog.ops:
    case op.kind
    of pgGetBlockId, pgYield:
      discard
    of pgLoadTile:
      if op.results[0] != mma.mmaA and op.results[0] != mma.mmaB:
        raise refuseBridge(prog.name, "load_view_tko %" & op.results[0], "the atom fragment program stages the mma operands only")
    of pgConstTile:
      if op.results[0] != mma.mmaAcc:
        raise refuseBridge(prog.name, "constant %" & op.results[0], "the atom fragment program initializes the accumulator from one splat constant only")
      if op.constValue.isList:
        raise refuseBridge(prog.name, "constant %" & op.results[0], "constant value lists are outside the atom subset, splats only")
    of pgMma:
      discard
    of pgStoreTile:
      if op.storeValue != mma.results[0]:
        raise refuseBridge(prog.name, "store_view_tko", "the atom fragment program stores the mma result only")
    else:
      raise refuseBridge(prog.name, $op.kind, "the atom fragment program covers one mmaf with its operands and store only")
  if store.kind != pgStoreTile:
    raise refuseBridge(prog.name, "atom fragment program", "the mma result has no store")

proc lowerAtomMmaBody(prog: Program, mma, store: ProgramOp, accInit: CirExpr,
                      loadMap: Table[string, LoadView],
                      env: Table[string, CirExpr]): seq[CirStmt] =
  ## Per-thread fragment program of one atom-selected mmaf:
  ## - nested loops over the atom repetitions
  ## - per-fragment operand staging from global memory
  ## - one simdgroup mma per (rm, rn, rk) cell triple
  ## - accumulator stores, one per owned accumulator coordinate
  ##
  ## Every fragment predicate is statically true: tile dims are positive
  ## multiples of the atom cell, checked at selection. The loops run the full
  ## repetition ranges with no runtime guards.
  ##
  ## Before (kernel gemm fragment, 16x16x16 tile, Apple 8x8x8 atom, threads = 32):
  ##   %acc = mmaf %tile_a, %tile_b, %zero : tile<16x16xf32> x3
  ## After (per thread, after the block offsets):
  ##   for tt_rm in 0 ..< 2:
  ##     for tt_rn in 0 ..< 2:
  ##       float tt_d0 = <accInit>, tt_d1 = <accInit>
  ##       for tt_rk in 0 ..< 2:
  ##         float tt_a0 = ptr_a[off_a(tt_rm, tt_rk, v=0)], tt_a1 = ...
  ##         float tt_b0 = ptr_b[off_b(tt_rk, tt_rn, v=0)], tt_b1 = ...
  ##         simdgroup_multiply_accumulate(sd, sa, sb, sd) over the staged fragments
  ##       ptr_c[off_c(tt_rm, tt_rn, v=0)] = tt_d0, ptr_c[off_c(..., v=1)] = tt_d1
  ##
  ## Raises:
  ## - SubsetError for an unbound name or an unbound accumulator operand
  let sel = prog.atomMmas[mma.results[0]]
  let aView = loadMap[mma.mmaA]
  let bView = loadMap[mma.mmaB]
  let elem = cirElemType(mma.mmaInfo.elem)
  let rm = sym("tt_rm", tInt(32))
  let rn = sym("tt_rn", tInt(32))
  let rk = sym("tt_rk", tInt(32))
  let pa = env[aView.base]
  let pb = env[bView.base]
  let cView = LoadView(viewName: store.opViewName(),
                       base: store.storeViewInfo.base,
                       viewInfo: store.storeViewInfo)
  let pc = env[cView.base]
  var rkBody: seq[CirStmt] = @[]
  for v in 0 ..< 2:
    rkBody.add varD("tt_a" & $v, elem,
      patE(pa, prog.atomFragOffset(sel.aBits, sel.aMap, v, rm, rn, rk,
                                   aView, env)))
  for v in 0 ..< 2:
    rkBody.add varD("tt_b" & $v, elem,
      patE(pb, prog.atomFragOffset(sel.bBits, sel.bMap, v, rm, rn, rk,
                                   bView, env)))
  rkBody.add atomMmaAsm(@["tt_d0", "tt_d1"], @["tt_a0", "tt_a1"],
                        @["tt_b0", "tt_b1"])
  var rnBody: seq[CirStmt] = @[
    varD("tt_d0", elem, accInit),
    varD("tt_d1", elem, accInit),
  ]
  rnBody.add whileLoop("tt_rk", sel.repsK, rkBody)
  for v in 0 ..< 2:
    rnBody.add asgnS(
      patE(pc, prog.atomFragOffset(sel.cBits, sel.cMap, v, rm, rn, rk,
                                   cView, env)),
      sym("tt_d" & $v, elem))
  result = whileLoop("tt_rm", sel.repsM, whileLoop("tt_rn", sel.repsN, rnBody))

proc lowerTileToCirv2*(prog: Program, atomMma = false): CirModule =
  ## Lowers one resolved Tile IR program to a CIRv2 module with one kernel proc.
  ## Output: the CIRv2 module handed to the pass pipeline.
  ##
  ## Statement order:
  ## - block-level statements first, the thread id and the block offsets
  ## - then one fused loop over the thread-value slots, carrying the tile
  ##   dataflow in program order, or the atom fragment program's nested
  ##   repetition loops in place of the loop
  ##
  ## Before (program ops of kernel axpy, threads=4, tile<8xf32>):
  ## - LoadTile(view: "pv_a", indices: @["bid_x"], info: tv(threads 4, values 2))
  ## - ProgramOp(kind: pgBinary, binKind: bkMulf, binLhs: "alpha", binRhs: "tile_a")
  ## After (CIRv2 body of `kernel void axpy`, one fused per-thread loop):
  ## - thread id: int tt_tid = (int)thread_position_in_threadgroup.x
  ## - per value slot tt_v in 0 ..< 2, owned coordinate tt_coord = tt_tid + tt_v * 4
  ##   float tile_a = ptr_a[tt_off_pv_a]
  ##   float scaled = (alpha * tile_a)
  ##   float sum = (scaled + tile_b), stored: ptr_c[tt_off_pv_c] = sum
  ##
  ## Raises:
  ## - SubsetError naming the construct and the kernel, per lowering step
  let info = prog.loopTripCount()
  var env = prog.blockEnv()

  var prelude: seq[CirStmt] = @[]
  prelude.add varD("tt_tid", tInt(32),
    builtinComponent(cbkThreadPositionInThreadgroup, "x"))
  prog.lowerBlockOffsets(env, prelude)

  var body = prelude
  if atomMma and prog.atomMmas.len > 0:
    if prog.atomMmas.len != 1:
      # atom-selected mmafs lower through the fragment program,
      # every other kernel shape keeps the abstract FMA fallback below
      raise refuseBridge(prog.name, "mma atom selection", "the atom fragment program lowers exactly one mmaf per kernel, got " & $prog.atomMmas.len & " selections")
    var mmaOp, storeOp, accOp: ProgramOp
    var sawAcc = false
    var defined: seq[string]
      ## SSA result names seen so far, enforcing the fused path's
      ## redefinition refusal on the atom path: a repeated accumulator
      ## constant would otherwise silently supply the last splat's value
    for op in prog.ops:
      for name in op.results:
        if name in defined:
          let construct = case op.kind
            of pgLoadTile: "load_view_tko %" & name
            of pgStoreTile: "store_view_tko %" & name
            of pgConstTile, pgConstScalar: "constant %" & name
            of pgBinary: binSpelling(op.binKind) & " %" & name
            of pgMma: "mmaf %" & name
            of pgGetBlockId, pgYield: $op.kind
          refuseRedefinition(prog.name, construct, name)
        defined.add name
      case op.kind
      of pgMma: mmaOp = op
      of pgStoreTile: storeOp = op
      of pgConstTile:
        accOp = op
        sawAcc = true
      else: discard
    prog.atomShapeCheck(mmaOp, storeOp, prog.loadViews())
    if not sawAcc:
      raise refuseBridge(prog.name, "mmaf %" & mmaOp.results[0], "accumulator %" & mmaOp.mmaAcc & " has no constant tile definition")
    let accInit = constToExpr(accOp.constValue.scalar, accOp.constElem)
    body.add prog.lowerAtomMmaBody(mmaOp, storeOp, accInit,
                                   prog.loadViews(), env)
  else:
    let statements = prog.lowerValueStatements(env, info, prog.loadViews())
    body.add whileLoop("tt_v", info.tv.values, statements)
  body.add retS()

  let p = procD(prog.name, prog.lowerParams(env), tVoid(), body)
  p.isKernel = true
  moduleD(prog.name, @[ivec3Type()], @[p])

proc runMetalPipeline(prog: Program, atomMma: bool): string =
  ## gpu_compiler.codegenMetal equivalent on a bridge-built CIRv2 module:
  ## - build the module through lowerTileToCirv2
  ## - run the shipped pass composition
  ## - print MSL
  ##
  ## Raises:
  ## - SubsetError naming the kernel and the refusing pass, per pass
  let m = prog.lowerTileToCirv2(atomMma)
  var reg = PassRegistry.new()
  registerNormalizationPasses(reg)
  registerSharedLoweringPasses(reg)
  # Rename position: before every hoist and explosion. The rename walks types,
  # procs and proc bodies, never `m.globals`. A declaration hoisted earlier
  # leaves the rename's reach unrenamed.
  reg.registerRenameReservedIdentifiers("metalReservedIdentifiers", MetalReserved)
  registerDecomposeArrayAssign(reg)
  registerExplodeStructWithPtrFields(reg)
  registerSmemInitRewrite(reg)
  registerMslHoistProgramScopeConstants(reg)
  registerMslHoistDeviceFnSmem(reg)
  registerMslRewriteSmemLocals(reg)
  registerMslThreadCoordinateBuiltins(reg)
  registerInstantiateAddressSpaceGenerics(reg)
  registerValidationPostPasses(reg)
  let perr = runPasses(m, reg)
  if perr.len > 0:
    raise refuseBridge(prog.name, "crucible MSL pass pipeline", perr)
  codegen(m, cgMetal)

proc runSpirvPipeline(prog: Program, threads: int): seq[uint32] =
  ## gpu_compiler.codegenSpirvShader equivalent on a bridge-built CIRv2 module:
  ## normalization and shared lowering, then the SPIR-V encoder.
  ## Binary SPIR-V words (Vulkan environment) out.
  ##
  ## The LocalSize execution mode bakes (threads, 1, 1) into the words, matching the Metal launch config `blk`.
  ## The MSL path ignores the baked workgroup size, the Metal launch config sets the shape at dispatch.
  ##
  ## Raises:
  ## - SubsetError naming the kernel and the refusing pass, per pass
  let m = prog.lowerTileToCirv2(false)
  m.procs[0].workgroup = [threads, 1, 1]
  var reg = PassRegistry.new()
  registerNormalizationPasses(reg)
  registerSharedLoweringPasses(reg)
  # No rename and no address-space callee instantiation: the encoder consumes typed words.
  registerValidationPostPasses(reg)
  let perr = runPasses(m, reg)
  if perr.len > 0:
    raise refuseBridge(prog.name, "crucible SPIR-V pass pipeline", perr)
  codegenSpirvShader(m)

proc compileTileIrToSpirv*(m: TileIrModule, entryName: string, threads: int): seq[uint32] {.compileTime.} =
  ## Compiles one entry of a parsed Tile IR module to binary SPIR-V words (Vulkan environment).
  ##
  ## Stages:
  ## - parse (caller), tile passes, CIRv2 bridge: the shared bridge output
  ## - crucible's SPIR-V pipeline: the Metal-path CIRv2 module compiles unchanged
  ##
  ## No tattletile lowering beyond the shared bridge: the same CIRv2 module
  ## the Metal path produces feeds crucible's SPIR-V backend.
  ## The abstract mma fallback lowers every shape, the atom fragment program is
  ## Metal-only (the simdgroup intrinsics have no SPIR-V form).
  ##
  ## Compile-time only: the crucible pass pipeline relies on compile-time evaluation.
  ## Call inside a `const`/`static` context with `parseTileIrFile`.
  ##
  ## Example:
  ##   const axpySpv = compileTileIrToSpirv(parseTileIrFile("fixtures/vec_add_axpy.tileir"), "axpy", threads = 4)
  ##
  ## Raises:
  ## - SubsetError naming the construct and the kernel, per pass
  let entry = m.getEntry(entryName)
  let prog = entry.runTilePipeline(threads)
  prog.runSpirvPipeline(threads)

proc compileTileIrToMsl*(m: TileIrModule, entryName: string, threads: int): string {.compileTime.} =
  ## Compiles one entry of a parsed Tile IR module through the whole pipeline
  ## to Metal Shading Language source.
  ## Stages: parse (caller) -> tile passes -> CIRv2 bridge -> crucible pipeline -> MSL printer.
  ##
  ## Compile-time only: the crucible pass pipeline relies on compile-time evaluation.
  ## Call inside a `const`/`static` context with `parseTileIrFile`.
  ##
  ## Example:
  ##   const axpyMsl = compileTileIrToMsl(parseTileIrFile("fixtures/vec_add_axpy.tileir"), "axpy", threads = 4)
  ##
  ## Raises:
  ## - SubsetError naming the construct and the kernel, per pass
  let entry = m.getEntry(entryName)
  let prog = entry.runTilePipeline(threads)
  prog.runMetalPipeline(prog.atomMmas.len > 0)
