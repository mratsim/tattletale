## MSL printer compile gate: emits MSL for a representative set
## of webgpu corpus kernels via the `metal:` macro and device-compiles every source
## through Metal's `newLibraryWithSource`, the same objc_abi bridge harness
## as tests/codegen/metal/test_metal_abi_smoke.nim. The acceptance bar is device-compile-clean, not mere printing.
##
## Covers 24 printer shapes:
## - add
## - external-struct device fn
## - scalar marshalling
## - all five index builtins (incl. synthesized gid)
## - shared memory + barrier
## - for loops
## - large-struct passByRef
## - ternary
## - constexpr tuple
## - CuTe Tile/tileAt
## - 2-D gid indexing
## - user-defined operator
## - multi-kernel sources
## - GEMM
## - var T params
## - ptr UncheckedArray device fn
## - while loop
## - let-block RHS
## - dummy-init constexpr
## - block-in-type
## - static-int ops (compile-time Int[N] arithmetic)
## - workgroup pragma (size stays dispatch-time)
## - tuple bracket access on a constexpr tuple
## - int64/uint64 buffer arithmetic
##
## The gate also pins the emission rules. The shader bakes no workgroup size.
## `blk` stays dispatch-time. The index-builtin mapping holds.
## The four attribute-qualified `uint3` params appear verbatim.
## `gid` maps to the composite `bid * bdim + tid`.
## A hand-built-IR kernel checks the atomic emission. Compile-time tripwires pin the loud rejects.
## fp64, reserved MSL keywords, and Metal builtin names never print.
##
## The `basicKernel` binop set (test_webgpu_user_defined_operator.nim)
## and `maxGeneric` (test_webgpu_add.nim) are covered by the execution suite
## (test_metal_user_defined_operator.nim, test_metal_add.nim), which runs
## both shapes through engine.run().
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/test_metal_msl_compile.nim

import std/[strutils, tables]

import workspace/crucible
import workspace/crucible/src/abis/objc_abi as objc
import workspace/crucible/src/codegen/ir/gpu_types
import workspace/crucible/src/codegen/targets/metal_lang

# ── DSL helper types (module scope, compile-time only) ──────────────────────

type
  Vec2* = object
    x: uint32
    y: uint32
  Int*[V: static int] = object
  Int2*[V: static int] = object
  Tuple2*[A, B] = object
    f0: A
    f1: B
  Tile*[M, N: static int] = object
    data: array[M * N, uint32]
  Wrapper* = object
    val: uint32
  FixMe*[V: static int] = object
  MySpan* = object
    idx: int32
    len: int32

template toIntVal*(x: int): int = x
template toIntVal*[V: static int](x: Int[V]): int = V

template genBinOp(op: untyped): untyped =
  template op*[V, U: static int](a: Int[V]; b: Int[U]): auto = Int[op(V, U)]()
  func op*[V: static int](a: Int[V]; b: static int): auto {.inline.} = Int[op(V, b)]()
  func op*[V: static int](a: static int; b: Int[V]): auto {.inline.} = Int[op(a, V)]()
  template op*[V: static int](a: Int[V]; b: int): int = op(V, b)
  template op*[V: static int](a: int; b: Int[V]): int = op(a, V)

genBinOp(`+`)
genBinOp(`*`)

proc vec2Add(a, b: Vec2): Vec2 =
  result.x = a.x + b.x
  result.y = a.y + b.y

proc tileAt[M, N: static int](t: Tile[M, N]; r, c: uint32): uint32 =
  t.data[r * uint32(N) + c]

proc `+`*(a, b: Wrapper): Wrapper = Wrapper(val: a.val + b.val)

template `()`(s: MySpan; a, b: int32): auto =
  block:
    let coord = (a, b)
    let offset = coord[0] * s.len + coord[1]
    var result: MySpan
    result.idx = s.idx + int32(offset)
    result.len = s.len
    result

proc deviceFn(span: MySpan) =
  discard span(0, 0).idx

# ── MSL kernel sources ──────────────────────────────────────────────────────

const addMsl = metal:
  proc addKernel(output: ptr UncheckedArray[uint32];
                 a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = a[0] + b[0]
    output[1] = a[1] + b[1]

const vec2Msl = metal:
  proc vec2AddKernel(output: ptr UncheckedArray[uint32];
                     a: ptr UncheckedArray[uint32];
                     b: ptr UncheckedArray[uint32]) {.global.} =
    let va = Vec2(x: a[0], y: a[1])
    let vb = Vec2(x: b[0], y: b[1])
    let vr = vec2Add(va, vb)
    output[0] = vr.x
    output[1] = vr.y

const scalarMsl = metal:
  proc scalarKernel(output: ptr UncheckedArray[uint32]; x: int32; f: float32; b: bool) {.global.} =
    if b:
      output[0] = uint32(x)
      output[1] = uint32(f)

const tidMsl = metal:
  proc tidKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = uint32(threadIdx.x)
    output[1] = uint32(blockIdx.x)
    output[2] = uint32(blockDim.x)
    output[3] = uint32(gridDim.x)
    output[4] = uint32(gid.x)
    output[5] = uint32(gid.y)

const sharedMsl = metal:
  proc sharedKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var scratch {.shared.}: array[64, uint32]
    scratch[threadIdx.x] = uint32(threadIdx.x)
    syncthreads()
    output[threadIdx.x] = scratch[uint32(63) - uint32(threadIdx.x)]

const forLoopMsl = metal:
  proc vec10_add(output: ptr UncheckedArray[uint32];
                 a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32]) {.global.} =
    for i in 0 ..< 10:
      output[i] = a[i] + b[i]

const largeStructMsl = metal:
  type LargeStruct = object
    data: array[8, uint32]
  proc takeLarge(s: LargeStruct): uint32 {.device.} =
    result = s.data[0] + s.data[1]
  proc kernelMain(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = takeLarge(LargeStruct(data: [10'u32, 20, 30, 40, 50, 60, 70, 80]))

const ternaryMsl = metal:
  proc ternaryKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    output[0] = if true: 1'u32 else: 0'u32

const constexprMsl = metal:
  proc testConstexpr(C: ptr UncheckedArray[uint32]) {.global.} =
    const tmp {.genSym.} = Tuple2[Int[8], Int[16]]()
    let L = tmp
    C[0] = 1'u32

const cuteMsl = metal:
  proc cuteKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    let t = Tile[2, 3](data: [10'u32, 20'u32, 30'u32, 40'u32, 50'u32, 60'u32])
    output[0] = tileAt(t, 0'u32, 0'u32)
    output[1] = tileAt(t, 0'u32, 1'u32)
    output[2] = tileAt(t, 1'u32, 2'u32)

const ndrangeMsl = metal:
  proc grid2d(C: ptr UncheckedArray[uint32]) {.global.} =
    C[uint32(gid.y) * 8'u32 + uint32(gid.x)] = uint32(gid.y) * 8'u32 + uint32(gid.x)

const opMsl = metal:
  proc structKernel(output: ptr UncheckedArray[uint32];
                    a: ptr UncheckedArray[uint32];
                    b: ptr UncheckedArray[uint32]) {.global.} =
    let x = Wrapper(val: a[0])
    let y = Wrapper(val: b[0])
    output[0] = (x + y).val

const multiKernelMsl = metal:
  proc vec10_add(output: ptr UncheckedArray[uint32];
                 a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32]) {.global.} =
    for i in 0 ..< 10:
      output[i] = a[i] + b[i]
  proc vec10_mul(output: ptr UncheckedArray[uint32];
                 a: ptr UncheckedArray[uint32];
                 b: ptr UncheckedArray[uint32]) {.global.} =
    for i in 0 ..< 10:
      output[i] = a[i] * b[i]

const gemmMsl = metal:
  type Tile[M, N: static int] = object
    data: array[M * N, uint32]
  proc tileAt[M, N: static int](t: Tile[M, N]; r, c: uint32): uint32 =
    t.data[r * uint32(N) + c]
  proc gemmKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    let A = Tile[2, 3](data: [1'u32, 2, 3, 4, 5, 6])
    let B = Tile[3, 2](data: [7'u32, 8, 9, 10, 11, 12])
    var Ct: Tile[2, 2]
    for i in 0 ..< 2:
      for j in 0 ..< 2:
        var sum: uint32 = 0
        for k in 0 ..< 3:
          sum += tileAt(A, uint32(i), uint32(k)) * tileAt(B, uint32(k), uint32(j))
        Ct.data[i * 2 + j] = sum
    C[0] = Ct.data[0]
    C[1] = Ct.data[1]
    C[2] = Ct.data[2]
    C[3] = Ct.data[3]

const varParamMsl = metal:
  type Pair = object
    x: uint32
    y: uint32
  proc setPair(p: var Pair; vx, vy: uint32) {.device.} =
    p.x = vx
    p.y = vy
  proc swap(a, b: var uint32) {.device.} =
    let tmp = a
    a = b
    b = tmp
  proc varParamKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var p: Pair
    setPair(p, 10'u32, 20'u32)
    output[0] = p.x
    output[1] = p.y
    var a, b: uint32 = 1
    b = 2
    swap(a, b)
    output[2] = a
    output[3] = b

const ptrUncheckedMsl = metal:
  proc fillArray(p: ptr UncheckedArray[uint32]; n: uint32) {.device.} =
    for i in 0 ..< n:
      p[i] = i + 10'u32
  proc fillKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    fillArray(output, 8)

const whileMsl = metal:
  proc whileKernel(output: ptr UncheckedArray[uint32]) {.global.} =
    var i: uint32 = 0
    while i < 4'u32:
      output[i] = i
      i = i + 1'u32

const letBlockRhsMsl = metal:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    let L = block:
      const tmp {.genSym.} = Tuple2[Int2[8], Int2[16]]()
      tmp
    C[0] = 1'u32

const dummyInitMsl = metal:
  proc dummyKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const x {.genSym.} = FixMe[8]()
    C[0] = 1'u32

const blockInTypeMsl = metal:
  proc reproKernel(output: ptr UncheckedArray[float32],
                   input: ptr UncheckedArray[float32],
                   M, N: int32) {.global.} =
    let s = MySpan(idx: M, len: N)
    deviceFn(s)

const staticIntOpsMsl = metal:
  proc staticIntOps(res: ptr UncheckedArray[int32];
                    dyn: ptr UncheckedArray[int32]) {.global.} =
    let h = int(dyn[0])
    let a = Int[10]() + 1        # Int[11]
    let b = 2 + Int[10]()        # Int[12]
    let c = Int[10]() * 3        # Int[30]
    let e = Int[2]() + Int[3]()  # Int[5]
    let g = Int[10]() + h        # 10 + h
    res[0] = int32(toIntVal a)
    res[1] = int32(toIntVal b)
    res[2] = int32(toIntVal c)
    res[3] = int32(toIntVal e)
    res[4] = int32(g)

const workgroupMsl = metal:
  proc grid2d(C: ptr UncheckedArray[uint32]) {.global, workgroup: (4, 2).} =
    C[global_id.y * 8'u32 + global_id.x] = global_id.y * 8'u32 + global_id.x

const tupleIndexMsl = metal:
  proc tupleKernel(C: ptr UncheckedArray[uint32]) {.global.} =
    const tup {.genSym.} = (Int[8](), Int[16]())
    let first = tup[0]
    C[0] = uint32(toIntVal first)

const int64Msl = metal:
  proc int64Kernel(output: ptr UncheckedArray[uint64];
                   a: ptr UncheckedArray[int64];
                   b: ptr UncheckedArray[uint64]) {.global.} =
    let x = a[0] + a[1]
    let y = b[0] * 2'u64 + 1000000000000'u64
    output[0] = uint64(x)
    output[1] = y

# ── Hand-built IR helpers ──────────────────────────────────────────────────

proc mkIdent(name: string; typ: GpuType; kind: GpuSymbolKind): GpuAst =
  ## GpuIdent node with a fresh symbol, for hand-built kernels.
  GpuAst(kind: gpuIdent, symbol: newSymbol(name, iSym = name, typ = typ, symKind = kind))

proc uint32Lit(v: string): GpuAst =
  ## gpuLit node for a uint32 literal.
  GpuAst(kind: gpuLit, lValue: v, lType: GpuType(kind: gtUint32))

proc buildAtomicSrc(): string =
  ## MSL for a kernel with an atomic counter param and one atomic_add call.
  ## Built as IR directly, because the shared atomic dummy builtins are not DSL-reachable
  ## in any backend (a shared-builtins gap).
  let outTyp = GpuType(kind: gtPtr, to: GpuType(kind: gtUA, uaTo: GpuType(kind: gtUint32)))
  let ctrTyp = GpuType(kind: gtPtr, to: GpuType(kind: gtUint32))
  let outParam = GpuParam(ident: mkIdent("output", outTyp, gsGlobalKernelParam), typ: outTyp)
  let ctrParam = GpuParam(ident: mkIdent("counter", ctrTyp, gsGlobalKernelParam), typ: ctrTyp)
  let call = GpuAst(kind: gpuCall,
                    cName: mkIdent("atomic_add", GpuType(kind: gtVoid), gsProc),
                    cArgs: @[mkIdent("counter", ctrTyp, gsGlobalKernelParam), uint32Lit("1")])
  let body = GpuAst(kind: gpuBlock, statements: @[call])
  let kernel = GpuAst(kind: gpuProc,
                      pName: mkIdent("atomicKernel", GpuType(kind: gtVoid), gsProc),
                      pRetType: GpuType(kind: gtVoid),
                      pParams: @[outParam, ctrParam],
                      pBody: body,
                      pAttributes: {attGlobal})
  var ctx = GpuContext()
  ctx.preprocess(kernel)
  result = ctx.codegen()

const atomicMsl = buildAtomicSrc()

# ── Loud-reject tripwires ──────────────────────────────────────────────────

static:
  # fp64. The printer's `gpuTypeToString` raises an AssertionDefect that names the unsupported type.
  # The check calls the printer directly and asserts the message, the same path
  # the `metal:` macro funnels every kernel type through. The tripwire fails
  # if a regression silently spells fp64 as `double` instead of raising.
  block:
    var rejected = false
    var rejectMsg = ""
    try:
      discard gpuTypeToString(GpuType(kind: gtFloat64))
    except AssertionDefect as e:
      rejected = true
      rejectMsg = e.msg
    doAssert rejected, "an fp64 kernel printed instead of rejecting loudly"
    doAssert "64-bit floating point" in rejectMsg,
      "fp64 reject message lost its 64-bit floating point note: " & rejectMsg

  # Reserved MSL keywords never print. The macro pass rejects function names.
  # The printer's guard, `checkReservedIdent`, covers params, locals, fields, and every other emitted identifier.
  # Each tripwire hand-builds the IR and runs it through the printer, asserting the loud raise.
  doAssert not compiles(block:
    const k = metal:
      proc kernel(output: ptr UncheckedArray[uint32]) {.global.} = discard
    discard k)

  block:
    # kernel param named `device`
    var rejected = false
    var rejectMsg = ""
    try:
      let ptyp = GpuType(kind: gtPtr, to: GpuType(kind: gtUA, uaTo: GpuType(kind: gtUint32)))
      let param = GpuParam(ident: mkIdent("device", ptyp, gsGlobalKernelParam), typ: ptyp)
      let kernel = GpuAst(kind: gpuProc,
                          pName: mkIdent("k", GpuType(kind: gtVoid), gsProc),
                          pRetType: GpuType(kind: gtVoid),
                          pParams: @[param],
                          pBody: GpuAst(kind: gpuBlock),
                          pAttributes: {attGlobal})
      var ctx = GpuContext()
      ctx.preprocess(kernel)
      discard ctx.codegen()
    except AssertionDefect as e:
      rejected = true
      rejectMsg = e.msg
    doAssert rejected, "a kernel param named `device` printed instead of rejecting loudly"
    doAssert "'device' is a reserved keyword in MSL. Rename the parameter." == rejectMsg,
      rejectMsg

  block:
    # kernel param named `thread` (MSL address-space keyword)
    var rejected = false
    var rejectMsg = ""
    try:
      let ptyp = GpuType(kind: gtPtr, to: GpuType(kind: gtUA, uaTo: GpuType(kind: gtUint32)))
      let param = GpuParam(ident: mkIdent("thread", ptyp, gsGlobalKernelParam), typ: ptyp)
      let kernel = GpuAst(kind: gpuProc,
                          pName: mkIdent("k", GpuType(kind: gtVoid), gsProc),
                          pRetType: GpuType(kind: gtVoid),
                          pParams: @[param],
                          pBody: GpuAst(kind: gpuBlock),
                          pAttributes: {attGlobal})
      var ctx = GpuContext()
      ctx.preprocess(kernel)
      discard ctx.codegen()
    except AssertionDefect as e:
      rejected = true
      rejectMsg = e.msg
    doAssert rejected, "a kernel param named `thread` printed instead of rejecting loudly"
    doAssert "'thread' is a reserved keyword in MSL. Rename the parameter." == rejectMsg,
      rejectMsg

  block:
    # kernel param named `threadIdx` (Metal builtin name)
    var rejected = false
    var rejectMsg = ""
    try:
      let ptyp = GpuType(kind: gtPtr, to: GpuType(kind: gtUA, uaTo: GpuType(kind: gtUint32)))
      let param = GpuParam(ident: mkIdent("threadIdx", ptyp, gsGlobalKernelParam), typ: ptyp)
      let kernel = GpuAst(kind: gpuProc,
                          pName: mkIdent("k", GpuType(kind: gtVoid), gsProc),
                          pRetType: GpuType(kind: gtVoid),
                          pParams: @[param],
                          pBody: GpuAst(kind: gpuBlock),
                          pAttributes: {attGlobal})
      var ctx = GpuContext()
      ctx.preprocess(kernel)
      discard ctx.codegen()
    except AssertionDefect as e:
      rejected = true
      rejectMsg = e.msg
    doAssert rejected, "a kernel param named `threadIdx` printed instead of rejecting loudly"
    doAssert "'threadIdx' is a Metal builtin name. Rename the parameter." == rejectMsg,
      rejectMsg

  block:
    # local variable named `threadgroup`
    var rejected = false
    var rejectMsg = ""
    try:
      let vTyp = GpuType(kind: gtUint32)
      let v = GpuAst(kind: gpuVar, vName: mkIdent("threadgroup", vTyp, gsLocal),
                     vType: vTyp,
                     vInit: GpuAst(kind: gpuLit, lValue: "3", lType: vTyp))
      let body = GpuAst(kind: gpuBlock, statements: @[v])
      let kernel = GpuAst(kind: gpuProc,
                          pName: mkIdent("k", GpuType(kind: gtVoid), gsProc),
                          pRetType: GpuType(kind: gtVoid),
                          pParams: @[],
                          pBody: body,
                          pAttributes: {attGlobal})
      var ctx = GpuContext()
      ctx.preprocess(kernel)
      discard ctx.codegen()
    except AssertionDefect as e:
      rejected = true
      rejectMsg = e.msg
    doAssert rejected, "a local variable named `threadgroup` printed instead of rejecting loudly"
    doAssert "'threadgroup' is a reserved keyword in MSL. Rename the variable." == rejectMsg,
      rejectMsg

  block:
    # local variable named `half` (MSL numeric type keyword)
    var rejected = false
    var rejectMsg = ""
    try:
      let vTyp = GpuType(kind: gtUint32)
      let v = GpuAst(kind: gpuVar, vName: mkIdent("half", vTyp, gsLocal),
                     vType: vTyp,
                     vInit: GpuAst(kind: gpuLit, lValue: "3", lType: vTyp))
      let body = GpuAst(kind: gpuBlock, statements: @[v])
      let kernel = GpuAst(kind: gpuProc,
                          pName: mkIdent("k", GpuType(kind: gtVoid), gsProc),
                          pRetType: GpuType(kind: gtVoid),
                          pParams: @[],
                          pBody: body,
                          pAttributes: {attGlobal})
      var ctx = GpuContext()
      ctx.preprocess(kernel)
      discard ctx.codegen()
    except AssertionDefect as e:
      rejected = true
      rejectMsg = e.msg
    doAssert rejected, "a local variable named `half` printed instead of rejecting loudly"
    doAssert "'half' is a reserved keyword in MSL. Rename the variable." == rejectMsg,
      rejectMsg

  block:
    # local variable named `gid` (Metal builtin name)
    var rejected = false
    var rejectMsg = ""
    try:
      let vTyp = GpuType(kind: gtUint32)
      let v = GpuAst(kind: gpuVar, vName: mkIdent("gid", vTyp, gsLocal),
                     vType: vTyp,
                     vInit: GpuAst(kind: gpuLit, lValue: "3", lType: vTyp))
      let body = GpuAst(kind: gpuBlock, statements: @[v])
      let kernel = GpuAst(kind: gpuProc,
                          pName: mkIdent("k", GpuType(kind: gtVoid), gsProc),
                          pRetType: GpuType(kind: gtVoid),
                          pParams: @[],
                          pBody: body,
                          pAttributes: {attGlobal})
      var ctx = GpuContext()
      ctx.preprocess(kernel)
      discard ctx.codegen()
    except AssertionDefect as e:
      rejected = true
      rejectMsg = e.msg
    doAssert rejected, "a local variable named `gid` printed instead of rejecting loudly"
    doAssert "'gid' is a Metal builtin name. Rename the variable." == rejectMsg,
      rejectMsg

  block:
    # Module-scope user symbol named `gid`. The hand-built ident carries the default `gsNone` kind.
    # That is exactly the shape a module-scope user `let gid` produces in the frontend.
    # The reference must raise. It must never rewrite to the builtin composite `bid * bdim + tid`.
    var rejected = false
    var rejectMsg = ""
    try:
      let vTyp = GpuType(kind: gtUint32)
      let gidRef = GpuAst(kind: gpuIdent, symbol: newSymbol("gid"))
      let v = GpuAst(kind: gpuVar, vName: mkIdent("x", vTyp, gsLocal),
                     vType: vTyp,
                     vInit: gidRef)
      let body = GpuAst(kind: gpuBlock, statements: @[v])
      let kernel = GpuAst(kind: gpuProc,
                          pName: mkIdent("k", GpuType(kind: gtVoid), gsProc),
                          pRetType: GpuType(kind: gtVoid),
                          pParams: @[],
                          pBody: body,
                          pAttributes: {attGlobal})
      var ctx = GpuContext()
      ctx.preprocess(kernel)
      discard ctx.codegen()
    except AssertionDefect as e:
      rejected = true
      rejectMsg = e.msg
    doAssert rejected, "a module-scope symbol named `gid` was rewritten to the builtin composite"
    doAssert "'gid' is a Metal builtin name. Rename the identifier." == rejectMsg, rejectMsg

  block:
    # struct field named `metal`
    var rejected = false
    var rejectMsg = ""
    try:
      let sTyp = GpuType(kind: gtObject, name: "S")
      let typeDef = GpuAst(kind: gpuTypeDef, tTyp: sTyp,
                           tFields: @[GpuTypeField(name: "metal", typ: GpuType(kind: gtUint32))])
      let kernel = GpuAst(kind: gpuProc,
                          pName: mkIdent("k", GpuType(kind: gtVoid), gsProc),
                          pRetType: GpuType(kind: gtVoid),
                          pParams: @[],
                          pBody: GpuAst(kind: gpuBlock),
                          pAttributes: {attGlobal})
      var ctx = GpuContext()
      ctx.types[sTyp] = typeDef
      ctx.preprocess(kernel)
      discard ctx.codegen()
    except AssertionDefect as e:
      rejected = true
      rejectMsg = e.msg
    doAssert rejected, "a struct field named `metal` printed instead of rejecting loudly"
    doAssert "'metal' collides with the MSL `metal::` namespace. Rename the field." == rejectMsg,
      rejectMsg

  block:
    # passByRef device-fn param named `device` (MSL keyword)
    var rejected = false
    var rejectMsg = ""
    try:
      let bigTyp = GpuType(kind: gtObject, name: "BigStruct")
      let param = GpuParam(ident: mkIdent("device", bigTyp, gsDeviceKernelParam),
                           typ: bigTyp, passByRef: true)
      let helperSym = mkIdent("helper", GpuType(kind: gtVoid), gsProc)
      let helper = GpuAst(kind: gpuProc,
                          pName: helperSym,
                          pRetType: GpuType(kind: gtVoid),
                          pParams: @[param],
                          pBody: GpuAst(kind: gpuBlock))
      let call = GpuAst(kind: gpuCall, cName: helperSym, cArgs: @[])
      let body = GpuAst(kind: gpuBlock, statements: @[call])
      let kernel = GpuAst(kind: gpuProc,
                          pName: mkIdent("k", GpuType(kind: gtVoid), gsProc),
                          pRetType: GpuType(kind: gtVoid),
                          pParams: @[],
                          pBody: body,
                          pAttributes: {attGlobal})
      var ctx = GpuContext()
      ctx.preprocess(GpuAst(kind: gpuBlock, statements: @[helper, kernel]))
      discard ctx.codegen()
    except AssertionDefect as e:
      rejected = true
      rejectMsg = e.msg
    doAssert rejected, "a passByRef device-fn param named `device` printed instead of rejecting loudly"
    doAssert "'device' is a reserved keyword in MSL. Rename the parameter." == rejectMsg,
      rejectMsg

  block:
    # atomic call with no target argument must raise
    var rejected = false
    var rejectMsg = ""
    try:
      let outTyp = GpuType(kind: gtPtr, to: GpuType(kind: gtUA, uaTo: GpuType(kind: gtUint32)))
      let outParam = GpuParam(ident: mkIdent("output", outTyp, gsGlobalKernelParam), typ: outTyp)
      let call = GpuAst(kind: gpuCall,
                        cName: mkIdent("atomic_add", GpuType(kind: gtVoid), gsProc),
                        cArgs: @[])
      let body = GpuAst(kind: gpuBlock, statements: @[call])
      let kernel = GpuAst(kind: gpuProc,
                          pName: mkIdent("k", GpuType(kind: gtVoid), gsProc),
                          pRetType: GpuType(kind: gtVoid),
                          pParams: @[outParam],
                          pBody: body,
                          pAttributes: {attGlobal})
      var ctx = GpuContext()
      ctx.preprocess(kernel)
      discard ctx.codegen()
    except AssertionDefect as e:
      rejected = true
      rejectMsg = e.msg
    doAssert rejected, "an atomic call with no target compiled instead of raising loudly"
    doAssert "requires a target argument" in rejectMsg, rejectMsg

  block:
    # atomic call targeting a literal (non-identifier) must raise
    var rejected = false
    var rejectMsg = ""
    try:
      let outTyp = GpuType(kind: gtPtr, to: GpuType(kind: gtUA, uaTo: GpuType(kind: gtUint32)))
      let outParam = GpuParam(ident: mkIdent("output", outTyp, gsGlobalKernelParam), typ: outTyp)
      let call = GpuAst(kind: gpuCall,
                        cName: mkIdent("atomic_add", GpuType(kind: gtVoid), gsProc),
                        cArgs: @[uint32Lit("1")])
      let body = GpuAst(kind: gpuBlock, statements: @[call])
      let kernel = GpuAst(kind: gpuProc,
                          pName: mkIdent("k", GpuType(kind: gtVoid), gsProc),
                          pRetType: GpuType(kind: gtVoid),
                          pParams: @[outParam],
                          pBody: body,
                          pAttributes: {attGlobal})
      var ctx = GpuContext()
      ctx.preprocess(kernel)
      discard ctx.codegen()
    except AssertionDefect as e:
      rejected = true
      rejectMsg = e.msg
    doAssert rejected, "an atomic call targeting a literal compiled instead of raising loudly"
    doAssert "targets a non-atomic identifier" in rejectMsg, rejectMsg

# ── Compile gate ────────────────────────────────────────────────────────────

template failLoud(msg: string) =
  ## Unified error policy: stacktrace + stderr + quit(1) with the caller's location.
  ## A template, so instantiationInfo() reports the call site.
  writeStackTrace()
  stderr.write($instantiationInfo() & " exited with error: " & msg & '\n')
  quit 1

proc compileOne(device: objc.ID; name, src: string) =
  ## Compiles `src` on the device via `newLibraryWithSource` and fails loudly
  ## on a nil library, surfacing the NSError `localizedDescription` when the compiler provides one.
  doAssert "threads_per_threadgroup(" notin src,
    name & " bakes a workgroup size; blk must stay dispatch-time"
  echo "  compiling ", name
  var compileError: objc.ID = objc.ID(nil)
  let library = objc.msgSend(device, objc.`$$`("newLibraryWithSource:options:error:"),
                        objc.nsStringFromNimString(src), objc.ID(nil), addr compileError)
  if objc.isNil(library):
    if objc.isNil(compileError):
      failLoud("MSL compile failed for " & name & ": no NSError object provided")
    let desc = objc.msgSend(compileError, objc.`$$`("localizedDescription"))
    failLoud("MSL compile failed for " & name & ": " & objc.nsStringToNimString(desc))
  echo "    OK"

proc runTest() =
  # Autorelease pool wraps the whole run, because Metal objects are autoreleased.
  # A missing pool would trip OBJC_DEBUG_MISSING_POOLS=YES.
  let pool = objc.msgSend(objc.ID(objc.getClass("NSAutoreleasePool")), objc.`$$`("alloc"))
  discard objc.msgSend(pool, objc.`$$`("init"))

  let device = objc.MTLCreateSystemDefaultDevice()
  if objc.isNil(device):
    failLoud("default Metal device lookup returned nil (no Metal device)")

  compileOne(device, "add", addMsl)
  compileOne(device, "vec2 external struct + device fn", vec2Msl)
  compileOne(device, "scalar marshalling (int32/float32/bool)", scalarMsl)
  compileOne(device, "tid/bid/bdim/gdim + synthesized gid", tidMsl)
  compileOne(device, "shared memory + syncthreads barrier", sharedMsl)
  compileOne(device, "for-loop vec10", forLoopMsl)
  compileOne(device, "large struct passByRef", largeStructMsl)
  compileOne(device, "ternary", ternaryMsl)
  compileOne(device, "constexpr tuple", constexprMsl)
  compileOne(device, "cute_layout Tile + tileAt", cuteMsl)
  compileOne(device, "2-D gid indexing", ndrangeMsl)
  compileOne(device, "user-defined operator struct", opMsl)
  compileOne(device, "multi-kernel source", multiKernelMsl)
  compileOne(device, "gemm nested loops + passByRef Tile", gemmMsl)
  compileOne(device, "var T params", varParamMsl)
  compileOne(device, "ptr UncheckedArray device fn", ptrUncheckedMsl)
  compileOne(device, "while loop", whileMsl)
  compileOne(device, "let-block RHS", letBlockRhsMsl)
  compileOne(device, "dummy-init constexpr", dummyInitMsl)
  compileOne(device, "block-in-type", blockInTypeMsl)
  compileOne(device, "static-int ops", staticIntOpsMsl)
  compileOne(device, "workgroup pragma", workgroupMsl)
  compileOne(device, "tuple bracket access", tupleIndexMsl)
  compileOne(device, "int64/uint64 buffers", int64Msl)
  compileOne(device, "atomic add (hand-built IR)", atomicMsl)

  # The asserts check string presence. A permuted attribute mapping would still satisfy all five and device-compile.
  # Semantic binding is verified by execution tests, which are not part of this compile gate.
  # Index-builtin mapping. The tid kernel source carries the four attribute-qualified uint3 params
  # and the synthesized gid composite, all verbatim.
  doAssert "uint3 tid [[thread_position_in_threadgroup]]" in tidMsl
  doAssert "uint3 bid [[threadgroup_position_in_grid]]" in tidMsl
  doAssert "uint3 bdim [[threads_per_threadgroup]]" in tidMsl
  doAssert "uint3 gdim [[threadgroups_per_grid]]" in tidMsl
  doAssert "(bid * bdim + tid)" in tidMsl
  echo "  OK — index-builtin mapping (tid/bid/bdim/gdim, gid = bid * bdim + tid)"

  doAssert "device atomic_uint* counter [[buffer(1)]]" in atomicMsl
  doAssert "atomic_fetch_add_explicit(counter, 1U, memory_order_relaxed)" in atomicMsl
  echo "  OK — atomic emission (atomic_uint param, atomic_fetch_add_explicit)"

  discard objc.msgSend(pool, objc.`$$`("drain"))
  echo "All 25 printed MSL sources compiled via newLibraryWithSource"

when isMainModule:
  runTest()
