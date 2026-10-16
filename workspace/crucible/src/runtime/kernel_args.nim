# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Kernel argument binding: how launch values cross a kernel boundary.
##
## One contract, two consumers that must never disagree:
## - the `{.global.}` bridge (wrap_and_run): one bridge parameter per dynamic kernel arg,
##   static subtrees rebuild from the type
## - the engine `run` paths: launch values bind as buffers or pack as byte chunks
##
## Crossing rules:
## - explosion: values explode into dynamic scalars in field order, output arg first,
##   one device param per scalar
## - classification: pointer values and backend device handles bind as buffers, other
##   scalars pack as bytes
## - static args: subtrees fully determined by their type cross as no bytes, the device
##   reads the value from the type
##
##   Fully determined means a `static` type, a typedesc application (`Int[128]`), an array
##   of such elements, or a composite of those
##
## Byte rules:
## - widths: chunks pack at the codegen-lowered scalar width
##   (`int`/`uint` lower to 32-bit), from toGpuTypeKind, the codegen's own scalar lowering
## - chunk layout: byte chunks pack contiguously into one blob, one chunk per exploded
##   device param
##
## Drift guard: the bridge descriptor and the engine-side layout are two projections
## of the rules here. Every wrap_and_run launch checks them against each other
## at compile time (checkedKernelArgLayout), a divergence is a compile-time error, never a misbound launch.

import std/[macros, typetraits]
import ../codegen/ir/gpu_types
import ../codegen/ir/gpu_type_constructors

export gpu_types, toGpuTypeKind

# ═════════════════════════════════════════════════════════════════════════
# ▸ Contract types
# ═════════════════════════════════════════════════════════════════════════

type
  KernelArgKind* = enum
    kaBuffer   ## device buffer binding: a pointer value or a backend handle
    kaBytes    ## by-value bytes at the crucible-lowered width
    kaStatic   ## compile-time constant, crosses as no bytes

  KernelArg* = object
    kind*: KernelArgKind
    irKind*: GpuTypeKind ## kaBytes: the codegen-lowered IR scalar kind
    count*: int          ## kaBytes of an array: the element count

  KernelArgLayout* = object
    ## Flattened kernel arg list in explosion order, output arg first.
    ## Static subtrees record nothing, they cross as no bytes.
    args*: seq[KernelArg]

# ═════════════════════════════════════════════════════════════════════════
# ▸ Widths
# ═════════════════════════════════════════════════════════════════════════

func kernelArgByteWidth*(k: GpuTypeKind, backend: static BackendKind): int =
  ## Byte width of an IR scalar kind in `backend`'s kernel ABI, the same
  ## lowering the codegen applies to kernel params (toGpuTypeKind):
  ## - `int`/`uint` lower to 32-bit
  ## - `bool` lowers to a 4-byte int everywhere but CUDA, where the C++
  ##   `bool` param is 1 byte
  case k
  of gtBool:
    when backend == bkCuda: result = 1
    else: result = 4
  of gtUint8: result = 1
  of gtUint16, gtInt16, gtFloat16, gtBf16: result = 2
  of gtUint32, gtInt32, gtFloat32: result = 4
  of gtUint64, gtInt64, gtFloat64, gtSize_t: result = 8
  else:
    raiseAssert "kernelArgByteWidth: not a scalar IR kind: " & $k

func kernelArgIrKind*[T](): GpuTypeKind {.compileTime.} =
  ## Returns the IR scalar kind the codegen lowers `T` to, matching
  ## toGpuTypeKind: `int`/`uint` lower to 32-bit.
  ##
  ## Every wrap_and_run launch check compares this chain
  ## against the AST walk's toGpuTypeKind.
  when T is bool: gtBool
  elif T is int: gtInt32
  elif T is uint: gtUint32
  elif T is uint8: gtUint8
  elif T is int16: gtInt16
  elif T is uint16: gtUint16
  elif T is int32: gtInt32
  elif T is uint32: gtUint32
  elif T is int64: gtInt64
  elif T is uint64: gtUint64
  elif T is float32: gtFloat32
  elif T is float64: gtFloat64
  elif T is SomeOrdinal and not (T is char):
    # enums are the ordinals left once the integer branches matched, they
    # resolve to uint32 storage in the codegen (resolveType, resolvers.nim)
    gtUint32
  else: gtInvalid

# ═════════════════════════════════════════════════════════════════════════
# ▸ AST domain (macro time): the bridge side
# ═════════════════════════════════════════════════════════════════════════

proc isStaticKernelArg*(t: NimNode): bool =
  ## True when the arg type `t` is fully determined by the type:
  ## - a `static` type or a typedesc application (`Int[128]`)
  ## - an array of such elements, the bounds are part of the type
  ## - an object or tuple built only from those
  ## Such args cross the kernel boundary as no bytes, the device reads
  ## the value from the type.
  case t.kind
  of nnkVarTy:
    error "kernel args: var fields are not supported: " & t.repr
  of nnkStaticTy:
    result = true
  of nnkBracketExpr:
    if t[0].typeKind == ntyTypeDesc:
      result = true
    elif t[0].typeKind == ntyArray:
      # the bounds are part of the type, always compile-time, the element
      # type alone decides
      result = isStaticKernelArg(t[^1])
    else:
      result = true
      for e in t[1 ..< t.len]:
        if not isStaticKernelArg(e):
          return false
  of nnkCharLit..nnkNilLit:
    result = true
  of nnkSym:
    if t.typeKind == ntyArray:
      # an array alias is a bare symbol, classify the resolved form
      result = isStaticKernelArg(t.getTypeImpl())
    else:
      result = false
  of nnkTupleTy:
    result = true
    for d in t:
      if d.kind == nnkIdentDefs and not isStaticKernelArg(d[^2]):
        return false
  of nnkPar, nnkTupleConstr:
    result = true
    for e in t:
      if not isStaticKernelArg(e):
        return false
  else:
    result = false

proc kernelArgArrayCount*(bounds: NimNode): int =
  ## Element count of an array type's bounds node: a plain literal counts
  ## elements (`array[3, T]`), a `lo .. hi` range includes both ends
  case bounds.kind
  of nnkIntLit..nnkUInt64Lit:
    result = int(bounds.intVal)
  of nnkRange:
    result = int(bounds[1].intVal - bounds[0].intVal) + 1
  of nnkInfix:
    doAssert $bounds[0] == "..", "kernel args: unsupported array bounds " & bounds.repr
    result = int(bounds[2].intVal - bounds[1].intVal) + 1
  else:
    error "kernel args: unsupported array bounds " & bounds.repr

proc constantKernelValue*(t: NimNode): NimNode =
  ## Returns the compile-time value of a static kernel arg type:
  ## - typedesc applications construct from the type (`Int[128]()` calls)
  ## - tuples nest element values
  ## - static arrays build N-element bracket literals of element constants
  if t.kind == nnkSym and t.typeKind == ntyArray:
    # an array alias is a bare symbol, build from the resolved form
    return constantKernelValue(t.getTypeImpl())
  if t.kind == nnkBracketExpr and t[0].typeKind == ntyArray:
    var r = nnkBracket.newTree()
    for _ in 1 .. kernelArgArrayCount(t[1]):
      r.add(constantKernelValue(t[^1]))
    return r
  if t.kind == nnkBracketExpr:
    return nnkCall.newTree(t.copy())
  if t.kind in {nnkTupleTy, nnkPar, nnkTupleConstr}:
    var r = nnkTupleConstr.newTree()
    if t.kind == nnkTupleTy:
      for d in t:
        if d.kind == nnkIdentDefs:
          r.add(constantKernelValue(d[^2]))
    else:
      for e in t:
        r.add(constantKernelValue(e))
    return r
  error "constantKernelValue: not a constant type node: " & t.repr

proc isBindManyArg*(t: NimNode): bool =
  ## True when the formal type is a `distinct` typeclass, such formals bind by value type
  if t.kind == nnkDistinctTy: return true
  if t.kind == nnkSym and t.getTypeInst().kind == nnkDistinctTy: return true
  return false

proc hasKernelPtrTy*(t: NimNode): bool =
  ## True when the type-impl node holds a pointer leaf anywhere
  if t.kind in {nnkPtrTy, nnkRefTy}:
    return true
  for c in t:
    if hasKernelPtrTy(c):
      return true
  return false

proc recordKernelArgLeaf*(t: NimNode, layout: var KernelArgLayout) =
  ## Records one launch arg's layout entry:
  ## - pointers bind as buffers
  ## - dynamic arrays cross as one gtArray chunk of their element count
  ## - other scalars pack as bytes at the codegen-lowered IR kind
  if t.typeKind in {ntyPtr, ntyPointer}:
    layout.args.add(KernelArg(kind: kaBuffer))
  elif t.typeKind == ntyArray:
    let impl = if t.kind == nnkBracketExpr: t else: t.getTypeImpl()
    layout.args.add(KernelArg(kind: kaBytes, irKind: gtArray,
                              count: kernelArgArrayCount(impl[1])))
  elif t.typeKind == ntyEnum:
    # enums resolve to uint32 storage in the codegen (resolveType, resolvers.nim)
    layout.args.add(KernelArg(kind: kaBytes, irKind: gtUint32))
  else:
    layout.args.add(KernelArg(kind: kaBytes, irKind: toGpuTypeKind(t.typeKind)))

proc reconstructKernelArg*(prefix: string, t, impl: NimNode, params: NimNode, layout: var KernelArgLayout): NimNode =
  ## Bridge-body value of type `t` (impl `impl` = `t.getTypeImpl()`)
  ## assembled from one bridge parameter per dynamic kernel arg, named
  ## `cru<prefix>...`, recording the same explosion into `layout`:
  ## - a static subtree rebuilds from the type, crossing as no bytes
  ## - a pointer leaf becomes one bridge param, the device binds a buffer
  ## - a dynamic scalar becomes one bridge param at the crucible-lowered width
  ## - an object or tuple explodes into its fields, recursion per field
  if isStaticKernelArg(t):
    return constantKernelValue(t)
  case impl.kind
  of nnkObjectTy:
    result = nnkObjConstr.newTree(t.copy())
    for k in 2 ..< impl.len:
      let recList = impl[k]
      doAssert recList.kind in {nnkRecList, nnkIdentDefs},
        "kernel args: unsupported object field node " & recList.repr
      let defs = if recList.kind == nnkRecList: recList else:
          nnkRecList.newTree(recList)
      for d in defs:
        doAssert d.kind == nnkIdentDefs,
          "kernel args: unsupported field " & d.repr
        let fldType = d[^2]
        if isStaticKernelArg(fldType):
          discard
        else:
          let fldImpl = fldType.getTypeImpl()
          doAssert fldImpl.kind notin {nnkObjectTy, nnkTupleTy, nnkPar, nnkTupleConstr} or
              not hasKernelPtrTy(fldImpl),
            "kernel args: nested pointer structs are not supported: " & t.repr
        for m in 0 ..< d.len - 2:
          result.add(nnkExprColonExpr.newTree(d[m].copy(),
            reconstructKernelArg(prefix & "_" & $d[m], fldType, fldType.getTypeImpl(), params, layout)))
  of nnkTupleTy, nnkPar, nnkTupleConstr:
    result = nnkTupleConstr.newTree()
    var i = 0
    for e in impl:
      if e.kind == nnkIdentDefs:
        for m in 0 ..< e.len - 2:
          result.add(reconstructKernelArg(prefix & "_" & $e[m], e[^2], e[^2].getTypeImpl(), params, layout))
      else:
        result.add(reconstructKernelArg(prefix & "_F" & $i, e, e.getTypeImpl(), params, layout))
      inc i
  else:
    # scalar, pointer or array leaf: one launch parameter
    let pname = ident("cru" & prefix)
    params.add(nnkIdentDefs.newTree(pname, t.copy(), newEmptyNode()))
    recordKernelArgLeaf(t, layout)
    result = pname

proc kernelArgLayoutAst*(l: KernelArgLayout): NimNode =
  ## Returns the compile-time literal of a kernel arg layout, emitted next
  ## to the bridge source and passed to `run` as the launch descriptor
  result = newLit(l)

# ═════════════════════════════════════════════════════════════════════════
# ▸ Typed domain: the engine side
# ═════════════════════════════════════════════════════════════════════════

func isStaticKernelArg*[T](): bool {.compileTime.} =
  ## True when the value type `T` is fully determined by the type:
  ## - an object or tuple whose every field is itself static
  ## - an array of such elements, the bounds are part of the type
  ##
  ## Such values cross the kernel boundary as no bytes.
  when T is object|tuple:
    result = true
    for f in fields(default(T)):
      if not isStaticKernelArg[typeof(f)]():
        return false
  elif T is array:
    # the bounds are part of the type, the element type alone decides
    when default(T).len == 0:
      result = true
    else:
      result = isStaticKernelArg[typeof(default(T)[0])]()
  else:
    result = false

func appendKernelArgEntries*[T, H](layout: var KernelArgLayout) {.compileTime.} =
  ## Records the layout entries of one launch value of type `T`:
  ## - device handles of type `H` and pointer values bind as buffers
  ## - composites explode per field in field order
  ## - static subtrees record nothing
  ##
  ## Dynamic arrays cross as one gtArray chunk of their element count, other
  ## scalars pack as bytes at the crucible-lowered IR kind.
  when T is H:
    layout.args.add(KernelArg(kind: kaBuffer))
  elif T is ptr or T is pointer:
    layout.args.add(KernelArg(kind: kaBuffer))
  elif T is object|tuple:
    when not isStaticKernelArg[T]():
      for f in fields(default(T)):
        appendKernelArgEntries[typeof(f), H](layout)
  elif T is array:
    when not isStaticKernelArg[T]():
      layout.args.add(KernelArg(kind: kaBytes, irKind: gtArray,
                                count: default(T).len))
  else:
    layout.args.add(KernelArg(kind: kaBytes, irKind: kernelArgIrKind[T]()))

func kernelArgLayout*[O, A, H](): KernelArgLayout {.compileTime.} =
  ## Engine-side layout of one launch: `output` first, then the args tuple
  ## of type `A`, exploded per the kernel arg contract.
  ##
  ## Args:
  ## - `H` is the backend device handle type, handles bind as one buffer arg each
  ## - the bridge descriptor has no handle vocabulary, compare against it with `H = void`
  appendKernelArgEntries[O, H](result)
  for f in fields(default(A)):
    appendKernelArgEntries[typeof(f), H](result)

func kernelArgLayoutMismatch*(own, given: KernelArgLayout): string {.compileTime.} =
  ## Returns the first divergence between engine-side layout and bridge descriptor,
  ## for the compile-time check's error message
  result = "arg count " & $own.args.len & " vs " & $given.args.len
  let n = min(own.args.len, given.args.len)
  for i in 0 ..< n:
    if own.args[i] != given.args[i]:
      result = "arg " & $i & ": engine binding " & $own.args[i].kind &
        (if own.args[i].kind == kaBytes: "/" & $own.args[i].irKind else: "") &
        " vs bridge descriptor " & $given.args[i].kind &
        (if given.args[i].kind == kaBytes: "/" & $given.args[i].irKind else: "")
      return

proc checkedKernelArgLayout*[O, A, H](
    provided: static KernelArgLayout, backend: static BackendKind): KernelArgLayout =
  ## Engine-side layout of a launch:
  ## - wrap_and_run launches pass the bridge descriptor, both are compared
  ##   at compile time, a mismatch is a compile-time error
  ## - direct runs pass the empty layout and bind from it alone
  const own = kernelArgLayout[O, A, H]()
  when provided.args.len > 0 and own != provided:
    {.error: "kernel arg binding diverges from the bridge descriptor: " &
        kernelArgLayoutMismatch(own, provided) & "."}
  own

# ─────────────────────────────────────────────────────────────────────────
# ▸ Runtime binder
# ─────────────────────────────────────────────────────────────────────────

type
  ArgSlot*[H] = object
    ## One launch arg as the backend API consumes it, in explosion order
    isBuffer*: bool
    buffer*: H     ## isBuffer: the device handle value
    offset*: int   ## else: byte offset of the chunk in `bytes`
    width*: int    ## else: crucible-lowered byte width

  KernelBinding*[H] = object
    ## Launch values bound per the kernel arg contract: one slot per
    ## exploded device arg in explosion order, byte chunks packed into
    ## `bytes`. Backends deliver from the slots, the blob layout is
    ## theirs (alignment included).
    slots*: seq[ArgSlot[H]]
    bytes*: seq[byte]

proc appendKernelArgBytes*[T](bytes: var seq[byte], v: T, W: static int) =
  ## Appends one scalar value at the crucible-lowered width `W`:
  ## - values wider than the device type truncate, Nim `int`/`uint`
  ##   lower to 32-bit
  ## - matching widths copy the raw value bytes
  when T is bool:
    when W == 1:
      let x = uint8(v)
      bytes.add(cast[ptr array[1, byte]](unsafeAddr x)[])
    else:
      let x = int32(v)
      bytes.add(cast[ptr array[4, byte]](unsafeAddr x)[])
  elif W < sizeof(T):
    when W == 4:
      when T is SomeSignedInt:
        let x = int32(v)
      else:
        let x = uint32(v)
      bytes.add(cast[ptr array[4, byte]](unsafeAddr x)[])
    else:
      {.error: "kernel arg bytes: unsupported narrowing from " & $T & " to " & $W & " bytes".}
  else:
    static: doAssert W == sizeof(T), "kernel arg bytes: width mismatch"
    bytes.add(cast[ptr array[W, byte]](unsafeAddr v)[])

proc appendKernelValue*[H](b: var KernelBinding[H], layout: KernelArgLayout,
                           backend: static BackendKind, pos: var int, v: auto,
                           ptrAsBuffer: static bool, argAlign: static int) =
  ## Binds one launch value (or value subtree) per the kernel arg contract,
  ## consuming `layout` entries in explosion order:
  ## - device handles bind as buffers
  ## - on Metal pointer leaves bind as buffers, the raw value is the buffer identity,
  ##   elsewhere they cross as their raw bytes
  ## - composites explode per field, static subtrees cross as no bytes
  ##
  ## Dynamic arrays cross as one raw chunk, element width times count.
  ## Scalars pack one chunk at the crucible-lowered width.
  when typeof(v) is H:
    doAssert pos < layout.args.len and layout.args[pos].kind == kaBuffer,
      "kernel arg binding: expected a buffer arg"
    b.slots.add(ArgSlot[H](isBuffer: true, buffer: v))
    inc pos
  elif typeof(v) is ptr or typeof(v) is pointer:
    doAssert pos < layout.args.len and layout.args[pos].kind == kaBuffer,
      "kernel arg binding: expected a buffer arg"
    when ptrAsBuffer:
      b.slots.add(ArgSlot[H](isBuffer: true, buffer: cast[H](v)))
    else:
      b.slots.add(ArgSlot[H](isBuffer: false, offset: b.bytes.len,
                             width: sizeof(pointer)))
      appendKernelArgBytes(b.bytes, v, sizeof(pointer))
    inc pos
  elif typeof(v) is object|tuple:
    when not isStaticKernelArg[typeof(v)]():
      var copy = v
      for f in fields(copy):
        appendKernelValue[H](b, layout, backend, pos, f, ptrAsBuffer, argAlign)
  elif typeof(v) is array:
    when isStaticKernelArg[typeof(v)]():
      discard
    else:
      doAssert pos < layout.args.len and layout.args[pos].kind == kaBytes and
        layout.args[pos].irKind == gtArray and
        layout.args[pos].count == v.len,
        "kernel arg binding: expected an array arg of " & $v.len & " elements"
      b.slots.add(ArgSlot[H](isBuffer: false, offset: b.bytes.len,
                             width: sizeof(v)))
      appendKernelArgBytes(b.bytes, v, sizeof(v))
      inc pos
  else:
    doAssert pos < layout.args.len and layout.args[pos].kind == kaBytes,
      "kernel arg binding: expected a byte arg"
    const w = kernelArgByteWidth(kernelArgIrKind[typeof(v)](), backend)
    let off = (b.bytes.len + argAlign - 1) div argAlign * argAlign
    b.bytes.setLen(off)
    b.slots.add(ArgSlot[H](isBuffer: false, offset: off, width: w))
    appendKernelArgBytes(b.bytes, v, w)
    inc pos

proc bindKernelArgs*[H](b: var KernelBinding[H], layout: KernelArgLayout,
                        backend: static BackendKind, output: auto, args: tuple,
                        ptrAsBuffer: static bool, argAlign: static int) =
  ## Binds `output` then the args tuple per the kernel arg contract,
  ## consuming `layout` in lockstep, the walk must land exactly at the layout end.
  ##
  ## Args:
  ## - `ptrAsBuffer`: pointer leaves bind as buffers on Metal
  ##   (the raw value is the buffer identity) or cross as raw bytes elsewhere
  ## - `argAlign`: byte alignment of each chunk in `bytes`
  ##   (1, or 256 for WebGPU's storage-buffer slots)
  var pos = 0
  appendKernelValue[H](b, layout, backend, pos, output, ptrAsBuffer, argAlign)
  var argsCopy = args
  for f in fields(argsCopy):
    appendKernelValue[H](b, layout, backend, pos, f, ptrAsBuffer, argAlign)
  doAssert pos == layout.args.len,
    "kernel arg binding: bound " & $pos & " args, layout holds " & $layout.args.len
