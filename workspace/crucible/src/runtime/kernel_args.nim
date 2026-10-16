# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/Apache-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Kernel argument binding: how launch values cross a kernel boundary.
##
## One rule: bind the args tuple to the kernel's declared signature.
## Signature sources, the rule enforced against each:
## - the `{.global.}` bridge (wrap_and_run launches): composites explode
##   into dynamic scalars in field order, output arg first, one device
##   param per scalar, exactly as the bridge descriptor records
## - the kernel's own declaration, reflected at pipeline-state build
##   time on Metal (argument reflection): the declared layout binds
##   a composite as one chunk placed at the declared member offsets
## - backends without reflection bind whole-value: the composite packs
##   flat in field order at the crucible-lowered widths when it holds
##   no buffer leaf
##
## Classification, identical under every source:
## - a pointer value or a backend device handle binds as a buffer
## - any other scalar packs as bytes
## - a static subtree, fully determined by its type, crosses as no bytes
##
##   Fully determined means a `static` type, a typedesc application
##   (`Int[128]`), an array of such elements, or a composite of those
##
## Byte rules:
## - widths: chunks pack at the codegen-lowered scalar width
##   (`int`/`uint` lower to 32-bit) per toGpuTypeKind, the codegen's
##   own scalar lowering
## - chunk layout: byte chunks pack contiguously into one blob, one
##   chunk per exploded device param, a whole-value chunk packs
##   its fields flat in field order
##
## Drift guard, projections of the rules here that must agree:
## - bridge descriptor vs engine-side layout: compared at compile time
##   (checkedKernelArgLayout), a divergence is a compile-time error,
##   never a misbound launch
## - engine-side layout vs the Metal-declared signature: compared at
##   launch time (failLoud) and bound from the declaration, so a bridge
##   or kernel divergence is a loud launch failure.

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
    kaStruct   ## whole-value chunk, one per declared kernel arg: fields packed
               ## flat in field order at the crucible-lowered widths
    kaStatic   ## compile-time constant, crosses as no bytes

  KernelArgMember* = object
    ## One declared scalar slot of a kaStruct chunk's flattened member layout:
    ## a struct member, or one element of an array member
    irKind*: GpuTypeKind ## the declared scalar kind, mapped from the backend's
                         ## own type enumeration
    offset*: int         ## declared byte offset in the chunk
    width*: int          ## declared byte width

  KernelArg* = object
    kind*: KernelArgKind
    irKind*: GpuTypeKind ## kaBytes: the codegen-lowered IR scalar kind
    count*: int          ## kaBytes of an array: the element count
    width*: int          ## kaBytes/kaStruct: declared byte width when the entry
                         ## reflects the kernel's signature, else 0, the ABI
                         ## width derives from irKind at bind time
    index*: int          ## the backend's declared arg index, from the kernel
                         ## reflection when the layout carries one, else 0
    members*: seq[KernelArgMember] ## kaStruct: the flattened declared member
                                   ## layout (signature reflection), empty
                                   ## means flat field-order packing

  KernelArgGranularity* = enum
    ## Binding granularity of a composite launch value, selected by
    ## the launch path (each member documents its selection rule)
    kagExploded
      ## composites explode per field in field order, one device param
      ## per leaf (wrap_and_run launches)
    kagWholeValue
      ## a composite without a buffer leaf binds as one chunk packed flat
      ## at the crucible-lowered field widths (direct runs, no reflection)

  KernelArgLayout* = object
    ## Flattened kernel arg list in explosion order, output arg first.
    ## Static subtrees record nothing, they cross as no bytes.
    args*: seq[KernelArg]

proc `$`*(a: KernelArg): string =
  ## Compact one-line form for loud-failure messages:
  ## `kaBytes/gtInt32 w4` or `kaStruct w72[3 members]`
  result = $a.kind
  if a.kind == kaBytes:
    result &= "/" & $a.irKind
  if a.width > 0:
    result &= " w" & $a.width
  if a.members.len > 0:
    result &= "[" & $a.members.len & " members]"

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

func hasBufferLeaf*[T, H](): bool {.compileTime.} =
  ## Returns true when the value type `T` holds a pointer or backend
  ## device handle leaf anywhere. Such a composite explodes per field
  ## even in whole-value mode, its buffer leaves bind as buffers.
  when T is H or T is ptr or T is pointer or T is ref:
    true
  elif T is object|tuple:
    for f in fields(default(T)):
      if hasBufferLeaf[typeof(f), H]():
        return true
  elif T is array:
    when default(T).len > 0:
      hasBufferLeaf[typeof(default(T)[0]), H]()
    else:
      false
  else:
    false

func kernelChunkByteWidth*[T, H](backend: static BackendKind): int {.compileTime.} =
  ## Returns the whole-value chunk byte width for a value of type `T`
  ## in `backend`'s kernel ABI: scalars at the crucible-lowered width,
  ## composites recurse flat in field order, arrays and pointer
  ## leaves copy raw
  ##
  ## `H` is the backend device handle type, `void` when the type has no
  ## handle vocabulary. Pointer and handle leaves never occur inside
  ## whole-value chunks, they are sized only so the width stays
  ## computable over any value tree.
  when T is H or T is ptr or T is pointer or T is ref:
    sizeof(pointer)
  elif T is object|tuple:
    for f in fields(default(T)):
      result += kernelChunkByteWidth[typeof(f), H](backend)
  elif T is array:
    sizeof(T)
  else:
    kernelArgByteWidth(kernelArgIrKind[T](), backend)

func appendKernelArgEntries*[T, H](layout: var KernelArgLayout,
                                   granularity: static KernelArgGranularity,
                                   backend: static BackendKind) {.compileTime.} =
  ## Records the layout entries of one launch value of type `T`:
  ## - device handles of type `H` and pointer values bind as buffers
  ## - composites explode per field in field order (kagExploded), or bind
  ##   as one chunk when they hold no buffer leaf (kagWholeValue, direct runs)
  ## - static subtrees record nothing
  ##
  ## Dynamic arrays cross as one gtArray chunk of their element count,
  ## other scalars pack as bytes at the crucible-lowered IR kind.
  when T is H:
    layout.args.add(KernelArg(kind: kaBuffer))
  elif T is ptr or T is pointer:
    layout.args.add(KernelArg(kind: kaBuffer))
  elif T is object|tuple:
    when isStaticKernelArg[T]():
      discard
    elif granularity == kagWholeValue and not hasBufferLeaf[T, H]():
      layout.args.add(KernelArg(kind: kaStruct,
                                width: kernelChunkByteWidth[T, H](backend)))
    else:
      for f in fields(default(T)):
        appendKernelArgEntries[typeof(f), H](layout, granularity, backend)
  elif T is array:
    when not isStaticKernelArg[T]():
      layout.args.add(KernelArg(kind: kaBytes, irKind: gtArray,
                                count: default(T).len))
  else:
    layout.args.add(KernelArg(kind: kaBytes, irKind: kernelArgIrKind[T]()))

func kernelArgLayout*[O, A, H](granularity: static KernelArgGranularity, backend: static BackendKind): KernelArgLayout {.compileTime.} =
  ## Engine-side layout of one launch: `output` first, then the args tuple
  ## of type `A`, classified per the kernel arg contract at `granularity`.
  ##
  ## Args:
  ## - `H` is the backend device handle type, handles bind as one buffer arg each
  ## - the bridge descriptor has no handle vocabulary, compare against it with `H = void`
  appendKernelArgEntries[O, H](result, granularity, backend)
  for f in fields(default(A)):
    appendKernelArgEntries[typeof(f), H](result, granularity, backend)

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
  ## Engine-side layout of a launch, the granularity selected by the launch
  ## path:
  ## - wrap_and_run launches pass the bridge descriptor, both exploded
  ##   layouts are compared at compile time, a mismatch is a compile-time error
  ## - direct runs on backends without signature reflection pass the empty
  ##   layout and bind whole-value, backends that reflect the kernel
  ##   signature bind from the reflected layout instead
  const own = kernelArgLayout[O, A, H](
    when provided.args.len > 0: kagExploded else: kagWholeValue, backend)
  when provided.args.len > 0 and own != provided:
    {.error: "kernel arg binding diverges from the bridge descriptor: " &
        kernelArgLayoutMismatch(own, provided) & "."}
  own

proc declaredIrKindCompatible(given, declared: GpuTypeKind): bool =
  ## A Nim scalar binds a declared slot of its own IR kind, plus two
  ## normalized pairs: a Nim bool binds the 4-byte `int` the codegen
  ## emits for bool kernel params, a Nim `size_t` binds the 64-bit
  ## unsigned kind behind MSL `size_t`
  given == declared or
    (given == gtBool and declared == gtInt32) or
    (given == gtSize_t and declared == gtUint64)

proc declaredLaunchLayout*(given, declared: KernelArgLayout): KernelArgLayout =
  ## Reconciles the value projection `given` (what the args tuple gives)
  ## against the declared signature `declared` (the backend's reflected view),
  ## position by position.
  ## The result binds at the declared indices, widths and member offsets:
  ## - counts must be equal
  ## - a pointer/handle entry binds any declared entry: on Metal every
  ##   arg is a buffer arg reporting its pointee type, and a constant
  ##   reference accepts a handle too
  ## - a byte entry binds a declared byte entry of a compatible IR kind,
  ##   or a one-member declared struct of a compatible member kind, where
  ##   the chunk is the member
  ## - a whole-value struct entry binds a declared struct entry whose
  ##   member kinds check at bind time, anything else fails loudly:
  ##   a declared scalar under a composite value, a declared
  ##   multi-member struct under a scalar value
  ##
  ## Usage:
  ##   let launch = declaredLaunchLayout(given, kernelDeclaredLayout)
  doAssert given.args.len == declared.args.len,
    "kernel arg binding: the launch binds " & $given.args.len &
    " args (" & $given.args & "), the kernel declares " &
    $declared.args.len & " (" & $declared.args & ")"
  for i in 0 ..< declared.args.len:
    let g = given.args[i]
    let d = declared.args[i]
    case g.kind
    of kaBuffer:
      # any declared entry accepts a handle
      discard
    of kaBytes:
      case d.kind
      of kaBytes:
        doAssert declaredIrKindCompatible(g.irKind, d.irKind),
          "kernel arg binding: arg " & $i & " binds " & $g &
          ", the kernel declares " & $d
      of kaStruct:
        doAssert d.members.len == 1,
          "kernel arg binding: arg " & $i & " binds one scalar " & $g &
          ", the kernel declares a " & $d.members.len & "-member struct " & $d
        doAssert declaredIrKindCompatible(g.irKind, d.members[0].irKind),
          "kernel arg binding: arg " & $i & " binds " & $g &
          ", the kernel declares member kind " & $d.members[0].irKind
      else:
        doAssert false,
          "kernel arg binding: arg " & $i & " binds " & $g &
          ", the kernel declares " & $d
    of kaStruct:
      doAssert d.kind == kaStruct,
        "kernel arg binding: arg " & $i & " binds a composite " & $g &
        ", the kernel declares " & $d
    of kaStatic:
      # static subtrees record no entry, a given layout never holds one
      doAssert false, "kernel arg binding: unexpected static entry at arg " & $i
    if g.kind == kaBytes and d.kind == kaStruct:
      # one scalar in a one-member declared struct: the chunk is the member
      result.args.add(KernelArg(kind: kaBytes, irKind: d.members[0].irKind,
                                width: d.members[0].width, index: d.index))
    else:
      result.args.add(d)

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
    index*: int    ## the backend's declared arg index: from the kernel
                   ## reflection when the layout carries one, else
                   ## the explosion position, backends that address args
                   ## sequentially may ignore it

  KernelBinding*[H] = object
    ## Launch values bound per the kernel arg contract: one slot per
    ## exploded device arg in explosion order, byte chunks packed into
    ## `bytes`. Backends deliver from the slots, the blob layout is
    ## theirs (alignment included).
    slots*: seq[ArgSlot[H]]
    bytes*: seq[byte]

proc putKernelArgBytes*[T](bytes: var seq[byte], off: int, v: T, w: int) =
  ## Stores one scalar value at byte offset `off`, lowered to `w` bytes:
  ## - bools store as a 1-byte bool or a 4-byte int
  ## - values wider than `w` truncate, Nim `int`/`uint` lower to 32-bit
  ## - matching widths copy the raw value bytes
  ##
  ## `off` must be within `bytes`, the caller sizes the blob.
  when T is bool:
    if w == 1:
      bytes[off] = byte(v)
    else:
      let x = int32(v)
      copyMem(bytes[off].addr, x.unsafeAddr, 4)
  else:
    if w < sizeof(T):
      when T is SomeSignedInt:
        let x = int32(v)
        copyMem(bytes[off].addr, x.unsafeAddr, 4)
      elif T is SomeUnsignedInt:
        let x = uint32(v)
        copyMem(bytes[off].addr, x.unsafeAddr, 4)
      else:
        doAssert false, "kernel arg bytes: unsupported narrowing from " & $T & " to " & $w & " bytes"
    else:
      doAssert w == sizeof(T), "kernel arg bytes: width mismatch, " & $w & " vs sizeof(" & $T & ")"
      copyMem(bytes[off].addr, v.unsafeAddr, w)

proc appendKernelArgBytes*[T](bytes: var seq[byte], v: T, W: static int) =
  ## Appends one scalar value at the crucible-lowered width `W`, see
  ## putKernelArgBytes for the lowering
  bytes.setLen(bytes.len + W)
  putKernelArgBytes(bytes, bytes.len - W, v, W)

proc appendWholeValueBytes*[T](bytes: var seq[byte], backend: static BackendKind, v: T) =
  ## Appends one whole-value chunk to `bytes`, one value of type `T`:
  ## - scalars pack at the crucible-lowered width (`int`/`uint` lower to 32-bit)
  ## - composites recurse flat in field order
  ## - arrays copy raw
  when T is object|tuple:
    var copy = v
    for f in fields(copy):
      appendWholeValueBytes(bytes, backend, f)
  else:
    const w = when T is array or T is ptr or T is pointer or T is ref: sizeof(T)
              else: kernelArgByteWidth(kernelArgIrKind[T](), backend)
    appendKernelArgBytes(bytes, v, w)

proc packDeclaredMembers[H, T](b: var KernelBinding[H], base: int, v: T,
                               members: seq[KernelArgMember], mi: var int,
                               backend: static BackendKind) =
  ## Packs value `v`'s field tree against a declared member layout
  ## flattened in field order:
  ## - scalars lower one member each at its declared offset
  ## - arrays recurse per element
  ## - composites recurse per field
  ## - a value kind that cannot bind the declared member kind fails loudly
  ##
  ## `mi` must land exactly at `members.len`, the caller checks.
  when T is object|tuple:
    var copy = v
    for f in fields(copy):
      packDeclaredMembers(b, base, f, members, mi, backend)
  elif T is array:
    for e in v:
      packDeclaredMembers(b, base, e, members, mi, backend)
  else:
    doAssert mi < members.len,
      "kernel arg binding: the value tree holds more scalars than the declared member layout"
    let m = members[mi]
    when T is bool:
      doAssert m.irKind in {gtBool, gtInt32},
        "kernel arg binding: declared member kind " & $m.irKind & " does not bind a bool value"
    else:
      const vKind = kernelArgIrKind[T]()
      doAssert m.irKind == vKind,
        "kernel arg binding: declared member kind " & $m.irKind & " vs value kind " & $vKind
    putKernelArgBytes(b.bytes, base + m.offset, v, m.width)
    inc mi

proc appendKernelValue*[H](b: var KernelBinding[H], layout: KernelArgLayout,
                           backend: static BackendKind, pos: var int, v: auto,
                           ptrAsBuffer: static bool, argAlign: static int) =
  ## Binds one launch value (or value subtree) per the kernel arg contract,
  ## consuming `layout` entries in explosion order:
  ## - device handles bind as buffers
  ## - on Metal pointer leaves bind as buffers, the raw value is the buffer identity,
  ##   elsewhere they cross as their raw bytes
  ## - composites bind at the granularity the layout records: one kaStruct
  ##   chunk packed flat at the lowered field widths, or exploded per field,
  ##   static subtrees cross as no bytes
  ##
  ## Dynamic arrays cross as one raw chunk, element width times count.
  ## Scalars pack one chunk at the crucible-lowered width.
  when typeof(v) is H:
    # a device handle binds a buffer: on Metal every declared arg is a buffer
    # arg and constant references accept handles too, so any entry kind
    # binds (the kernel's pointee type is not host-checkable), elsewhere
    # only a declared buffer arg binds
    when ptrAsBuffer:
      doAssert pos < layout.args.len and layout.args[pos].kind in {kaBuffer, kaBytes, kaStruct},
        "kernel arg binding: no declared arg left for a device handle"
    else:
      doAssert pos < layout.args.len and layout.args[pos].kind == kaBuffer,
        "kernel arg binding: expected a buffer arg"
    b.slots.add(ArgSlot[H](isBuffer: true, index: layout.args[pos].index,
                           buffer: v))
    inc pos
  elif typeof(v) is ptr or typeof(v) is pointer:
    # a pointer value binds a buffer: on Metal every declared arg is a buffer
    # arg and constant references accept handles too, so any entry kind
    # binds (the kernel's pointee type is not host-checkable), elsewhere
    # only a declared buffer arg binds, the raw value crosses as its bytes
    when ptrAsBuffer:
      doAssert pos < layout.args.len and layout.args[pos].kind in {kaBuffer, kaBytes, kaStruct},
        "kernel arg binding: no declared arg left for a pointer value"
      b.slots.add(ArgSlot[H](isBuffer: true, index: layout.args[pos].index,
                             buffer: cast[H](v)))
    else:
      doAssert pos < layout.args.len and layout.args[pos].kind == kaBuffer,
        "kernel arg binding: expected a buffer arg"
      b.slots.add(ArgSlot[H](isBuffer: false, index: layout.args[pos].index,
                             offset: b.bytes.len, width: sizeof(pointer)))
      appendKernelArgBytes(b.bytes, v, sizeof(pointer))
    inc pos
  elif typeof(v) is object|tuple:
    when isStaticKernelArg[typeof(v)]():
      discard
    else:
      doAssert pos < layout.args.len,
        "kernel arg binding: no layout entry left for a composite arg"
      if layout.args[pos].kind == kaStruct:
        if layout.args[pos].members.len > 0:
          # declared member layout (the kernel's reflected signature): one
          # chunk at the declared byte width, each field lowered at
          # its declared offset, padding bytes stay zero
          let off = (b.bytes.len + argAlign - 1) div argAlign * argAlign
          b.bytes.setLen(off + layout.args[pos].width)
          b.slots.add(ArgSlot[H](isBuffer: false,
                                 index: layout.args[pos].index,
                                 offset: off, width: layout.args[pos].width))
          var mi = 0
          packDeclaredMembers(b, off, v, layout.args[pos].members, mi, backend)
          doAssert mi == layout.args[pos].members.len,
            "kernel arg binding: the value tree holds fewer scalars (" & $mi &
            ") than the declared member layout (" & $layout.args[pos].members.len & ")"
        else:
          # whole-value chunk: one blob at the crucible-lowered field widths,
          # the kernel reads it as one declared arg
          const w = kernelChunkByteWidth[typeof(v), H](backend)
          doAssert layout.args[pos].width == w,
            "kernel arg binding: struct chunk width " & $layout.args[pos].width &
            " vs " & $w
          let off = (b.bytes.len + argAlign - 1) div argAlign * argAlign
          b.bytes.setLen(off)
          b.slots.add(ArgSlot[H](isBuffer: false, index: layout.args[pos].index,
                                 offset: off, width: w))
          appendWholeValueBytes(b.bytes, backend, v)
        inc pos
      else:
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
      doAssert layout.args[pos].width in {0, sizeof(v)},
        "kernel arg binding: declared array width " & $layout.args[pos].width &
        " vs raw value width " & $sizeof(v)
      b.slots.add(ArgSlot[H](isBuffer: false, index: layout.args[pos].index,
                             offset: b.bytes.len, width: sizeof(v)))
      appendKernelArgBytes(b.bytes, v, sizeof(v))
      inc pos
  else:
    doAssert pos < layout.args.len and layout.args[pos].kind == kaBytes,
      "kernel arg binding: expected a byte arg, the declared layout holds " &
      $layout.args[pos]
    when typeof(v) is bool:
      # a Nim bool binds a declared bool slot or the 4-byte int slot
      # the codegen emits for bool kernel params
      doAssert layout.args[pos].irKind in {gtBool, gtInt32},
        "kernel arg binding: declared kind " & $layout.args[pos].irKind &
        " does not bind a bool value"
    else:
      const vKind = kernelArgIrKind[typeof(v)]()
      doAssert layout.args[pos].irKind == vKind,
        "kernel arg binding: declared kind " & $layout.args[pos].irKind &
        " vs value kind " & $vKind
    const abiW = kernelArgByteWidth(kernelArgIrKind[typeof(v)](), backend)
    let w = if layout.args[pos].width > 0: layout.args[pos].width else: abiW
    let off = (b.bytes.len + argAlign - 1) div argAlign * argAlign
    b.bytes.setLen(off + w)
    b.slots.add(ArgSlot[H](isBuffer: false, index: layout.args[pos].index,
                           offset: off, width: w))
    putKernelArgBytes(b.bytes, off, v, w)
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
