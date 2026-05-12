# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/macros,
  # Internal
  workspace/libtorch/src/raw_libtorch as F,
  workspace/libtorch/src/raw/support/[ast_utils, indexing_macros],
  workspace/libtorch/src/vecs/vecs

# Export Nim-friendly types (no C++ types leak past this boundary)
export F.ScalarKind, F.DeviceKind, F.Device, F.TensorOptions,
       F.Scalar, F.SomeTorchType, F.TorchComplex
# Indexing sugar
export F.`_`, F.ellipsis, F.`...`

# Generic sandwich with ArrayRef when fancy indexing via `[]` and `[]=` macro
export F.shape, F.`[]`, F.len

# #######################################################################
#
#                            Core Types
#
# #######################################################################

type Tensor* = ref object
  raw: TorchTensor

# Construction
# ----------------------------------------------------------

proc placementNew[T](p: ptr T): ptr T {.importcpp: "(new (#) '*0(@))", nodecl, discardable.}
  ## Default-construct an object at the given memory location via placement-new.

proc wrapTorchTensorImpl(a: sink TorchTensor): Tensor {.inline, nodestroy.} =
  new result
  placementNew(result.raw.addr)
  `=sink`(result.raw, a)

template wrapTorchTensor(body: untyped): untyped =
  when typeof(body) is TorchTensor:
    wrapTorchTensorImpl(body)
  elif typeof(body) is CppVector[TorchTensor]:
    let raws = block: body
    var tensors = newSeq[Tensor](len(raws))
    for i in 0 ..< len(raws):
      tensors[i] = wrapTorchTensorImpl(raws[i])
    tensors
  elif typeof(body) is CppTuple2[TorchTensor, TorchTensor]:
    let raws = block: body
    (wrapTorchTensorImpl(get(raws, 0)), wrapTorchTensorImpl(get(raws, 1)))
  else:
    body

# Existence
# ----------------------------------------------------------

func isDefined*(a: Tensor): bool {.inline.} =
  if a.isNil():
    return false
  return F.isDefined(a.raw)

# C++ Exception handling
# ----------------------------------------------------------

type LibTorchDefect* = object of Defect

template convertLibTorchExceptions(body: untyped): untyped =
  ## Catches C++ torch::Error and convert it to Nim exceptions
  ## so they don't leak into downstream Nim packages.
  ## and also are properly reported in CLI.
  when not defined(cpp) and defined(nimCheck):
    {.error: "You are running 'nim check' in C mode. It will misreport that C++ exceptions can't be caught because they aren't ref objects.".}

  try:
    body
  except TorchError as e:
    raise newException(
      LibTorchDefect,
      "\n❌ Caught low-level C++ libtorch exception:\n" &
      "───────────────────────────────────────────────────────────────────────────────\n" &
      $e.what() &
      "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )

# #######################################################################
#
#                         Debugging
#
# #######################################################################

proc `$`*(t: Tensor): string =
  "Tensor\n" & $(F.toCppString(t.raw))

proc print*(t: Tensor) {.sideeffect, inline.} =
  F.print(t.raw)

# #######################################################################
#
#                       Wrapping routines
#
# #######################################################################
#
# We want to ease wrapping as much as possible as there are hundreds
# of procedures to wrap. And this is a reference file for everything available.
#
# We don't want to pollute LLM agents context with redundancy when a call is
# transparently forwarded with just TorchTensor<->Tensor conversion.
# Or having {.inline.} everywhere.
#
# We want 4 things:
# 1. Tag all proc and func (func == proc {.noSIdeEffect.}) as {.inline.}
# 2. Convert torch::Error exceptions into Nim exceptions
# 3. Auto-wrap TorchTensor return values into Tensor
# 4. Auto-forward inputs to torch_tensors.nim overload, unwrapping Tensor->TorchTensor
#    iff there is no function body defined. I.e. we can always write the function body

template unwrapArg(arg: untyped): untyped =
  # Convert Nim -> libtorch
  when arg is Tensor:
    arg.raw
  elif arg is typedesc[SomeTorchType]:
    toScalarKind(arg)
  elif arg is varargs[int] or arg is openArray[int]:
    asTorchView(arg)
  elif arg is varargs[Tensor] or arg is openArray[Tensor]:
    block:
      var raws {.gensym.} = new(Vec[TorchTensor], arg.len)
      for i in 0 ..< arg.len:
        raws[i] = arg[i].raw
      asTorchView(raws)
  else:
    arg

{.experimental: "dynamicbindsym".}

proc autoForward(fnDef: NimNode): NimNode =
  ## Take a function signature for example
  ##
  ##   proc foo(a: Tensor, b: int): Tensor
  ##
  ## and transform it into a forwarding call to the underlying libtorch C++ call
  ##
  ##   proc foo(a: Tensor, b: int): Tensor =
  ##     convertLibTorchExceptions:
  ##       wrapTorchTensor:
  ##         foo(a.raw, b)
  fnDef.expectKind {nnkProcDef, nnkFuncDef}

  if fnDef.body.kind != nnkEmpty:
    return fnDef

  let fnName = fnDef.name

  result = fnDef
  result.addPragma ident"inline"

  # Call F.myFunction(...) to disambiguate when types can't
  var body = newCall(nnkDotExpr.newTree(ident"F", fnName))

  fnDef[3].expectKind nnkFormalParams
  for i in 1 ..< fnDef[3].len:
    # Skip arg 0 - the return type
    for j in 0 ..< fnDef[3][i].len - 2:
      # Handle proc(a, b: int = 1)
      body.add getAST(unwrapArg(fnDef[3][i][j]))

  body = getAst(wrapTorchTensor(body))
  body = getAst(convertLibTorchExceptions(body))

  result.body = body

  when false:
    # View proc signature.
    debugEcho "Rewrapped:\n",
          result.toStrLit(),
          "\n───────────────────────────────────────────────────────────────────────────────"

macro wrapLibtorch(body: untyped): untyped =
  result = newStmtList()

  for statement in body:
    if statement.kind notin {nnkProcDef, nnkFuncDef}:
      result.add statement
      continue

    result.add autoForward(statement)

# #######################################################################
#
#                          Array → Tensor
#
# #######################################################################

func toTensor*[T: SomeTorchType](oa: openarray[T]): Tensor {.inline.} =
  ## Convert a flat Nim array/seq to a 1D Tensor (owning copy).
  wrapTorchTensor(toTorchTensor(oa))

func toTensor*[T: seq | array](oa: openarray[T]): Tensor {.inline.} =
  ## Convert a nested Nim array/seq to a multi-dimensional Tensor (owning copy).
  wrapTorchTensor(toTorchTensor(oa))

wrapLibtorch:

  # #######################################################################
  #
  #                              Factory
  #
  # #######################################################################

  # empty
  func empty*(size: varargs[int]): Tensor
  func empty*(size: varargs[int], T: typedesc[SomeTorchType]): Tensor
  func empty*(size: varargs[int], scalarKind: ScalarKind): Tensor
  func empty*(size: varargs[int], device: DeviceKind): Tensor
  func empty*(size: varargs[int], options: TensorOptions): Tensor

  # zeros
  func zeros*(size: varargs[int]): Tensor
  func zeros*(size: varargs[int], T: typedesc[SomeTorchType]): Tensor
  func zeros*(size: varargs[int], scalarKind: ScalarKind): Tensor
  func zeros*(size: varargs[int], device: DeviceKind): Tensor
  func zeros*(size: varargs[int], options: TensorOptions): Tensor

  # ones
  func ones*(size: varargs[int]): Tensor
  func ones*(size: varargs[int], T: typedesc[SomeTorchType]): Tensor
  func ones*(size: varargs[int], scalarKind: ScalarKind): Tensor
  func ones*(size: varargs[int], device: DeviceKind): Tensor
  func ones*(size: varargs[int], options: TensorOptions): Tensor

  # full
  func full*(size: varargs[int], fillValue: Scalar): Tensor
  func full*(size: varargs[int], fillValue: Scalar, T: typedesc[SomeTorchType]): Tensor
  func full*(size: varargs[int], fillValue: Scalar, scalarKind: ScalarKind): Tensor
  func full*(size: varargs[int], fillValue: Scalar, device: DeviceKind): Tensor
  func full*(size: varargs[int], fillValue: Scalar, options: TensorOptions): Tensor

  # rand
  func rand*(size: varargs[int]): Tensor
  func rand*(size: varargs[int], T: typedesc[SomeTorchType]): Tensor
  func rand*(size: varargs[int], scalarKind: ScalarKind): Tensor
  func rand*(size: varargs[int], device: DeviceKind): Tensor
  func rand*(size: varargs[int], options: TensorOptions): Tensor

  # randn
  func randn*(size: varargs[int]): Tensor
  func randn*(size: varargs[int], T: typedesc[SomeTorchType]): Tensor
  func randn*(size: varargs[int], scalarKind: ScalarKind): Tensor
  func randn*(size: varargs[int], device: DeviceKind): Tensor
  func randn*(size: varargs[int], options: TensorOptions): Tensor

  # rand_like
  func rand_like*(a: Tensor): Tensor
  func rand_like*(a: Tensor, T: typedesc[SomeTorchType]): Tensor
  func rand_like*(a: Tensor, scalarKind: ScalarKind): Tensor
  func rand_like*(a: Tensor, device: DeviceKind): Tensor
  func rand_like*(a: Tensor, options: TensorOptions): Tensor

  # eye
  func eye*(n: int): Tensor
  func eye*(n: int, T: typedesc[SomeTorchType]): Tensor
  func eye*(n: int, scalarKind: ScalarKind): Tensor
  func eye*(n: int, device: DeviceKind): Tensor
  func eye*(n: int, options: TensorOptions): Tensor

  # arange
  func arange*(stop: Scalar): Tensor
  func arange*(stop: Scalar, T: typedesc[SomeTorchType]): Tensor
  func arange*(stop: Scalar, scalarKind: ScalarKind): Tensor
  func arange*(stop: Scalar, device: DeviceKind): Tensor
  func arange*(stop: Scalar, options: TensorOptions): Tensor
  func arange*(start, stop: Scalar): Tensor
  func arange*(start, stop: Scalar, T: typedesc[SomeTorchType]): Tensor
  func arange*(start, stop: Scalar, scalarKind: ScalarKind): Tensor
  func arange*(start, stop: Scalar, device: DeviceKind): Tensor
  func arange*(start, stop: Scalar, options: TensorOptions): Tensor
  func arange*(start, stop, step: Scalar): Tensor
  func arange*(start, stop, step: Scalar, T: typedesc[SomeTorchType]): Tensor
  func arange*(start, stop, step: Scalar, scalarKind: ScalarKind): Tensor
  func arange*(start, stop, step: Scalar, device: DeviceKind): Tensor
  func arange*(start, stop, step: Scalar, options: TensorOptions): Tensor

  # linspace
  func linspace*(start, stop: Scalar): Tensor
  func linspace*(start, stop: Scalar, steps: int): Tensor
  func linspace*(start, stop: Scalar, steps: int, T: typedesc[SomeTorchType]): Tensor
  func linspace*(start, stop: Scalar, steps: int, scalarKind: ScalarKind): Tensor
  func linspace*(start, stop: Scalar, steps: int, device: DeviceKind): Tensor
  func linspace*(start, stop: Scalar, steps: int, options: TensorOptions): Tensor

  # logspace
  func logspace*(start, stop: Scalar, steps: int): Tensor
  func logspace*(start, stop: Scalar, steps: int, base: int): Tensor
  func logspace*(start, stop: Scalar, steps: int, base: int, T: typedesc[SomeTorchType]): Tensor
  func logspace*(start, stop: Scalar, steps: int, base: int, scalarKind: ScalarKind): Tensor
  func logspace*(start, stop: Scalar, steps: int, base: int, device: DeviceKind): Tensor
  func logspace*(start, stop: Scalar, steps: int, base: int, options: TensorOptions): Tensor

  # randint
  func randint*(start, stopEx: int, size: varargs[int]): Tensor
  func randint*(start, stopEx: int, size: varargs[int], T: typedesc[SomeTorchType]): Tensor

  # from_blob

  func from_blob*(data: pointer, sizes: openArray[int], T: typedesc[SomeTorchType]): Tensor
    ## Create a non-owning tensor view from a data pointer.
    ## The data MUST remain valid for the lifetime of the view.
    ## The sizes are copied and managed by the new view and do not need to remain valid.

  func from_blob*(data: pointer, sizes: openArray[int], scalarKind: ScalarKind): Tensor
    ## Create a non-owning tensor view from a data pointer.
    ## The data MUST remain valid for the lifetime of the view.
    ## The sizes are copied and managed by the new view and do not need to remain valid.

  func from_blob*(data: pointer, sizes: openArray[int], device: DeviceKind): Tensor
    ## Create a non-owning tensor view from a data pointer.
    ## The data MUST remain valid for the lifetime of the view.
    ## The sizes are copied and managed by the new view and do not need to remain valid.

  func from_blob*(data: pointer, sizes: openArray[int], options: TensorOptions): Tensor
    ## Create a non-owning tensor view from a data pointer.
    ## The data MUST remain valid for the lifetime of the view.
    ## The sizes are copied and managed by the new view and do not need to remain valid.

  func from_blob*(data: pointer, sizes, strides: openArray[int], T: typedesc[SomeTorchType]): Tensor
    ## Create a non-owning tensor view from a data pointer.
    ## The data MUST remain valid for the lifetime of the view.
    ## The sizes and strides are copied and managed by the new view and do not need to remain valid.

  func from_blob*(data: pointer, sizes, strides: openArray[int], scalarKind: ScalarKind): Tensor
    ## Create a non-owning tensor view from a data pointer.
    ## The data MUST remain valid for the lifetime of the view.
    ## The sizes and strides are copied and managed by the new view and do not need to remain valid.

  func from_blob*(data: pointer, sizes, strides: openArray[int], device: DeviceKind): Tensor
    ## Create a non-owning tensor view from a data pointer.
    ## The data MUST remain valid for the lifetime of the view.
    ## The sizes and strides are copied and managed by the new view and do not need to remain valid.

  func from_blob*(data: pointer, sizes, strides: openArray[int], options: TensorOptions): Tensor
    ## Create a non-owning tensor view from a data pointer.
    ## The data MUST remain valid for the lifetime of the view.
    ## The sizes and strides are copied and managed by the new view and do not need to remain valid.

  # clone
  func clone*(a: Tensor): Tensor

  # #######################################################################
  #
  #                         Methods / Shapeshifting
  #
  # #######################################################################

  # Shape manipulation
  func reshape*(a: Tensor, size: varargs[int]): Tensor
  func view*(a: Tensor, size: varargs[int]): Tensor
  func permute*(a: Tensor, dims: varargs[int]): Tensor
  func expand*(a: Tensor, size: varargs[int], implicit: bool = false): Tensor
  func transpose*(a: Tensor, dim0, dim1: int64): Tensor
  func t*(a: Tensor): Tensor
  func repeat_interleave*(a: Tensor, repeats: int, dim: int = -1): Tensor
  func narrow*(a: Tensor, dim: int, start: int, length: int): Tensor
  func flip*(a: Tensor, dims: varargs[int]): Tensor
  func squeeze*(a: Tensor): Tensor
  func squeeze*(a: Tensor, axis: int): Tensor
  func unsqueeze*(a: Tensor, axis: int): Tensor

  # Backend / dtype
  func to*(a: Tensor, device: Device, non_blocking = false, copy = false): Tensor
  func to*(a: Tensor, device: DeviceKind, non_blocking = false, copy = false): Tensor
  func to*(a: Tensor, dtype: ScalarKind): Tensor
  func to*(a: Tensor, device: DeviceKind, dtype: ScalarKind, non_blocking = false, copy = false): Tensor
  func contiguous*(a: Tensor): Tensor
  func toSparse*(a: Tensor): Tensor
  func toSparse*(a: Tensor, sparseDim: int): Tensor

  # Complex
  func view_as_real*(a: Tensor): Tensor
  func view_as_complex*(a: Tensor): Tensor

  # Device transfers
  func cpu*(a: Tensor): Tensor
  func cuda*(a: Tensor): Tensor
  func hip*(a: Tensor): Tensor
  func vulkan*(a: Tensor): Tensor

  # #######################################################################
  #
  #                            Metadata
  #
  # #######################################################################

  template sizes*(a: Tensor): openArray[int] =
    F.sizes(a.raw).asNimView()

  template shape*(a: Tensor): openArray[int] =
    F.shape(a.raw).asNimView()

  template strides*(a: Tensor): openArray[int] =
    F.strides(a.raw).asNimView()

  func dim*(a: Tensor): int
  func ndimension*(a: Tensor): int
  func nbytes*(a: Tensor): uint
  func numel*(a: Tensor): int
  func size*(a: Tensor, axis: int): int
  func itemsize*(a: Tensor): uint
  func element_size*(a: Tensor): int
  func scalarType*(a: Tensor): ScalarKind
  func get_device*(a: Tensor): int

  # Backend checks
  func is_cuda*(a: Tensor): bool
  func is_hip*(a: Tensor): bool
  func is_sparse*(a: Tensor): bool
  func is_mkldnn*(a: Tensor): bool
  func is_vulkan*(a: Tensor): bool
  func is_quantized*(a: Tensor): bool
  func is_meta*(a: Tensor): bool
  func has_storage*(a: Tensor): bool

  # Reference checks
  func is_same*(a, b: Tensor): bool
  func is_alias_of*(a, b: Tensor): bool

  # #######################################################################
  #
  #                          Math Unary
  #
  # #######################################################################

  func abs*(a: Tensor): Tensor
  func absolute*(a: Tensor): Tensor
  func angle*(a: Tensor): Tensor
  func sgn*(a: Tensor): Tensor
  func conj*(a: Tensor): Tensor
  func acos*(a: Tensor): Tensor
  func arccos*(a: Tensor): Tensor
  func acosh*(a: Tensor): Tensor
  func arccosh*(a: Tensor): Tensor
  func asinh*(a: Tensor): Tensor
  func arcsinh*(a: Tensor): Tensor
  func atanh*(a: Tensor): Tensor
  func arctanh*(a: Tensor): Tensor
  func asin*(a: Tensor): Tensor
  func arcsin*(a: Tensor): Tensor
  func atan*(a: Tensor): Tensor
  func arctan*(a: Tensor): Tensor
  func cos*(a: Tensor): Tensor
  func sin*(a: Tensor): Tensor
  func tan*(a: Tensor): Tensor
  func exp*(a: Tensor): Tensor
  func exp2*(a: Tensor): Tensor
  func log*(a: Tensor): Tensor
  func log2*(a: Tensor): Tensor
  func log10*(a: Tensor): Tensor
  func erf*(a: Tensor): Tensor
  func erfc*(a: Tensor): Tensor
  func reciprocal*(a: Tensor): Tensor
  func neg*(a: Tensor): Tensor
  func square*(a: Tensor): Tensor
  func sqrt*(a: Tensor): Tensor

  # With scalar params
  func clamp*(a: Tensor, minVal, maxVal: Scalar): Tensor
  func clampMin*(a: Tensor, minVal: Scalar): Tensor
  func clampMax*(a: Tensor, maxVal: Scalar): Tensor

  # #######################################################################
  #
  #                          Math Binary
  #
  # #######################################################################

  func dot*(a, other: Tensor): Tensor
  func pow*(a, exponent: Tensor): Tensor
  func pow*(a, exponent: Scalar): Tensor

  # #######################################################################
  #
  #                         Linear Algebra
  #
  # #######################################################################

  func add*(a: Tensor, other: Tensor, alpha: Scalar = 1): Tensor
  func add*(a: Tensor, other: Scalar, alpha: Scalar = 1): Tensor
  func addmv*(a, mat, vec: Tensor, beta, alpha: Scalar = 1): Tensor
  func addmm*(a, mat1, mat2: Tensor, beta, alpha: Scalar = 1): Tensor
  func mm*(a, other: Tensor): Tensor
  func matmul*(a, other: Tensor): Tensor
  func bmm*(a, other: Tensor): Tensor
  func luSolve*(a, data, pivots: Tensor): Tensor

  # #######################################################################
  #
  #                         Comparison
  #
  # #######################################################################

  func equal*(a: Tensor, b: Tensor): bool
  func eq*(a: Tensor, b: Tensor): Tensor

  func `==.`*(a: Tensor, b: Tensor): Tensor {.inline.} =
    eq(a, b)

  func `<.`*(a: Tensor, b: Tensor): Tensor {.inline.} =
    convertLibTorchExceptions:
      wrapTorchTensorImpl:
        F.lt(a.raw, b.raw)

  func `>.`*(a: Tensor, b: Tensor): Tensor {.inline.} =
    convertLibTorchExceptions:
      wrapTorchTensorImpl:
        F.gt(a.raw, b.raw)

  func `<=.`*(a: Tensor, b: Tensor): Tensor {.inline.} =
    convertLibTorchExceptions:
      wrapTorchTensorImpl:
        F.le(a.raw, b.raw)

  func `>=.`*(a: Tensor, b: Tensor): Tensor {.inline.} =
    convertLibTorchExceptions:
      wrapTorchTensorImpl:
        F.ge(a.raw, b.raw)

  func `!=.`*(a: Tensor, b: Tensor): Tensor {.inline.} =
    convertLibTorchExceptions:
      wrapTorchTensorImpl:
        F.ne(a.raw, b.raw)

  func `<.`*(a: Tensor, b: Scalar): Tensor {.inline.} =
    convertLibTorchExceptions:
      wrapTorchTensorImpl:
        F.lt(a.raw, b)

  func `>.`*(a: Tensor, b: Scalar): Tensor {.inline.} =
    convertLibTorchExceptions:
      wrapTorchTensorImpl:
        F.gt(a.raw, b)

  func `<=.`*(a: Tensor, b: Scalar): Tensor {.inline.} =
    convertLibTorchExceptions:
      wrapTorchTensorImpl:
        F.le(a.raw, b)

  func `>=.`*(a: Tensor, b: Scalar): Tensor {.inline.} =
    convertLibTorchExceptions:
      wrapTorchTensorImpl:
        F.ge(a.raw, b)

  func `!=.`*(a: Tensor, b: Scalar): Tensor {.inline.} =
    convertLibTorchExceptions:
      wrapTorchTensorImpl:
        F.ne(a.raw, b)

  func `<.`*(a: Scalar, b: Tensor): Tensor {.inline.} =
    convertLibTorchExceptions:
      wrapTorchTensorImpl:
        F.lt(a, b.raw)

  func `>.`*(a: Scalar, b: Tensor): Tensor {.inline.} =
    convertLibTorchExceptions:
      wrapTorchTensorImpl:
        F.gt(a, b.raw)

  func `<=.`*(a: Scalar, b: Tensor): Tensor {.inline.} =
    convertLibTorchExceptions:
      wrapTorchTensorImpl:
        F.le(a, b.raw)

  func `>=.`*(a: Scalar, b: Tensor): Tensor {.inline.} =
    convertLibTorchExceptions:
      wrapTorchTensorImpl:
        F.ge(a, b.raw)

  func `!=.`*(a: Scalar, b: Tensor): Tensor {.inline.} =
    convertLibTorchExceptions:
      wrapTorchTensorImpl:
        F.ne(a, b.raw)

  func allClose*(a: Tensor, b: Tensor, rtol = 1e-5, abstol = 1e-8, equalNan = false): bool

  # #######################################################################
  #
  #                        Arithmetic Operators
  #
  # #######################################################################

  # Binary (Tensor + Tensor)

  func `+`*(a: Tensor, b: Tensor): Tensor
  func `-`*(a: Tensor, b: Tensor): Tensor
  func `*`*(a: Tensor, b: Tensor): Tensor
  func `/`*(a: Tensor, b: Tensor): Tensor
  func `%`*(a: Tensor, b: Tensor): Tensor
  func `^`*(a: Tensor, b: Tensor): Tensor
  func `**`*(a: Tensor, b: Tensor): Tensor

  # Scalar mixed

  func `+`*(a: SomeNumber, b: Tensor): Tensor
  func `+`*(a: Tensor, b: SomeNumber): Tensor
  func `-`*(a: Tensor, b: SomeNumber): Tensor
  func `*`*(a: SomeNumber, b: Tensor): Tensor
  func `*`*(a: Tensor, b: SomeNumber): Tensor
  func `/`*(a: Tensor, b: SomeNumber): Tensor
  func `%`*(a: SomeNumber, b: Tensor): Tensor
  func `%`*(a: Tensor, b: SomeNumber): Tensor
  func `^`*(a: Tensor, b: Scalar): Tensor
  func `**`*(a: Tensor, b: Scalar): Tensor

  # Unary

  func `-`*(a: Tensor): Tensor
  func `not`*(a: Tensor): Tensor

  # Bitwise (returns Tensor)

  func `and`*(a: Tensor, b: Tensor): Tensor
  func `or`*(a: Tensor, b: Tensor): Tensor
  func `xor`*(a: Tensor, b: Tensor): Tensor

  # In-place (var Tensor)

  proc `+=`*(a: var Tensor, b: Tensor)
  proc `+=`*(a: var Tensor, s: Scalar)
  proc `-=`*(a: var Tensor, b: Tensor)
  proc `-=`*(a: var Tensor, s: Scalar)
  proc `*=`*(a: var Tensor, b: Tensor)
  proc `*=`*(a: var Tensor, s: Scalar)
  proc `/=`*(a: var Tensor, b: Tensor)
  proc `/=`*(a: var Tensor, s: Scalar)
  proc bitand_mut*(a: var Tensor, b: Tensor)
  proc bitor_mut*(a: var Tensor, b: Tensor)
  proc bitxor_mut*(a: var Tensor, b: Tensor)
  proc assign*(a: var Tensor, other: Tensor)

  # #######################################################################
  #
  #                        Reductions (Scalar)
  #
  # #######################################################################

  func sum*(a: Tensor): Tensor
  func sum*(a: Tensor, T: typedesc[SomeTorchType]): Tensor
  func sum*(a: Tensor, dtype: ScalarKind): Tensor
  func sum*(a: Tensor, axis: int, keepdim = false): Tensor
  func sum*(a: Tensor, axis: int, keepdim = false, dtype: ScalarKind): Tensor
  func sum*(a: Tensor, axes: openArray[int], keepdim = false): Tensor
  func sum*(a: Tensor, axes: openArray[int], keepdim = false, dtype: ScalarKind): Tensor


  func mean*(a: Tensor): Tensor
  func mean*(a: Tensor, T: typedesc[SomeTorchType]): Tensor
  func mean*(a: Tensor, axis: int, keepdim = false): Tensor
  func mean*(a: Tensor, axis: int, keepdim = false, dtype: ScalarKind): Tensor
  func mean*(a: Tensor, axes: openArray[int], keepdim = false): Tensor
  func mean*(a: Tensor, axes: openArray[int], keepdim = false, dtype: ScalarKind): Tensor

  func prod*(a: Tensor): Tensor
  func prod*(a: Tensor, T: typedesc[SomeTorchType]): Tensor
  func prod*(a: Tensor, axis: int, keepdim = false): Tensor
  func prod*(a: Tensor, axis: int, keepdim = false, dtype: ScalarKind): Tensor

  func min*(a: Tensor): Tensor
  func max*(a: Tensor): Tensor

  func all*(a: Tensor): Tensor
  func all*(a: Tensor, axis: int): Tensor
  func all*(a: Tensor, axis: int, keepdim: bool): Tensor

  func any*(a: Tensor, axis: int): Tensor
  func any*(a: Tensor, axis: int, keepdim: bool): Tensor

  func argmax*(a: Tensor): Tensor
  func argmax*(a: Tensor, axis: int, keepdim = false): Tensor
  func argmin*(a: Tensor): Tensor
  func argmin*(a: Tensor, axis: int, keepdim = false): Tensor

  func variance*(a: Tensor, unbiased = true): Tensor
  func variance*(a: Tensor, axis: int, unbiased = true): Tensor

  func stddev*(a: Tensor, unbiased = true): Tensor
  func stddev*(a: Tensor, axis: int, unbiased = true): Tensor

  # #######################################################################
  #
  #                       Reductions (Tuple)
  #
  # #######################################################################

  func min*(a: Tensor, axis: int, keepdim = false): tuple[values, indices: Tensor]
  func max*(a: Tensor, axis: int, keepdim = false): tuple[values, indices: Tensor]
  func sort*(a: Tensor, axis = -1, descending = false): tuple[values, indices: Tensor]
    ## Sorts the elements of the input tensor along a given dimension in ascending order by value.
    ## If dim is not given, the last dimension of the input is chosen (dim=-1).
    ## Returns (values, originalIndices) of type (TensorT, TensorInt64)
    ## where originalIndices is the original index of each values (before sorting)
  func argsort*(a: Tensor, axis = -1, descending = false): Tensor
  func qr*(a: Tensor, some: bool): tuple[q, r: Tensor]

  # #######################################################################
  #
  #                          Algorithms
  #
  # #######################################################################

  func cat*(tensors: varargs[Tensor], axis = 0): Tensor
  func stack*(tensors: varargs[Tensor], dim = 0): Tensor
  func chunk*(a: Tensor, chunks: int, dim = 0): seq[Tensor]
  func unbind*(a: Tensor, dim = 0): seq[Tensor]

  # #######################################################################
  #
  #                        Fancy Indexing
  #
  # #######################################################################

  func index_select*(a: Tensor, axis: int, indices: Tensor): Tensor
  func masked_select*(a: Tensor, mask: Tensor): Tensor

  # #######################################################################
  #
  #                        In-place Mutation
  #
  # #######################################################################

  proc random_mut*(a: var Tensor, start, stopEx: int)
  proc index_fill_mut*(a: var Tensor, mask: Tensor, value: Scalar)
  proc index_fill_mut*(a: var Tensor, mask: Tensor, value: Tensor)
  proc masked_fill_mut*(a: var Tensor, mask: Tensor, value: Scalar)
  proc masked_fill_mut*(a: var Tensor, mask: Tensor, value: Tensor)

  # #######################################################################
  #
  #                           Indexing
  #
  # #######################################################################

  func item*(t: Tensor, T: typedesc): T {.inline.} =
    F.item(t.raw, T)

  template `[]`*(t: Tensor{call}, args: varargs[untyped]): untyped =
    # Due to generic sandwich bug, this needs export of F.shape, F.`[]`, F.len
    let tmp = t
    convertLibTorchExceptions:
      wrapTorchTensor:
        tmp.raw[args]

  template `[]`*(t: Tensor{`let`|`var`|`const`|lvalue}, args: varargs[untyped]): untyped =
    # Due to generic sandwich bug, this needs export of F.shape, F.`[]`, F.len
    convertLibTorchExceptions:
      wrapTorchTensor:
        t.raw[args]

  macro `[]=`*(t: var Tensor, args: varargs[untyped]): untyped =
    var tmp = args
    let valAST = tmp.pop()
    let new_args = getAST(desugarSlices(tmp))
    result = quote do:
      convertLibTorchExceptions:
        let tmpVal = `valAST`
        when tmpVal is Tensor:
          slice_typed_dispatch_mut(`t`.raw, `new_args`, tmpVal.raw)
        else:
          slice_typed_dispatch_mut(`t`.raw, `new_args`, tmpVal)

  # #######################################################################
  #
  #                           Data Access
  #
  # #######################################################################

  func data_ptr*(a: Tensor): pointer
  func data_ptr*(a: Tensor, T: typedesc): ptr UncheckedArray[T] {.inline.} =
    F.data_ptr(a.raw, T)

  # #######################################################################
  #
  #                              FFT
  #
  # #######################################################################

  # 1-D
  func fft*(a: Tensor): Tensor
  func fft*(a: Tensor, n: int, dim = -1): Tensor
  func ifft*(a: Tensor): Tensor
  func rfft*(a: Tensor): Tensor
  func irfft*(a: Tensor): Tensor
  func hfft*(a: Tensor): Tensor
  func ihfft*(a: Tensor): Tensor

  # N-D
  func fft2*(a: Tensor): Tensor
  func fft2*(a: Tensor, s: openArray[int]): Tensor
  func ifft2*(a: Tensor): Tensor
  func ifft2*(a: Tensor, s: openArray[int]): Tensor
  func fftn*(a: Tensor): Tensor
  func fftn*(a: Tensor, s: openArray[int]): Tensor
  func ifftn*(a: Tensor): Tensor
  func ifftn*(a: Tensor, s: openArray[int]): Tensor
  func rfft2*(a: Tensor): Tensor
  func rfft2*(a: Tensor, s: openArray[int]): Tensor
  func irfft2*(a: Tensor): Tensor
  func irfft2*(a: Tensor, s: openArray[int]): Tensor
  func rfftn*(a: Tensor): Tensor
  func rfftn*(a: Tensor, s: openArray[int]): Tensor
  func irfftn*(a: Tensor): Tensor
  func irfftn*(a: Tensor, s: openArray[int]): Tensor

  # Shift
  func fftshift*(a: Tensor): Tensor
  func fftshift*(a: Tensor, dim: varargs[int]): Tensor
  func ifftshift*(a: Tensor): Tensor
  func ifftshift*(a: Tensor, dim: varargs[int]): Tensor
