# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/[complex, enumerate, macros, strformat],
  # Internal
  workspace/libtorch/src/std_cpp,
  workspace/libtorch/src/c10,
  workspace/libtorch/src/torch_tensors,
  workspace/libtorch/src/dynamic_stack_arrays,
  workspace/libtorch/src/[ast_utils, indexing_macros]

# #######################################################################
#
#               Syntactic sugar for Torch and Nim interop
#
# #######################################################################

type Metadata* = DynamicStackArray[int64]

# Shape
# -----------------------------------------------------

func product*(a: openArray[SomeInteger]): SomeInteger {.inline.} =
  if unlikely(a.len == 0):
    return 0
  result = 1
  for value in items(a):
    result *= value

# ArrayRefs
# -----------------------------------------------------
# libtorch/include/c10/util/ArrayRef.h

{.experimental: "views".}

template asNimView*[T](ar: ArrayRef[T]): openArray[T] =
  toOpenArray(ar.data.unsafeAddr, 0, ar.size.int - 1)

template asTorchView*[T](oa: openArray[T]): ArrayRef[T] =
  # Don't remove. This makes @[1, 2, 3].asTorchView works
  let a = @oa
  ArrayRef[T].init(a[0].unsafeAddr, a.len)

template asTorchView*(meta: Metadata): ArrayRef[int64] =
  ArrayRef[int64].init(meta.data[0].unsafeAddr, meta.len)

proc `$`*[T](ar: ArrayRef[T]): string =
  `$`(ar.asNimView())

func len*[T](ar: ArrayRef[T]): int =
  # Nim idiomatic proc for seq
  ar.size().int

iterator items*[T](ar: ArrayRef[T]): T =
  # Iterate over ArrayRef
  var i: int = 0
  while i < ar.len():
    yield ar.data()[i]
    inc i

func `[]`*[T](ar: ArrayRef[T], idx: SomeInteger): T =
  when compileOption("boundChecks"):
    if idx < 0 or idx >= ar.len():
      raise newException(
        IndexDefect,
        &"ArrayRef `[]` access out-of-bounds. Index constrained by 0 <= {idx} <= ArrayRef.len() = {ar.len()}.",
      )
  result = getAt(ar, idx)

func `[]=`*[T](ar: var ArrayRef[T], idx: SomeInteger, val: T) =
  when compileOption("boundChecks"):
    if idx < 0 or idx >= ar.len():
      raise newException(
        IndexDefect,
        &"ArrayRef `[]` access out-of-bounds. Index constrained by 0 <= {idx} <= ArrayRef.len() = {ar.len()}.",
      )
  let p = ar.data()
  p[idx] = val

# Type map
# -----------------------------------------------------
func toTypedesc*(scalarKind: ScalarKind): typedesc =
  ## Maps a Torch ScalarKind to Nim type
  case scalarKind
  of kUint8:
    typedesc(uint8)
  of kInt8:
    typedesc(int8)
  of kInt16:
    typedesc(int16)
  of kInt32:
    typedesc(int32)
  of kInt64:
    typedesc(int64)
  of kFloat32:
    typedesc(float32)
  of kFloat64:
    typedesc(float64)
  of kComplexF32:
    typedesc(Complex32)
  of kComplexF64:
    typedesc(Complex64)
  of kBool:
    typedesc(bool)
  else:
    raise newException(ValueError, "Unsupported libtorch type in Nim: " & $scalarKind)

func toScalarKind*(T: typedesc[SomeTorchType]): static ScalarKind =
  ## Maps a Nim type to Torch scalar kind
  when T is uint8 | byte:
    kUint8
  elif T is int8:
    kInt8
  elif T is int16:
    kInt16
  elif T is int32 or (T is int and sizeof(int) == sizeof(int32)):
    kInt32
  elif T is int64 or (T is int and sizeof(int) == sizeof(int64)):
    kInt64
  elif T is float32:
    kFloat32
  elif T is float64:
    kFloat64
  elif T is Complex[float32]:
    kComplexF32
  elif T is Complex[float64]:
    kComplexF64
  elif T is bool:
    kBool
  else:
    {.error: "Unsupported type in libtorch: " & $T.}

converter convertTypeDef*(T: typedesc[SomeTorchType]): static ScalarKind =
  toScalarKind(T)

# Nim openarrays -> Torch Tensors
# -----------------------------------------------------

func getShape*[T](s: openarray[T], parent_shape = Metadata()): Metadata =
  ## Get the shape of nested seqs/arrays
  ## Important ⚠: at each nesting level, only the length
  ##   of the first element is used for the shape.
  ##   Ensure before or after that seqs have the expected length
  ##   or that the total number of elements matches the product of the dimensions.

  result = parent_shape
  result.add(s.len)

  when (T is seq | array):
    result = getShape(s[0], result)

macro getBaseType*(T: typedesc): untyped =
  # Get the base T of a seq[T] input
  result = T.getTypeInst()[1]
  while result.kind == nnkBracketExpr and (result[0].eqIdent"seq" or result[0].eqIdent"array"):
    # We can also have nnkBracketExpr(Complex, float32)
    if result[0].eqIdent"seq":
      result = result[1]
    else: # array
      result = result[2]

iterator flatIter*[T](s: openarray[T]): auto {.noSideEffect.} =
  ## Inline iterator on any-depth seq or array
  ## Returns values in order
  for item in s:
    when item is array | seq:
      for subitem in flatIter(item):
        yield subitem
    else:
      yield item

func toTorchTensorView*[T: SomeTorchType](oa: openarray[T]): lent TorchTensor =
  ## Interpret an openarray as a CPU Tensor
  ## Important:
  ##   the buffer is shared.
  ##   There is no copy but modifications are shared
  ##   and the view cannot outlive its buffer.
  ##
  ## Input:
  ##      - An array or a seq (can be nested)
  ## Result:
  ##      - A view Tensor of the same shape
  return from_blob(oa[0].unsafeAddr, oa.len.int64, toScalarKind(T))

func toTorchTensor*[T: SomeTorchType](oa: openarray[T]): TorchTensor =
  ## Convert an openarray to CPU Tensor
  ##
  ## Input:
  ##      - An array or a seq
  ## Result:
  ##      - A view Tensor of the same shape
  let shape = getShape(oa)
  result = empty(shape.asTorchView(), T.toScalarKind())

  return from_blob(oa[0].unsafeAddr, oa.len.int64, toScalarKind(T)).clone()

func toTorchTensor*[T: seq | array](oa: openarray[T]): TorchTensor =
  ## Convert an openarray of openarrays to a CPU Tensor
  ##
  ## Input:
  ##      - A nested array or a seq
  ## Result:
  ##      - A view Tensor of the same shape
  let shape = getShape(oa)
  type BaseType = getBaseType(T)

  result = empty(shape.asTorchView(), BaseType.toScalarKind())

  let data = result.data_ptr(BaseType)
  for i, val in enumerate(flatIter(oa)):
    data[i] = val

# TorchTensor -> Nim string
# -----------------------------------------------------

func toCppString(t: TorchTensor): CppString =
  ## Tensors don't have a `$` equivilent so we have to put it into
  ## a ostringstream and convert it to a CppString.
  {.emit:
  """
  std::ostringstream stream;
  stream << `t`;
  result = stream.str();
  """
  .}

proc `$`*(t: TorchTensor): string =
  "TorchTensor\n" & $(toCppString(t))


# #######################################################################
#
#                        Public fancy indexers
#
# #######################################################################
# Checkers func to Raise IndexDefect
# -----------------------------------------------------------------------

macro `[]`*(t: TorchTensor, args: varargs[untyped]): untyped =
  ## Slice a Tensor
  ## Input:
  ##   - a Tensor
  ##   - and:
  ##     - specific coordinates (``varargs[int]``)
  ##     - or a slice (cf. tutorial)
  ## Returns:
  ##   - a value or a tensor corresponding to the slice
  ##
  ## Usage:
  ##    - Basic indexing - foo[2, 3]
  ##    - Basic indexing - foo[1+1, 2*2*1]
  ##    - Basic slicing - foo[1..2, 3]
  ##    - Basic slicing - foo[1+1..4, 3-2..2]
  ##    - Span slices - foo[_, 3]
  ##    - Span slices - foo[1.._, 3]
  ##    - Span slices - foo[_..3, 3]
  ##    - Span slices - foo[_.._, 3]
  ##    - Stepping - foo[1..3\|2, 3]
  ##    - Span stepping - foo[_.._\|2, 3]
  ##    - Span stepping - foo[_.._\|+2, 3]
  ##    - Span stepping - foo[1.._\|1, 2..3]
  ##    - Span stepping - foo[_..<4\|2, 3]
  ##    - Slicing until at n from the end - foo[0..^4, 3]
  ##    - Span Slicing until at n from the end - foo[_..^2, 3]
  ##    - Stepped Slicing until at n from the end - foo[1..^1\|2, 3]
  ##    - Slice from the end - foo[^1..0\|-1, 3]
  ##    - Slice from the end - expect non-negative step error - foo[^1..0, 3]
  ##    - Slice from the end - foo[^(2*2)..2*2, 3]
  ##    - Slice from the end - foo[^3..^2, 3]
  let new_args = getAST(desugarSlices(args))

  result = quote do:
    slice_typed_dispatch(`t`, `new_args`)

macro `[]=`*(t: var TorchTensor, args: varargs[untyped]): untyped =
  ## Modifies a tensor inplace at the corresponding location or slice
  ##
  ##
  ## Input:
  ##   - a ``var`` tensor
  ##   - a location:
  ##     - specific coordinates (``varargs[int]``)
  ##     - or a slice (cf. tutorial)
  ##   - a value:
  ##     - a single value that will
  ##       - replace the value at the specific coordinates
  ##       - or be applied to the whole slice
  ##     - an openarray with a shape that matches the slice
  ##     - a tensor with a shape that matches the slice
  ## Result:
  ##   - Nothing, the tensor is modified in-place
  ## Usage:
  ##   - Assign a single value - foo[1..2, 3..4] = 999
  ##   - Assign an array/seq of values - foo[0..1,0..1] = [[111, 222], [333, 444]]
  ##   - Assign values from a view/Tensor - foo[^2..^1,2..4] = bar
  ##   - Assign values from the same Tensor - foo[^2..^1,2..4] = foo[^1..^2|-1, 4..2|-1]

  # varargs[untyped] consumes all arguments so the actual value should be popped
  # https://github.com/nim-lang/Nim/issues/5855

  var tmp = args
  let val = tmp.pop
  let new_args = getAST(desugarSlices(tmp))

  result = quote do:
    slice_typed_dispatch_mut(`t`, `new_args`, `val`)
