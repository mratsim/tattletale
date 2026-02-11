# Flambeau
# Copyright (c) 2020 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  workspace/libtorch/src/abi/std_cpp,
  workspace/libtorch/vendor/libtorch

# c10 is a collection of utilities in PyTorch

# C++ interop
# -----------------------------------------------------------------------

{.push cdecl, header: TorchHeader.}

# ArrayRef
# -----------------------------------------------------------------------
#
# LibTorch is using "ArrayRef" through the codebase in particular
# for shapes and strides.
#
# It has the following definition in
# libtorch/include/c10/util/ArrayRef.h
#
# template <typename T>
# class ArrayRef final {
#  private:
#   /// The start of the array, in an external buffer.
#   const T* Data;
#
#   /// The number of elements.
#   size_type Length;
#
# It is noted that the class does not own the underlying data.
# We can model that in a zero-copy and safely borrow-checked way
# with "openarray[T]"

{.experimental: "views".} # TODO this is ignored

type
  ArrayRef*[T] {.importcpp: "c10::ArrayRef", bycopy.} = object
    # The field are private so we can't use them, but `lent` enforces borrow checking
    #p: lent UncheckedArray[T]
    #len: csize_t

  IntArrayRef* = ArrayRef[int64]

func data*[T](ar: ArrayRef[T]): lent UncheckedArray[T] {.importcpp: "const_cast<'*1*>(#.data())".}
func size*(ar: ArrayRef): csize_t {.importcpp: "#.size()".}

func init*[T](AR: type ArrayRef[T], p: ptr T, len: SomeInteger): ArrayRef[T] {.constructor, importcpp: "c10::ArrayRef<'*0>(@)".}
func init*[T](AR: type ArrayRef[T], vec: CppVector[T]): ArrayRef[T] {.constructor, importcpp: "c10::ArrayRef<'*0>(@)".}
func init*[T](AR: type ArrayRef[T]): ArrayRef[T] {.constructor, varargs, importcpp: "c10::ArrayRef<'*0>({@})".}

# Simple indexing for use in check_index (exported from rawinterop too, but different name to avoid conflicts)
func getAt*[T](ar: ArrayRef[T], idx: SomeInteger): T {.importcpp: "#[#]".}
func `==`*[T](ar1, ar2: ArrayRef[T]): bool {.importcpp: "(# == #)".}

# c10::complex
# -----------------------------------------------------------------------
type TorchComplex*[T: SomeFloat] {.importcpp: "c10::complex".} = object

func torchComplex*[T: SomeFloat](re, im: T): TorchComplex[T] {.constructor, importcpp: "c10::complex<'*0>(@)".}
func real*[T: SomeFloat](self: TorchComplex[T]): T {.importcpp: "#.real()".}
func imag*[T: SomeFloat](self: TorchComplex[T]): T {.importcpp: "#.imag()".}

proc `+`*[T: SomeFloat](a, b: TorchComplex[T]): TorchComplex[T] {.importcpp: "(# + #)".}
proc `-`*[T: SomeFloat](a, b: TorchComplex[T]): TorchComplex[T] {.importcpp: "(# - #)".}
proc `*`*[T: SomeFloat](a, b: TorchComplex[T]): TorchComplex[T] {.importcpp: "(# * #)".}
proc `/`*[T: SomeFloat](a, b: TorchComplex[T]): TorchComplex[T] {.importcpp: "(# / #)".}

proc `=+`*[T: SomeFloat](self: var TorchComplex[T], arg: TorchComplex[T]) {.importcpp: "(# += #)".}
proc `=-`*[T: SomeFloat](self: var TorchComplex[T], arg: TorchComplex[T]) {.importcpp: "(# -= #)".}
proc `=*`*[T: SomeFloat](self: var TorchComplex[T], arg: TorchComplex[T]) {.importcpp: "(# *= #)".}
proc `=/`*[T: SomeFloat](self: var TorchComplex[T], arg: TorchComplex[T]) {.importcpp: "(# /= #)".}

proc `==`*[T: SomeFloat](a, b: TorchComplex[T]): bool {.importcpp: "(# == #)".}
proc `!=`*[T: SomeFloat](a, b: TorchComplex[T]): bool {.importcpp: "(# != #)".}

{.pop.}
