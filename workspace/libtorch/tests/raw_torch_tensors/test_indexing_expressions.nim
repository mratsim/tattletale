# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/math,
  std/strformat,
  workspace/libtorch/src/raw_libtorch as torch,
  ./utils/[torch_tensors_overloads, raw_libtorch_testutils]

proc genShiftedVandermonde5x5*(dtype: ScalarKind): TorchTensor =
  torch.arange(1, 6).reshape(-1, 1) ** torch.arange(1, 6)

func formatName*(desc, indexingExample: string): string =
  fmt"{desc:<40}  {indexingExample}"

proc main() =
  # IMPORTANT: Tensors AND launchMissile must be INSIDE each runCppTest body.
  # Capturing TorchTensor in a closure corrupts the C++ object.

  # -----------------------------------------------------------------------
  # Single evaluation - integer indexing (isAllInt path)

  runCppTest formatName("Point indexing", "launchMissile(t)[1, 2]"):
    proc(): bool =
      var i = 0
      proc launchMissile(a: TorchTensor): TorchTensor =
        i += 1
        a
      let vandermonde = genShiftedVandermonde5x5(kFloat64)
      let val = launchMissile(vandermonde)[1, 2]
      doAssert i == 1  # launchMissile called exactly once
      doAssert val.item(float64) == 8.0
      true

  runCppTest formatName("Negative point indexing", "launchMissile(t)[-1, -1]"):
    proc(): bool =
      var i = 0
      proc launchMissile(a: TorchTensor): TorchTensor =
        i += 1
        a
      let vandermonde = genShiftedVandermonde5x5(kFloat64)
      let val = launchMissile(vandermonde)[-1, -1]
      doAssert i == 1
      doAssert val.item(float64) == 3125.0
      true

  runCppTest formatName("Expression indices", "launchMissile(t)[1+1, 2*2]"):
    proc(): bool =
      var i = 0
      proc launchMissile(a: TorchTensor): TorchTensor =
        i += 1
        a
      let vandermonde = genShiftedVandermonde5x5(kFloat64)
      let val = launchMissile(vandermonde)[1+1, 2*2]
      doAssert i == 1
      doAssert val.item(float64) == 243.0  # (2+1)^(4+1) = 3^5 = 243
      true

  # -----------------------------------------------------------------------
  # Single evaluation - full slice (sliceSpan path)

  runCppTest formatName("Full span", "launchMissile(t)[_, _]"):
    proc(): bool =
      var i = 0
      proc launchMissile(a: TorchTensor): TorchTensor =
        i += 1
        a
      let vandermonde = genShiftedVandermonde5x5(kFloat64)
      let sliced = launchMissile(vandermonde)[_, _]
      doAssert i == 1
      doAssert sliced.shape[0] == 5
      doAssert sliced.shape[1] == 5
      true

  runCppTest formatName("Full span shorthand _.._", "launchMissile(t)[_.._, _]"):
    proc(): bool =
      var i = 0
      proc launchMissile(a: TorchTensor): TorchTensor =
        i += 1
        a
      let vandermonde = genShiftedVandermonde5x5(kFloat64)
      let sliced = launchMissile(vandermonde)[_.._, _]
      doAssert i == 1
      doAssert sliced == vandermonde
      true

  # -----------------------------------------------------------------------
  # Single evaluation - slice indexing (normalizedSlice path)

  runCppTest formatName("Slice from start", "launchMissile(t)[_..<3, _]"):
    proc(): bool =
      var i = 0
      proc launchMissile(a: TorchTensor): TorchTensor =
        i += 1
        a
      let vandermonde = genShiftedVandermonde5x5(kFloat64)
      let sliced = launchMissile(vandermonde)[_..<3, _]
      doAssert i == 1
      doAssert sliced.shape[0] == 3
      doAssert sliced.shape[1] == 5
      true

  runCppTest formatName("Slice to end", "launchMissile(t)[1..<_]"):
    proc(): bool =
      var i = 0
      proc launchMissile(a: TorchTensor): TorchTensor =
        i += 1
        a
      let vandermonde = genShiftedVandermonde5x5(kFloat64)
      let sliced = launchMissile(vandermonde)[1..<_]
      doAssert i == 1
      doAssert sliced.shape[0] == 4
      doAssert sliced.shape[1] == 5
      true

  runCppTest formatName("Stepped slice with ..|", "launchMissile(t)[1..|2]"):
    proc(): bool =
      var i = 0
      proc launchMissile(a: TorchTensor): TorchTensor =
        i += 1
        a
      let vandermonde = genShiftedVandermonde5x5(kFloat64)
      let sliced = launchMissile(vandermonde)[1..|2]
      doAssert i == 1
      doAssert sliced.shape[0] == 2
      doAssert sliced.shape[1] == 5
      true

  runCppTest formatName("Span with step _.._|N", "launchMissile(t)[_.._|2, _]"):
    proc(): bool =
      var i = 0
      proc launchMissile(a: TorchTensor): TorchTensor =
        i += 1
        a
      let vandermonde = genShiftedVandermonde5x5(kFloat64)
      let sliced = launchMissile(vandermonde)[_.._|2, _]
      doAssert i == 1
      doAssert sliced.shape[0] == 3
      doAssert sliced.shape[1] == 5
      true

  runCppTest formatName("Slice with start, stop, step", "launchMissile(t)[1..<4|2, _]"):
    proc(): bool =
      var i = 0
      proc launchMissile(a: TorchTensor): TorchTensor =
        i += 1
        a
      let vandermonde = genShiftedVandermonde5x5(kFloat64)
      let sliced = launchMissile(vandermonde)[1..<4|2, _]
      doAssert i == 1
      doAssert sliced.shape[0] == 2
      doAssert sliced.shape[1] == 5
      true

  runCppTest formatName("Negative end-relative slice", "launchMissile(t)[_..-1, _]"):
    proc(): bool =
      var i = 0
      proc launchMissile(a: TorchTensor): TorchTensor =
        i += 1
        a
      let vandermonde = genShiftedVandermonde5x5(kFloat64)
      let sliced = launchMissile(vandermonde)[_..-1, _]
      doAssert i == 1
      doAssert sliced.shape[0] == 4
      doAssert sliced.shape[1] == 5
      true

  runCppTest formatName("Negative start slice", "launchMissile(t)[-3.._, _]"):
    proc(): bool =
      var i = 0
      proc launchMissile(a: TorchTensor): TorchTensor =
        i += 1
        a
      let vandermonde = genShiftedVandermonde5x5(kFloat64)
      let sliced = launchMissile(vandermonde)[-3.._, _]
      doAssert i == 1
      doAssert sliced.shape[0] == 3
      doAssert sliced.shape[1] == 5
      true

  runCppTest formatName("Negative slice with step", "launchMissile(t)[_..-1|2, _]"):
    proc(): bool =
      var i = 0
      proc launchMissile(a: TorchTensor): TorchTensor =
        i += 1
        a
      let vandermonde = genShiftedVandermonde5x5(kFloat64)
      let sliced = launchMissile(vandermonde)[_..-1|2, _]
      doAssert i == 1
      doAssert sliced.shape[0] == 2
      doAssert sliced.shape[1] == 5
      true

  # -----------------------------------------------------------------------
  # Single evaluation - mixed indexing

  runCppTest formatName("Span on first dim, int on second", "launchMissile(t)[_, 2]"):
    proc(): bool =
      var i = 0
      proc launchMissile(a: TorchTensor): TorchTensor =
        i += 1
        a
      let vandermonde = genShiftedVandermonde5x5(kFloat64)
      let sliced = launchMissile(vandermonde)[_, 2]
      doAssert i == 1
      doAssert sliced.shape.len == 1
      doAssert sliced == [1, 8, 27, 64, 125].toTorchTensor.to(kFloat64)
      true

  runCppTest formatName("Slice on first dim, span on second", "launchMissile(t)[1..3, _]"):
    proc(): bool =
      var i = 0
      proc launchMissile(a: TorchTensor): TorchTensor =
        i += 1
        a
      let vandermonde = genShiftedVandermonde5x5(kFloat64)
      let sliced = launchMissile(vandermonde)[1..3, _]
      doAssert i == 1
      doAssert sliced.shape[0] == 3
      doAssert sliced.shape[1] == 5
      true

  runCppTest formatName("Two slices", "launchMissile(t)[1..<3, 1..<3]"):
    proc(): bool =
      var i = 0
      proc launchMissile(a: TorchTensor): TorchTensor =
        i += 1
        a
      let vandermonde = genShiftedVandermonde5x5(kFloat64)
      let sliced = launchMissile(vandermonde)[1..<3, 1..<3]
      doAssert i == 1
      doAssert sliced ==
        [[ 4,  8],
         [ 9, 27]].toTorchTensor.to(kFloat64)
      true

  runCppTest formatName("Unary pipe step with span", "launchMissile(t)[|2, _]"):
    proc(): bool =
      var i = 0
      proc launchMissile(a: TorchTensor): TorchTensor =
        i += 1
        a
      let vandermonde = genShiftedVandermonde5x5(kFloat64)
      let sliced = launchMissile(vandermonde)[|2, _]
      doAssert i == 1
      doAssert sliced.shape[0] == 3
      doAssert sliced.shape[1] == 5
      true

  runCppTest formatName("Stepped span with index", "launchMissile(t)[|2, 0]"):
    proc(): bool =
      var i = 0
      proc launchMissile(a: TorchTensor): TorchTensor =
        i += 1
        a
      let vandermonde = genShiftedVandermonde5x5(kFloat64)
      let sliced = launchMissile(vandermonde)[|2, 0]
      doAssert i == 1
      doAssert sliced.shape.len == 1
      doAssert sliced == [1, 3, 5].toTorchTensor.to(kFloat64)
      true

  # -----------------------------------------------------------------------
  # Single evaluation - normalizedSlice path (FancyNone with t.shape[i])
  # These codepaths access t MULTIPLE times inside slice_typed_dispatch:
  #   - index(t) for the call
  #   - t.shape[0], t.shape[1], ... for each normalizedSlice axisLen
  # Without the `let tmp = t` guard, launchMissile would fire 3+ times.

  runCppTest formatName("Two slices on same tensor", "launchMissile(t)[_..<3, _..<2]"):
    proc(): bool =
      ## normalizedSlice accesses t.shape[0] AND t.shape[1] inside the macro
      ## t is referenced 3 times total: index(t), t.shape[0], t.shape[1]
      var i = 0
      proc launchMissile(a: TorchTensor): TorchTensor =
        i += 1
        a
      let vandermonde = genShiftedVandermonde5x5(kFloat64)
      let sliced = launchMissile(vandermonde)[_..<3, _..<2]
      doAssert i == 1  # must be 1, not 3
      doAssert sliced.shape[0] == 3
      doAssert sliced.shape[1] == 2
      true

  runCppTest formatName("Three slices (3D tensor)", "launchMissile(t3d)[_..<2, _..<2, _..<2]"):
    proc(): bool =
      ## normalizedSlice on 3 axes: t.shape[0], t.shape[1], t.shape[2]
      ## t is referenced 4 times: index(t) + 3x t.shape[i]
      var i = 0
      proc launchMissile(a: TorchTensor): TorchTensor =
        i += 1
        a
      let t3d = torch.arange(24, kFloat64).reshape(2, 3, 4)
      let sliced = launchMissile(t3d)[_..<2, _..<2, _..<2]
      doAssert i == 1  # must be 1, not 4
      doAssert sliced.shape[0] == 2
      doAssert sliced.shape[1] == 2
      doAssert sliced.shape[2] == 2
      true

  runCppTest formatName("Negative index slice (runtime normalization)", "launchMissile(t)[_..-1, _..-1]"):
    proc(): bool =
      ## Negative indices require normalizedSlice which accesses t.shape[i]
      ## Two negative-indexed slices: t.shape[0] and t.shape[1] accessed
      var i = 0
      proc launchMissile(a: TorchTensor): TorchTensor =
        i += 1
        a
      let vandermonde = genShiftedVandermonde5x5(kFloat64)
      let sliced = launchMissile(vandermonde)[_..-1, _..-1]
      doAssert i == 1  # must be 1, not 3
      doAssert sliced.shape[0] == 4
      doAssert sliced.shape[1] == 4
      true

  runCppTest formatName("Slice with mixed int + slice", "launchMissile(t)[_..<3, 2]"):
    proc(): bool =
      ## Only the first dim is a slice, so t.shape[0] is accessed once
      ## t referenced 2 times: index(t) + t.shape[0]
      var i = 0
      proc launchMissile(a: TorchTensor): TorchTensor =
        i += 1
        a
      let vandermonde = genShiftedVandermonde5x5(kFloat64)
      let sliced = launchMissile(vandermonde)[_..<3, 2]
      doAssert i == 1  # must be 1, not 2
      doAssert sliced.shape.len == 1  # int index squeezes axis
      true

  # -----------------------------------------------------------------------
  # Single evaluation - fancy indexing path (masked_select / index_select)
  # These codepaths access t MULTIPLE times inside the quote block:
  #   - t.scalarType to check dtype
  #   - t passed to masked_select / index_select
  # Without the guard, launchMissile would fire 2+ times.

  runCppTest "Boolean mask full (masked_select) [SKIPPED]":
    proc(): bool =
      ## FancyUnknownFull path: quote block accesses t.scalarType then masked_select(t, ...)
      ## t referenced 2 times inside the quote block
      ## TODO: Re-enable when TorchTensor comparison operators are implemented
      true  # skip: no comparison operators on TorchTensor yet
  runCppTest "Integer fancy index via [] [SKIPPED: macro generates raw array, ABI needs Tensor]":
    proc(): bool =
      ## FancyIndex path via [] macro: index_select(t, axis, [0, 2])
      ## TODO: Re-enable when the macro converts [0, 2] to toTorchTensor().to(kInt64)
      true  # skip: macro generates index_select(tmp, 2, [0, 2]) but ABI needs Tensor
when isMainModule:
  main()
