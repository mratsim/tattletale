# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/unittest,
  std/math,
  std/strformat,
  workspace/safetensors,
  workspace/libtorch as torch

proc genShiftedVandermonde5x5*(dtype: ScalarKind): TorchTensor =
  ## Generate 5x5 shifted Vandermonde matrix: v[i, j] = i^(j+1)
  ## [[   1    1    1    1    1]
  ##  [   2    4    8   16   32]
  ##  [   3    9   27   81  243]
  ##  [   4   16   64  256 1024]
  ##  [   5   25  125  625 3125]]
  torch.arange(1, 6).reshape(-1, 1) ** torch.arange(1, 6)

proc display(t: TorchTensor) =
  echo "Test matrix:"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo t
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"


proc main() =
  let vandermonde = genShiftedVandermonde5x5(kFloat64)

  suite "Torch Tensor Indexing - PyTorch/libtorch Documentation Examples":
    ## Reference: https://pytorch.org/cppdocs/notes/tensor_indexing.html
    ## All tests include equivalent PyTorch and libtorch syntax for documentation

    vandermonde.display()

    test "Integer indexing - a[1, 2]":
      ## PyTorch: a[1, 2]
      ## C++ libtorch: a.index({1, 2})
      ## Nim libtorch: a[1, 2]
      let val = vandermonde[1, 2]
      check: val.item(float64) == 8.0  # 2^3 = 8

    test "Strided monodimensional indexing - a[1..|2]":
      ## PyTorch: a[1::2]
      ## C++ libtorch: q.index({Slice(1, None, 2)})
      ## Nim libtorch: a[1..|2]
      let sliced = vandermonde[1..|2]
      # Slices rows 1 and 3 (bases 2 and 4), all columns
      check: sliced.shape[0] == 2
      check: sliced.shape[1] == 5

  suite "Slice Types (Python ':' equivalent to libtorch Slice)":
    ## Python `:` / `::` maps to `torch::indexing::Slice()`

    vandermonde.display()

    test "Full slice - a[_ ,_]":
      ## PyTorch: a[:,:]
      ## C++ libtorch: a.index({Slice(), Slice()})
      ## Nim libtorch: a[_,_]
      let full = vandermonde[_, _]
      check: full.shape[0] == 5
      check: full.shape[1] == 5

    test "Slice from start - a[_..<3, _]":
      ## PyTorch: a[:3]
      ## C++ libtorch: a.index({Slice(None, 3)})
      ## Nim libtorch: a[_..<3,_]
      ## TODO - allow  a[..<3,_]
      let sliced = vandermonde[_..<3, _]
      check: sliced.shape[0] == 3
      check: sliced.shape[1] == 5

    test "Slice to end - a[1..<]":
      ## PyTorch: a[1:]
      ## C++ libtorch: a.index({Slice(1, None)})
      ## Nim libtorch: a[1..<_]
      let sliced = vandermonde[1..<_]
      check: sliced.shape[0] == 4
      check: sliced.shape[1] == 5

    test "Slice with step only - a[|2]":
      ## PyTorch: a[::2]
      ## C++ libtorch: a.index({Slice(None, None, 2)})
      ## Nim libtorch: a[|2]
      let sliced = vandermonde[|2]
      check: sliced.shape[0] == 3  # rows 0, 2, 4
      check: sliced.shape[1] == 5

    test "Slice with start, stop, step - a[1..<3|2]":
      ## PyTorch: a[1:3:2]
      ## libtorch: a.index({Slice(1, 3, 2)})
      ## Nim libtorch: a[1..<3|2]
      let sliced = vandermonde[1..<3|2]
      check: sliced.shape[0] == 1  # only row 1
      check: sliced.shape[1] == 5

  # suite "Python Slice Syntax to Nim Translation Reference":
  #   ## This suite documents the mapping between Python slice syntax and Nim syntax.
  #   ##
  #   ## Python slices are EXCLUSIVE on the end (like C++/Python standard).
  #   ## Nim slices are INCLUSIVE on both ends (..) or exclusive (..<).
  #   ##
  #   ## Python: t[start:stop]    -> elements from start (inclusive) to stop (exclusive)
  #   ## Nim:    t[start..<stop]   -> elements from start to stop-1
  #   ##
  #   ## Python: t[start:]       -> elements from start to end
  #   ## Nim:    t[start..^1]    -> ^1 is last element (inclusive)
  #   ##
  #   ## Python: t[:stop]        -> elements from 0 to stop-1
  #   ## Nim:    t[_..<stop]     -> _ means all of dimension, then exclusive
  #   ##
  #   ## Python: t[:]           -> all elements
  #   ## Nim:    t[_.._]        -> full span
  #   ##
  #   ## Python: t[::step]      -> every step-th element
  #   ## Nim:    t[_.._|step]   -> full span with step
  #   ##
  #   ## Python: t[1:5:2]       -> elements 1, 3 (start=1, stop=5, step=2)
  #   ## Nim:    t[1..<5|2]     -> elements 1, 3

  #   test "Python t[:2] (exclusive stop) -> Nim t[_..<2]":
  #     ## Python: tensor[:2] gets indices 0, 1
  #     ## Nim: tensor[_..<2] gets indices 0, 1 (exclusive)
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[_..<2, _]
  #     check: sliced.shape[0] == 2
  #     check: sliced.shape[1] == 5

  #   test "Python t[3:] (start to end) -> Nim t[3..^1]":
  #     ## Python: tensor[3:] gets indices 3, 4
  #     ## Nim: tensor[3..^1] gets indices 3, 4 (^1 = last element)
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[3..^1, _]
  #     check: sliced.shape[0] == 2
  #     check: sliced.shape[1] == 5
  #     check: sliced[0, 0].item(float64) == 4.0  # Row 3 (base 4)
  #     check: sliced[1, 0].item(float64) == 5.0  # Row 4 (base 5)

  #   test "Python t[:] (full slice) -> Nim t[_.._]":
  #     ## Python: tensor[:] gets all elements
  #     ## Nim: tensor[_.._] gets all elements
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[_.._, _]
  #     check: sliced.shape[0] == 5
  #     check: sliced.shape[1] == 5
  #     check: sliced == t

  #   test "Python t[::2] (step 2) -> Nim t[_.._|2] or t[|2]":
  #     ## Python: tensor[::2] gets every 2nd element
  #     ## Nim: tensor[_.._|2] or tensor[|2] (cleaner!) gets every 2nd element
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced1 = t[_.._|2, _]
  #     let sliced2 = t[|2, _]
  #     check: sliced1.shape[0] == 3  # indices 0, 2, 4
  #     check: sliced1[0, 0].item(float64) == 1.0  # Row 0
  #     check: sliced1[1, 0].item(float64) == 3.0  # Row 2
  #     check: sliced1[2, 0].item(float64) == 5.0  # Row 4
  #     check: sliced1 == sliced2  # Both syntaxes are equivalent

  #   test "New: tensor[|3] (unary pipe step) -> every 3rd element":
  #     ## The unary `|step` syntax is cleaner than `_.._|step`
  #     ## tensor[|3] -> Slice(None, None, 3)
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[|3, _]
  #     check: sliced.shape[0] == 2  # indices 0, 3
  #     check: sliced[0, 0].item(float64) == 1.0  # Row 0
  #     check: sliced[1, 0].item(float64) == 4.0  # Row 3

  #   test "New: tensor[|2, 0] (stepped span with index) -> Slice(None, None, 2), 0":
  #     ## tensor[|2, 0] -> every 2nd element of dim 0, index 0 of dim 1
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[|2, 0]
  #     check: sliced.shape[0] == 3
  #     check: sliced.shape[1] == 1
  #     check: sliced[0, 0].item(float64) == 1.0  # Row 0, col 0
  #     check: sliced[1, 0].item(float64) == 3.0  # Row 2, col 0
  #     check: sliced[2, 0].item(float64) == 5.0  # Row 4, col 0

  #   test "New: tensor[1, |2, _] (mixed indexing with stepped span)":
  #     ## tensor[1, |2, _] -> index 1, every 2nd of dim 1, all of dim 2
  #     let t = arange(24, kFloat64).reshape(@[2, 3, 4])
  #     let sliced = t[1, |2, _]
  #     check: sliced.shape[0] == 2  # Every 2nd of dim 1 (indices 0, 2)
  #     check: sliced.shape[1] == 4  # All of dim 2
  #     check: sliced[0, 0].item(float64) == 8.0   # t[1, 0, 0]
  #     check: sliced[1, 0].item(float64) == 16.0  # t[1, 2, 0]

  #   test "Python t[1:4] (start, exclusive stop) -> Nim t[1..<4]":
  #     ## Python: tensor[1:4] gets indices 1, 2, 3
  #     ## Nim: tensor[1..<4] gets indices 1, 2, 3 (exclusive)
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[1..<4, _]
  #     check: sliced.shape[0] == 3
  #     check: sliced[0, 0].item(float64) == 2.0  # Row 1
  #     check: sliced[2, 0].item(float64) == 4.0  # Row 3

  #   test "Python t[1:4:2] (start, stop, step) -> Nim t[1..<4|2]":
  #     ## Python: tensor[1:4:2] gets indices 1, 3
  #     ## Nim: tensor[1..<4|2] gets indices 1, 3
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[1..<4|2, _]
  #     check: sliced.shape[0] == 2
  #     check: sliced[0, 0].item(float64) == 2.0  # Row 1
  #     check: sliced[1, 0].item(float64) == 4.0  # Row 3

  #   test "Python t[:-1] (to before last) -> Nim t[_..<^1]":
  #     ## Python: tensor[:-1] gets all but last element
  #     ## Nim: tensor[_..<^1] gets all but last (^1 = last, < makes it exclusive)
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[_..<^1, _]
  #     check: sliced.shape[0] == 4
  #     check: sliced[3, 0].item(float64) == 4.0  # Row 3 (not row 4)

  #   test "Python t[-3:] (last 3) -> Nim t[^3..^1]":
  #     ## Python: tensor[-3:] gets last 3 indices
  #     ## Nim: tensor[^3..^1] gets last 3 (^3 = 3rd from end, ^1 = last)
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[^3..^1, _]
  #     check: sliced.shape[0] == 3  # indices 2, 3, 4
  #     check: sliced[0, 0].item(float64) == 3.0  # Row 2
  #     check: sliced[2, 0].item(float64) == 5.0  # Row 4

  #   test "Python t[::-1] (reverse) -> Use flip() in Nim":
  #     ## PyTorch/Python: tensor[::-1] reverses the tensor along a dimension
  #     ## libtorch/Nim: Negative steps are NOT supported in Slice()
  #     ##            Use flip() instead
  #     ##
  #     ## What Python t[::-1] would give: first element becomes last, etc.
  #     ## What Nim flip() gives: same result
  #     var t = genShiftedVandermonde5x5(kFloat64)
  #     let reversed = t.flip(@[0])

  #     ## flip() along dim 0 should give same as t[::-1] in Python
  #     check: reversed[0, 0].item(float64) == 5.0   # Last row of original
  #     check: reversed[4, 0].item(float64) == 1.0   # First row of original
  #     check: reversed[0, 4].item(float64) == 3125.0 # 5^5 = 3125
  #     check: reversed[4, 4].item(float64) == 1.0    # 1^5 = 1

  #   test "Negative steps t[|-2] are NOT supported - use flip()":
  #     ## libtorch's Slice() does NOT support negative steps
  #     ## Python: tensor[::2] would work, tensor[::-2] would reverse with step 2
  #     ## Nim: t[_.._|2] works, t[_.._|-2] raises compile error
  #     ##
  #     ## To reverse and step, use: tensor.flip(dim).slice(...)
  #     var t = genShiftedVandermonde5x5(kFloat64)
  #     let reversed = t.flip(@[0])
  #     let stepped = reversed[_.._|2, _]  # Reverse, then take every 2nd
  #     check: stepped.shape[0] == 3  # rows 0, 2, 4 of reversed = rows 4, 2, 0 of original
  #     check: stepped[0, 0].item(float64) == 5.0  # Row 4 (first of reversed)
  #     check: stepped[2, 0].item(float64) == 1.0  # Row 0 (last of reversed)

  # suite "Ellipsis `...` or ellipsis - Python '...' equivalent to libtorch Ellipsis":
  #   ## Note: In Nim, `...` must be used quoted a[0..<2, `...`] or a[0..<2, ellipsis]
  #   ##
  #   ## Both are equivalent:
  #   ##   tensor[`...`, 0]
  #   ##   tensor[ellipsis, 0]

  #   test "Single ellipsis - tensor[...] -> tensor.index({Ellipsis})":
  #     ## PyTorch: tensor[...]
  #     ## libtorch: tensor.index({torch::indexing::Ellipsis})
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     check: t[`...`].shape[0] == 5
  #     check: t[`...`].shape[1] == 5

  #   test "Ellipsis with other indices - tensor[`...`, 0]":
  #     ## PyTorch: tensor[..., 0]
  #     ## libtorch: tensor.index({Ellipsis, 0})
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[`...`, 0]
  #     check: sliced.shape[0] == 5
  #     check: sliced[0, 0].item(float64) == 1.0  # (1+1)^(0+1) = 2^1 = 2

  #   test "IndexEllipsis constant works same as quoted ellipsis":
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let with_const = t[ellipsis, 0]
  #     let with_quoted = t[`...`, 0]
  #     check: with_const.shape == with_quoted.shape
  #     check: with_const == with_quoted

  #   test "Ellipsis expansion - tensor[..., 0] equivalent to tensor[:, :, 0]":
  #     ## Demonstrates ellipsis expansion
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let ellipsis_result = t[`...`, 0]
  #     let explicit = t[_, _, 0]
  #     check: ellipsis_result == explicit

  #   test "Leading ellipsis - tensor[0, ...]":
  #     ## PyTorch: tensor[0, ...]
  #     ## libtorch: tensor.index({0, Ellipsis})
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[0, `...`]
  #     check: sliced.shape[0] == 5
  #     check: sliced[0].item(float64) == 1.0  # Row 0, all columns (powers of 1 = 1)

  #   test "Middle ellipsis - tensor[1, ..., 0]":
  #     ## PyTorch: tensor[1, ..., 0]
  #     ## libtorch: tensor.index({1, Ellipsis, 0})
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[1, `...`, 0]
  #     check: sliced.shape[0] == 5
  #     check: sliced[0].item(float64) == 2.0  # Row 1 (base 2), column 0 = 2^1 = 2

  # suite "Arraymancer-style Indexing (Span `_`)":
  #   ## Arraymancer uses `_` for full span (equivalent to Python `:`)
  #   ## Note: libtorch `_` maps to `Slice()`

  #   test "Single span - tensor[_]":
  #     ## Arraymancer: tensor[_] / tensor[_, _]
  #     ## Maps to: Slice() / Slice(None, None)
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[_, _]
  #     check: sliced.shape[0] == 5
  #     check: sliced.shape[1] == 5

  #   test "Span on first dimension only - tensor[_, 2]":
  #     ## Arraymancer: tensor[_, 2]
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[_, 2]
  #     check: sliced.shape[0] == 5
  #     check: sliced.shape[1] == 1

  #   test "Span with slice - tensor[1..3, _]":
  #     ## Arraymancer: tensor[1..3, _]
  #     ## PyTorch: tensor[1:3, :]
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[1..<3, _]
  #     check: sliced.shape[0] == 2
  #     check: sliced.shape[1] == 5

  #   test "Span with partial - tensor[_..2, 2]":
  #     ## Arraymancer: tensor[_..2, 2]
  #     ## PyTorch: tensor[:2, 2]
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[_..<2, 2]
  #     check: sliced.shape[0] == 2
  #     check: sliced.shape[1] == 1

  #   test "Full span shorthand - tensor[_.._]":
  #     ## Arraymancer: tensor[_.._]
  #     ## This should be equivalent to Slice() i.e. full dimension
  #     ## Currently the desugar might map this to Ellipsis - needs fixing
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[_.._, _]
  #     check: sliced.shape[0] == 5
  #     check: sliced.shape[1] == 5

  #   test "Span with step - tensor[_.._|2]":
  #     ## Arraymancer: tensor[_.._|2]
  #     ## PyTorch: tensor[::2]
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[_.._|2, _]
  #     check: sliced.shape[0] == 3  # rows 0, 2, 4
  #     check: sliced.shape[1] == 5

  # suite "Negative Indexing (End-relative with ^N)":
  #   ## Arraymancer and Nim use ^N for end-relative indexing:
  #   ## ^1 = last element (index size-1)
  #   ## ^2 = second-to-last element (index size-2)
  #   ## ^3 = third-to-last element (index size-3)
  #   ## etc.
  #   ##
  #   ## This is equivalent to Python's negative indexing: tensor[-1]
  #   ## But for slices, the end is INCLUSIVE in Nim's .. syntax.
  #   ## So ^1 means the actual last element.

  #   test "Single negative index - tensor[^1] (last element)":
  #     ## ^1 = last element (PyTorch: tensor[-1])
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let val = t[^1, ^1]
  #     check: val.item(float64) == 3125.0  # 5^5 = 3125

  #   test "Second-to-last - tensor[^2, ^2]":
  #     ## ^2 = second-to-last element
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let val = t[^2, ^2]
  #     check: val.item(float64) == 625.0  # 5^4 = 625

  #   test "Third-to-last - tensor[^3, ^3]":
  #     ## ^3 = third-to-last element
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let val = t[^3, ^3]
  #     check: val.item(float64) == 125.0  # 5^3 = 125

  #   test "Inclusive slice to end - tensor[0..^1] (includes last)":
  #     ## 0..^1 means from 0 to last element (inclusive)
  #     ## PyTorch: tensor[0:] or tensor[:]
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[0..^1, _]
  #     check: sliced.shape[0] == 5

  #   test "Exclusive slice to end - tensor[0..<^1] (excludes last)":
  #     ## 0..<^1 means from 0 to before last element
  #     ## PyTorch: tensor[0:-1]
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[0..<^1, _]
  #     check: sliced.shape[0] == 4

  #   test "Negative slice - tensor[^3..^1] (third-to-last to last)":
  #     ## ^3..^1 means from third-to-last to last (inclusive)
  #     ## This gets rows 3 and 4 (bases 4 and 5)
  #     ## PyTorch: tensor[-3:]
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[^3..^1, _]
  #     check: sliced.shape[0] == 2  # rows 3 and 4
  #     check: sliced[0, 0].item(float64) == 4.0  # base 4
  #     check: sliced[1, 0].item(float64) == 5.0  # base 5

  #   test "Negative exclusive slice - tensor[^3..<^1]":
  #     ## ^3..<^1 means from third-to-last to before last
  #     ## Gets only row 3 (base 4)
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[^3..<^1, _]
  #     check: sliced.shape[0] == 1
  #     check: sliced[0, 0].item(float64) == 4.0  # base 4

  #   test "Span with negative index - tensor[_..^2]":
  #     ## _..^2 means from start to second-to-last (inclusive)
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[_..^2, _]
  #     check: sliced.shape[0] == 4  # rows 0, 1, 2, 3

  #   test "Negative span with step - tensor[^4.._|2]":
  #     ## ^4.._|2 from fourth-from-end to end, step 2
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[^4.._|2, _]
  #     check: sliced.shape[0] == 2  # rows 1 (^4=1) and 3 (^2=3)

  # suite "Multidimensional Slice Behavior":
  #   ## Reference: https://pytorch.org/cppdocs/notes/tensor_indexing.html
  #   ## For 3D tensor: a[0:3] is equivalent to a[0:3, ...] / a[0:3, :, :]

  #   test "Partial slice on 3D tensor - matches first dimension":
  #     ## a[0:3] equivalent to a[0:3, ...] on 3D tensor
  #     let t3d = arange(24, kFloat64).reshape(@[2, 3, 4])
  #     let sliced = t3d[0..<2, _, _]
  #     check: sliced.shape[0] == 2
  #     check: sliced.shape[1] == 3
  #     check: sliced.shape[2] == 4

  #   test "Slice equivalent to explicit full spans - t[0..<3] vs t[0..<3, _, :]":
  #     let t3d = arange(24, kFloat64).reshape(@[2, 3, 4])
  #     let implicit = t3d[0..<2]
  #     let explicit = t3d[0..<2, _, _]
  #     check: implicit.shape == explicit.shape

  #   test "Slice last dimension only - t[..., 0:2]":
  #     ## PyTorch: t[..., 0:2]
  #     ## libtorch: t.index({Ellipsis, Slice(0, 2)})
  #     let t3d = arange(24, kFloat64).reshape(@[2, 3, 4])
  #     let sliced = t3d[`...`, 0..<2]
  #     check: sliced.shape[0] == 2
  #     check: sliced.shape[1] == 3
  #     check: sliced.shape[2] == 2

  # suite "Assignment Operations (index_put_)":
  #   test "Point assignment - tensor[0, 0] = 999":
  #     ## PyTorch: tensor[0, 0] = 999
  #     ## libtorch: tensor.index_put_({0, 0}, 999)
  #     var t = genShiftedVandermonde5x5(kFloat64)
  #     t[0, 0] = 999.0
  #     check: t[0, 0].item(float64) == 999.0

  #   test "Slice assignment - tensor[0..2, 0..2] = 0":
  #     ## PyTorch: tensor[0:2, 0:2] = 0
  #     ## libtorch: tensor.index_put_({Slice(0, 2), Slice(0, 2)}, 0)
  #     var t = genShiftedVandermonde5x5(kFloat64)
  #     t[0..<2, 0..<2] = 0.0
  #     check: t[0, 0].item(float64) == 0.0
  #     check: t[1, 1].item(float64) == 0.0
  #     check: t[2, 2].item(float64) == 81.0  # Unchanged

  #   test "Ellipsis assignment - tensor[...] = 0":
  #     ## PyTorch: tensor[...] = 0
  #     ## libtorch: tensor.index_put_({Ellipsis}, 0)
  #     var t = genShiftedVandermonde5x5(kFloat64)
  #     t[IndexEllipsis] = 0.0
  #     check: t[0, 0].item(float64) == 0.0
  #     check: t[4, 4].item(float64) == 0.0

  #   test "Assignment with step - tensor[::2, ::2] = 999":
  #     ## PyTorch: tensor[::2, ::2] = 999
  #     var t = genShiftedVandermonde5x5(kFloat64)
  #     t[_..<5|2, _..<5|2] = 999.0
  #     check: t[0, 0].item(float64) == 999.0
  #     check: t[2, 2].item(float64) == 999.0
  #     check: t[1, 1].item(float64) == 81.0  # Unchanged

  # suite "Common Attention Mechanism Patterns":
  #   ## These patterns appear frequently in transformer attention implementations

  #   test "Q/K/V slicing for multi-head attention":
  #     ## Pattern: tensor[:, start_idx:start_idx+head_dim, :]
  #     var t = arange(100, kFloat64).reshape(@[2, 5, 10])
  #     let head_dim = 4
  #     let start_idx = 0
  #     let q = t[_, start_idx..<start_idx+head_dim, _]
  #     check: q.shape[0] == 2
  #     check: q.shape[1] == 4
  #     check: q.shape[2] == 10

  #   test "Slicing all heads for a position - tensor[:, :, pos_idx]":
  #     ## Extract a single position across all heads
  #     let t = arange(100, kFloat64).reshape(@[2, 5, 10])
  #     let pos_idx = 2
  #     let sliced = t[_, _, pos_idx]
  #     check: sliced.shape[0] == 2
  #     check: sliced.shape[1] == 5
  #     check: sliced.shape[2] == 1

  #   test "Causal mask slicing - tensor[t, :, t_end:]":
  #     ## Pattern for causal attention (upper triangular)
  #     var t = arange(30, kFloat64).reshape(@[2, 3, 5])
  #     let t_idx = 1
  #     let sliced = t[t_idx, _, t_idx+1..<5]
  #     check: sliced.shape[0] == 1
  #     check: sliced.shape[1] == 3
  #     check: sliced.shape[2] == 3

  #   test "Attention score masking with ellipsis - mask[..., q_idx, k_idx]":
  #     ## Pattern for attention weight masking
  #     var t = arange(60, kFloat64).reshape(@[2, 3, 2, 5])
  #     let q_idx = 1
  #     let k_idx = 2
  #     let sliced = t[`...`, q_idx, k_idx]
  #     check: sliced.shape[0] == 2
  #     check: sliced.shape[1] == 3

  #   test "Slicing last dimension for residual - x[:, :, -head_size:]":
  #     ## Pattern: keep only the last head_size channels
  #     var t = arange(100, kFloat64).reshape(@[2, 5, 10])
  #     let head_size = 2
  #     let sliced = t[_, _, ^head_size..^0]
  #     check: sliced.shape[2] == 2

  #   test "Interleaved slicing for RoPE/AliBi - tensor[batch, seq, ::2]":
  #     ## Pattern: every other element (used in positional encoding)
  #     var t = arange(40, kFloat64).reshape(@[2, 4, 5])
  #     let sliced = t[_, _, _..<5|2]
  #     check: sliced.shape[2] == 3  # 5 elements, step 2 = 3

  #   test "Empty slice - tensor[0..0]":
  #     ## PyTorch: tensor[0:0] returns empty tensor
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[0..<0, _]
  #     check: sliced.shape[0] == 0

  #   test "Full range slice - tensor[0..<5]":
  #     ## Slice covering entire dimension
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[0..<5, _]
  #     check: sliced.shape[0] == 5
  #     check: sliced.shape[1] == 5

  #   test "Single element slice - tensor[2..<3]":
  #     ## Slice producing single element
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[2..<3, _]
  #     check: sliced.shape[0] == 1
  #     check: sliced.shape[1] == 5

  #   test "Large step - tensor[::100] with small tensor":
  #     ## Step larger than dimension size
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[_..<5|100, _]
  #     check: sliced.shape[0] == 1  # Only first element

  #   test "Reverse with negative step - use flip() not t[::-1]":
  #     ## Python: tensor[::-1] reverses along a dimension
  #     ## Nim: Negative step syntax `|_` is NOT supported
  #     ##      Use flip() instead
  #     ##
  #     ## Example: What Python t[:, :, ::-1] would give in Nim:
  #     var t = genShiftedVandermonde5x5(kFloat64)
  #     let reversed = t.flip(@[1])  # Reverse along dim 1
  #     let first_col = reversed[_, 0]
  #     check: first_col[0, 0].item(float64) == 1.0    # Column 4 reversed = column 0
  #     check: first_col[4, 0].item(float64) == 3125.0 # Column 4

  #   test "Integer and slice mix - tensor[0, 1:4]":
  #     ## PyTorch: tensor[0, 1:4]
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[0, 1..<4]
  #     check: sliced.shape[0] == 3

  #   test "Slice and ellipsis - tensor[1:3, ..., 0:2]":
  #     ## PyTorch: tensor[1:3, ..., 0:2]
  #     let t3d = arange(60, kFloat64).reshape(@[2, 3, 10])
  #     let sliced = t3d[1..<3, `...`, 0..<2]
  #     check: sliced.shape[0] == 2
  #     check: sliced.shape[1] == 10

  #   test "Integer array indexing - tensor[[0, 2, 4]]":
  #     ## PyTorch: tensor[[0, 2, 4]] - fancy indexing
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let indices_seq = @[0, 2, 4].to(kInt64)
  #     let indices = indices_seq.toTorchTensor().clone()
  #     let sliced = t.index_select(0, indices)
  #     check: sliced.shape[0] == 3

  #   test "Boolean mask indexing - tensor[tensor > 10]":
  #     ## PyTorch: tensor[tensor > 10] - masked indexing
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let mask = t > 10.0
  #     let sliced = t[mask]
  #     # Count of elements > 10 in Vandermonde matrix

  #   test "Single slice on 3D - affects first dimension only":
  #     ## PyTorch: t3d[0:2]
  #     ## Equivalent to: t3d[0:2, :, :]
  #     let sliced = t3d[0..<2]
  #     check: sliced.shape[0] == 2
  #     check: sliced.shape[1] == 3
  #     check: sliced.shape[2] == 4

  #   test "Slice middle dimension - t3d[:, 0:2, :]":
  #     ## PyTorch: t3d[:, 0:2, :]
  #     let sliced = t3d[_, 0..<2, _]
  #     check: sliced.shape[0] == 2
  #     check: sliced.shape[1] == 2
  #     check: sliced.shape[2] == 4

  #   test "Slice last dimension - t3d[:, :, 0:2]":
  #     ## PyTorch: t3d[:, :, 0:2]
  #     let sliced = t3d[_, _, 0..<2]
  #     check: sliced.shape[0] == 2
  #     check: sliced.shape[1] == 3
  #     check: sliced.shape[2] == 2

  #   test "Slice with ellipsis - t3d[0, ...]":
  #     ## PyTorch: t3d[0, ...]
  #     ## All of remaining dimensions
  #     let sliced = t3d[0, `...`]
  #     check: sliced.shape[0] == 3
  #     check: sliced.shape[1] == 4

  #   test "Ellipsis expansion - t3d[...] equals t3d":
  #     let sliced = t3d[IndexEllipsis]
  #     check: sliced.shape[0] == 2
  #     check: sliced.shape[1] == 3
  #     check: sliced.shape[2] == 4

  #   test "Multiple indices with slice - t3d[0, 0:2, 1:3]":
  #     ## PyTorch: t3d[0, 0:2, 1:3]
  #     let sliced = t3d[0, 0..<2, 1..<3]
  #     check: sliced.shape[0] == 2
  #     check: sliced.shape[1] == 2

  #   test "Single slice on 4D - affects first dimension":
  #     let sliced = t4d[0..<2]
  #     check: sliced.shape[0] == 2
  #     check: sliced.shape[1] == 3
  #     check: sliced.shape[2] == 4
  #     check: sliced.shape[3] == 5

  #   test "Ellipsis in 4D - t4d[..., 0]":
  #     ## PyTorch: t4d[..., 0]
  #     let sliced = t4d[`...`, 0]
  #     check: sliced.shape[0] == 2
  #     check: sliced.shape[1] == 3
  #     check: sliced.shape[2] == 4

  #   test "Leading ellipsis - t4d[0, ..., 0]":
  #     ## PyTorch: t4d[0, ..., 0]
  #     let sliced = t4d[0, `...`, 0]
  #     check: sliced.shape[0] == 3
  #     check: sliced.shape[1] == 4

  #   test "All spans - t4d[_, _, _, _]":
  #     let sliced = t4d[_, _, _, _]
  #     check: sliced.shape == @[2, 3, 4, 5]

  #   test "Slice head dimension - t4d[:, 0..<2, :, :]":
  #     let sliced = t4d[_, 0..<2, _, _]
  #     check: sliced.shape[0] == 2
  #     check: sliced.shape[1] == 2
  #     check: sliced.shape[2] == 4
  #     check: sliced.shape[3] == 5

  #   test "Ellipsis in 4D - t4d[..., 0]":
  #     ## PyTorch: t4d[..., 0]
  #     let sliced = t4d[`...`, 0]
  #     check: sliced.shape[0] == 2
  #     check: sliced.shape[1] == 3
  #     check: sliced.shape[2] == 4

  #   test "Leading ellipsis - t4d[0, ..., 0]":
  #     ## PyTorch: t4d[0, ..., 0]
  #     let sliced = t4d[0, `...`, 0]
  #     check: sliced.shape[0] == 3
  #     check: sliced.shape[1] == 4

  #   test "All spans - t4d[_, _, _, _]":
  #     let sliced = t4d[_, _, _, _]
  #     check: sliced.shape == @[2, 3, 4, 5]

  #   test "Slice head dimension - t4d[:, 0..<2, :, :]":
  #     let sliced = t4d[_, 0..<2, _, _]
  #     check: sliced.shape[0] == 2
  #     check: sliced.shape[1] == 2
  #     check: sliced.shape[2] == 4
  #     check: sliced.shape[3] == 5

  # suite "Arraymancer Test Suite Compatibility":
  #   ## Ported tests from Arraymancer to ensure compatibility

  #   let vandermonde = genShiftedVandermonde5x5(kFloat64)
  #   # vandermonde[0, 0] = 1, vandermonde[1, 1] = 4, vandermonde[2, 2] = 27

  #   test "Basic indexing - v[2, 3]":
  #     ## Arraymancer: v[2, 3]
  #     check: vandermonde[2, 3].item(float64) == 81.0

  #   test "Basic slicing - v[1..2, 3]":
  #     ## Arraymancer: v[1..2, 3]
  #     let sliced = vandermonde[1..<2, 3]
  #     check: sliced.shape[0] == 1
  #     check: sliced[0, 0].item(float64) == 16.0  # (1+1)^(3+1) = 2^4 = 16

  #   test "Span slicing - v[_, 3]":
  #     ## Arraymancer: v[_, 3]
  #     let sliced = vandermonde[_, 3]
  #     check: sliced.shape[0] == 5
  #     check: sliced.shape[1] == 1

  #   test "Span slicing with range - v[1.._, 3]":
  #     ## Arraymancer: v[1.._, 3]
  #     let sliced = vandermonde[1..<5, 3]
  #     check: sliced.shape[0] == 4
  #     check: sliced[0, 0].item(float64) == 16.0

  #   test "Span slicing from start - v[_..3, 3]":
  #     ## Arraymancer: v[_..3, 3]
  #     let sliced = vandermonde[_..<3, 3]
  #     check: sliced.shape[0] == 3
  #     check: sliced[0, 0].item(float64) == 1.0

  #   test "Stepping - v[1..3|2, 3]":
  #     ## Arraymancer: v[1..3|2, 3]
  #     let sliced = vandermonde[1..<3|2, 3]
  #     check: sliced.shape[0] == 1
  #     check: sliced[0, 0].item(float64) == 16.0

  #   test "Span stepping - v[_.._|2, 3]":
  #     ## Arraymancer: v[_.._|2, 3]
  #     let sliced = vandermonde[_..<5|2, 3]
  #     check: sliced.shape[0] == 3
  #     check: sliced[0, 0].item(float64) == 1.0
  #     check: sliced[1, 0].item(float64) == 81.0

  #   test "Full span - v[_.._, 3]":
  #     ## Arraymancer: v[_.._, 3]
  #     let sliced = vandermonde[_.._, 3]
  #     check: sliced.shape[0] == 5
  #     check: sliced.shape[1] == 1

  # suite "Assignment Compatibility":
  #   var vandermonde_mut = genShiftedVandermonde5x5(kFloat64)

  #   test "Slice to single value - v[1..2, 3..4] = 999":
  #     ## Arraymancer: v[1..2, 3..4] = 999
  #     vandermonde_mut[1..<2, 3..<4] = 999.0
  #     check: vandermonde_mut[1, 3].item(float64) == 999.0
  #     check: vandermonde_mut[1, 4].item(float64) == 999.0
  #     check: vandermonde_mut[2, 3].item(float64) == 999.0
  #     check: vandermonde_mut[2, 4].item(float64) == 999.0

  #   test "Slice to array - v[0..1, 0..1] = [[111, 222], [333, 444]]":
  #     ## Arraymancer: v[0..1, 0..1] = [[111, 222], [333, 444]]
  #     var data = newSeq[seq[float64]](2)
  #     data[0] = @[111.0, 222.0]
  #     data[1] = @[333.0, 444.0]
  #     vandermonde_mut[0..<1, 0..<1] = data.toTorchTensor()
  #     check: vandermonde_mut[0, 0].item(float64) == 111.0
  #     check: vandermonde_mut[1, 1].item(float64) == 444.0

  # suite "Ellipsis Behavior Verification":
  #   test "Ellipsis equivalence - t[2, ...] == t[2, _, _, _]":
  #     let t5d = arange(100, kFloat64).reshape(@[2, 2, 5, 5, 1])
  #     let with_ellipsis = t5d[2, `...`]
  #     let explicit = t5d[2, _, _, _, _]
  #     check: with_ellipsis.shape == explicit.shape
  #     check: with_ellipsis == explicit

  #   test "Leading ellipsis equivalence - t[..., 0] == t[_, _, _, 0]":
  #     let t5d = arange(100, kFloat64).reshape(@[2, 2, 5, 5, 1])
  #     let with_ellipsis = t5d[`...`, 0]
  #     let explicit = t5d[_, _, _, _, 0]
  #     check: with_ellipsis.shape == explicit.shape
  #     check: with_ellipsis == explicit

  #   test "Middle ellipsis - t[1, ..., 0]":
  #     let t5d = arange(100, kFloat64).reshape(@[2, 2, 5, 5, 1])
  #     let sliced = t5d[1, `...`, 0]
  #     check: sliced.shape[0] == 2
  #     check: sliced.shape[1] == 5

  #   test "Double ellipsis is invalid (should error or behave as single)":
  #     ## PyTorch doesn't allow multiple ellipsis
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     discard

  # suite "Slice Span vs Ellipsis Distinction":
  #   ## This tests the fix for the issue where _.._ was wrongly mapped to Ellipsis
  #   test "Full span with _.._ should be Slice() not Ellipsis":
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[_.._, _]
  #     check: sliced.shape == @[5, 5]
  #     check: sliced == t

  #   test "Single _ should be Slice() not Ellipsis":
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[_, _]
  #     check: sliced.shape == @[5, 5]
  #     check: sliced == t

  #   test "Span with integer - t[0.._, _] should slice first dimension":
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[0..<5, _]
  #     check: sliced.shape == @[5, 5]
  #     check: sliced == t

  #   test "Slice with span - t[0..<3, _.._] should combine slices":
  #     let t = genShiftedVandermonde5x5(kFloat64)
  #     let sliced = t[0..<3, _.._]
  #     check: sliced.shape == @[3, 5]
  #     check: sliced[0, 0].item(float64) == 1.0

  # suite "Span vs Ellipsis - anti-regression":
  #   ## This suite verifies the fix for the bug where _.._ was mapped to Ellipsis instead of Slice()
  #   ##
  #   ## Summary of the fix:
  #   ## - `_` (single) maps to Slice() (full span for one dimension)
  #   ## - `_.._` (double) maps to Slice() (full span, NOT Ellipsis!)
  #   ## - `...` (ellipsis) maps to torch::indexing::Ellipsis (expands to fill remaining dims)
  #   ##
  #   ## Example showing the difference on a 3D tensor:
  #   ##   t3d[_, 0, 0]  = Slice(), 0, 0   -> First dim all, second dim index 0, third dim index 0
  #   ##   t3d[..., 0, 0] = Ellipsis, 0, 0 -> Last two dims index 0
  #   ##   t3d[0, ..., 0] = 0, Ellipsis, 0 -> First dim index 0, last dim index 0
  #   ##
  #   ## PyTorch equivalents:
  #   ##   t3d[:, 0, 0]        (using :) = t3d[_, 0, 0] in Nim
  #   ##   t3d[..., 0, 0]     (using ...) = t3d[Ellipsis, 0, 0] in Nim
  #   ##   t3d[0, ..., 0]     (using ...) = t3d[0, Ellipsis, 0] in Nim

  #   test "Span (_) affects only ONE dimension - t[_, 0] on 3D tensor":
  #     ## PyTorch: t3d[:, 0]
  #     ## Gets all of dimension 0, index 0 of dimension 1, all of dimension 2
  #     let t3d = arange(24, kFloat64).reshape(@[2, 3, 4])
  #     let sliced = t3d[_, 0, _]
  #     check: sliced.shape[0] == 2
  #     check: sliced.shape[1] == 1
  #     check: sliced.shape[2] == 4
  #     check: sliced[0, 0, 0].item(float64) == 0.0  # t3d[0, 0, 0] = 0

  #   test "Ellipsis (...) fills ALL remaining dimensions - t[..., 0] on 3D tensor":
  #     ## PyTorch: t3d[..., 0]
  #     ## Gets all dimensions for the ellipsis part, index 0 only for the last dim
  #     let t3d = arange(24, kFloat64).reshape(@[2, 3, 4])
  #     let sliced = t3d[`...`, 0]
  #     check: sliced.shape[0] == 2
  #     check: sliced.shape[1] == 3
  #     check: sliced.shape[2] == 1
  #     check: sliced[0, 0, 0].item(float64) == 0.0  # t3d[0, 0, 0] = 0
  #     check: sliced[0, 1, 0].item(float64) == 4.0  # t3d[0, 1, 0] = 4

  #   test "Leading ellipsis - t[0, ...] on 3D tensor":
  #     ## PyTorch: t3d[0, ...]
  #     let t3d = arange(24, kFloat64).reshape(@[2, 3, 4])
  #     let sliced = t3d[0, `...`]
  #     check: sliced.shape[0] == 3
  #     check: sliced.shape[1] == 4
  #     check: sliced[0, 0].item(float64) == 0.0   # t3d[0, 0, 0] = 0
  #     check: sliced[1, 0].item(float64) == 4.0   # t3d[0, 1, 0] = 4
  #     check: sliced[3, 3].item(float64) == 15.0  # t3d[0, 3, 3] = 15

  #   test "Span with _.._ on 3D - t[_.._, _, _] should equal full tensor":
  #     let t3d = arange(24, kFloat64).reshape(@[2, 3, 4])
  #     let sliced = t3d[_.._, _, _]
  #     check: sliced.shape == @[2, 3, 4]
  #     check: sliced == t3d

  #   test "Compare _.._ vs Ellipsis on 2D - they should be DIFFERENT":
  #     let t2d = genShiftedVandermonde5x5(kFloat64)
  #     let with_underscore = t2d[_.._, _]
  #     let with_ellipsis = t2d[IndexEllipsis]
  #     check: with_underscore.shape == @[5, 5]
  #     check: with_ellipsis.shape == @[5, 5]
  #     check: with_underscore == t2d  # _.._ should be full span
  #     check: with_ellipsis == t2d   # Ellipsis should also be full (expands to all dims)

  #   test "Compare _.._ vs Ellipsis on 3D - they should be DIFFERENT":
  #     let t3d = arange(24, kFloat64).reshape(@[2, 3, 4])
  #     let with_underscore = t3d[_.._, _]
  #     let with_ellipsis = t3d[IndexEllipsis]
  #     check: with_underscore.shape == @[5, 4]  # _.._ is one dim, _ is all remaining
  #     check: with_ellipsis.shape == @[2, 3, 4] # Ellipsis expands to all dims
  #     check: with_underscore != with_ellipsis

  #   test "Mixed: slice + ellipsis + span - t[0..<2, ..., _]":
  #     ## First dim slice 0..<2, middle dims ellipsis (all), last dim span (all)
  #     let t4d = arange(120, kFloat64).reshape(@[2, 3, 4, 5])
  #     let sliced = t4d[0..<2, `...`, _]
  #     check: sliced.shape[0] == 2
  #     check: sliced.shape[1] == 3
  #     check: sliced.shape[2] == 4
  #     check: sliced.shape[3] == 5

  # suite "^N End-relative Indexing Reference":
  #   ## Summary: ^1 is the LAST element, ^2 is second-to-last, etc.
  #   ##
  #   ## For a 5-element array/dimension (indices 0, 1, 2, 3, 4):
  #   ##   ^1 refers to index 4 (last)
  #   ##   ^2 refers to index 3 (second-to-last)
  #   ##   ^3 refers to index 2 (third-to-last)
  #   ##   ^4 refers to index 1 (fourth-to-last)
  #   ##   ^5 refers to index 0 (first element, but this is unusual)
  #   ##
  #   ## This is equivalent to Python's -1, -2, -3, etc. negative indexing.

  #   let arr5 = @[10.0, 20.0, 30.0, 40.0, 50.0].toTorchTensor()

  #   test "^1 is the last element (index 4)":
  #     check: arr5[^1].item(float64) == 50.0

  #   test "^2 is second-to-last (index 3)":
  #     check: arr5[^2].item(float64) == 40.0

  #   test "^3 is third-to-last (index 2)":
  #     check: arr5[^3].item(float64) == 30.0

  #   test "^4 is fourth-to-last (index 1)":
  #     check: arr5[^4].item(float64) == 20.0

  #   test "^5 equals ^1 (index 0, same as first)":
  #     check: arr5[^5].item(float64) == 10.0

  #   test "Slice from ^3 to ^1 (inclusive)":
  #     ## ^3..^1 means indices 2, 3, 4
  #     let sliced = arr5[^3..^1]
  #     check: sliced.shape[0] == 3
  #     check: sliced[0].item(float64) == 30.0
  #     check: sliced[2].item(float64) == 50.0

  #   test "Slice from 0 to ^1 (all elements, includes last)":
  #     ## 0..^1 includes the last element
  #     let sliced = arr5[0..^1]
  #     check: sliced.shape[0] == 5
  #     check: sliced[^1].item(float64) == 50.0

  #   test "Slice from ^4 to ^2 (exclusive)":
  #     ## ^4..<^2 means indices 1, 2, 3
  #     let sliced = arr5[^4..<^2]
  #     check: sliced.shape[0] == 3
  #     check: sliced[0].item(float64) == 20.0
  #     check: sliced[^1].item(float64) == 40.0

  # suite "Attention Mechanism Patterns - Real-world Examples":
  #   ## These tests verify common patterns used in transformer implementations

  #   test "Multi-head attention: split heads":
  #     ## Pattern: tensor[:, head_idx*head_size:(head_idx+1)*head_size, :]
  #     var t = arange(240, kFloat64).reshape(@[2, 6, 20])  # batch=2, 6 heads, features=20
  #     let head_idx = 2
  #     let head_size = 4
  #     let head_slice = t[_, head_idx*head_size..<(head_idx+1)*head_size, _]
  #     check: head_slice.shape[0] == 2
  #     check: head_slice.shape[1] == 4
  #     check: head_slice.shape[2] == 20

  #   test "Multi-head attention: all heads":
  #     ## Pattern: tensor[:, :, :] (no slicing, get all heads)
  #     var t = arange(240, kFloat64).reshape(@[2, 6, 20])
  #     let all_heads = t[_, _, _]
  #     check: all_heads.shape == @[2, 6, 20]

  #   test "Attention: causal mask preparation":
  #     ## Pattern: upper triangle mask for seq_len positions
  #     let seq_len = 5
  #     var mask = zeros(@[seq_len, seq_len], kFloat64)
  #     for i in 0..<seq_len:
  #       for j in i+1..<seq_len:
  #         mask[i, j] = float64(-1e9)  # Mask out future positions
  #     # Verify mask shape and some values
  #     check: mask.shape[0] == 5
  #     check: mask.shape[1] == 5
  #     check: mask[0, 4].item(float64) == -1e9  # Last column masked
  #     check: mask[4, 4].item(float64) == 0.0   # Diagonal not masked

  #   test "Rotary Position Embedding (RoPE): slice with step 2":
  #     ## Pattern: interleaved positions for cos/sin computation
  #     var t = arange(40, kFloat64).reshape(@[2, 4, 5])
  #     let cos_sin = t[_, _, _..<5|2]  # Every other position
  #     check: cos_sin.shape[2] == 3  # 5 elements, step 2 = 3 values

  #   test "Flash Attention-like slicing: load block":
  #     ## Pattern: t[q_start..q_end, :, k_start..k_end]
  #     var t = arange(200, kFloat64).reshape(@[10, 8, 10])
  #     let q_start = 2
  #     let q_end = 5
  #     let k_start = 1
  #     let k_end = 4
  #     let sliced = t[q_start..<q_end, _, k_start..<k_end]
  #     check: sliced.shape[0] == 3
  #     check: sliced.shape[1] == 8
  #     check: sliced.shape[2] == 3

  #   test "Output projection slice - last N features":
  #     ## Pattern: tensor[:, :, -proj_dim:]
  #     var t = arange(100, kFloat64).reshape(@[2, 5, 10])
  #     let proj_dim = 4
  #     let projected = t[_, _, ^proj_dim..^0]
  #     check: projected.shape[2] == 4

  # suite "Complete Vandermonde Reference Tests":
  #   ## Comprehensive tests using the 5x5 Vandermonde matrix
  #   ## v[i, j] = (i+1)^j for 0 <= i,j < 5
  #   ## Row 0: 1, 1, 1, 1, 1
  #   ## Row 1: 2, 4, 8, 16, 32
  #   ## Row 2: 3, 9, 27, 81, 243
  #   ## Row 3: 4, 16, 64, 256, 1024
  #   ## Row 4: 5, 25, 125, 625, 3125

  #   let v = genShiftedVandermonde5x5(kFloat64)

  #   test "Vandermonde: row access with span":
  #     let row1 = v[1, _]
  #     check: row1.shape[0] == 5
  #     check: row1[0].item(float64) == 2.0
  #     check: row1[4].item(float64) == 32.0

  #   test "Vandermonde: column access with span":
  #     let col2 = v[_, 2]
  #     check: col2.shape[0] == 5
  #     check: col2[0].item(float64) == 1.0  # 1^3 = 1
  #     check: col2[4].item(float64) == 125.0  # 5^3 = 125

  #   test "Vandermonde: submatrix slice":
  #     let sub = v[1..<4, 1..<4]
  #     check: sub.shape[0] == 3
  #     check: sub.shape[1] == 3
  #     check: sub[0, 0].item(float64) == 4.0   # 2^2 = 4
  #     check: sub[2, 2].item(float64) == 64.0  # 4^2 = 64

  #   test "Vandermonde: every other row":
  #     let evens = v[_..<5|2, _]
  #     check: evens.shape[0] == 3
  #     check: evens[0, 0].item(float64) == 1.0   # Row 0
  #     check: evens[1, 0].item(float64) == 3.0   # Row 2
  #     check: evens[2, 0].item(float64) == 5.0   # Row 4

  #   test "Vandermonde: last 2 columns":
  #     let last2 = v[_, ^2..^0]
  #     check: last2.shape[1] == 2
  #     check: last2[0, 0].item(float64) == 1.0   # 1^4 = 1
  #     check: last2[4, 1].item(float64) == 3125.0  # 5^5 = 3125

  #   test "Vandermonde: anti-diagonal with ^":
  #     ## ^N counts from end, so ^4..^0 is all indices
  #     let diag = v[^5..^1, ^5..^1]
  #     check: diag.shape[0] == 5
  #     check: diag[0, 0].item(float64) == 5.0   # 5^0 = 1, but reversed!
  #     check: diag[4, 4].item(float64) == 5.0   # 1^4 = 1, reversed

  #   test "Vandermonde: top-right triangle":
  #     let upper = v[_..<5, 3..<5]
  #     check: upper.shape[0] == 5
  #     check: upper.shape[1] == 2
  #     check: upper[0, 1].item(float64) == 1.0   # row 0, col 4

  #   test "Vandermonde: bottom-left triangle with span":
  #     let lower = v[2..<5, _..<3]
  #     check: lower.shape[0] == 3
  #     check: lower.shape[1] == 3
  #     check: lower[0, 0].item(float64) == 9.0   # row 2, col 0

  #   test "Vandermonde: assign to slice":
  #     var v2 = v.clone()
  #     v2[0..<2, 0..<2] = 0.0
  #     check: v2[0, 0].item(float64) == 0.0
  #     check: v2[1, 1].item(float64) == 0.0
  #     check: v2[2, 2].item(float64) == 27.0  # Unchanged

  #   test "Vandermonde: ellipsis row access":
  #     let row1_ellipsis = v[1, `...`]
  #     check: row1_ellipsis.shape[0] == 5
  #     check: row1_ellipsis[0].item(float64) == 2.0

when isMainModule:
  main()
