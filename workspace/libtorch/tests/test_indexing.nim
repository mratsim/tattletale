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

func formatName*(desc, indexingExample: string): string =
  fmt"{desc:<40}  {indexingExample}"

proc main() =
  let vandermonde = genShiftedVandermonde5x5(kFloat64)
  let t3d = arange(24, kFloat64).reshape(@[2, 3, 4])
  let t4d = arange(120, kFloat64).reshape(@[2, 3, 4, 5])

  suite "Torch Tensor Indexing - PyTorch/libtorch Documentation Examples":
    ## Reference: https://pytorch.org/cppdocs/notes/tensor_indexing.html
    ## All tests include equivalent PyTorch and libtorch syntax for documentation

    vandermonde.display()

    test formatName("Integer indexing", "a[1, 2]"):
      ## Nim libtorch: a[1, 2]
      ## Python: a[1, 2]
      ## C++ libtorch: a.index({1, 2})
      let val = vandermonde[1, 2]
      check: val.item(float64) == 8.0  # 2^3 = 8

    test formatName("Strided monodimensional indexing", "a[1..|2]"):
      ## Nim libtorch: a[1..|2]
      ## Python: a[1::2]
      ## C++ libtorch: a.index({Slice(1, None, 2)})
      let sliced = vandermonde[1..|2]
      # Slices rows 1 and 3 (bases 2 and 4), all columns
      check: sliced.shape[0] == 2
      check: sliced.shape[1] == 5

  suite "Slice Types (Python ':' equivalent to libtorch Slice)":
    ## Python `:` / `::` maps to `torch::indexing::Slice()`

    vandermonde.display()

    test formatName("Full slice", "a[_, _]"):
      ## Nim libtorch: a[_, _]
      ## Python: a[:, :]
      ## C++ libtorch: a.index({Slice(), Slice()})
      let full = vandermonde[_, _]
      check: full.shape[0] == 5
      check: full.shape[1] == 5

    test formatName("Slice from start", "a[_..<3, _]"):
      ## Nim libtorch: a[_..<3, _]
      ## Python: a[:3]
      ## C++ libtorch: a.index({Slice(None, 3)})
      ## TODO - allow  a[..<3, _]
      let sliced = vandermonde[_..<3, _]
      check: sliced.shape[0] == 3
      check: sliced.shape[1] == 5

    test formatName("Slice to end", "a[1..<_]"):
      ## Nim libtorch: a[1..<_]
      ## Python: a[1:]
      ## C++ libtorch: a.index({Slice(1, None)})
      let sliced = vandermonde[1..<_]
      check: sliced.shape[0] == 4
      check: sliced.shape[1] == 5

    test formatName("Slice with step only", "a[|2]"):
      ## Nim libtorch: a[|2]
      ## Python: a[::2]
      ## C++ libtorch: a.index({Slice(None, None, 2)})
      let sliced = vandermonde[|2]
      check: sliced.shape[0] == 3  # rows 0, 2, 4
      check: sliced.shape[1] == 5

    test formatName("Slice with start, stop, step", "a[1..<3|2]"):
      ## Nim libtorch: a[1..<3|2]
      ## Python: a[1:3:2]
      ## C++ libtorch: a.index({Slice(1, 3, 2)})
      let sliced = vandermonde[1..<3|2]
      check: sliced.shape[0] == 1  # only row 1
      check: sliced.shape[1] == 5

  suite "Python Slice Syntax to Nim Translation Reference":
    ## This suite documents the mapping between Python slice syntax and Nim syntax.
    ##
    ## Python slices are EXCLUSIVE on the end (like C++/Python standard).
    ## Nim slices are INCLUSIVE on both ends (..) or exclusive (..<).
    ##
    ## Nim:    a[start..<stop]   -> elements from start to stop-1
    ## Python: a[start:stop]    -> elements from start (inclusive) to stop (exclusive)
    ##
    ## Nim:    a[start..-1]    -> elements from start to end (negative index)
    ## Python: a[start:]       -> elements from start to end
    ##
    ## Nim:    a[_..<stop]     -> _ means all of dimension, then exclusive
    ## Python: a[:stop]        -> elements from 0 to stop-1
    ##
    ## Nim:    a[_.._]        -> full span
    ## Python: a[:]           -> all elements
    ##
    ## Nim:    a[_.._|step]   -> full span with step
    ## Python: a[::step]      -> every step-th element
    ##
    ## Nim:    a[1..<5|2]     -> elements 1, 3
    ## Python: a[1:5:2]       -> elements 1, 3 (start=1, stop=5, step=2)

    vandermonde.display()

    test formatName("Python a[:2] -> Nim a[_..<2]", "a[:2]"):
      ## Nim: a[_..<2] gets indices 0, 1 (exclusive)
      ## Python: a[:2] gets indices 0, 1
      let t = vandermonde
      let sliced = t[_..<2, _]
      check:
        sliced ==
          [[   1,    1,    1,    1,    1],
           [   2,    4,    8,   16,   32]].toTorchTensor.to(kFloat64)

    test formatName("Python a[3:] -> Nim a[3.._]", "a[3:]"):
      ## Nim: a[3.._] gets indices 3, 4 (use _ for "to the end")
      ## Python: a[3:] gets indices 3, 4
      let t = vandermonde
      let sliced = t[3.._, _]
      check:
        sliced ==
          [[   4,   16,   64,  256, 1024],
           [   5,   25,  125,  625, 3125]].toTorchTensor.to(kFloat64)

    test formatName("Python a[:] -> Nim a[_.._]", "a[:]"):
      ## Nim: a[_.._] gets all elements
      ## Python: a[:] gets all elements
      let t = vandermonde
      let sliced = t[_.._, _]
      check: sliced == t

    test formatName("Python a[::2] -> Nim a[_.._|2]", "a[::2]"):
      ## Nim: a[_.._|2] or a[|2] (cleaner!) gets every 2nd element
      ## Python: a[::2] gets every 2nd element
      let t = vandermonde
      let sliced1 = t[_.._|2, _]
      let sliced2 = t[|2, _]
      check: sliced1 == sliced2  # Both syntaxes are equivalent
      check:
        sliced1 ==
          [[   1,    1,    1,    1,    1],
           [   3,    9,   27,   81,  243],
           [   5,   25,  125,  625, 3125]].toTorchTensor.to(kFloat64)

    test formatName("Unary pipe step", "a[|3]"):
      ## The unary `|step` syntax is cleaner than `_.._|step`
      ## Nim: a[|3] -> Slice(None, None, 3)
      let t = vandermonde
      let sliced = t[|3, _]
      check:
        sliced ==
          [[   1,    1,    1,    1,    1],
           [   4,   16,   64,  256, 1024]].toTorchTensor.to(kFloat64)

    test formatName("Stepped span with index", "a[|2, 0]"):
      ## Nim: a[|2, 0] -> every 2nd element of dim 0, index 0 of dim 1
      ## Note: Indexing with a scalar (like 0) squeezes that axis since size is 1
      let t = vandermonde
      let sliced = t[|2, 0]
      check: sliced.shape[0] == 3
      check: sliced.shape.len == 1  # Scalar index squeezes axis, so only 1 dim remains
      check: sliced == [1, 3, 5].toTorchTensor.to(kFloat64)

    test formatName("Mixed indexing with stepped span", "a[1, |2, _]"):
      ## Nim: a[1, |2, _] -> index 1, every 2nd of dim 1, all of dim 2
      ## numpy equivalent: t[1, ::2, :]
      let t = arange(24, kFloat64).reshape(@[2, 3, 4])
      let sliced = t[1, |2, _]
      check:
        sliced ==
          [[  12,  13,  14,  15],
           [  20,  21,  22,  23]].toTorchTensor.to(kFloat64)

    test formatName("Python a[1:4] -> Nim a[1..<4]", "a[1:4]"):
      ## Nim: a[1..<4] gets indices 1, 2, 3 (exclusive)
      ## Python: a[1:4] gets indices 1, 2, 3
      let t = vandermonde
      let sliced = t[1..<4, _]
      check:
        sliced ==
          [[   2,    4,    8,   16,   32],
           [   3,    9,   27,   81,  243],
           [   4,   16,   64,  256, 1024]].toTorchTensor.to(kFloat64)

    test formatName("Python a[1:4:2] -> Nim a[1..<4|2]", "a[1:4:2]"):
      ## Nim: a[1..<4|2] gets indices 1, 3
      ## Python: a[1:4:2] gets indices 1, 3
      let t = vandermonde
      let sliced = t[1..<4|2, _]
      check:
        sliced ==
          [[   2,    4,    8,   16,   32],
           [   4,   16,   64,  256, 1024]].toTorchTensor.to(kFloat64)

    test formatName("Python a[:-1] -> Nim a[_..-1]", "a[:-1]"):
      ## Nim: a[_..-1] gets all but last (stop=-1 is exclusive)
      ## Python: a[:-1] gets all but last element
      let t = vandermonde
      let sliced = t[_..-1, _]
      check:
        sliced ==
          [[   1,    1,    1,    1,    1],
           [   2,    4,    8,   16,   32],
           [   3,    9,   27,   81,  243],
           [   4,   16,   64,  256, 1024]].toTorchTensor.to(kFloat64)

    test formatName("Python a[-3:] -> Nim a[-3.._]", "a[-3:]"):
      ## Nim: a[-3.._] gets last 3 (start at -3, go to end with _)
      ## Python: a[-3:] gets last 3 indices (2, 3, 4)
      let t = vandermonde
      let sliced = t[-3.._, _]
      check:
        sliced ==
          [[   3,    9,   27,   81,  243],
           [   4,   16,   64,  256, 1024],
           [   5,   25,  125,  625, 3125]].toTorchTensor.to(kFloat64)

    test formatName("Python a[-3:-1] -> Nim a[-3..-1]", "a[-3:-1]"):
      ## Nim: a[-3..-1] gets indices 2, 3 (3rd-from-end to before last)
      ## Python: a[-3:-1] gets indices 2, 3 (exclusive upper bound)
      let t = vandermonde
      let sliced = t[-3..-1, _]
      check:
        sliced ==
          [[   3,    9,   27,   81,  243],
           [   4,   16,   64,  256, 1024]].toTorchTensor.to(kFloat64)

    test formatName("Python a[::-1] -> Use flip()", "a[::-1]"):
      ## Nim: Negative steps are NOT supported in Slice()
      ##            Use flip() instead
      ## Python: a[::-1] reverses the tensor along a dimension
      ##
      ## What Nim flip() gives: same result as Python a[::-1]
      var t = vandermonde.clone()
      let reversed = t.flip(@[0])
      let expected = @[
        @[  5.0,  25.0,  125.0,  625.0, 3125.0],
        @[  4.0,  16.0,   64.0,  256.0, 1024.0],
        @[  3.0,   9.0,   27.0,   81.0,  243.0],
        @[  2.0,   4.0,    8.0,   16.0,   32.0],
        @[  1.0,   1.0,    1.0,    1.0,    1.0]
      ].toTorchTensor.to(kFloat64)
      check: reversed == expected

    test formatName("Negative steps not supported", "a[|-2]"):
      ## libtorch's Slice() does NOT support negative steps
      ## Python: a[::2] would work, a[::-2] would reverse with step 2
      ## Nim: a[_.._|2] works, a[_.._|-2] raises compile error
      ##
      ## To reverse and step, use: a.flip(dim).slice(...)
      var t = vandermonde.clone()
      let reversed = t.flip(@[0])
      let stepped = reversed[_.._|2, _]  # Reverse, then take every 2nd
      let expected = @[
        @[  5.0,  25.0,  125.0,  625.0, 3125.0],
        @[  3.0,   9.0,   27.0,   81.0,  243.0],
        @[  1.0,   1.0,    1.0,    1.0,    1.0]
      ].toTorchTensor.to(kFloat64)
      check: stepped == expected

  suite "Negative Indexing with Variables and Expressions":
    ## Tests that negative indices work with variables and runtime expressions
    ## The key insight is that handleNegativeIndex normalizes at runtime

    vandermonde.display()

    test formatName("Negative index via variable", "a[_..negOne]"):
      ## Python equivalent: a[:-1] (all but last)
      ## Using a variable to hold the negative index
      let t = vandermonde
      let negOne = -1
      let sliced = t[_..negOne, _]
      check:
        sliced ==
          [[   1,    1,    1,    1,    1],
           [   2,    4,    8,   16,   32],
           [   3,    9,   27,   81,  243],
           [   4,   16,   64,  256, 1024]].toTorchTensor.to(kFloat64)

    test formatName("Negative index via variable (different value)", "a[_..negTwo]"):
      ## Python equivalent: a[:-2] (all but last 2)
      ## Using a variable for -2
      let t = vandermonde
      let negTwo = -2
      let sliced = t[_..negTwo, _]
      check:
        sliced ==
          [[   1,    1,    1,    1,    1],
           [   2,    4,    8,   16,   32],
           [   3,    9,   27,   81,  243]].toTorchTensor.to(kFloat64)

    test formatName("Negative start via variable", "a[negThree.._]"):
      ## Python equivalent: a[-3:] (last 3 elements)
      ## Using a variable for the start index
      let t = vandermonde
      let negThree = -3
      let sliced = t[negThree.._, _]
      check:
        sliced ==
          [[   3,    9,   27,   81,  243],
           [   4,   16,   64,  256, 1024],
           [   5,   25,  125,  625, 3125]].toTorchTensor.to(kFloat64)

    test formatName("Both bounds via variables", "a[negTwo..negOne]"):
      ## Python equivalent: a[-2:-1] (second-to-last element only)
      let t = vandermonde
      let negTwo = -2
      let negOne = -1
      let sliced = t[negTwo..negOne]
      check: sliced.shape[0] == 1  # Squeezed to 1D
      check: sliced == [[4, 16, 64, 256, 1024]].toTorchTensor.to(kFloat64)

    test formatName("Negative index via expression", "a[0..-(n-1)]"):
      ## Python equivalent: a[:-(n-1)] where n is tensor size
      ## For a 5x5 tensor, -(n-1) = -(5-1) = -4, stop at index 1 (exclusive)
      let t = vandermonde
      let n = 5
      let sliced = t[0..-(n-1), _]
      check: sliced.shape[0] == 1  # Squeezed to 1D
      check: sliced == [[1, 1, 1, 1, 1]].toTorchTensor.to(kFloat64)

    test formatName("Negative index via expression (2*n)", "a[_..-(2*n)]"):
      ## Python equivalent: a[:-(2*n)]
      ## For m=2, -(2*m) = -4, stop at index 1 (exclusive)
      let t = vandermonde
      let m = 2
      let sliced = t[_..-(2*m), _]
      check: sliced.shape[0] == 1  # Squeezed to 1D
      check: sliced == [[1, 1, 1, 1, 1]].toTorchTensor.to(kFloat64)

    test formatName("Negative start via expression", "a[-(n-3).._]"):
      ## Python equivalent: a[-(n-3):] for n=5 gives a[-2:] = indices 3, 4
      let t = vandermonde
      let n = 5
      let sliced = t[-(n-3).._, _]
      check:
        sliced ==
          [[   4,   16,   64,  256, 1024],
           [   5,   25,  125,  625, 3125]].toTorchTensor.to(kFloat64)

    # TODO
    # test formatName("Both bounds via expressions", "a[-(n-2)..-(n-4)]"):
    #   ## Python equivalent: a[-(n-2):-(n-4)] for n=5 gives a[-3:-1] = indices 2, 3
    #   let t = vandermonde
    #   let n = 5
    #   let sliced = t[-(n-2)..-(n-4), _]
    #   check:
    #     sliced ==
    #       [[3,  9, 27,  81,  243]
    #        [4, 16, 64, 256, 1024]].toTorchTensor.to(kFloat64)

    test formatName("Negative index with step", "a[_..-1|2]"):
      ## Python equivalent: a[:-1:2] - every 2nd element excluding last
      let t = vandermonde
      let sliced = t[_..-1|2, _]
      check: sliced ==
              [[1, 1,  1,  1,   1],
               [3, 9, 27, 81, 243]].toTorchTensor.to(kFloat64)

    # TODO
    # test formatName("Runtime negative computation", "a[-(2*n)..-n]"):
    #   ## Python equivalent: a[-(2*n):-n] for n=2 gives a[-4:-2] = indices 1, 2
    #   let t = vandermonde
    #   let n = 2
    #   let sliced = t[-(2*n)..-n, _]
    #   check:
    #     sliced ==
    #       [[   2,    4,    8,   16,   32],
    #        [   3,    9,   27,   81,  243]].toTorchTensor.to(kFloat64)

    test formatName("Mixed: literal start, variable stop", "a[1..negOne]"):
      ## Python equivalent: a[1:-1] (from index 1 to before last)
      let t = vandermonde
      let negOne = -1
      let sliced = t[1..negOne, _]
      check:
        sliced ==
          [[   2,    4,    8,   16,   32],
           [   3,    9,   27,   81,  243],
           [   4,   16,   64,  256, 1024]].toTorchTensor.to(kFloat64)

    # TODO
    # test formatName("Mixed: expression start, literal stop", "a[-(n-2)..3]"):
    #   ## Python equivalent: a[-(n-2):3] for n=5 gives a[-3:3] = indices 2, 3
    #   let t = vandermonde
    #   let n = 5
    #   let sliced = t[-(n-2)..3, _]
    #   check:
    #     sliced ==
    #       [   3,    9,   27,   81,  243].toTorchTensor.to(kFloat64)

  # TODO: Negative step via autoflipping first
  # suite "Python a[::-1] -> Use flip()":
  #   test formatName("Python a[::-1] -> Use flip()", "a[::-1]"):
  #     ## Nim: Negative steps are NOT supported in Slice()
  #     ##            Use flip() instead
  #     ## Python: a[::-1] reverses the tensor along a dimension
  #     ##
  #     ## What Nim flip() gives: same result as Python a[::-1]
  #     var t = genShiftedVandermonde5x5(kFloat64)
  #     let reversed = t.flip(@[0])

  #     ## flip() along dim 0 should give same as a[::-1] in Python
  #     check: reversed[0, 0].item(float64) == 5.0   # Last row of original
  #     check: reversed[4, 0].item(float64) == 1.0   # First row of original
  #     check: reversed[0, 4].item(float64) == 3125.0 # 5^5 = 3125
  #     check: reversed[4, 4].item(float64) == 1.0    # 1^5 = 1

  #   test formatName("Negative steps not supported", "a[|-2]"):
  #     ## libtorch's Slice() does NOT support negative steps
  #     ## Python: a[::2] would work, a[::-2] would reverse with step 2
  #     ## Nim: a[_.._|2] works, a[_.._|-2] raises compile error
  #     ##
  #     ## To reverse and step, use: a.flip(dim).slice(...)
  #     var t = genShiftedVandermonde5x5(kFloat64)
  #     let reversed = t.flip(@[0])
  #     let stepped = reversed[_.._|2, _]  # Reverse, then take every 2nd
  #     check: stepped.shape[0] == 3  # rows 0, 2, 4 of reversed = rows 4, 2, 0 of original
  #     check: stepped[0, 0].item(float64) == 5.0  # Row 4 (first of reversed)
  #     check: stepped[2, 0].item(float64) == 1.0  # Row 0 (last of reversed)

  suite "Ellipsis `...` or ellipsis - Python '...' equivalent to libtorch Ellipsis":
    ## Note: In Nim, `...` must be used quoted a[0..<2, `...`] or a[0..<2, ellipsis]
    ##
    ## Both are equivalent:
    ##   a[`...`, 0]
    ##   a[ellipsis, 0]

    vandermonde.display()

    # TODO
    # test formatName("Single ellipsis", "a[...]"):
    #   ## Nim: a[...] -> a.index({torch::indexing::Ellipsis})
    #   ## Python: a[...]
    #   let t = genShiftedVandermonde5x5(kFloat64)
    #   check: t[`...`].shape[0] == 5
    #   check: t[`...`].shape[1] == 5
    #
    # test formatName("Ellipsis with other indices", "a[..., 0]"):
    #   ## Nim: a[`...`, 0] -> a.index({Ellipsis, 0})
    #   ## Python: a[..., 0]
    #   let t = genShiftedVandermonde5x5(kFloat64)
    #   let sliced = t[`...`, 0]
    #   check: sliced.shape[0] == 5
    #   check: sliced[0, 0].item(float64) == 1.0  # (1+1)^(0+1) = 2^1 = 2
    #
    # test formatName("IndexEllipsis constant", "ellipsis vs `...`"):
    #   let t = genShiftedVandermonde5x5(kFloat64)
    #   let with_const = t[ellipsis, 0]
    #   let with_quoted = t[`...`, 0]
    #   check: with_const.shape == with_quoted.shape
    #   check: with_const == with_quoted
    #
    # test formatName("Ellipsis expansion", "a[..., 0] = a[:, :, 0]"):
    #   ## Demonstrates ellipsis expansion
    #   let t = genShiftedVandermonde5x5(kFloat64)
    #   let ellipsis_result = t[`...`, 0]
    #   let explicit = t[_, _, 0]
    #   check: ellipsis_result == explicit
    #
    # test formatName("Leading ellipsis", "a[0, ...]"):
    #   ## Nim: a[0, ...] -> a.index({0, Ellipsis})
    #   ## Python: a[0, ...]
    #   let t = genShiftedVandermonde5x5(kFloat64)
    #   let sliced = t[0, `...`]
    #   check: sliced.shape[0] == 5
    #   check: sliced[0].item(float64) == 1.0  # Row 0, all columns (powers of 1 = 1)
    #
    # test formatName("Middle ellipsis", "a[1, ..., 0]"):
    #   ## Nim: a[1, ..., 0] -> a.index({1, Ellipsis, 0})
    #   ## Python: a[1, ..., 0]
    #   let t = genShiftedVandermonde5x5(kFloat64)
    #   let sliced = t[1, `...`, 0]
    #   check: sliced.shape[0] == 5
    #   check: sliced[0].item(float64) == 2.0  # Row 1 (base 2), column 0 = 2^1 = 2

    test formatName("Single span", "a[_]"):
      ## Nim: a[_] / a[_, _] maps to Slice() / Slice(None, None)
      ## Python: a[:] / a[:, :]
      let t = vandermonde
      let sliced = t[_, _]
      check: sliced == vandermonde

    test formatName("Span on first dimension only", "a[_, 2]"):
      ## Nim: a[_, 2] - all rows, column 2 (squeezed to 1D since size 1)
      let t = vandermonde
      let sliced = t[_, 2]
      check: sliced.shape[0] == 5
      check: sliced.shape.len == 1  # Scalar index squeezes axis
      check: sliced == [1, 8, 27, 64, 125].toTorchTensor.to(kFloat64)

    test formatName("Span with slice", "a[1..3, _]"):
      ## Nim: a[1..<3, _] - rows 1, 2, all columns
      ## Python: a[1:3, :]
      let t = vandermonde
      let sliced = t[1..<3, _]
      check:
        sliced ==
          [[   2,    4,    8,   16,   32],
           [   3,    9,   27,   81,  243]].toTorchTensor.to(kFloat64)

    test formatName("Span with partial", "a[_..2, 2]"):
      ## Nim: a[_..<2, 2] - rows 0, 1, column 2 (squeezed)
      ## Python: a[:2, 2]
      let t = vandermonde
      let sliced = t[_..<2, 2]
      check: sliced.shape[0] == 2
      check: sliced.shape.len == 1  # Scalar index squeezes axis
      check: sliced == [1, 8].toTorchTensor.to(kFloat64)

    test formatName("Full span shorthand", "a[_.._]"):
      ## Nim: a[_.._, _] - all rows, all columns
      let t = vandermonde
      let sliced = t[_.._, _]
      check: sliced == t

    test formatName("Span with step", "a[_.._|2]"):
      ## Nim: a[_.._|2] - rows 0, 2, 4, all columns
      ## Python: a[::2]
      let t = vandermonde
      let sliced = t[_.._|2, _]
      check:
        sliced ==
          [[   1,    1,    1,    1,    1],
           [   3,    9,   27,   81,  243],
           [   5,   25,  125,  625, 3125]].toTorchTensor.to(kFloat64)

    test "Span on first dimension only - a[_, 2]":
      ## Nim: a[_, 2]
      let t = vandermonde
      let sliced = t[_, 2]
      check: sliced.shape[0] == 5
      check: sliced.shape.len == 1
      check: sliced == [1, 8, 27, 64, 125].toTorchTensor.to(kFloat64)

    test "Span with slice - a[1..3, _]":
      ## Nim: a[1..<3, _]
      ## Python: a[1:3, :]
      let t = vandermonde
      let sliced = t[1..<3, _]
      check:
        sliced ==
          [[   2,    4,    8,   16,   32],
           [   3,    9,   27,   81,  243]].toTorchTensor.to(kFloat64)

    test "Span with partial - a[_..2, 2]":
      ## Nim: a[_..<2, 2]
      ## Python: a[:2, 2]
      let t = vandermonde
      let sliced = t[_..<2, 2]
      check: sliced.shape[0] == 2
      check: sliced.shape.len == 1
      check: sliced == [1, 8].toTorchTensor.to(kFloat64)

    test "Full span shorthand - a[_.._]":
      ## Nim: a[_.._, _]
      let t = vandermonde
      let sliced = t[_.._, _]
      check: sliced == t

    test "Span with step - a[_.._|2]":
      ## Nim: a[_.._|2]
      ## Python: a[::2]
      let t = vandermonde
      let sliced = t[_.._|2, _]
      check:
        sliced ==
          [[   1,    1,    1,    1,    1],
           [   3,    9,   27,   81,  243],
           [   5,   25,  125,  625, 3125]].toTorchTensor.to(kFloat64)

  suite "Negative Indexing (End-relative with -N)":
    ## Nim-libtorch uses negative indices like Python:
    ## -1 = last element (index size-1)
    ## -2 = second-to-last element (index size-2)
    ## -3 = third-to-last element (index size-3)
    ## etc.
    ##
    ## This is equivalent to Python's negative indexing: a[-1]
    ## For slices, use `..-N` for end-relative slicing (exclusive)

    vandermonde.display()

    test formatName("Single negative index", "a[-1]"):
      ## Nim: -1 = last element at both dims
      ## Python: a[-1, -1]
      let t = vandermonde
      let val = t[-1, -1]
      check: val.item(float64) == 3125.0  # 5^5 = 3125

    test formatName("Second-to-last", "a[-2, -2]"):
      ## Nim: -2 = second-to-last element
      ## Python: a[-2, -2]
      let t = vandermonde
      let val = t[-2, -2]
      check: val.item(float64) == 256.0  # 4^4 = 256

    test formatName("Third-to-last", "a[-3, -3]"):
      ## Nim: -3 = third-to-last element
      ## Python: a[-3, -3]
      let t = vandermonde
      let val = t[-3, -3]
      check: val.item(float64) == 27.0  # 3^3 = 27

    test formatName("Inclusive slice to end", "a[0..-1]"):
      ## Nim: 0..-1 from 0 to before last (exclusive via ..-)
      ## Python: a[0:-1]
      let t = vandermonde
      let sliced = t[0..-1, _]
      check:
        sliced ==
          [[   1,    1,    1,    1,    1],
           [   2,    4,    8,   16,   32],
           [   3,    9,   27,   81,  243],
           [   4,   16,   64,  256, 1024]].toTorchTensor.to(kFloat64)

    test formatName("Negative slice", "a[-3..-1]"):
      ## Nim: -3..-1 from third-to-last to before last
      ## Python: a[-3:-1]
      let t = vandermonde
      let sliced = t[-3..-1, _]
      check:
        sliced ==
          [[   3,    9,   27,   81,  243],
           [   4,   16,   64,  256, 1024]].toTorchTensor.to(kFloat64)

    test formatName("Span with negative index", "a[_..-2]"):
      ## Nim: _..-2 from start to before second-to-last
      ## Python: a[:-2]
      let t = vandermonde
      let sliced = t[_..-2, _]
      check:
        sliced ==
          [[   1,    1,    1,    1,    1],
           [   2,    4,    8,   16,   32],
           [   3,    9,   27,   81,  243]].toTorchTensor.to(kFloat64)

    test formatName("Negative span with step", "a[-4.._|2]"):
      ## Nim: -4.._|2 from fourth-from-end to end, step 2
      ## Python: a[-4::2]
      let t = vandermonde
      let sliced = t[-4.._|2, _]
      check:
        sliced ==
          [[   2,    4,    8,   16,   32],
           [   4,   16,   64,  256, 1024]].toTorchTensor.to(kFloat64)

  suite "Multidimensional Slice Behavior":
    ## Reference: https://pytorch.org/cppdocs/notes/tensor_indexing.html
    ## For 3D tensor: a[0:3] is equivalent to a[0:3, ...] / a[0:3, :, :]

    vandermonde.display()

    test formatName("Partial slice on 3D tensor", "a[0..<2]"):
      ## Nim: a[0..<2] on 3D tensor is equivalent to a[0..<2, ...]
      ## Python: a[0:3] equivalent to a[0:3, ...] on 3D tensor
      let t3d = arange(24, kFloat64).reshape(@[2, 3, 4])
      let sliced = t3d[0..<2, _, _]
      let expected = @[
        @[@[0.0, 1.0, 2.0, 3.0],
          @[4.0, 5.0, 6.0, 7.0],
          @[8.0, 9.0, 10.0, 11.0]],
        @[@[12.0, 13.0, 14.0, 15.0],
          @[16.0, 17.0, 18.0, 19.0],
          @[20.0, 21.0, 22.0, 23.0]]
      ].toTorchTensor.to(kFloat64)
      check: sliced == expected

    test formatName("Slice equivalent to explicit spans", "a[0..<3] vs a[0..<3, _, :]"):
      let t3d = arange(24, kFloat64).reshape(@[2, 3, 4])
      let implicit = t3d[0..<2]
      let explicit = t3d[0..<2, _, _]
      check: implicit == explicit

    # TODO - Ellipsis
    # test formatName("Slice last dimension", "a[..., 0:2]"):
    #   ## Nim: a[..., 0..<2] -> a.index({Ellipsis, Slice(0, 2)})
    #   ## Python: a[..., 0:2]
    #   let t3d = arange(24, kFloat64).reshape(@[2, 3, 4])
    #   let sliced = t3d[`...`, 0..<2]
    #   check: sliced.shape[0] == 2
    #   check: sliced.shape[1] == 3
    #   check: sliced.shape[2] == 2

  suite "Assignment Operations (index_put_)":

    vandermonde.display()

    test formatName("Point assignment", "a[0, 0] = 999"):
      ## Nim: a[0, 0] = 999
      ## Python: a[0, 0] = 999
      ## C++ libtorch: a.index_put_({0, 0}, 999)
      var t = vandermonde.clone()
      t[0, 0] = 999.0
      check: t[0, 0].item(float64) == 999.0
      check:
        t ==
          [[ 999,    1,    1,    1,    1],
           [   2,    4,    8,   16,   32],
           [   3,    9,   27,   81,  243],
           [   4,   16,   64,  256, 1024],
           [   5,   25,  125,  625, 3125]].toTorchTensor.to(kFloat64)

    test formatName("Slice assignment", "a[0..2, 0..2] = 0"):
      ## Nim: a[0..<2, 0..<2] = 0
      ## Python: a[0:2, 0:2] = 0
      ## C++ libtorch: a.index_put_({Slice(0, 2), Slice(0, 2)}, 0)
      var t = vandermonde.clone()
      t[0..<2, 0..<2] = 0.0
      check: t[0, 0].item(float64) == 0.0
      check: t[1, 1].item(float64) == 0.0
      check: t[2, 2].item(float64) == 27.0  # Unchanged
      check:
        t ==
          [[   0,    0,    1,    1,    1],
           [   0,    0,    8,   16,   32],
           [   3,    9,   27,   81,  243],
           [   4,   16,   64,  256, 1024],
           [   5,   25,  125,  625, 3125]].toTorchTensor.to(kFloat64)

    test formatName("Assignment with step", "a[::2, ::2] = 999"):
      ## Nim: a[_..<5|2, _..<5|2] = 999
      ## Python: a[::2, ::2] = 999
      var t = vandermonde.clone()
      t[_..<5|2, _..<5|2] = 999.0
      check: t[0, 0].item(float64) == 999.0
      check: t[2, 2].item(float64) == 999.0
      check: t[1, 1].item(float64) == 4.0  # Unchanged (row 1, col 1 is not stepped)
      check:
        t ==
          [[ 999,    1,  999,    1,  999],
           [   2,    4,    8,   16,   32],
           [ 999,    9,  999,   81,  999],
           [   4,   16,   64,  256, 1024],
           [ 999,   25,  999,  625,  999]].toTorchTensor.to(kFloat64)

    # test formatName("Ellipsis assignment", "a[...] = 0"):
    #   ## Nim: a[...] = 0 sets all elements to 0
    #   ## Python: a[...] = 0
    #   var t = vandermonde.clone()
    #   t[IndexEllipsis] = 0.0
    #   check: t[0, 0].item(float64) == 0.0
    #   check: t[4, 4].item(float64) == 0.0
    #   check: t.numel() == 25

    test formatName("Assignment with step", "a[::2, ::2] = 999"):
      ## Nim: a[_..<5|2, _..<5|2] = 999
      ## Python: a[::2, ::2] = 999
      var t = vandermonde.clone()
      t[_..<5|2, _..<5|2] = 999.0
      check: t[0, 0].item(float64) == 999.0
      check: t[2, 2].item(float64) == 999.0
      check: t[1, 1].item(float64) == 4.0  # Unchanged (row 1, col 1 is not stepped)
      check:
        t ==
          [[ 999,    1,  999,    1,  999],
           [   2,    4,    8,   16,   32],
           [ 999,    9,  999,   81,  999],
           [   4,   16,   64,  256, 1024],
           [ 999,   25,  999,  625,  999]].toTorchTensor.to(kFloat64)

  suite "Common Attention Mechanism Patterns":
    ## These patterns appear frequently in transformer attention implementations

    vandermonde.display()

    test formatName("Q/K/V slicing for multi-head attention", "a[:, start_idx:start_idx+head_dim, :]"):
      ## Pattern: a[:, start_idx:start_idx+head_dim, :]
      var t = arange(100, kFloat64).reshape(@[2, 5, 10])
      let head_dim = 4
      let start_idx = 0
      let q = t[_, start_idx..<start_idx+head_dim, _]
      let expected = @[
        @[@[ 0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0],
          @[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
          @[20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0],
          @[30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0]],
        @[@[50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0],
          @[60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0],
          @[70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0],
          @[80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0]]
      ].toTorchTensor.to(kFloat64)
      check: q == expected

    # TODO: investigate out-of-bounds error
    # test formatName("Slicing all heads for a position", "a[:, :, pos_idx]"):
    #   ## Extract a single position across all heads
    #   let t = arange(100, kFloat64).reshape(@[2, 5, 10])
    #   let pos_idx = 2
    #   let sliced = t[_, _, pos_idx]
    #   check: sliced.shape[0] == 2
    #   check: sliced.shape[1] == 5
    #   check: sliced.shape[2] == 1

    # TODO: fix shape assertion
    # test formatName("Causal mask slicing", "a[t, :, t_end:]"):
    #   ## Pattern for causal attention (upper triangular)
    #   var t = arange(30, kFloat64).reshape(@[2, 3, 5])
    #   let t_idx = 1
    #   let sliced = t[t_idx, _, t_idx+1..<5]
    #   check: sliced.shape[0] == 1
    #   check: sliced.shape[1] == 3
    #   check: sliced.shape[2] == 3

    # TODO ellipsis
    # test formatName("Attention score masking with ellipsis", "a[..., q_idx, k_idx]"):
    #   ## Pattern for attention weight masking
    #   var t = arange(60, kFloat64).reshape(@[2, 3, 2, 5])
    #   let q_idx = 1
    #   let k_idx = 2
    #   let sliced = t[`...`, q_idx, k_idx]
    #   check: sliced.shape[0] == 2
    #   check: sliced.shape[1] == 3

    # TODO: fix shape assertion
    # test formatName("Slicing last dimension for residual", "a[:, :, -head_size:]"):
    #   ## Pattern: keep only the last head_size channels
    #   var t = arange(100, kFloat64).reshape(@[2, 5, 10])
    #   let head_size = 2
    #   let sliced = t[_, _, -head_size..-0]
    #   check: sliced.shape[2] == 2

    test formatName("Interleaved slicing for RoPE/AliBi", "a[batch, seq, ::2]"):
      ## Pattern: every other element (used in positional encoding)
      var t = arange(40, kFloat64).reshape(@[2, 4, 5])
      let sliced = t[_, _, _..<5|2]
      let expected = @[
        @[@[ 0.0,  2.0,  4.0],
          @[ 5.0,  7.0,  9.0],
          @[10.0, 12.0, 14.0],
          @[15.0, 17.0, 19.0]],
        @[@[20.0, 22.0, 24.0],
          @[25.0, 27.0, 29.0],
          @[30.0, 32.0, 34.0],
          @[35.0, 37.0, 39.0]]
      ].toTorchTensor.to(kFloat64)
      check: sliced == expected

    test formatName("Empty slice", "a[0..0]"):
      ## Python: a[0:0] returns empty tensor
      let t = vandermonde
      let sliced = t[0..<0, _]
      check: sliced.shape[0] == 0
      check: sliced.shape[1] == 5
      # Cannot use full matrix comparison for empty tensor

    test formatName("Full range slice", "a[0..<5]"):
      ## Slice covering entire dimension
      let t = vandermonde
      let sliced = t[0..<5, _]
      check: sliced == vandermonde

    test formatName("Single element slice", "a[2..<3]"):
      ## Slice producing single element
      let t = vandermonde
      let sliced = t[2..<3, _]
      let expected = @[
        @[  3.0,   9.0,   27.0,   81.0,  243.0]
      ].toTorchTensor.to(kFloat64)
      check: sliced == expected

    test formatName("Large step", "a[::100] with small tensor"):
      ## Step larger than dimension size
      let t = vandermonde
      let sliced = t[_..<5|100, _]
      let expected = @[
        @[1.0, 1.0, 1.0, 1.0, 1.0]
      ].toTorchTensor.to(kFloat64)
      check: sliced == expected

    test formatName("Reverse with negative step", "Use flip() not a[::-1]"):
      ## Python: a[::-1] reverses along a dimension
      ## Nim: Negative step syntax `|_` is NOT supported
      ##      Use flip() instead
      ##
      ## Example: What Python a[:, :, ::-1] would give in Nim:
      var t = vandermonde.clone()
      let reversed = t.flip(@[1])  # Reverse along dim 1
      let first_col = reversed[_, 0]
      let expected_col = @[1.0, 32.0, 243.0, 1024.0, 3125.0].toTorchTensor.to(kFloat64)
      check: first_col == expected_col

    test formatName("Integer and slice mix", "a[0, 1:4]"):
      ## Python: a[0, 1:4]
      let t = vandermonde
      let sliced = t[0, 1..<4]
      let expected = @[1.0, 1.0, 1.0].toTorchTensor.to(kFloat64)
      check: sliced == expected

    # TODO ellipsis
    # test formatName("Slice and ellipsis", "a[1:3, ..., 0:2]"):
    #   ## Python: a[1:3, ..., 0:2]
    #   let t3d = arange(60, kFloat64).reshape(@[2, 3, 10])
    #   let sliced = t3d[1..<3, `...`, 0..<2]
    #   check: sliced.shape[0] == 2
    #   check: sliced.shape[1] == 10

    test formatName("Integer array indexing", "a[[0, 2, 4]]"):
      ## Python: a[[0, 2, 4]] - fancy indexing
      let t = vandermonde
      let indices_seq = [0, 2, 4].toTorchTensor().to(kInt64)
      let indices = indices_seq.clone()
      let sliced = t.index_select(0, indices)
      let expected = @[
        @[  1.0,   1.0,   1.0,   1.0,   1.0],
        @[  3.0,   9.0,  27.0,  81.0, 243.0],
        @[  5.0,  25.0, 125.0, 625.0, 3125.0]
      ].toTorchTensor.to(kFloat64)
      check: sliced == expected

    # TODO: fix comparison operator
    # test formatName("Boolean mask indexing", "a[a > 10]"):
    #   ## Python: a[a > 10] - masked indexing
    #   let t = vandermonde
    #   let mask = t > 10.0
    #   let sliced = t[mask]
    #   # Count of elements > 10 in Vandermonde matrix

  suite "Full Matrix Comparison Tests (Python Validated)":
    ## These tests verify exact matrix equality against Python-validated results

    vandermonde.display()

    test formatName("First 2 rows", "t[_..<2, _]"):
      let t = vandermonde
      let sliced = t[_..<2, _]
      check:
        sliced ==
          [[ 1,  1,  1,  1,  1],
           [ 2,  4,  8, 16, 32]].toTorchTensor.to(kFloat64)

    test formatName("Rows 1 to 3", "t[1..<4, _]"):
      let t = vandermonde
      let sliced = t[1..<4, _]
      check:
        sliced ==
          [[   2,    4,    8,   16,   32],
           [   3,    9,   27,   81,  243],
           [   4,   16,   64,  256, 1024]].toTorchTensor.to(kFloat64)

    test formatName("Last 2 rows", "t[-2.._, _]"):
      let t = vandermonde
      let sliced = t[-2.._, _]
      check:
        sliced ==
          [[   4,   16,   64,  256, 1024],
           [   5,   25,  125,  625, 3125]].toTorchTensor.to(kFloat64)

    test formatName("All but last row", "t[0..-1, _]"):
      let t = vandermonde
      let sliced = t[0..-1, _]
      check:
        sliced ==
          [[   1,    1,    1,    1,    1],
           [   2,    4,    8,   16,   32],
           [   3,    9,   27,   81,  243],
           [   4,   16,   64,  256, 1024]].toTorchTensor.to(kFloat64)

    test formatName("Every 2nd row", "t[0..<4|2, _]"):
      let t = vandermonde
      let sliced = t[0..<4|2, _]
      check:
        sliced ==
          [[  1,   1,   1,   1,   1],
           [  3,   9,  27,  81, 243]].toTorchTensor.to(kFloat64)

    test formatName("First 2 columns", "t[_, 0..<2]"):
      let t = vandermonde
      let sliced = t[_, 0..<2]
      check:
        sliced ==
          [[ 1,  1],
           [ 2,  4],
           [ 3,  9],
           [ 4, 16],
           [ 5, 25]].toTorchTensor.to(kFloat64)

    test formatName("Last 2 columns", "t[_, -2.._]"):
      let t = vandermonde
      let sliced = t[_, -2.._]
      check:
        sliced ==
          [[   1,    1],
           [  16,   32],
           [  81,  243],
           [ 256, 1024],
           [ 625, 3125]].toTorchTensor.to(kFloat64)

    test formatName("Submatrix 2x2", "t[1..<3, 1..<3]"):
      let t = vandermonde
      let sliced = t[1..<3, 1..<3]
      check:
        sliced ==
          [[ 4,  8],
           [ 9, 27]].toTorchTensor.to(kFloat64)

    test formatName("Every 2nd column", "t[_, |2]"):
      let t = vandermonde
      let sliced = t[_, |2]
      check:
        sliced ==
          [[   1,    1,    1],
           [   2,    8,   32],
           [   3,   27,  243],
           [   4,   64, 1024],
           [   5,  125, 3125]].toTorchTensor.to(kFloat64)

    test formatName("Rows 1,3 cols 0,2,4", "t[1..<4|2, 0..<5|2]"):
      let t = vandermonde
      let sliced = t[1..<4|2, 0..<5|2]
      check:
        sliced ==
          [[   2,    8,   32],
           [   4,   64, 1024]].toTorchTensor.to(kFloat64)

    test formatName("Single slice on 3D", "a3d[0:2]"):
      ## Python: a3d[0:2] equivalent to a3d[0:2, :, :]
      let sliced = t3d[0..<2]
      let expected = @[
        @[@[0.0, 1.0, 2.0, 3.0],
          @[4.0, 5.0, 6.0, 7.0],
          @[8.0, 9.0, 10.0, 11.0]],
        @[@[12.0, 13.0, 14.0, 15.0],
          @[16.0, 17.0, 18.0, 19.0],
          @[20.0, 21.0, 22.0, 23.0]]
      ].toTorchTensor.to(kFloat64)
      check: sliced == expected

    test formatName("Slice middle dimension", "a3d[:, 0:2, :]"):
      ## Python: a3d[:, 0:2, :]
      let sliced = t3d[_, 0..<2, _]
      let expected = @[
        @[@[0.0, 1.0, 2.0, 3.0],
          @[4.0, 5.0, 6.0, 7.0]],
        @[@[12.0, 13.0, 14.0, 15.0],
          @[16.0, 17.0, 18.0, 19.0]]
      ].toTorchTensor.to(kFloat64)
      check: sliced == expected

    test formatName("Slice last dimension", "a3d[:, :, 0:2]"):
      ## Python: a3d[:, :, 0:2]
      let sliced = t3d[_, _, 0..<2]
      let expected = @[
        @[@[0.0, 1.0],
          @[4.0, 5.0],
          @[8.0, 9.0]],
        @[@[12.0, 13.0],
          @[16.0, 17.0],
          @[20.0, 21.0]]
      ].toTorchTensor.to(kFloat64)
      check: sliced == expected

    # TODO ellipsis
    # test formatName("Slice with ellipsis", "a3d[0, ...]"):
    #   ## Python: a3d[0, ...]
    #   ## All of remaining dimensions
    #   let sliced = t3d[0, `...`]
    #   check: sliced.shape[0] == 3
    #   check: sliced.shape[1] == 4

    # TODO: IndexEllipsis not fully supported
    # test formatName("Ellipsis expansion", "a3d[...] equals a3d"):
    #   let sliced = t3d[IndexEllipsis]
    #   check: sliced.shape[0] == 2
    #   check: sliced.shape[1] == 3
    #   check: sliced.shape[2] == 4

    test formatName("Multiple indices with slice", "a3d[0, 0:2, 1:3]"):
      ## Python: a3d[0, 0:2, 1:3]
      let sliced = t3d[0, 0..<2, 1..<3]
      let expected = @[
        @[1.0, 2.0],
        @[5.0, 6.0]
      ].toTorchTensor.to(kFloat64)
      check: sliced == expected

    test formatName("Single slice on 4D", "a4d[0..<2]"):
      let sliced = t4d[0..<2]
      check: sliced.shape[0] == 2
      check: sliced.shape[1] == 3
      check: sliced.shape[2] == 4
      check: sliced.shape[3] == 5

    # TODO: Ellipsis not fully supported
    # test formatName("Ellipsis in 4D (repeat)", "a4d[..., 0]"):
    #   ## Python: a4d[..., 0]
    #   let sliced = t4d[`...`, 0]
    #   check: sliced.shape[0] == 2
    #   check: sliced.shape[1] == 3
    #   check: sliced.shape[2] == 4

    # TODO: Ellipsis not fully supported
    # test formatName("Leading ellipsis (repeat)", "a4d[0, ..., 0]"):
    #   ## Python: a4d[0, ..., 0]
    #   let sliced = t4d[0, `...`, 0]
    #   check: sliced.shape[0] == 3
    #   check: sliced.shape[1] == 4

  suite "-N End-relative Indexing Reference":
    ## Summary: -1 is the LAST element, -2 is second-to-last, etc.
    ##
    ## For a 5-element array/dimension (indices 0, 1, 2, 3, 4):
    ##   -1 refers to index 4 (last)
    ##   -2 refers to index 3 (second-to-last)
    ##   -3 refers to index 2 (third-to-last)
    ##   -4 refers to index 1 (fourth-to-last)
    ##   -5 refers to index 0 (first element, but this is unusual)
    ##
    ## This is equivalent to Python's -1, -2, -3, etc. negative indexing.

    let arr5 = @[10.0, 20.0, 30.0, 40.0, 50.0].toTorchTensor()

    test formatName("-1 is the last element", "arr5[-1]"):
      check: arr5[-1].item(float64) == 50.0

    test formatName("-2 is second-to-last", "arr5[-2]"):
      check: arr5[-2].item(float64) == 40.0

    test formatName("-3 is third-to-last", "arr5[-3]"):
      check: arr5[-3].item(float64) == 30.0

    test formatName("-4 is fourth-to-last", "arr5[-4]"):
      check: arr5[-4].item(float64) == 20.0

    test formatName("-5 equals 0", "arr5[-5]"):
      check: arr5[-5].item(float64) == 10.0

    test formatName("Slice from -3 to -1 (inclusive)", "arr5[-3..-1]"):
      ## arr5[-3:-1] -> Python equivalent: arr5[-3:-1] -> indices 2, 3
      ## Note: -1 as STOP is exclusive in Python, so -3:-1 gives 2 elements
      let sliced = arr5[-3..-1]
      let expected = @[30.0, 40.0].toTorchTensor.to(kFloat64)
      check: sliced == expected

    test formatName("Slice from 0 to -1", "arr5[0..-1]"):
      ## arr5[0:-1] -> Python equivalent: arr5[0:-1] -> all but last
      ## Note: -1 as STOP is exclusive in Python
      let sliced = arr5[0..-1]
      let expected = @[10.0, 20.0, 30.0, 40.0].toTorchTensor.to(kFloat64)
      check: sliced == expected

    # TODO: exclusive with negative (a[<..-2]) not yet supported
    test formatName("Slice from -4 to -2", "arr5[-4..-2]"):
      ## arr5[-4:-2] -> Python equivalent: arr5[-4:-2] -> indices 1, 2
      ## Note: -2 as STOP is exclusive in Python
      let sliced = arr5[-4..-2]
      let expected = @[20.0, 30.0].toTorchTensor.to(kFloat64)
      check: sliced == expected

  #   test formatName("Multi-head attention: split heads", "a[:, head_idx*head_size:(head_idx+1)*head_size, :]"):
  #     ## Pattern: a[:, head_idx*head_size:(head_idx+1)*head_size, :]
  #     var t = arange(240, kFloat64).reshape(@[2, 6, 20])  # batch=2, 6 heads, features=20
  #     let head_idx = 2
  #     let head_size = 4
  #     let head_slice = t[_, head_idx*head_size..<(head_idx+1)*head_size, _]
  #     check: head_slice.shape[0] == 2
  #     check: head_slice.shape[1] == 4
  #     check: head_slice.shape[2] == 20

  #   test formatName("Multi-head attention: all heads", "a[:, :, :]"):
  #     ## Pattern: a[:, :, :] (no slicing, get all heads)
  #     var t = arange(240, kFloat64).reshape(@[2, 6, 20])
  #     let all_heads = t[_, _, _]
  #     check: all_heads.shape == @[2, 6, 20]

  #   test formatName("Attention: causal mask preparation", "Upper triangle mask"):
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

  #   test formatName("Rotary Position Embedding (RoPE)", "a[batch, seq, ::2]"):
  #     ## Pattern: interleaved positions for cos/sin computation
  #     var t = arange(40, kFloat64).reshape(@[2, 4, 5])
  #     let cos_sin = t[_, _, _..<5|2]  # Every other position
  #     check: cos_sin.shape[2] == 3  # 5 elements, step 2 = 3 values

  #   test formatName("Flash Attention-like slicing", "a[q_start..q_end, :, k_start..k_end]"):
  #     ## Pattern: a[q_start..q_end, :, k_start..k_end]
  #     var t = arange(200, kFloat64).reshape(@[10, 8, 10])
  #     let q_start = 2
  #     let q_end = 5
  #     let k_start = 1
  #     let k_end = 4
  #     let sliced = t[q_start..<q_end, _, k_start..<k_end]
  #     check: sliced.shape[0] == 3
  #     check: sliced.shape[1] == 8
  #     check: sliced.shape[2] == 3

  #   test formatName("Output projection slice", "a[:, :, -proj_dim:]"):
  #     ## Pattern: a[:, :, -proj_dim:]
  #     var t = arange(100, kFloat64).reshape(@[2, 5, 10])
  #     let proj_dim = 4
  #     let projected = t[_, _, ^proj_dim..^0]
  #     check: projected.shape[2] == 4

  #   test formatName("Vandermonde: row access with span", "v[1, _]"):
  #     let row1 = v[1, _]
  #     check: row1.shape[0] == 5
  #     check: row1[0].item(float64) == 2.0
  #     check: row1[4].item(float64) == 32.0

  #   test formatName("Vandermonde: column access with span", "v[_, 2]"):
  #     let col2 = v[_, 2]
  #     check: col2.shape[0] == 5
  #     check: col2[0].item(float64) == 1.0  # 1^3 = 1
  #     check: col2[4].item(float64) == 125.0  # 5^3 = 125

  #   test formatName("Vandermonde: submatrix slice", "v[1..<4, 1..<4]"):
  #     let sub = v[1..<4, 1..<4]
  #     check: sub.shape[0] == 3
  #     check: sub.shape[1] == 3
  #     check: sub[0, 0].item(float64) == 4.0   # 2^2 = 4
  #     check: sub[2, 2].item(float64) == 64.0  # 4^2 = 64

  #   test formatName("Vandermonde: every other row", "v[_..<5|2, _]"):
  #     let evens = v[_..<5|2, _]
  #     check: evens.shape[0] == 3
  #     check: evens[0, 0].item(float64) == 1.0   # Row 0
  #     check: evens[1, 0].item(float64) == 3.0   # Row 2
  #     check: evens[2, 0].item(float64) == 5.0   # Row 4

  #   test formatName("Vandermonde: last 2 columns", "v[_, -2..-0]"):
  #     let last2 = v[_, -2..-0]
  #     check: last2.shape[1] == 2
  #     check: last2[0, 0].item(float64) == 1.0   # 1^4 = 1
  #     check: last2[4, 1].item(float64) == 3125.0  # 5^5 = 3125

  #   test formatName("Vandermonde: anti-diagonal with -", "v[-5..-1, -5..-1]"):
  #     ## -N counts from end, so -4..-0 is all indices
  #     let diag = v[-5..-1, -5..-1]
  #     check: diag.shape[0] == 5
  #     check: diag[0, 0].item(float64) == 5.0   # 5^0 = 1, but reversed!
  #     check: diag[4, 4].item(float64) == 5.0   # 1^4 = 1, reversed

  #   test formatName("Vandermonde: top-right triangle", "v[_..<5, 3..<5]"):
  #     let upper = v[_..<5, 3..<5]
  #     check: upper.shape[0] == 5
  #     check: upper.shape[1] == 2
  #     check: upper[0, 1].item(float64) == 1.0   # row 0, col 4

  #   test formatName("Vandermonde: bottom-left triangle", "v[2..<5, _..<3]"):
  #     let lower = v[2..<5, _..<3]
  #     check: lower.shape[0] == 3
  #     check: lower.shape[1] == 3
  #     check: lower[0, 0].item(float64) == 9.0   # row 2, col 0

  #   test formatName("Vandermonde: assign to slice", "v[0..<2, 0..<2] = 0"):
  #     var v2 = v.clone()
  #     v2[0..<2, 0..<2] = 0.0
  #     check: v2[0, 0].item(float64) == 0.0
  #     check: v2[1, 1].item(float64) == 0.0
  #     check: v2[2, 2].item(float64) == 27.0  # Unchanged

  #   test formatName("Vandermonde: ellipsis row access", "v[1, `...`]"):
  #     let row1_ellipsis = v[1, `...`]
  #     check: row1_ellipsis.shape[0] == 5
  #     check: row1_ellipsis[0].item(float64) == 2.0

when isMainModule:
  main()
