# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

import
  std/unittest,
  std/os,
  std/math,
  std/strformat,
  std/strutils,
  std/tables,
  workspace/safetensors,
  std/importutils,
  workspace/safetensors/src/safetensors {.all.},
  workspace/libtorch as torch

privateAccess(SafetensorObj)

const FIXTURES_DIR = currentSourcePath().parentDir() / "fixtures"

const Patterns = ["gradient", "alternating", "repeating"]

const Shapes: array[4, seq[int]] = [
  @[8],
  @[4, 4],
  @[2, 3, 4],
  @[3, 2, 2, 2]
]

const TestedDtypes = [F64, F32, F16, I64, I32, I16, I8, U8]

func product(a: seq[int]): int =
  result = 1
  for i in 0 ..< a.len:
    result *= a[i]

proc generateExpectedTensor*(pattern: string, shape: seq[int], dtype: ScalarKind): Tensor =
  let numel = shape.product()

  case pattern
  of "gradient":
    arange(numel, dtype).reshape(shape).to(dtype)
  of "alternating":
    let flat = arange(numel, kInt64)
    let modVal = (flat % 2).to(kFloat64)
    modVal.reshape(shape).to(dtype)
  of "repeating":
    let flat = arange(numel, kInt64)
    let modVal = ((flat % 10) + 1).to(kFloat64)
    modVal.reshape(shape).to(dtype)
  else:
    raise newException(ValueError, "Unknown pattern: " & pattern)

proc genShiftedVandermonde5x5*(dtype: ScalarKind): Tensor =
  ## Generate 5x5 shifted Vandermonde matrix: v[i, j] = i^(j+1)
  ## [[   1    1    1    1    1]
  ##  [   2    4    8   16   32]
  ##  [   3    9   27   81  243]
  ##  [   4   16   64  256 1024]
  ##  [   5   25  125  625 3125]]
  let v = torch.arange(1, 6).reshape(-1, 1) ** torch.arange(1, 6)
  return v.to(dtype)

proc main() =
  suite "safetensors fixtures tests (owned tensors)":
    test "vandermonde single fixture test (owned)":
      let fixturePath = FIXTURES_DIR / "vandermonde.safetensors"
      check fileExists(fixturePath)

      var st = Safetensor.open(fixturePath)

      let key = "F64_vandermonde_5x5"
      check st.tensors.hasKey(key)

      let shape = @[5, 5]
      let info = st.tensors[key]
      check info.shape == shape

      let expectedTensor = genShiftedVandermonde5x5(kFloat64)
      let actualTensor = st.getTensorOwned(key)
      check actualTensor.equal(expectedTensor)

    test "vandermonde BF16 fixture test (owned)":
      let fixturePath = FIXTURES_DIR / "vandermonde.safetensors"
      check fileExists(fixturePath)

      var st = Safetensor.open(fixturePath)

      let key = "BF16_vandermonde_5x5"
      check st.tensors.hasKey(key)

      let shape = @[5, 5]
      let info = st.tensors[key]
      check info.shape == shape

      let expectedTensor = genShiftedVandermonde5x5(kBFloat16)
      let actualTensor = st.getTensorOwned(key)
      check actualTensor.equal(expectedTensor)

    test "getTensorOwned copies survive scope exit":
      # Contract of the owned surface.
      # `open(path)` acquires the mapping and the destructor releases it
      # with the last reference. Owned copies from a live reader
      # stay valid beyond the destructor.
      let fixturePath = FIXTURES_DIR / "vandermonde.safetensors"
      check fileExists(fixturePath)

      let expectedTensor = genShiftedVandermonde5x5(kFloat64)
      var ownedTensor: Tensor
      block readerScope:
        var st = Safetensor.open(fixturePath)
        ownedTensor = st.getTensorOwned("F64_vandermonde_5x5")
      check ownedTensor.equal(expectedTensor)

    test "scope exit releases the mapping":
      # Contract of the ownership surface: the destructor of a reader
      # runs exactly once, with the last reference, releasing the owned
      # mapping. `munmap(nil, 0)` fails, so a destructor that never ran
      # or ran twice raises OSError or corrupts memory when the same
      # fixture is mapped again. The checks below traverse both release
      # points without error.
      let fixturePath = FIXTURES_DIR / "vandermonde.safetensors"
      check fileExists(fixturePath)

      var first = Safetensor.open(fixturePath)
      check first.tensors.hasKey("F64_vandermonde_5x5")
      block secondReaderScope:
        # A second mapping of the same file, copied from the live header
        # tables before either value dies.
        var second = Safetensor.open(fixturePath)
        check second.tensors.hasKey("F64_vandermonde_5x5")
      # The second reader's destructor ran here.
      check first.getTensorOwned("F64_vandermonde_5x5").equal(
        genShiftedVandermonde5x5(kFloat64))
      # first's destructor runs at test-body end.

    test "open python-generated safetensor fixtures (owned)":
      let fixturePath = FIXTURES_DIR / "fixtures.safetensors"
      check fileExists(fixturePath)

      var st = Safetensor.open(fixturePath)

      var count = 0
      for dtype in TestedDtypes:
        for pattern in Patterns:
          for shape in Shapes:
            let key = &"""{dtype}_{pattern}_{shape.join("x")}"""
            if not st.tensors.hasKey(key):
              echo &"[tests_safetensors] Warning: key missing '{key}'"
              continue

            let info = st.tensors[key]
            check info.shape == shape

            let expectedTensor = generateExpectedTensor(pattern, shape, dtype.toTorchType())
            let actualTensor = st.getTensorOwned(key)
            check actualTensor.equal(expectedTensor)
            count += 1

      doAssert count == Patterns.len * Shapes.len * TestedDtypes.len

when isMainModule:
  main()