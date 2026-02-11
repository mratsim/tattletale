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
  std/memfiles,
  std/strformat,
  std/strutils,
  std/tables,
  workspace/safetensors,
  workspace/libtorch

const FIXTURES_DIR = currentSourcePath().parentDir() / "fixtures"

const Patterns = ["gradient", "alternating", "repeating"]

const Shapes: array[4, seq[int]] = [
  @[8],
  @[4, 4],
  @[2, 3, 4],
  @[3, 2, 2, 2]
]

const TestedDtypes = [F64, F32, F16, I64, I32, I16, I8, U8]

proc generateExpectedTensor*(pattern: string, shape: seq[int], dtype: ScalarKind): TorchTensor =
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

proc generateVandermondeExpected*(shape: openArray[int], dtype: ScalarKind): TorchTensor =
  let numel = shape.product()
  var data = newSeq[float64](numel)
  for i in 0..<5:
    for j in 0..<5:
      data[j * 5 + i] = pow(float64(i + 1), float64(j))
  let sizes = [numel]
  let flat = from_blob(data[0].unsafeAddr, sizes, dtype)
  flat.reshape(shape)

proc main() =
  suite "safetensors fixtures tests":
    test "vandermonde single fixture test":
      let fixturePath = FIXTURES_DIR / "vandermonde.safetensors"
      check fileExists(fixturePath)

      var memFile = memFiles.open(fixturePath, mode = fmRead)
      defer: close(memFile)

      let (st, dataSectionOffset) = safetensors.load(memFile)

      let key = "F64_vandermonde_5x5"
      check st.tensors.hasKey(key)

      let shape = @[5, 5]
      let info = st.tensors[key]
      check info.shape == shape

      let expectedTensor = generateVandermondeExpected(shape, kFloat64)
      let actualTensor = st.getTensor(memFile, dataSectionOffset, key)
      check actualTensor == expectedTensor

    test "load python-generated safetensors fixtures":
      let fixturePath = FIXTURES_DIR / "fixtures.safetensors"
      check fileExists(fixturePath)

      var memFile = memFiles.open(fixturePath, mode = fmRead)
      defer: close(memFile) # TODO - close them after data is loaded but before testing to ensure we own the buffer

      let (st, dataSectionOffset) = safetensors.load(memFile)

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
            let actualTensor = st.getTensor(memFile, dataSectionOffset, key)
            check actualTensor == expectedTensor
            count += 1

      doAssert count == Patterns.len * Shapes.len * TestedDtypes.len

when isMainModule:
  main()