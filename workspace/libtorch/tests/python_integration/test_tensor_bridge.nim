# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Nim ↔ Python Tensor Bridge Tests

# Tests for tensors_py bidirectional conversion between Nim Tensor and Python torch.Tensor.

import
  std/strutils,
  nimpy,
  nimpy/py_lib as pyl,
  workspace/libtorch,
  workspace/libtorch/src/tensors_py {.all.},
  workspace/libtorch/libtorch_testutils

proc main() =
  echo ""
  echo "Testing tensors_py bridge..."
  echo ""

  # -----------------------------------------------------------------------
  # Python → Nim

  runTest "Python → Nim: basic float32 tensor":
    proc(): bool =
      let torch = pyImport("torch")
      let pyTensor = callMethod(torch, "tensor", @[1.0, 2.0, 3.0])
      let nim = tensorFromPyObject(pyTensor)
      nim.isDefined() and nim.shape == @[3]

  runTest "Python → Nim: int64 tensor (batch input shape)":
    proc(): bool =
      let torch = pyImport("torch")
      let pyTensor = callMethod(torch, "tensor", @[@[1'i64, 2, 3, 4, 5]])
      let nim = tensorFromPyObject(pyTensor)
      nim.isDefined() and nim.shape == @[1, 5]

  runTest "Python → Nim: multi-dimensional shape preservation":
    proc(): bool =
      let torch = pyImport("torch")
      let py = callMethod(torch, "rand", @[2, 3, 4])
      let nim = tensorFromPyObject(py)
      nim.shape == @[2, 3, 4]

  runTest "Python → Nim: non-torch object raises ValueError":
    proc(): bool =
      let listFn = pyImport("builtins").getAttr("list")
      let pyList = callMethod(listFn, "__call__", @[@[1, 2, 3]])
      try:
        discard tensorFromPyObject(pyList)
        false  # Should have raised
      except ValueError:
        true

  # -----------------------------------------------------------------------
  # Nim → Python

  runTest "Nim → Python: basic int64 tensor":
    proc(): bool =
      let ids = @[1'i64, 2, 3]
      let nim = ids.toTensor().unsqueeze(0)
      let py = tensorToPyObject(nim)
      let pyInt = callMethod(py, "tolist")
      pyInt.to(seq[seq[int]]) == @[@[1, 2, 3]]

  runTest "Nim → Python: float32 tensor":
    proc(): bool =
      let t = @[1.0'f32, 2.0, 3.0].toTensor()
      let py = tensorToPyObject(t)
      let pyList = callMethod(py, "tolist")
      let values = pyList.to(seq[float])
      abs(values[0] - 1.0) < 1e-6 and
      abs(values[1] - 2.0) < 1e-6 and
      abs(values[2] - 3.0) < 1e-6

  runTest "Nim nil tensor → Python None":
    proc(): bool =
      let t: Tensor = nil
      let py = tensorToPyObject(t)
      let isNone = cast[pointer](py.privateRawPyObj()) == cast[pointer](pyl.pyLib.Py_None)
      isNone

  # -----------------------------------------------------------------------
  # Roundtrip

  runTest "Roundtrip preserves values (float32)":
    proc(): bool =
      let torch = pyImport("torch")
      let py = callMethod(torch, "tensor", @[@[1.5, 2.7, 3.1]])
      let roundTrip = tensorToPyObject(tensorFromPyObject(py))
      callMethod(torch, "allclose", py, roundTrip).to(bool)

  runTest "Roundtrip preserves values (int64)":
    proc(): bool =
      let torch = pyImport("torch")
      let py = callMethod(torch, "tensor", @[@[151643'i64, 42, 9876]])
      let roundTrip = tensorToPyObject(tensorFromPyObject(py))
      callMethod(torch, "equal", py, roundTrip).to(bool)

  # -----------------------------------------------------------------------
  # Capsule helpers

  runTest "capsuleNew + capsuleGetPointer roundtrip":
    proc(): bool =
      var stored: int = 42
      let c = capsuleNew(addr stored, "test".cstring)
      let retrievedPtr = capsuleGetPointer(c, "test".cstring)
      retrievedPtr == addr stored

when isMainModule:
  main()
  echo ""
  echo "All tensors_py tests passed ✅"
