# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Test for safetensors shape aliasing bug
##
## BUG: from_blob() stores a pointer to the shape array, not a copy.
## When using temporary arrays like [shape[0], shape[1]], the array
## is stack-allocated and deallocated at end of scope.
## Subsequent calls overwrite the same stack memory, causing all tensors
## to share the last shape values.
##
## FIX: Use toOpenArray on the seq[int] field that persists in SafeTensor.
## See TensorInfo.shape documentation in safetensors.nim.

import
  std/memfiles,
  std/os,
  workspace/safetensors,
  workspace/libtorch

const FIXTURES_DIR = currentSourcePath().parentDir() / "fixtures"
const FIXTURE_PATH = FIXTURES_DIR / "shape_aliasing_multi_tensor.safetensors"

# Expected shapes for each tensor
const
  expected_shape_a = [2048, 1024]
  expected_shape_b = [1024, 1024]
  expected_shape_c = [1024, 2048]
  expected_shape_d = [3072, 1024]
  expected_shape_e = [1024, 3072]

proc main() =
  echo "Testing safetensors shape aliasing bug"
  echo "Fixture: ", FIXTURE_PATH
  echo ""
  
  if not fileExists(FIXTURE_PATH):
    echo "❌ Fixture not found. Run generate_multi_shape_tensors.py first."
    quit(1)
  
  var mf = memFiles.open(FIXTURE_PATH, mode = fmRead)
  defer: mf.close()
  
  var st = safetensors.load(mf)
  
  echo "Loading tensors and verifying shapes:"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  
  # Load all tensors and verify shapes on initial load
  let tensor_a = st.getTensorOwned("tensor_a")
  let shape_a = tensor_a.shape
  echo "tensor_a loaded: [", shape_a[0], ", ", shape_a[1], "] (expected: [", expected_shape_a[0], ", ", expected_shape_a[1], "])"
  doAssert shape_a[0] == expected_shape_a[0] and shape_a[1] == expected_shape_a[1], "tensor_a initial shape wrong"
  
  let tensor_b = st.getTensorOwned("tensor_b")
  let shape_b = tensor_b.shape
  echo "tensor_b loaded: [", shape_b[0], ", ", shape_b[1], "] (expected: [", expected_shape_b[0], ", ", expected_shape_b[1], "])"
  doAssert shape_b[0] == expected_shape_b[0] and shape_b[1] == expected_shape_b[1], "tensor_b initial shape wrong"
  
  let tensor_c = st.getTensorOwned("tensor_c")
  let shape_c = tensor_c.shape
  echo "tensor_c loaded: [", shape_c[0], ", ", shape_c[1], "] (expected: [", expected_shape_c[0], ", ", expected_shape_c[1], "])"
  doAssert shape_c[0] == expected_shape_c[0] and shape_c[1] == expected_shape_c[1], "tensor_c initial shape wrong"
  
  let tensor_d = st.getTensorOwned("tensor_d")
  let shape_d = tensor_d.shape
  echo "tensor_d loaded: [", shape_d[0], ", ", shape_d[1], "] (expected: [", expected_shape_d[0], ", ", expected_shape_d[1], "])"
  doAssert shape_d[0] == expected_shape_d[0] and shape_d[1] == expected_shape_d[1], "tensor_d initial shape wrong"
  
  let tensor_e = st.getTensorOwned("tensor_e")
  let shape_e = tensor_e.shape
  echo "tensor_e loaded: [", shape_e[0], ", ", shape_e[1], "] (expected: [", expected_shape_e[0], ", ", expected_shape_e[1], "])"
  doAssert shape_e[0] == expected_shape_e[0] and shape_e[1] == expected_shape_e[1], "tensor_e initial shape wrong"
  
  echo ""
  echo "Rechecking saved shape variables after all loads:"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "tensor_a shape: [", shape_a[0], ", ", shape_a[1], "] (expected: [", expected_shape_a[0], ", ", expected_shape_a[1], "])"
  echo "tensor_b shape: [", shape_b[0], ", ", shape_b[1], "] (expected: [", expected_shape_b[0], ", ", expected_shape_b[1], "])"
  echo "tensor_c shape: [", shape_c[0], ", ", shape_c[1], "] (expected: [", expected_shape_c[0], ", ", expected_shape_c[1], "])"
  echo "tensor_d shape: [", shape_d[0], ", ", shape_d[1], "] (expected: [", expected_shape_d[0], ", ", expected_shape_d[1], "])"
  echo "tensor_e shape: [", shape_e[0], ", ", shape_e[1], "] (expected: [", expected_shape_e[0], ", ", expected_shape_e[1], "])"
  
  echo ""
  echo "Verification:"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  
  doAssert shape_a[0] == expected_shape_a[0] and shape_a[1] == expected_shape_a[1], "tensor_a shape corrupted after loading all tensors"
  doAssert shape_b[0] == expected_shape_b[0] and shape_b[1] == expected_shape_b[1], "tensor_b shape corrupted after loading all tensors"
  doAssert shape_c[0] == expected_shape_c[0] and shape_c[1] == expected_shape_c[1], "tensor_c shape corrupted after loading all tensors"
  doAssert shape_d[0] == expected_shape_d[0] and shape_d[1] == expected_shape_d[1], "tensor_d shape corrupted after loading all tensors"
  doAssert shape_e[0] == expected_shape_e[0] and shape_e[1] == expected_shape_e[1], "tensor_e shape corrupted after loading all tensors"
  
  echo "✅ All shapes correct - no aliasing bug detected"

when isMainModule:
  main()