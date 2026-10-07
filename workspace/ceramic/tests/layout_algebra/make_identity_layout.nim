# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## make_basis_like, unit coordinate strides for a shape profile,
## and make_identity_layout, the identity layout built from them.
## Run: nim c -r --hints:off --warnings:off -p:"$PWD" workspace/ceramic/tests/layout_algebra/make_identity_layout.nim
## Reference: pycute atuple.py (make_basis_like), layout.py (make_identity_layout)

import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra/layout_constructors

proc runMakeBasisLikeTests: void =
  let atoms = make_basis_like((10, 20))
  doAssert $atoms == "(1@0, 1@1)"
  doAssert typeof(atoms[0]) is CoordStride[(1,)]

  let nested = make_basis_like((10, (20, 30)))
  doAssert $nested == "(1@0, (1@0@1, 1@1@1))"
  doAssert typeof(nested[1][1]) is CoordStride[(0, (0, 1))]

  doAssert make_basis_like(10) == 1

proc runMakeIdentityLayoutTests: void =
  let flat = make_identity_layout((4, 8))
  doAssert flat.shape === (4, 8)
  doAssert $flat.stride == "(1@0, 1@1)"
  doAssert typeof(flat.stride[0]) is CoordStride[(1,)]
  doAssert typeof(flat.stride[1]) is CoordStride[(0, 1)]

  let nested = make_identity_layout((4, (2, 3)))
  doAssert nested.shape === (4, (2, 3))
  doAssert $nested.stride == "(1@0, (1@0@1, 1@1@1))"
  doAssert typeof(nested.stride[1][0]) is CoordStride[(0, (1,))]

  let scalar = make_identity_layout(6)
  doAssert scalar.shape === 6
  doAssert scalar.stride === 1

proc runTests =
  echo "\n── make_basis_like: unit coordinate strides ──"
  runMakeBasisLikeTests()
  echo "\n── make_identity_layout: a shape as its own layout ──"
  runMakeIdentityLayoutTests()

runTests()
echo "make_identity_layout: all tests passed"
