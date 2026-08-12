## Tattletale
## Copyright (c) 2026 Mamy André-Ratsimbazafy
## Licensed and distributed under either of
##   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
##   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
## at your option. This file may not be copied, modified, or distributed except according to those terms.
##
## OpenCL: empty-object construction must emit `{0}`, never `{}`.
## OpenCL C is C99 and rejects an empty initializer list for a struct
## ("scalar initializer cannot be empty"). A stateless object with no
## fields previously emitted `(struct X){}`, which fails to build on the
## device. This test compiles and executes a kernel that constructs an
## empty object, and pins the emitted initializer text.
##
## Run:
##   cd tattletale
##   nim c -r --hints:off --warnings:off \
##     workspace/crucible/tests/codegen/opencl/test_opencl_empty_object_init.nim

import std/strutils
import workspace/crucible

type
  Stateless = object
    discard

const kernelCode = opencl:
  proc makeStateless(): Stateless {.device.} =
    result = Stateless()
  proc kernelMain(output: ptr UncheckedArray[uint32]) {.global.} =
    var s = makeStateless()
    output[0] = 42'u32

echo kernelCode

# The empty struct initializer must be `{0}`: `{}` is invalid C99.
doAssert kernelCode.contains("{0}"), "expected a {0} zero-initializer for the empty object"
doAssert not kernelCode.contains("){}"), "empty initializer list {} is invalid OpenCL C"

block:
  var engine = bkOpenCL.init()
  engine.ingest(kernelCode)
  var outVal: array[1, uint32]
  engine.run("kernelMain", outVal, ())
  doAssert outVal[0] == 42
  echo "  OK"
