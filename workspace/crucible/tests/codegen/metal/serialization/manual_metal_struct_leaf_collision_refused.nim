## Metal: a struct-argument leaf name colliding with a kernel name refuses at compile time.
##
## Test that a kernel whose struct parameter explodes to a leaf name a local already uses
## dies in the struct explosion before any shader source is emitted.
## The refusal names the pass, kernel, and leaf:
##
##   Error: pass explodeGlobalStructWithPtrFields: kernel kern: leaf 'view_p' collides with an existing kernel parameter or local
##
## Run:
##   nim c --hints:off --warnings:off \
##     --outdir:build/tests --nimcache:nimcache/tests \
##     workspace/crucible/tests/codegen/metal/serialization/manual_metal_struct_leaf_collision_refused.nim

import workspace/crucible

type View = object
  p: ptr UncheckedArray[float32]

const collisionMetal = metal:
  proc kern(view: View, output: ptr UncheckedArray[float32]) {.global.} =
    var view_p: float32 = 1.0'f32
    output[0] = view_p

proc runTest() =
  var engine = newEngine(runMetal)
  engine.ingest(collisionMetal)
  echo "UNREACHABLE — the leaf-name collision must be refused at compile time"

when isMainModule:
  runTest()
