{.experimental: "views".}

## Prealloc dedup-poison falsifier: alloc-without-enqueue reads, narrow-chain offset materialization
##
## Test that a tensor allocated without enqueueing its kernel reads back garbage, while narrow-chain offset materialization stays correct
##
## Run:
##   timeout 95 nim c -r -d:release --hints:off --warnings:off --outdir:build/wip \
##     --nimcache:nimcache/wip400 workspace/tensorcery/tests/graph/t_prealloc_falsifier.nim

from std/importutils import privateAccess
import workspace/crucible
import workspace/iface
import workspace/ceramic
import workspace/tensorcery
import workspace/tensorcery/src/tensors_globals

privateAccess(Tensor)

proc concreteImpl(t: Tensor, T: typedesc): auto =
  t.node.getImpl().to(gMetalEngine().getTensorImplType(T))

proc main =
  const run = runMetal

  block:
    let y = full([4], 5.0'f32, run)
    gMetalEngine().allocDataIn(concreteImpl(y, float32))
    let got = y.item(float32)
    doAssert got == 5.0'f32,
      "allocated-without-enqueue full, got " & $got & " want 5.0"

  block:
    let y = full([4], 5.0'f32, run)
    gMetalEngine().allocDataIn(concreteImpl(y, float32))
    let v = y.narrow(0, 1, 2)
    let got = v.item(float32)
    doAssert got == 5.0'f32,
      "narrow over allocated-without-enqueue full, got " & $got & " want 5.0"

  block:
    let y = full([2], 5.0'f32, run)
    gMetalEngine().allocDataIn(concreteImpl(y, float32))
    let z = y + y
    let got = z.item(float32)
    doAssert got == 10.0'f32,
      "add over allocated-without-enqueue full computed " & $got & " want 10.0"

  block:
    let t = @[1.0'f32, 2, 3, 4, 5, 6, 7, 8].toTensor([8], run, float32)
    let a = t.narrow(0, 2, 4)
    let b = a.narrow(0, 1, 2)
    let got = b.item(float32)
    doAssert got == 4.0'f32,
      "narrow-of-narrow over an allocated leaf, got " & $got & " want 4.0"

  block:
    let t = @[1.0'f32, 2, 3, 4, 5, 6, 7, 8].toTensor([8], run, float32)
    let y = t + t
    let a = y.narrow(0, 4, 4)
    let b = a.narrow(0, 1, 2)
    let got = b.item(float32)
    doAssert got == 12.0'f32,
      "narrow-of-narrow over an unallocated op output, got " & $got & " want 12.0"

  block:
    let t = @[1.0'f32, 2, 3, 4, 5, 6, 7, 8].toTensor([8], run, float32)
    let y = t + t
    let a = y.narrow(0, 4, 4)
    let b = a.narrow(0, 1, 2)
    discard a.item(float32)
    let got = b.item(float32)
    doAssert got == 12.0'f32,
      "narrow-of-narrow after an earlier eval of the intermediate, got " & $got & " want 12.0"

  block:
    let t = @[1.0'f32, 2, 3, 4].toTensor([4], run, float32)
    let v = t.narrow(0, 1, 2)
    let s1 = v + v
    let got1 = s1.item(float32)
    doAssert got1 == 4.0'f32,
      "diamond first eval, got " & $got1 & " want 4.0"
    let s2 = v + v + v
    let got2 = s2.item(float32)
    doAssert got2 == 6.0'f32,
      "diamond second eval, got " & $got2 & " want 6.0"

when isMainModule:
  main()
