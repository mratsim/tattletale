{.experimental: "views".}

## Diamond dedup: one enqueue per batch, stable data across batches
##
## Test that a diamond's shared node enqueues its kernel exactly once per batch,
## and a second batch reuses it without recomputing or reallocating its buffer
##
## Run:
##   timeout 95 nim c -r -d:TTT_tensorgraphMetrics -d:release --hints:off --warnings:off --outdir:build/wip \
##     --nimcache:nimcache/wip404 workspace/tensorcery/tests/graph/t_graph_diamond.nim

from std/importutils import privateAccess
import workspace/crucible
import workspace/iface
import workspace/ceramic
import workspace/tensorcery
import workspace/tensorcery/src/tensors_globals

privateAccess(Tensor)

proc concreteImpl(t: Tensor, T: typedesc): auto =
  t.node.getImpl().to(gMetalEngine().getTensorImplType(T))

proc dataPtr(t: Tensor): pointer =
  cast[pointer](concreteImpl(t, float32).data)

proc main =
  const run = runMetal

  block:
    let a = @[1.0'f32, 2].toTensor([2], run, float32)
    let b = @[10.0'f32, 20].toTensor([2], run, float32)
    let x = @[100.0'f32, 200].toTensor([2], run, float32)
    let d = @[1000.0'f32, 2000].toTensor([2], run, float32)
    let c = a + b
    let e = x + c
    let f = c + d
    let before = graphCounters()
    eval(e, f)
    let after = graphCounters()
    let enqueued = after.kernelsEnqueued - before.kernelsEnqueued
    doAssert enqueued == 3,
      "diamond in one batch enqueued " & $enqueued &
      " kernels want 3: add(A,B) once, add(X,C), add(C,D)"
    let gotC = c.item(float32)
    doAssert gotC == 11.0'f32,
      "diamond shared node, got " & $gotC & " want 11.0"
    let gotE = e.item(float32)
    doAssert gotE == 111.0'f32,
      "diamond first root, got " & $gotE & " want 111.0"
    let gotF = f.item(float32)
    doAssert gotF == 1011.0'f32,
      "diamond second root, got " & $gotF & " want 1011.0"

  block:
    let a = @[1.0'f32, 2].toTensor([2], run, float32)
    let b = @[10.0'f32, 20].toTensor([2], run, float32)
    let x = @[100.0'f32, 200].toTensor([2], run, float32)
    let d = @[1000.0'f32, 2000].toTensor([2], run, float32)
    let c = a + b
    let e = x + c
    let f = c + d
    let before1 = graphCounters()
    eval(e)
    let after1 = graphCounters()
    let enqueued1 = after1.kernelsEnqueued - before1.kernelsEnqueued
    doAssert enqueued1 == 2,
      "first diamond batch enqueued " & $enqueued1 &
      " kernels want 2: add(A,B), add(X,C)"
    let gotE = e.item(float32)
    doAssert gotE == 111.0'f32,
      "diamond first batch, got " & $gotE & " want 111.0"
    let cBefore = dataPtr(c)
    let before2 = graphCounters()
    eval(f)
    let after2 = graphCounters()
    let enqueued2 = after2.kernelsEnqueued - before2.kernelsEnqueued
    doAssert enqueued2 == 1,
      "second diamond batch enqueued " & $enqueued2 &
      " kernels want 1: add(C,D) only, C must not recompute"
    let gotF = f.item(float32)
    doAssert gotF == 1011.0'f32,
      "diamond second batch, got " & $gotF & " want 1011.0"
    doAssert dataPtr(c) == cBefore, "C's buffer was reallocated across batches"

when isMainModule:
  main()
