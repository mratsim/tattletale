{.experimental: "views".}

## View offset materialization values across allocation timing on Metal
##
## Test that a view reads the same values however its offset materializes:
## over an eager or an unallocated source,
## across narrow/reshape chains, eval timings, and window edges.
##
## Run:
##   timeout 95 nim c -r -d:release --hints:off --warnings:off \
##     --outdir:build/wip --nimcache:nimcache/wip401 \
##     workspace/tensorcery/tests/t_prealloc_view_offsets_metal.nim

from std/importutils import privateAccess
from std/sequtils import mapIt
import workspace/crucible
import workspace/ceramic
import workspace/tensorcery

privateAccess(Tensor)

proc main =
  const run = runMetal
  const xd = @[1.0'f32, 2, 3, 4, 5, 6, 7, 8]

  proc expect(start, width: int, scale = 1.0'f32): Tensor =
    mapIt(start ..< start + width, (scale * xd[it]).to(bfloat16)).toTensor([width], run, bfloat16)

  proc doubled: Tensor =
    var x = xd.toTensor([8], run, bfloat16)
    x + x

  block:
    let y = doubled()
    let v = y.narrow(0, 2, 4)
    eval(v)
    doAssert v.allClose(expect(2, 4, 2.0)),
      "narrow over an unallocated op output: the offset folds at enqueue"

  block:
    let v = xd.toTensor([8], run, bfloat16).narrow(0, 2, 4)
    eval(v)
    doAssert v.allClose(expect(2, 4)),
      "narrow over an eager leaf: materialized at enqueue, the source kernel never enqueues"

  block:
    let a = doubled().narrow(0, 2, 4)
    let r = reshape(a, [2, 2])
    doAssert r.allClose(expect(2, 4, 2.0).reshape([2, 2])),
      "two-level fold, narrow then reshape over an unallocated op output"
    let b = a.narrow(0, 1, 2)
    doAssert b.allClose(expect(3, 2, 2.0)),
      "two-level fold, narrow then narrow over an unallocated op output"

  block:
    let y = doubled()
    eval(y)
    let v = y.narrow(0, 2, 4)
    doAssert v.allClose(expect(2, 4, 2.0)),
      "narrow over a source evaluated in an earlier batch: the materialized offset derives from the collapsed leaf"

  block:
    let y = doubled()
    let v1 = y.narrow(0, 1, 2)
    let v2 = y.narrow(0, 5, 3)
    eval(v1, v2)
    doAssert v1.allClose(expect(1, 2, 2.0)),
      "diamond in one batch, first narrow over the shared op output"
    doAssert v2.allClose(expect(5, 3, 2.0)),
      "diamond in one batch, second narrow over the shared op output"

  block:
    let y = doubled()
    let v1 = y.narrow(0, 1, 2)
    eval(v1)
    let v2 = y.narrow(0, 5, 3)
    eval(v2)
    doAssert v1.allClose(expect(1, 2, 2.0)),
      "diamond split batches, first narrow eval'd before the source allocation"
    doAssert v2.allClose(expect(5, 3, 2.0)),
      "diamond split batches, second narrow materializes once after the first eval allocated the source"

  block:
    let v = doubled().narrow(0, 2, 4)
    eval(v)
    eval(v)
    doAssert v.allClose(expect(2, 4, 2.0)),
      "the same view eval'd twice keeps its values"

  block:
    let x = xd.toTensor([8], run, bfloat16)
    doAssert (x + x).narrow(0, 0, 8).allClose(expect(0, 8, 2.0)),
      "edge window, full-axis narrow over an unallocated op output"
    doAssert (x + x).narrow(0, 7, 1).allClose(expect(7, 1, 2.0)),
      "edge window, narrow ending on the last element over an unallocated op output"
    doAssert x.narrow(0, 0, 8).allClose(expect(0, 8)),
      "edge window, full-axis narrow over an eager leaf"
    doAssert x.narrow(0, 7, 1).allClose(expect(7, 1)),
      "edge window, narrow ending on the last element over an eager leaf"
    let v = (x + x).narrow(0, 1, 6)
    eval(v)
    doAssert v.narrow(0, 0, 6).allClose(expect(1, 6, 2.0)),
      "edge window, full-axis narrow over an already-eval'd narrow"
    doAssert v.narrow(0, 5, 1).allClose(expect(6, 1, 2.0)),
      "edge window, narrow ending on the last element over an already-eval'd narrow"

  block:
    let x = xd.toTensor([8], run, bfloat16)
    let y = x + x
    doAssertRaises AssertionDefect:
      discard y.narrow(0, 7, 2)
    doAssertRaises AssertionDefect:
      discard x.narrow(0, 7, 2)
    eval(y)
    doAssertRaises AssertionDefect:
      discard y.narrow(0, 7, 2)
    let raw = Tensor(engine: x.engine, node: x.engine.newViewNode(y.node, [4], [1], 6, "raw"))
    doAssertRaises AssertionDefect:
      discard raw.narrow(0, 0, 4)

  block:
    let x = xd[0 ..< 4].toTensor([4], run, bfloat16)
    let y = x + x
    let v = Tensor(engine: x.engine, node: x.engine.newViewNode(y.node, [4], [-1], 3, "negstride"))
    eval(v)
    let expected = mapIt(0 ..< 4, (2.0'f32 * xd[3 - it]).to(bfloat16)).toTensor([4], run, bfloat16)
    doAssert v.allClose(expected),
      "negative stride over an unallocated op output: a window ending at the source's last element reads reversed"
    let bad = Tensor(engine: x.engine, node: x.engine.newViewNode(y.node, [4], [-1], 4, "negstride"))
    doAssertRaises AssertionDefect:
      discard bad.narrow(0, 0, 4)

when isMainModule:
  main()
