## Binding styles for the transform_layout argument, one MSL byte count each.
##
## - T1 symbol arg, the tip's unbound form
## - T2 expression arg, unbound form, the arg tree spliced per dimension use
## - T3 expression arg, raw `let lyt =` prologue, hand-expanded
##
## T4 hand-expands `evalOnceAs(lyt,)`. The body uses it_l three times,
## per-use re-emission shows up as byte deltas.
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra

const t1Msl = metal:
  proc t1Kernel(C: ptr UncheckedArray[float32]) {.global.} =
    let a = make_layout((32, 16), (1, 32))
    let r = transform_layout(a, (4, 4)):
      make_layout(it_l.shape, it_l.stride * it_l.stride + it_l.shape)
    C[0] = float32 toIntVal crd2idx(r, (1, 1))

const t2Msl = metal:
  proc t2Kernel(C: ptr UncheckedArray[float32]) {.global.} =
    let r = transform_layout(make_layout((32, 16), (1, 32)), (4, 4)):
      make_layout(it_l.shape, it_l.stride * it_l.stride + it_l.shape)
    C[0] = float32 toIntVal crd2idx(r, (1, 1))

const t3Msl = metal:
  proc t3Kernel(C: ptr UncheckedArray[float32]) {.global.} =
    let lyt = make_layout((32, 16), (1, 32))
    let r0 = lyt.dimension(0)
    let r1 = lyt.dimension(1)
    let r = make_layout((r0.shape, r1.shape), (r0.stride * r0.stride + r0.shape, r1.stride * r1.stride + r1.shape))
    C[0] = float32 toIntVal crd2idx(r, (1, 1))

const t4Msl = metal:
  proc t4Kernel(C: ptr UncheckedArray[float32]) {.global.} =
    block:
      evalOnceAs(lyt, make_layout((32, 16), (1, 32)))
      let r0 = lyt.dimension(0)
      let r1 = lyt.dimension(1)
      let r = make_layout((r0.shape, r1.shape), (r0.stride * r0.stride + r0.shape, r1.stride * r1.stride + r1.shape))
    let x = 1
    C[0] = float32 x

proc size(msl: string): int = msl.len

writeFile("/tmp/probe_t1.msl", t1Msl)
writeFile("/tmp/probe_t2.msl", t2Msl)
writeFile("/tmp/probe_t3.msl", t3Msl)
writeFile("/tmp/probe_t4.msl", t4Msl)
echo "T1 symbol-arg unbound:        ", size(t1Msl)
echo "T2 expr-arg unbound:          ", size(t2Msl)
echo "T3 expr-arg raw-let binding:  ", size(t3Msl)
echo "T4 expr-arg evalOnceAs:       ", size(t4Msl)
