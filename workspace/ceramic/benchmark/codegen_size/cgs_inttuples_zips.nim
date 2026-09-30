## Codesize ledger, int-tuple zip family.
##
## Every kernel compiles one real call site to Metal Shading Language with the crucible metal backend, one row per kernel.
## zipLeavesWith and zipDimensionsWith get one runtime row each over runtime leaf values.
##
## Run from the tattletale/ dir, plain release protocol, usage in benchmark/codegen_size/README.md.
##
## Baselines live in .scratchspace/20260929-1527-C07D02-ldivide-emission/reports/codesize_ledger.md.
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# zipLeavesWith over a stride/shape pair, filter_zeros body, static Int strides branch per type, runtime strides compare against 0
const zipLeavesMsl = metal:
# tiles-allow measured kernel, the call site reads a tuple leaf raw
  proc zipLeavesKernel(C: ptr UncheckedArray[float32]; M, S: int32) {.global.} =
    let r = zipLeavesWith((int(S), 1), (int(M), 16)):
      block:
        when it_a is Int:
          when it_a.V == 0: Int[1]() else: it_b
        elif it_b is Int: it_b
        else: (if it_a == 0: 1 else: it_b)
    C[0] = float32 toIntVal(r[0])

# zipDimensionsWith over two runtime tuples, top-level pairwise map, leftover elements append unchanged
const zipDimensionsMsl = metal:
# tiles-allow measured kernel, the call site reads a tuple leaf raw
  proc zipDimensionsKernel(C: ptr UncheckedArray[float32]; M, N, S: int32) {.global.} =
    let r = zipDimensionsWith((int(M), int(N), 16), (int(S), 32)):
      it_a + it_b
    C[0] = float32 toIntVal(r[0])

# ── kernel rows ──

cgsReport("cgs_inttuples_zips", [
  cgsReceipt("zipLeavesKernel", zipLeavesMsl),
  cgsReceipt("zipDimensionsKernel", zipDimensionsMsl)])
