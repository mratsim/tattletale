## Codesize ledger, int-tuple fold family.
##
## Every kernel compiles one real call site to Metal Shading Language with the crucible metal backend, one row per kernel.
##
## fold, prefix_scanIt, and suffix_scanIt get one runtime row each over runtime leaf values.
##
## fold_recurse, prefix_scanIt_recurse, suffix_scanIt_recurse, head_accumulator, and tail_accumulator carry no separate rows.
##
## Run from the tattletale/ dir, plain release protocol, usage in benchmark/codegen_size/README.md.
##
## Baselines live in .scratchspace/20260929-1527-C07D02-ldivide-emission/reports/codesize_ledger.md.
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# fold over a runtime product chain, size/product call-site shape, divide-chain pair-tuple input
const foldMsl = metal:
# tiles-allow measured kernel, the call site reads a tuple leaf raw
  proc foldKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let t = ((int(M div 16), 16), (int(N div 16), 16))
    C[0] = float32 fold(t, 1, acc * it)

# prefix_scanIt over a runtime product scan, stride-from-shape call-site shape, recurse template and tail_accumulator inside
const prefixScanMsl = metal:
# tiles-allow measured kernel, the call site reads a tuple leaf raw
  proc prefixScanKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let t = ((int(M), 16), (int(N), 16))
    let r = prefix_scanIt(t, 1, acc * it)
    C[0] = float32 toIntVal(r[0][0])

# suffix_scanIt over the mirrored runtime scan, recurse template and head_accumulator inside
const suffixScanMsl = metal:
# tiles-allow measured kernel, the call site reads a tuple leaf raw
  proc suffixScanKernel(C: ptr UncheckedArray[float32]; M, N: int32) {.global.} =
    let t = ((int(M), 16), (int(N), 16))
    let r = suffix_scanIt(t, 1, acc * it)
    C[0] = float32 toIntVal(r[0][0])

# ── kernel rows ──

cgsReport("cgs_inttuples_folds", [
  cgsReceipt("foldKernel", foldMsl),
  cgsReceipt("prefixScanKernel", prefixScanMsl),
  cgsReceipt("suffixScanKernel", suffixScanMsl)])
