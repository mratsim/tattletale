## Codesize ledger, concat family.
##
## Every kernel compiles one real call site to Metal Shading Language with the crucible
## metal backend and prints its MSL byte size.
##
## Run from the tattletale/ dir with the suite flags of config.nims testerCmd,
## Usage in benchmark/codegen_size/README.md.
##
## Baselines live in .scratchspace/20260929-1527-C07D02-ldivide-emission/reports/codesize_ledger.md.
## Kernels cover the call sites of the concat track.
import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layout_algebra
import workspace/ceramic/src/tensors
import workspace/ceramic/benchmark/codegen_size/codegen_size_analysis

# direct concat of two tuple layouts into one layout
const concatDirectMsl = metal:
  proc concatDirectKernel(C: ptr UncheckedArray[float32]) {.global.} =
    let r = make_layout(concat((2, 3), (4,)), concat((1, 3), (12,)))
    C[0] = float32 toIntVal crd2idx(r, (1, 1, 1))

# tiled_product, concat with the block dimension kept grouped
const tiledProductMsl = metal:
  proc tiledProductKernel(C: ptr UncheckedArray[float32]; M: int32) {.global.} =
    let blk = make_layout((int(M), 4), (1, 4))
    let p = tiled_product(blk, make_layout((2, 2)))
    C[0] = float32 toIntVal size(p)

# flat_product, concat with both dimensions unpacked
const flatProductMsl = metal:
  proc flatProductKernel(C: ptr UncheckedArray[float32]; M: int32) {.global.} =
    let blk = make_layout((int(M), 4), (1, 4))
    let p = flat_product(blk, make_layout((2, 2)))
    C[0] = float32 toIntVal size(p)

echo "concatDirectKernel: ", concatDirectMsl.len
echo "tiledProductKernel: ", tiledProductMsl.len
echo "flatProductKernel: ", flatProductMsl.len


# ── standard codegen-size report ──

cgsReport([
  ("concatDirectKernel", concatDirectMsl),
  ("tiledProductKernel", tiledProductMsl),
  ("flatProductKernel", flatProductMsl)])
