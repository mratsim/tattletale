## Phase 1 host test — partition math (thrfrg equivalent) + golden checks.
##
## Validates partition_A/B/C against:
##   1. Known CUTLASS fragment numbers (12-cutlass-layers.md §3):
##      single atom 4/2/4 regs per thread; 2×2 tiled → 64/64/64 for a
##      (128,32)×(128,32)→(128,64) problem.
##   2. tensor-layouts golden values: the tf32 A layout offset map
##      ((4,8),(2,2)):((16,1),(8,64)) gives thread t values at
##      (t%4)*16 + (t//4) + {0, 8, 64, 72}.
##   3. Algebraic disjointness + coverage (verifyFragments).
## Run with: nim cpp -r workspace/ceramic/experiments/wip_mma_gemm/test_partition_host.nim

import std/[strformat, sequtils]
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_constructors
import workspace/ceramic/src/layout_indexing
import workspace/ceramic/src/atoms
import workspace/ceramic/src/atoms_nvidia
import workspace/ceramic/src/atoms_mma_partitioning

const atom = SM80_16x8x8_F32TF32TF32F32_TN

# ── Single atom: 1×1×1 thread tiling, 32 threads ──
const single = TiledMma[typeof(atom), typeof(make_layout((1, 1, 1)))](
  atom: atom, threadLayout: make_layout((1, 1, 1)))

# fragment counts per thread: A 4, B 2, C 4 (tile 16×8, 16×8, 16×8)
doAssert single.partitionA((16, 8), 0).len == 4
doAssert single.partitionB((8, 8), 0).len == 2
doAssert single.partitionC((16, 8), 0).len == 4

# golden: thread 0 A offsets = {0, 8, 64, 72}; thread 5 = 17 + {0,8,64,72}
let a0 = single.partitionA((16, 8), 0)
doAssert a0[0].offset == 0 and a0[1].offset == 8 and a0[2].offset == 64 and a0[3].offset == 72,
  &"A t0 offsets: {a0.mapIt(it.offset)}"
let a5 = single.partitionA((16, 8), 5)
doAssert a5[0].offset == 17 and a5[1].offset == 25 and a5[2].offset == 81 and a5[3].offset == 89,
  &"A t5 offsets: {a5.mapIt(it.offset)}"
# golden: thread 0 C offsets = {0, 16, 8, 24} (SM80_16x8_Row)
let c0 = single.partitionC((16, 8), 0)
doAssert c0[0].offset == 0 and c0[1].offset == 16 and c0[2].offset == 8 and c0[3].offset == 24,
  &"C t0 offsets: {c0.mapIt(it.offset)}"
# golden: thread 0 B offsets = {0, 32}
let b0 = single.partitionB((8, 8), 0)
doAssert b0[0].offset == 0 and b0[1].offset == 32, &"B t0 offsets: {b0.mapIt(it.offset)}"

# disjointness + coverage for the single atom
single.verifyFragments((16, 8), opA)
single.verifyFragments((8, 8), opB)
single.verifyFragments((16, 8), opC)
echo "  OK — single atom partition + golden offsets"

# ── 2×2 tiled: (128,32)×(128,32) → (128,64), 128 threads ──
#   12-cutlass-layers.md §3c: tiled 2×2 over (128,32) → 64/64/64 per thread
#   C tile (128, 64) = 8192 / 128 threads = 64 per thread ✓
const tiled2 = TiledMma[typeof(atom), typeof(make_layout((2, 2, 1)))](
  atom: atom, threadLayout: make_layout((2, 2, 1)))

let tA = tiled2.partitionA((32, 8), 0)   # A tile (2*16, 1*8) = (32, 8)
let tB = tiled2.partitionB((16, 8), 0)   # B tile (2*8, 1*8) = (16, 8)
let tC = tiled2.partitionC((32, 16), 0)  # C tile (2*16, 2*8) = (32, 16)
doAssert tA.len == 4, &"tiled A: {tA.len}"
doAssert tB.len == 2, &"tiled B: {tB.len}"
doAssert tC.len == 4, &"tiled C: {tC.len}"

# per-thread counts across the whole 128-thread warp-group
var totalA = 0
var totalC = 0
for t in 0 ..< 128:
  totalA += tiled2.partitionA((32, 8), t).len
  totalC += tiled2.partitionC((32, 16), t).len
# A is shared across ThrN=2 N-atoms: each element held by 2 threads
doAssert totalA == 2 * 32 * 8, &"A total {totalA} (2 copies of 32×8)"
doAssert totalC == 32 * 16, &"C total {totalC} (1 copy of 32×16)"

tiled2.verifyFragments((32, 8), opA)
tiled2.verifyFragments((16, 8), opB)
tiled2.verifyFragments((32, 16), opC)
echo "  OK — 2×2 tiled partition (128 threads) + coverage"

# ── 4×4 tiled over a (128,32)×(128,32) → (128,64) problem ──
#   12-cutlass-layers.md §3c verified numbers: 64/64/64 per thread, 2048 FMA
const tiled4 = TiledMma[typeof(atom), typeof(make_layout((4, 4, 1)))](
  atom: atom, threadLayout: make_layout((4, 4, 1)))
let bigA = tiled4.partitionA((64, 8), 3)     # (4*16, 8)
let bigB = tiled4.partitionB((32, 8), 3)     # (4*8, 8)
let bigC = tiled4.partitionC((64, 32), 3)    # (4*16, 4*8)
doAssert bigA.len == 4 and bigB.len == 2 and bigC.len == 4
var perThread = 0
for t in 0 ..< 512:
  perThread += tiled4.partitionC((64, 32), t).len
doAssert perThread == 64 * 32, "4×4 C tile covered by 512 threads"
tiled4.verifyFragments((64, 8), opA)
tiled4.verifyFragments((32, 8), opB)
tiled4.verifyFragments((64, 32), opC)
echo "  OK — 4×4 tiled partition (512 threads, 64/64/64 pattern)"
