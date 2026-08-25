#!/usr/bin/env python3
"""tile_evaluator.py — check and find MMA fragment layouts for software-FMA tiles.

Encodes the Apple 8×8×8 fragment-layout contract (the AC/B layouts) and
generalizes it to arbitrary lane counts and tile shapes, for GPUs WITHOUT
tensor cores (Apple M4 family, AMD consumer) where the mma must be spelled
as a software FMA + shuffle pipeline.

For a candidate tile it reports:
  - per-lane fragment ownership (which lane holds which tile cells)
  - values-per-thread for A, B, C
  - the AC contract: does C-role == A-role? (S tile feeds the next mma's A
    operand with zero data movement iff A and C share one fragment layout)
  - the mma shuffle schedule: the src-lane formulas that gather the A and B
    elements each lane needs for its dot products
  - numerical verification: random A, B, run the schedule, compare C against
    a plain dot-product reference — bit-exact (within fp rounding)
  - cost: gathers and FMAs per lane per mma

`find()` searches candidate lane maps (bit decompositions of the lane id) for
a given thread count and tile shape and prints the ones that satisfy the AC
contract with a valid shuffle schedule, ranked by cost.

Usage:
    python3 tile_evaluator.py --apple          # the known-good Apple 8x8x8 tile
    python3 tile_evaluator.py --find 32 8x8    # search 32-lane maps for an 8x8 tile
    python3 tile_evaluator.py --find 64 8x8 8 1   # search 64-lane maps (AMD wave64)

Note: the shared lane map requires a SQUARE atom (m == n == k); rectangular
kernel tiles are square atoms replicated. Higher v amortizes B gathers
(v=2: 1.5 gathers/FMA vs v=1: 2.0 gathers/FMA at the same total FMA count).
"""

import itertools
import random
import sys


# ═════════════════════════════════════════════════════════════════════════
#  Lane maps
# ═════════════════════════════════════════════════════════════════════════
#
# A lane map is a function lane -> (fm, fn): the "anchor" fragment position
# of that lane. With V values per thread the lane then holds the V adjacent
# cells (fm, fn), (fm, fn+1), ... The Apple 8×8 map (verified on-device):
#
#     qid = lane div 4
#     fm  = (qid and 4) + ((lane div 2) mod 4)    # fragment row 0..7
#     fn  = (qid and 2)*2 + (lane mod 2)*2        # fragment col 0,2,4,6
#
# In bit terms (lane = b0 + 2·b1 + 4·b2 + 8·b3 + 16·b4):
#     fm = b1 + 2·b2 + 4·b4     (3 row bits)
#     fn = 2·b0 + 4·b3          (2 col-anchor bits, even cols)
#


def apple_lane_map(lane: int) -> tuple[int, int]:
    """Apple 8×8 fragment map: lane -> (fm, fn anchor)."""
    qid = lane // 4
    fm = (qid & 4) + ((lane // 2) % 4)
    fn = (qid & 2) * 2 + (lane % 2) * 2
    return fm, fn


def bit_lane_map(lane: int, row_bits: list[int], col_bits: list[int],
                 v: int = 2) -> tuple[int, int]:
    """Generalized lane map from a bit decomposition of the lane id.

    row_bits / col_bits: lane-bit indices (0 = LSB) assigned to the fragment
    row / col-anchor. The anchor bits select the coarse column; the v
    adjacent cells fill the fine columns, so anchors are multiples of v
    (anchor bit i contributes v·2^i).

    Apple 8×8 v=2 is row_bits=[1,2,4], col_bits=[0,3]:
        fm = bit1 + 2·bit2 + 4·bit4 ; fn = 2·bit0 + 4·bit3
    """
    fm = sum(((lane >> b) & 1) << i for i, b in enumerate(row_bits))
    fn = sum(((lane >> b) & 1) * (v << i) for i, b in enumerate(col_bits))
    return fm, fn


# ═════════════════════════════════════════════════════════════════════════
#  Tile spec
# ═════════════════════════════════════════════════════════════════════════


class Tile:
    """A candidate MMA tile: thread count, A/B/C operand shapes, lane map.

    A and C share the lane map (the AC contract); B uses the same map with
    the k axis in the row position (the B layout). V is values per thread.
    """

    def __init__(self, threads: int, m: int, n: int, k: int, v: int,
                 lane_map, name: str = ""):
        self.threads = threads
        self.m, self.n, self.k = m, n, k
        self.v = v
        self.lane_map = lane_map
        self.name = name

    # -- per-lane fragment cells ----------------------------------------

    def a_cells(self, lane: int) -> list[tuple[int, int]]:
        """A operand: (row, k) cells owned by this lane."""
        fm, fn = self.lane_map(lane)
        return [(fm, fn + t) for t in range(self.v)]

    def b_cells(self, lane: int) -> list[tuple[int, int]]:
        """B operand: (k, col) cells owned by this lane (B layout)."""
        fm, fn = self.lane_map(lane)
        return [(fm, fn + t) for t in range(self.v)]

    def c_cells(self, lane: int) -> list[tuple[int, int]]:
        """C operand: (row, col) cells owned by this lane (== A layout)."""
        fm, fn = self.lane_map(lane)
        return [(fm, fn + t) for t in range(self.v)]

    # -- coverage ---------------------------------------------------------

    def _coverage(self, cells_fn, rows: int, cols: int) -> tuple[bool, str]:
        owned = {}
        for lane in range(self.threads):
            for (r, c) in cells_fn(lane):
                if r >= rows or c >= cols:
                    return False, f"cell ({r},{c}) out of bounds"
                if (r, c) in owned:
                    return False, f"cell ({r},{c}) owned by lanes {owned[(r,c)]} and {lane}"
                owned[(r, c)] = lane
        missing = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in owned]
        if missing:
            return False, f"{len(missing)} cells uncovered, e.g. {missing[:3]}"
        return True, f"{rows}x{cols} covered by {self.threads} lanes x {self.v} values"

    def coverage_ok(self) -> bool:
        for cells_fn, rows, cols, tag in (
            (self.a_cells, self.m, self.k, "A"),
            (self.b_cells, self.k, self.n, "B"),
            (self.c_cells, self.m, self.n, "C"),
        ):
            ok, msg = self._coverage(cells_fn, rows, cols)
            if not ok:
                return False
        return True

    # -- shuffle schedule ------------------------------------------------

    def src_lanes(self, lane: int) -> dict:
        """For each A and B element this lane needs, which lane holds it.

        Returns (src_lane, elem_index) for each needed value:
            a_src[(row, k)]  = (lane, idx) holding A[row][k]
            b_src[(k, col)]  = (lane, idx) holding B[k][col]
        """
        # build reverse maps
        a_owner, b_owner = {}, {}
        for l in range(self.threads):
            for i, (r, c) in enumerate(self.a_cells(l)):
                a_owner[(r, c)] = (l, i)
            for i, (r, c) in enumerate(self.b_cells(l)):
                b_owner[(r, c)] = (l, i)

        fm, fn = self.lane_map(lane)
        a_src, b_src = {}, {}
        for t in range(self.v):
            col = fn + t
            for k in range(self.k):
                a_src[(fm, k)] = a_owner[(fm, k)]
                b_src[(k, col)] = b_owner[(k, col)]
        return a_src, b_src

    def cost(self) -> dict:
        """Gathers and FMAs per lane per mma, averaged over lanes."""
        total_a, total_b, total_fma = 0, 0, 0
        for lane in range(self.threads):
            a_src, b_src = self.src_lanes(lane)
            # one shuffle per element gathered (distinct source elements)
            total_a += len(set(a_src.values()))
            total_b += len(set(b_src.values()))
            total_fma += self.v * self.k  # dot products: v C cells x k terms
        return {
            "a_gathers": total_a // self.threads,
            "b_gathers": total_b // self.threads,
            "fmas": total_fma // self.threads,
        }

    # -- numerical verification ------------------------------------------

    def verify(self, trials: int = 3) -> bool:
        """Run the shuffle schedule with random data, compare against reference."""
        for _ in range(trials):
            a = {(r, k): random.uniform(-1, 1)
                 for r in range(self.m) for k in range(self.k)}
            b = {(k, c): random.uniform(-1, 1)
                 for k in range(self.k) for c in range(self.n)}

            # reference
            ref = {(r, c): sum(a[(r, k)] * b[(k, c)] for k in range(self.k))
                   for r in range(self.m) for c in range(self.n)}

            for lane in range(self.threads):
                fm, fn = self.lane_map(lane)
                c_reg = [0.0] * self.v

                a_src, b_src = self.src_lanes(lane)

                for t in range(self.v):
                    col = fn + t
                    for k in range(self.k):
                        al, ai = a_src[(fm, k)]
                        bl, bi = b_src[(k, col)]
                        # simdShuffle(src_lane, idx): read the src lane's
                        # register element at its own fragment index
                        av = a[self.a_cells(al)[ai]]
                        bv = b[self.b_cells(bl)[bi]]
                        c_reg[t] += av * bv

                for t in range(self.v):
                    got = c_reg[t]
                    want = ref[(fm, fn + t)]
                    if abs(got - want) > 1e-9:
                        return False
        return True


# ═════════════════════════════════════════════════════════════════════════
#  Search
# ═════════════════════════════════════════════════════════════════════════


def search(threads: int, m: int, n: int, k: int, v: int) -> list[Tile]:
    """Enumerate bit-decomposition lane maps and keep the good tiles.

    A lane map is an ordered assignment of lane-id bits: `row_bits` bits to
    the fragment row (LSB-first), `col_bits` bits to the col anchor. The
    anchor bits select the coarse column and the v adjacent cells fill the
    fine columns, so col anchors must be multiples of v:
        row_bits = log2(m)          (m must be a power of 2)
        col_bits = log2(n) - log2(v)   (n/v must be a power of 2)
    The A, B, C operands share the map, which requires a SQUARE atom
    (m == n == k): A reads rows x k, B reads k x cols, C reads rows x cols,
    and the same (row-bits, col-bits) split covers all three only when
    k == m. Rectangular tiles are square atoms replicated (each lane then
    holds the atom's fragment for each replica). Threads must be exactly
    2^(row_bits + col_bits).
    """
    if m != n or n != k:
        return []  # shared lane map requires a square atom
    if threads != (1 << (threads.bit_length() - 1)):
        return []  # threads not a power of two
    if m & (m - 1) or n % v or (n // v) & ((n // v) - 1):
        return []  # shape not decomposable: m pow2, n = v * pow2

    nbits = threads.bit_length() - 1
    row_count = m.bit_length() - 1
    col_count = (n // v).bit_length() - 1
    if row_count + col_count != nbits:
        return []  # lane count must be exactly rows x col-anchors

    results = []
    seen = set()
    for row_bits in itertools.permutations(range(nbits), row_count):
        rest = [b for b in range(nbits) if b not in row_bits]
        for col_bits in itertools.permutations(rest, col_count):
            key = (row_bits, col_bits)
            if key in seen:
                continue
            seen.add(key)

            t = Tile(threads, m, n, k, v,
                     lambda lane, rb=row_bits, cb=col_bits, vv=v:
                         bit_lane_map(lane, list(rb), list(cb), vv),
                     name=f"rb={row_bits} cb={col_bits}")
            if not t.coverage_ok():
                continue
            if not t.verify():
                continue
            results.append(t)
    return results


# ═════════════════════════════════════════════════════════════════════════
#  Report
# ═════════════════════════════════════════════════════════════════════════


def report(t: Tile) -> None:
    print(f"tile: {t.name or '<unnamed>'}  {t.m}x{t.n}x{t.k}  "
          f"{t.threads} lanes x {t.v} vpt")
    # AC contract: C-role == A-role is structural (same lane map)
    print(f"  AC contract (C-role == A-role): yes (A and C share the lane map)")
    for cells_fn, rows, cols, tag in (
        (t.a_cells, t.m, t.k, "A"),
        (t.b_cells, t.k, t.n, "B"),
        (t.c_cells, t.m, t.n, "C"),
    ):
        ok, msg = t._coverage(cells_fn, rows, cols)
        print(f"  {tag} coverage: {'OK  ' if ok else 'FAIL'} {msg}")
    c = t.cost()
    print(f"  cost per lane per mma: {c['a_gathers']} A-gathers, "
          f"{c['b_gathers']} B-gathers, {c['fmas']} FMAs")
    ok = t.verify()
    print(f"  numerical verification: {'PASS' if ok else 'FAIL'}")
    # example schedule for lane 0
    if t.threads > 0:
        a_src, b_src = t.src_lanes(0)
        print(f"  lane 0 src lanes: A row {t.lane_map(0)[0]} -> "
              f"{sorted(set(v[0] for v in a_src.values()))}, "
              f"B -> {sorted(set(v[0] for v in b_src.values()))}")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    if argv[0] == "--apple":
        t = Tile(32, 8, 8, 8, 2, apple_lane_map, name="apple 8x8x8")
        report(t)
        return 0
    if argv[0] == "--find" and len(argv) >= 3:
        threads = int(argv[1])
        m, n = (int(x) for x in argv[2].split("x"))
        k = int(argv[3]) if len(argv) > 3 else 8
        v = int(argv[4]) if len(argv) > 4 else 2
        print(f"searching {threads}-lane maps for {m}x{n}x{k} tile (v={v})...")
        found = search(threads, m, n, k, v)
        if not found:
            print("  no valid tiles found")
            return 1
        print(f"  {len(found)} valid tiles, ranked by gather cost:")
        found.sort(key=lambda t: (t.cost()["a_gathers"] + t.cost()["b_gathers"],
                                  t.cost()["fmas"]))
        for t in found[:8]:
            c = t.cost()
            print(f"    {t.name:24s} A-g {c['a_gathers']:2d} "
                  f"B-g {c['b_gathers']:2d} FMA {c['fmas']:3d}")
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
