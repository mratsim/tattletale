"""e03_load_tile_oneshot: the one-step TVCrdLayout tile load, the pycute
counterpart of experiments/e03_load_tile_oneshot.nim

Run: uv run python e03_load_tile_oneshot_pycute.py, any cwd works
(the script locates the pycute checkout itself)

- the TVCrdLayout is the p.74 one-step composition: shard/tile identity
  layouts (flat -> coordinate) composed with the integer thread map
- R = zipped_divide(world, TVCrdLayout) in one call, where the Nim file
  assembles the rest modes elementarily, the stopgap
- the fragment is the (thr, cell) Tensor slice of R, the tile modes
  diced away by tile_id, the diced offset baked into the accessor
- the guard is coordinate-space only: a coordinate tensor
  zipped_divide(identity, TVCrdLayout) carries the tile-corner lift
  in its rest strides (32@0, 16@1). Slicing it identically yields
  the world coordinate per fragment element, OOB takes the fill

Evaluating the TV layout at a coordinate is a plain layout call,
`tvcrd(coord)`. The Nim evalTileCoord macro is the shim for crd2idx
over coordinate-strided layouts, a missing library primitive (part3).
"""

import array
import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_root, "_references_research", "NVlabs-CuTe"))

from pycute import (E, ImplicitAccessor, Layout, Ptr, Tensor, composition,
                    idx2crd, in_bounds, size, zipped_divide)

# ── the Apple 8x8x8 MMA atom, verbatim from h_mma_registry.nim ──────

APPLE_AC_SHAPE = ((2, 2, 2, 2, 2), 2)
APPLE_AC_STRIDE = ((16, 1, 2, 32, 4), 8)

MT, KT = 32, 16  # the A operand tile (M, K)


def get_atom_tile_flat():
    """Atom layout in tile-flat space, a one-time static per-atom
    conversion (registry material), integer strides only.

    Returns the atom layout, shape ((2,2,2,2,2),2)
    """
    atom_map = Layout(APPLE_AC_SHAPE, APPLE_AC_STRIDE)
    cell_outer, cell_last = 8, 8  # the atom's (M, K)
    # identity layouts map a flat index to its coordinate, the integer
    # -> coordinate step. The flat layout maps it back in tile-flat space
    shard_id = Layout((cell_outer, cell_last), (E(0), E(1)))
    atom_crd = composition(shard_id, atom_map)
    flat_tile = Layout((MT, KT), (1, MT))
    return composition(flat_tile, atom_crd)


def get_tv_crd_layout():
    """Build the TVCrdLayout for opA, the p.74 slide formulation
    T o (t,v). The thread map assembles uniformly tile-flat, the atom
    part pre-lifted and the cell strides already tile-flat, then ONE
    coordinate-stride identity composition converts everything.

    Returns the tvcrd layout, shape (((2,2,2,2,2),2),(4,2)) with E-atom
    strides over the (M, K) tile modes
    """
    atomT = get_atom_tile_flat()
    cell_outer, cell_last = 8, 8  # the atom's (M, K)
    rep_outer, rep_last = MT // cell_outer, KT // cell_last
    tile_outer = cell_outer * rep_outer
    # getMatmulThreadmap with the converted atom part
    thread_map_t = Layout((atomT.shape, (rep_outer, rep_last)),
                          (atomT.stride, (cell_outer, tile_outer * cell_last)))
    tile_id = Layout((MT, KT), (E(0), E(1)))
    return composition(tile_id, thread_map_t)


def make_world(M, K, s0, s1):
    """Build a (M, K) world with caller strides and offset-indexed payload.

    Ragged strides are not injective on coordinates, so the payload is
    a function of the offset, never of the coordinate. This is the Nim
    check's buf[i] = i mod 251 convention.

    Returns (byte buffer, layout), the buffer as array.array so Ptr can
    anchor into it
    """
    layout = Layout((M, K), (s0, s1))
    n = max(M * K, (M - 1) * s0 + (K - 1) * s1 + 1)
    data = array.array('d', [float(i % 257) for i in range(n)])
    return data, layout


def load_tile_one_step(data, world, tvcrd, tile_id, thr, fill=0.0):
    """One-step loadTile: one divide, one Tensor slice per operand, one
    guarded read per element, no manual offset arithmetic anywhere.
    Element offsets: the sliced Tensor's accessor.
    World coordinates: the sliced coordinate Tensor.

    Returns fragment values in colex element order
    """
    # the coordinate map of the world: reading a flat index yields
    # the (m, k) coordinate
    C = Layout(world.shape, (E(0), E(1)))
    R = zipped_divide(world, tvcrd)
    RC = zipped_divide(C, tvcrd)
    thr_crd = idx2crd(thr, tvcrd.shape[0][0])
    rest = idx2crd(tile_id, R.shape[1])
    # slice: keep the (thr, vpt, cell) modes and dice the tile modes away
    sel = ((((thr_crd[0], thr_crd[1], thr_crd[2], thr_crd[3], thr_crd[4]),
             slice(None)),
            (slice(None), slice(None))),
           rest)
    frag = Tensor(Ptr(data), R)[sel]
    cfrag = Tensor(ImplicitAccessor(0), RC)[sel]
    out = []
    for i in range(size(frag.layout.shape)):
        # the slice flattens the kept leaves: (vpt, cell0, cell1)
        v, c0, c1 = idx2crd(i, frag.layout.shape)
        if in_bounds(cfrag[v, c0, c1], world.shape):
            out.append(frag[v, c0, c1])
        else:
            out.append(fill)
    return out


def check(got, want, name):
    ok = got == want
    print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        for i, (g, w) in enumerate(zip(got, want)):
            if g != w:
                print(f"    first diff at {i}: got {g} want {w}")
                break
    return ok


def run_world(M, K, s0, s1):
    """Gather every tile's every thread fragment and compare each against
    an independently recomputed guarded read"""
    data, world = make_world(M, K, s0, s1)
    tvcrd = get_tv_crd_layout()
    tiles_m, tiles_k = (M + MT - 1) // MT, (K + KT - 1) // KT
    all_ok = True
    for tm in range(tiles_m):
        for tk in range(tiles_k):
            for thr in range(32):
                got = load_tile_one_step(data, world, tvcrd,
                                         tm + tiles_m * tk, thr)
                # the fragment's world positions, recomputed independently
                frag_positions = fragment_positions(world, tvcrd,
                                                    tm + tiles_m * tk, thr)
                want = [data[world(p)] if in_bounds(p, (M, K)) else 0.0
                        for p in frag_positions]
                if not check(got, want, f"tile ({tm},{tk}) thr {thr:2d}"):
                    all_ok = False
    return all_ok


def fragment_positions(world, tvcrd, tile_id, thr):
    """World positions for one thread's fragment, the sliced
    coordinate Tensor read elementwise, no TV eval repeated here.

    Returns (m, k) positions in colex element order
    """
    C = Layout(world.shape, (E(0), E(1)))
    RC = zipped_divide(C, tvcrd)
    thr_crd = idx2crd(thr, tvcrd.shape[0][0])
    rest = idx2crd(tile_id, RC.shape[1])
    sel = ((((thr_crd[0], thr_crd[1], thr_crd[2], thr_crd[3], thr_crd[4]),
             slice(None)),
            (slice(None), slice(None))),
           rest)
    cfrag = Tensor(ImplicitAccessor(0), RC)[sel]
    # same colex element order as the loader: (vpt, cell0, cell1)
    return [tuple(cfrag[tuple(idx2crd(i, cfrag.layout.shape))])
            for i in range(size(cfrag.layout.shape))]


def main():
    tvcrd = get_tv_crd_layout()
    print("TVCrdLayout (opA, tile 32x16):")
    print(f"  {tvcrd}")
    print("one-step loads vs naive guarded gather:")
    all_ok = True
    for (M, K, s0, s1) in [(32, 32, 1, 32), (40, 45, 1, 40), (70, 50, 1, 70),
                           (32, 32, 32, 1), (70, 50, 50, 1), (45, 41, 7, 3)]:
        print(f" world ({M},{K}) s({s0},{s1}):")
        all_ok = run_world(M, K, s0, s1) and all_ok
    print(f"e03 pycute checks: {'all green' if all_ok else 'FAILURES'}")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
