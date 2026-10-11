"""e04_load_tile_oneshot: the one-step TVCrdLayout tile load, the pycute
counterpart of experiments/e04_load_tile_oneshot.nim

Run: uv run python e04_load_tile_oneshot_pycute.py, any cwd works
(the script locates the pycute checkout itself)

- the TVCrdLayout is the p.74 one-step composition: the atom layout
  pre-lifted to tile-flat, the cell strides from the tile/atom divide,
  then ONE coordinate-stride identity composition (T o (t,v))
- built for every operand: A (M, K), B (N, K) implicit-transpose, C (M, N)
- the fragment is the (thr, cell) Tensor slice of R, the tile modes
  diced away by tile_id, the diced offset baked into the accessor
- the guard is coordinate-space only: a coordinate tensor
  zipped_divide(identity, TVCrdLayout) carries the tile-corner lift
  in its rest strides. Slicing it identically yields the world
  coordinate per fragment element, OOB takes the fill

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
                    get, idx2crd, in_bounds, repeat_like, size, zipped_divide)

# ── the Apple 8x8x8 MMA atom, verbatim from h_mma_registry.nim ──────

APPLE_AC_SHAPE = ((2, 2, 2, 2, 2), 2)
APPLE_AC_STRIDE = ((16, 1, 2, 32, 4), 8)
APPLE_B_SHAPE = ((2, 2, 2, 2, 2), 2)
APPLE_B_STRIDE = ((2, 8, 16, 4, 32), 1)
ATOM_M, ATOM_N, ATOM_K = 8, 8, 8

MT, NT, KT = 32, 32, 16  # the GEMM tile (M, N, K)


def get_atom_tile_flat(atom_shape, atom_stride, shard, tile):
    """Atom layout in tile-flat space, a one-time static per-atom
    conversion (registry material), integer strides only.

    Returns the atom layout with atom_shape's shape
    """
    atom_map = Layout(atom_shape, atom_stride)
    # identity layouts map a flat index to its coordinate, the integer
    # -> coordinate step. The flat layout maps it back in tile-flat space
    shard_id = Layout(shard, (E(0), E(1)))
    atom_crd = composition(shard_id, atom_map)
    flat_tile = Layout(tile, (1, tile[0]))
    return composition(flat_tile, atom_crd)


def get_tv_crd_layout(op):
    """Build the TVCrdLayout for one operand, the p.74 slide
    formulation T o (t,v):
    - A is (M, K), B is (N, K) implicit transpose, C is (M, N)
    - the thread map assembles uniformly tile-flat, the atom part
      pre-lifted and the cell strides drawn from the tile/atom divide
    - ONE coordinate-stride identity composition converts everything

    Returns the tvcrd layout with E-atom strides over the operand's
    two tile modes
    """
    cell_outer = ATOM_N if op == "B" else ATOM_M
    cell_last = ATOM_N if op == "C" else ATOM_K
    shard = (cell_outer, cell_last)
    tile = ((NT if op == "B" else MT), (NT if op == "C" else KT))
    atom_shape, atom_stride = ((APPLE_B_SHAPE, APPLE_B_STRIDE) if op == "B"
                               else (APPLE_AC_SHAPE, APPLE_AC_STRIDE))
    atomT = get_atom_tile_flat(atom_shape, atom_stride, shard, tile)
    # the rep counts and cell strides are the rest modes of the tile
    # divided by the atom, no hand division, already in tile-flat space
    tiled_tile = zipped_divide(Layout(tile), shard)
    cell_map = get(tiled_tile, mode=1)
    # getMatmulThreadmap with the converted atom part
    thread_map_t = Layout((atomT.shape, cell_map.shape),
                          (atomT.stride, cell_map.stride))
    tile_id = Layout(tile, (E(0), E(1)))
    return composition(tile_id, thread_map_t)


def make_world(shape, s0, s1):
    """Build a world with caller strides and offset-indexed payload.

    Ragged strides are not injective on coordinates, so the payload is
    a function of the offset, never of the coordinate. This is the Nim
    check's buf[i] = i mod 251 convention.

    Returns (byte buffer, layout), the buffer as array.array so Ptr can
    anchor into it
    """
    layout = Layout(shape, (s0, s1))
    n = max(shape[0] * shape[1], (shape[0] - 1) * s0 + (shape[1] - 1) * s1 + 1)
    data = array.array('d', [float(i % 257) for i in range(n)])
    return data, layout


def fragment_sel(R, thr, tile_id):
    """Zipped-divide slice selector, dice/keep by mode:
    - atom leaves: diced with the thread coords
    - cells: kept
    - rest modes: diced with the tile coords

    The mode paths are R's own zipped structure, the shape tree
    (((atom, vpt), cells), rest), rank-generic for any atom arity
    and rest rank

    Returns the selector tree
    """
    thr_crd = idx2crd(thr, R.shape[0][0][0])
    cells_keep = repeat_like(slice(None), R.shape[0][1])
    rest_crd = idx2crd(tile_id, R.shape[1])
    return (((thr_crd, slice(None)), cells_keep), rest_crd)


def load_tile_one_step(data, world, tvcrd, tile_shape, tile_id, thr, fill=0.0):
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
    sel = fragment_sel(R, thr, tile_id)
    frag = Tensor(Ptr(data), R)[sel]
    cfrag = Tensor(ImplicitAccessor(0), RC)[sel]
    out = []
    for i in range(size(frag.layout.shape)):
        # the natural coordinate passes straight through, any fragment rank, no leaf unpacking
        crd = idx2crd(i, frag.layout.shape)
        if in_bounds(cfrag[crd], world.shape):
            out.append(frag[crd])
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


def fragment_positions(world, tvcrd, tile_id, thr):
    """World positions for one thread's fragment, the sliced
    coordinate Tensor read elementwise, no TV eval repeated here.

    Returns (m, k) positions in colex element order
    """
    C = Layout(world.shape, (E(0), E(1)))
    RC = zipped_divide(C, tvcrd)
    sel = fragment_sel(RC, thr, tile_id)
    cfrag = Tensor(ImplicitAccessor(0), RC)[sel]
    return [tuple(cfrag[idx2crd(i, cfrag.layout.shape)])
            for i in range(size(cfrag.layout.shape))]


def run_world(shape, s0, s1, tvcrd, tile_shape):
    """Gather every tile's every thread fragment and compare each against
    an independently recomputed guarded read"""
    data, world = make_world(shape, s0, s1)
    tiles = ((shape[0] + tile_shape[0] - 1) // tile_shape[0],
             (shape[1] + tile_shape[1] - 1) // tile_shape[1])
    all_ok = True
    for tm in range(tiles[0]):
        for tk in range(tiles[1]):
            for thr in range(32):
                got = load_tile_one_step(data, world, tvcrd, tile_shape,
                                         tm + tiles[0] * tk, thr)
                # the fragment's world positions, recomputed independently
                frag_positions = fragment_positions(world, tvcrd,
                                                    tm + tiles[0] * tk, thr)
                want = [data[world(p)] if in_bounds(p, shape) else 0.0
                        for p in frag_positions]
                if not check(got, want, f"tile ({tm},{tk}) thr {thr:2d}"):
                    all_ok = False
    return all_ok


def main():
    print("one-step loads vs naive guarded gather, per operand:")
    all_ok = True
    worlds = [(32, 32, 1, 32), (40, 45, 1, 40), (70, 50, 1, 70),
              (32, 32, 32, 1), (70, 50, 50, 1), (45, 41, 7, 3)]
    for op in "ABC":
        tvcrd = get_tv_crd_layout(op)
        tile_shape = ((NT if op == "B" else MT), (NT if op == "C" else KT))
        print(f"op{op} TVCrdLayout: {tvcrd}")
        for (d0, d1, s0, s1) in worlds:
            print(f" world ({d0},{d1}) s({s0},{s1}):")
            all_ok = run_world((d0, d1), s0, s1, tvcrd, tile_shape) and all_ok
    print(f"e03 pycute checks: {'all green' if all_ok else 'FAILURES'}")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
