"""load_tile: predicated GEMM tile loads via in_bounds, the pycute twin
of experiments/e01_load_tile.nim

Run: uv run python workspace/ceramic/experiments/e01_load_tile_pycute.py
(requires the pycute checkout on sys.path, see the NVlabs-CuTe reference)

Same layout, same guards, same edge cases as the ceramic twin:
- A is (M, K) col-major, strides (1, M), the GEMM A convention
- a tile at block (bm, bk) is a (tileM, tileK) view, offset by crd2idx
- the world coordinate of a tile element is tileCrd + blockOrigin
- the load guard is in_bounds(worldCrd, worldShape), OOB takes the fill

Two predication granularities:
- element granularity: per-element guard, nothing is ever read OOB
- chunk granularity: one bool per 16B chunk, an OOB chunk zero-fills
  whole, the cp.async ZFILL contract
"""

import sys

sys.path.insert(0, "_references_research/NVlabs-CuTe")

from pycute import Layout, crd2idx, idx2crd, in_bounds, size

TILE_M, TILE_K = 8, 8
CHUNK = 4  # 4 float32 = 16B, the cp.async chunk


def make_world(M, K):
    """A (M, K) col-major world, element (m, k) = 100*m + k"""
    layout = Layout((M, K), (1, M))
    data = [0.0] * (M * K)
    for m in range(M):
        for k in range(K):
            data[m + k * M] = float(100 * m + k)
    return data, layout


def ref_tile(M, K, bm, bk, tileM=TILE_M, tileK=TILE_K):
    """The expected tile contents, the naive guarded read"""
    out = []
    for j in range(tileK):
        for i in range(tileM):
            m, k = bm * tileM + i, bk * tileK + j
            out.append(float(100 * m + k)
                       if 0 <= m < M and 0 <= k < K else 0.0)
    return out


def ref_tile_chunked(M, K, bm, bk, chunk=CHUNK,
                     tileM=TILE_M, tileK=TILE_K):
    """The expected chunked-load tile: a chunk that straddles the world
    edge zero-fills whole, its in-bounds elements included"""
    out = ref_tile(M, K, bm, bk, tileM, tileK)
    for c in range(len(out) // chunk):
        straddles = any(
            not (0 <= bm * tileM + (c * chunk + e) % tileM < M
                 and 0 <= bk * tileK + (c * chunk + e) // tileM < K)
            for e in range(chunk))
        if straddles:
            for e in range(chunk):
                out[c * chunk + e] = 0.0
    return out


def load_tile_element(data, world, buf, block_origin,
                      tileM=TILE_M, tileK=TILE_K):
    """Tile <- world, one guarded read per element"""
    world_shape = world.shape
    tile = Layout((tileM, tileK), (1, tileM))
    for f in range(size(tile)):
        tile_crd = idx2crd(f, tile.shape)
        world_crd = tuple(c + o for c, o in zip(tile_crd, block_origin))
        off = crd2idx(tile_crd, tile.shape)
        buf[off] = data[crd2idx(world_crd, world.shape)] \
            if in_bounds(world_crd, world_shape) else 0.0


def chunk_predicates(world, block_origin, tileM=TILE_M, tileK=TILE_K,
                     chunk=CHUNK):
    """One bool per chunk along the tile's M rows, true iff every
    element of the chunk is in bounds of the world"""
    world_shape = world.shape
    preds = []
    for c in range(tileM * tileK // chunk):
        ok = True
        for e in range(chunk):
            f = c * chunk + e
            tile_crd = (f % tileM, f // tileM)
            world_crd = tuple(c + o for c, o in zip(tile_crd, block_origin))
            ok = ok and in_bounds(world_crd, world_shape)
        preds.append(ok)
    return preds


def load_tile_chunked(data, world, buf, block_origin, preds,
                      chunk=CHUNK, tileM=TILE_M, tileK=TILE_K):
    """Tile <- world, one guarded read per 16B chunk, ZFILL contract:
    pred false zero-fills the whole chunk, pred true copies it whole"""
    world_shape = world.shape
    tile = Layout((tileM, tileK), (1, tileM))
    for f in range(size(tile)):
        c = f // chunk
        tile_crd = idx2crd(f, tile.shape)
        world_crd = tuple(c + o for c, o in zip(tile_crd, block_origin))
        off = crd2idx(tile_crd, tile.shape)
        if preds[c]:
            buf[off] = data[crd2idx(world_crd, world.shape)]
        else:
            buf[off] = 0.0


def check(got, want, name):
    assert got == want, "%s: element %d got %r want %r" % (
        name, next(i for i, (a, b) in enumerate(zip(got, want)) if a != b),
        got[next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)],
        want[next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)])


def run_element(M, K, bm, bk, name):
    data, world = make_world(M, K)
    buf = [0.0] * (TILE_M * TILE_K)
    load_tile_element(data, world, buf, (bm * TILE_M, bk * TILE_K))
    check(buf, ref_tile(M, K, bm, bk), name)


def run_chunked(M, K, bm, bk, name):
    data, world = make_world(M, K)
    buf = [0.0] * (TILE_M * TILE_K)
    origin = (bm * TILE_M, bk * TILE_K)
    preds = chunk_predicates(world, origin)
    load_tile_chunked(data, world, buf, origin, preds)
    check(buf, ref_tile_chunked(M, K, bm, bk), name)


print("element-granularity loads:")
run_element(32, 32, 1, 1, "interior")
run_element(13, 32, 1, 0, "M edge")
run_element(32, 17, 0, 2, "K edge")
run_element(13, 17, 1, 2, "corner")
run_element(5, 8, 0, 0, "world smaller than tile")
run_element(8, 8, 0, 0, "exact fit")
run_element(8, 8, -1, -1, "negative origin")

print("chunk-granularity loads (copyFromIfAsync / ZFILL contract):")
run_chunked(32, 32, 1, 1, "interior")
run_chunked(13, 32, 1, 0, "M edge")
run_chunked(13, 17, 1, 2, "corner")

# The granularity contrast: the straddling chunk zero-fills whole
# where the elementwise load keeps the in-bounds elements
data, world = make_world(13, 32)
origin = (TILE_M, 0)
preds = chunk_predicates(world, origin)
assert preds[1] is False   # chunk m 4..7, world 12..15 straddles M = 13
assert preds[0] is True    # chunk m 0..3, world 8..11 fully in bounds
assert preds[2] is True    # chunk k = 1, m 0..3, fully in bounds
buf = [0.0] * (TILE_M * TILE_K)
load_tile_chunked(data, world, buf, origin, preds)
assert buf[4] == 0.0       # straddling chunk zero-filled whole,
                           # elementwise load keeps 100*12+0 = 1200 here
assert buf[0] == 800.0     # interior chunk copied

print("GEMM load pattern, all A and B blocks:")

# The full GEMM load legs: A (M, K) and B (N, K) col-major, every block
# loads its tile, every element verified against the naive guarded read
M, N, K = 13, 11, 17
TILE_N = 8
a_data, a_world = make_world(M, K)
b_data = [0.0] * (N * K)
for n in range(N):
    for k in range(K):
        b_data[n + k * N] = float(1000 * n + k)
b_world = Layout((N, K), (1, N))

for bk in range((K + TILE_K - 1) // TILE_K):
    for bm in range((M + TILE_M - 1) // TILE_M):
        buf = [0.0] * (TILE_M * TILE_K)
        load_tile_element(a_data, a_world, buf, (bm * TILE_M, bk * TILE_K))
        check(buf, ref_tile(M, K, bm, bk), "A leg")
    for bn in range((N + TILE_N - 1) // TILE_N):
        buf = [0.0] * (TILE_N * TILE_K)
        load_tile_element(b_data, b_world, buf, (bn * TILE_N, bk * TILE_K),
                          tileM=TILE_N)
        want = [float(1000 * (bn * TILE_N + i) + bk * TILE_K + j)
                if 0 <= bn * TILE_N + i < N and 0 <= bk * TILE_K + j < K
                else 0.0
                for j in range(TILE_K) for i in range(TILE_N)]
        check(buf, want, "B leg")

print("load_tile_pycute: all green")
