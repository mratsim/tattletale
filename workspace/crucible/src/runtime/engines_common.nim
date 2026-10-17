# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Dependency-free leaf: the launch modules and engines.nim both import it.

type
  Dim3* = object
    ## A 3D launch dims, `Dim3(x: n)` defaults y/z to 1.
    x*, y*, z* = 1

  LaunchConfig*[E] = object
    ## Engine + launch geometry, the value `<<` hands to `>>`.
    engine*: E
    grid*, blk*: Dim3

template dim3*(a, b, c: SomeInteger = 1): Dim3 =
  Dim3(x: int(a), y: int(b), z: int(c))

type
  HwEngine* = concept engine, type E
    proc ingest(engine: E, source: string)
    proc getArtifact(engine: E)
    proc deviceName(engine: E): string
