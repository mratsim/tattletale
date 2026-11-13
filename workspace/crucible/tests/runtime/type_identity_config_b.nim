# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## One of two `Config` definitions under one identifier, one per module, with different shapes.
## t_launch_device_fn_type_identity expands both through the bridge-name hash.
## Paired with type_identity_config_a.nim.

type Config* = object
  x: float32
  y: float32
