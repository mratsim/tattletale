## Helper for test_metal_type_alias_canonicalization: the apply-style
## device proc and the alias spelling of the tile type, defined
## in a separate module from the kernel that declares the tile var.
##
## The alias spells the atom explicitly as the canonical type application:
## an alias whose RHS is a template call (the defaulted-atom `rt_l` form)
## cannot be expanded at resolve time, so an alias spelling the tile type
## must use the canonical `RtLeft`/`RtRight` application.
##
## Not a test file: the name has no `test_`/`t_` prefix, so the harness
## does not pick it up.

import workspace/crucible
import workspace/ceramic/src/int_tuples
import workspace/ceramic/src/layouts
import workspace/ceramic/src/layout_constructors
import workspace/ceramic/src/layout_indexing
import workspace/ceramic/src/tile_algebra/tile_config
import workspace/ceramic/src/tile_algebra/tiles
import workspace/ceramic/src/kernel_gemm/atoms_apple
import workspace/ceramic/src/layouts_datatypes
import workspace/ceramic/src/atoms

type RTileF32*[R, C: static int] = RtLeft[float32, R, C, APPLE_8x8x8_F32]
  ## The fp32 R-outer tile under its alias spelling: the form an epilogue
  ## user writes for a fixed-element tile. The atom is explicit, keeping
  ## the alias RHS a type application the resolver can canonicalize.

type RTileF32DefaultedAtom*[R, C: static int] = RtLeft[float32, R, C, TileConfigFor(float32)]
  ## The same tile with the atom selected by the per-element mapping inside
  ## the canonical bracket: the resolver must fold the defaulted atom
  ## (`TileConfigFor` expands to the const sym in the typed AST) and emit
  ## the identical struct name.

proc applyTile*[T; R, C: static int; A: static MmaAtom](
    d: var RtLeft[T, R, C, A]) {.device.} =
  ## The generic apply-style device proc over the canonical tile type:
  ## writes lane 0's first fragment element. The test kernel calls it
  ## with a var declared through the alias. The emitted MSL parameter
  ## type must be the canonical tile struct, matching the declaration.
  threadElements(d.frags[0][0].frag, 0'u32) = T(7)
