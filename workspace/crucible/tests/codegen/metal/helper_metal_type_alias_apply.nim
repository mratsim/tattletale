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

# ── Local fixture: the tile/atom surface, crucible types only ──────────────
# The ceramic tile layer's atom records, fragments, and tile types define
# the shapes the alias canonicalization pins. These stand-ins reproduce
# them with crucible's SimdgroupMatrix, in this module so the kernel
# module and the apply module share one tile type: the cross-module
# generic call resolves only over a single RtLeft.

type
  MmaAtomKind* = enum
    bkGPU_TensorCore
  AppleLayout* = object
    ## Placeholder layout type of the atom's compile-time layout params.
  MmaAtom*[LA, LB, LC] = object
    ## Compile-time MMA atom record: the tile's fourth generic arg,
    ## carrying the subtile dims for the frags grid.
    name*: string
    mnk*: tuple[m, n, k: int]
    kind*: MmaAtomKind
  FragmentOf*[A: static MmaAtom; T] = object
    ## Per-lane register fragment of one atom subtile.
    frag*: SimdgroupMatrix[T, false]
  RtLeft*[T; R, C: static int; A: static MmaAtom] = object
    ## R-outer register tile: the subtile grid of per-lane fragments.
    frags*: array[R div A.mnk.m, array[C div A.mnk.n, FragmentOf[A, T]]]

const APPLE_8x8x8_F32* = MmaAtom[AppleLayout, AppleLayout, AppleLayout](
  name: "APPLE_8x8x8_F32", mnk: (m: 8, n: 8, k: 8), kind: bkGPU_TensorCore)
  ## The 8×8×8 simdgroup atom: one per-lane simdgroup matrix per subtile.

template TileConfigFor*(T: typedesc): untyped =
  ## Maps the tile element type to its MMA atom. fp32 selects the Apple atom.
  when T is float32:
    APPLE_8x8x8_F32
  else:
    {.error: "TileConfigFor: no atom for element type " & $T & ".".}

template rt_l*(T: typedesc; R, C: static int; A: untyped = TileConfigFor(T)): untyped =
  ## R-outer tile type constructor with the defaulted atom.
  RtLeft[T, R, C, A]

type Int*[V: static int] = object
  ## Compile-time integer type: the `Int[4]` instantiation emits `Int4`.

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
