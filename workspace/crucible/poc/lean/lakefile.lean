import Lake
open Lake DSL

package «leanpoc» where
  -- no dependencies: core Lean only

@[default_target]
lean_lib «LeanPoc» where
  -- explicit roots, not a glob: Negative.lean stays out of `lake build`
  roots := #[`LeanPoc, `AddKernel]
