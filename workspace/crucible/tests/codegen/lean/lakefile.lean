import Lake
open Lake DSL

package «crucible_lean» where
  -- no dependencies: core Lean only

@[default_target]
lean_lib «ToolchainSmoke» where
  -- explicit roots, not a glob: Negative.lean stays out of `lake build`
  roots := #[`ToolchainSmoke, `AddKernel]
