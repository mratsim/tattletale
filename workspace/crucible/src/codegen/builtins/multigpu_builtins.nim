## Cross-platform GPU builtins available on CUDA, OpenCL, GLSL, and WGSL.
## These are defined with {.builtin.} so the codegen registers them without
## parsing system module bodies. {.noRewrite.} prevents template expansion
## from circumventing the {.builtin.} pragma.
import ./builtins_pragmas

template genMinMaxAbs(T) =
  template min*(x, y: T): T {.builtin.} =
    # This builtin name collides between Nim stdlib and the GPU library.
    # The flow:
    # - Nim resolves concrete types before generics, so we generate a concrete function
    # - The GPU compiler sees `builtin` and ignore the body.
    # - The Nim compiler sees builtin, skips it as it has no special meaning
    # - Then in the body it sees that there is a call to `min` but given that it is gated by {.norewrite.}
    #   it doesn't try to replace min by its own body
    {.noRewrite.}:
      system.min(x, y)

  template max*(x, y: T): T {.builtin.} =
    # This builtin name collides between Nim stdlib and the GPU library.
    # The flow:
    # - Nim resolves concrete types before generics, so we generate a concrete function
    # - The GPU compiler sees `builtin` and ignore the body.
    # - The Nim compiler sees builtin, skips it as it has no special meaning
    # - Then in the body it sees that there is a call to `min` but given that it is gated by {.norewrite.}
    #   it doesn't try to replace min by its own body
    {.noRewrite.}:
      system.max(x, y)

  template abs*(x, y: T): T {.builtin.} =
    # This builtin name collides between Nim stdlib and the GPU library.
    # The flow:
    # - Nim resolves concrete types before generics, so we generate a concrete function
    # - The GPU compiler sees `builtin` and ignore the body.
    # - The Nim compiler sees builtin, skips it as it has no special meaning
    # - Then in the body it sees that there is a call to `min` but given that it is gated by {.norewrite.}
    #   it doesn't try to replace min by its own body
    {.noRewrite.}:
      system.abs(x, y)

genMinMaxAbs(int32)
genMinMaxAbs(float32)