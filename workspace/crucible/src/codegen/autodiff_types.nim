# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#            Backward-differentiation metadata types
#
# ############################################################

type
  BwdParamKind* = enum
    pkBuffer    ## Read-only input buffer, gradient is a same-shape buffer
    pkScalar    ## Scalar parameter (integer or float), integer scalars carry no gradient
    pkResidual  ## Forward-output buffer the forward kernel writes, available to the backward kernel at its block-2 position

  BwdGradKind* = enum
    gkNone    ## No gradient slot in the backward signature
    gkBuffer  ## Gradient buffer, same shape as the primal argument
    gkBuffer1 ## 1-element gradient device buffer, the scalar gradient written by the generated reduce kernel

  BwdExtentSource* = enum
    xeWrapperField ## Buffer length read off the caller's host wrapper, `DeviceMem` has no length field
    xeStaticShape  ## Pointer to a static array, the length is compile-time
    xeLenParam     ## Buffer length read off a separate integer kernel parameter

  BwdParamMeta* = object
    ## Signature record of one forward parameter.
    name*: string            ## Forward parameter name
    kind*: BwdParamKind
    gradName*: string        ## `d_<name>`, empty when `gradKind == gkNone`
    gradKind*: BwdGradKind
    extent*: BwdExtentSource ## Buffer-length source, buffers only, unused for scalars
    extentParam*: string     ## `xeLenParam`: the integer parameter's name
    staticLen*: int          ## `xeStaticShape`: the compile-time element count

  ScalarWorkspace* = object
    ## Reduce-partials scratch of one scalar gradient: one partial per index
    ## loop iteration and thread, folded by the single-thread reduce kernel
    ## into the scalar's 1-element gradient buffer.
    gradName*: string        ## `d_<name>`, the 1-element gradient buffer
    workspaceName*: string   ## `d_<name>_partials`, the backward kernel's scratch
    tripCount*: int          ## Static index-loop trip count, -1 when runtime-derived
    tripCountParam*: string  ## Name of the kernel parameter carrying the trip count, runtime-derived
    tripCount64*: bool       ## Trip count is a 64-bit value, else 32-bit
    threadCount*: int        ## Thread product of the pair's normalized workgroup, unset axes count 1, the scratch length is tripCount * threadCount

  AdKernelsMetas* = object
    ## Transposer's single metadata derivation of one kernel pair: one signature record per forward parameter, plus the reduce-partials workspaces.
    params*: seq[BwdParamMeta]
    workspaces*: seq[ScalarWorkspace]
