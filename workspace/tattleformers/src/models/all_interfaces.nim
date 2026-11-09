# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Model registry: architecture name to loader.
##
## Contract:
##
## - registration: each model module registers its loader in a static
##   block keyed by the config.json `architectures[0]` name
## - loadModel (models.nim) dispatches through the populated registry
## - loaders return RootRef and loadModel casts to the concrete model

import
  std/tables,
  workspace/tensorcery/src/tensor_datatypes

type ModelLoader* = proc (engine: AnyEngine, modelPath: string): RootRef {.nimcall.}

var ModelRegistry* {.compileTime.}: Table[string, ModelLoader]

proc registerModel*(name: string, loader: ModelLoader) =
  ## Register a model loader under its config.json architecture name.
  ## Called from a model module's static block at import time.
  ModelRegistry[name] = loader
