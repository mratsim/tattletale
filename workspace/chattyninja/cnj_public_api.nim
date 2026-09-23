# Tattletale Copyright (c) 2026 Mamy Ratsimbazafy Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Chattyninja public API. This file IS the package surface, everything a consumer
## may name, nothing else.
##
## - the render tier, `parseJinjaTemplate`, `startJinjaRender`, `pullInto`,
##   `renderToString`, over the compiled artifact and the caller's `JinjaRenderContext`
## - the data tier, `JinjaVal` with `ValueKind`, the value constructors, `eqVal`,
##   `dictSet`/`dictGet` over `DictVal`, the `tojson` form `toJson` with `JsonOpts`,
##   `JinjaError` with its cause
## - the depth, nesting, step and element caps
##
## Source modules sit under `import x {.all.}`. The surface above is exported by name,
## so the home modules carry no export stars. The internals of every module, including
## `jinja_interpolation` and `jinja_builtins`, stay reachable only through `import x {.all.}`.

import ./src/cnj_engine {.all.}
import ./src/cnj_types {.all.}
import ./src/jinja_data_model {.all.}
import ./src/jinja_serialize {.all.}
import ./src/cnj_parse {.all.}

# Render driver and parser.
export cnj_engine.startJinjaRender, cnj_engine.pullInto,
    cnj_engine.renderToString, cnj_parse.parseJinjaTemplate

# Compiled artifact, render session and the knobs.
export cnj_types.JinjaRenderContext, cnj_types.CompiledTemplate, cnj_types.CompiledSymbols,
    cnj_types.TTT_CNJ_MacroDepthCap, cnj_types.TTT_CNJ_ExprDepthCap,
    cnj_types.TTT_CNJ_ParseNestingCap, cnj_types.TTT_CNJ_StepBudget

# Values and their construction, equality, the dict reads, the error and the JSON form.
export jinja_data_model.JinjaVal, jinja_data_model.ValueKind,
    jinja_data_model.DictVal,
    jinja_data_model.JinjaError, jinja_data_model.JinjaCause, jinja_data_model.JsonOpts,
    jinja_data_model.TTT_CNJ_RangeElemCap,
    jinja_data_model.TTT_CNJ_ValueDepthCap,
    jinja_data_model.undefinedVal, jinja_data_model.noneVal, jinja_data_model.boolVal,
    jinja_data_model.intVal, jinja_data_model.floatVal, jinja_data_model.strVal,
    jinja_data_model.seqVal, jinja_data_model.dictVal, jinja_data_model.nsVal,
    jinja_data_model.rangeVal, jinja_data_model.cutVal, jinja_data_model.eqVal,
    jinja_data_model.dictSet, jinja_data_model.dictGet

export jinja_serialize.toJson
