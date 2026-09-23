# Tattletale Copyright (c) 2026 Mamy Ratsimbazafy Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Chattyninja package umbrella.
## Explicit re-export list, the package surface. A module's own `*` surface outside these
## names is Nim's cross-module plumbing, not API, and nothing under `src/` re-exports.
##
## - the render tier, `parseTemplate`, `startRender`, `pullInto`, `pullAll`
##   and `renderToString`, over the compiled artifact and the caller's `JinjaRenderContext`
## - the data tier, `JinjaVal` with `ValueKind`, the value constructors, `eqVal`,
##   `dictSet`/`dictGet`, the `tojson` form (`toJson`, `JsonOpts`), `JinjaError` with its cause
## - the depth, nesting, step and element caps
##
## `jinja_interpolation`, `jinja_builtins` and every module's internals stay internal,
## cross-module code importing them through `import x {.all.}`.

import ./src/cnj_engine
import ./src/cnj_types, ./src/jinja_data_model, ./src/jinja_serialize, ./src/cnj_parse

# Render driver and parser.
export cnj_engine.startRender, cnj_engine.pullInto,
    cnj_engine.pullAll, cnj_engine.renderToString, cnj_parse.parseTemplate

# Compiled artifact, render session and the knobs.
export cnj_types.JinjaRenderContext, cnj_types.CompiledTemplate, cnj_types.CompiledSymbols,
    cnj_types.TTT_CNJ_MacroDepthCap, cnj_types.TTT_CNJ_ExprDepthCap,
    cnj_types.TTT_CNJ_ParseNestingCap, cnj_types.TTT_CNJ_StepBudget

# Values and their construction, equality, the dict reads, the error and the JSON form.
export jinja_data_model.JinjaVal, jinja_data_model.ValueKind, jinja_data_model.SeqVal,
    jinja_data_model.DictVal, jinja_data_model.RangeVal, jinja_data_model.LoopState,
    jinja_data_model.MacroVal, jinja_data_model.DeferredMacroCall,
    jinja_data_model.JinjaError, jinja_data_model.JinjaCause, jinja_data_model.JsonOpts,
    jinja_data_model.NoOffset, jinja_data_model.TTT_CNJ_RangeElemCap,
    jinja_data_model.TTT_CNJ_ValueDepthCap,
    jinja_data_model.undefinedVal, jinja_data_model.noneVal, jinja_data_model.boolVal,
    jinja_data_model.intVal, jinja_data_model.floatVal, jinja_data_model.strVal,
    jinja_data_model.seqVal, jinja_data_model.dictVal, jinja_data_model.nsVal,
    jinja_data_model.loopVal, jinja_data_model.macroVal, jinja_data_model.callVal,
    jinja_data_model.rangeVal, jinja_data_model.cutVal, jinja_data_model.eqVal,
    jinja_data_model.dictSet, jinja_data_model.dictGet

export jinja_serialize.toJson
