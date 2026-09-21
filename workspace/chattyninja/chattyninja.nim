# Tattletale Copyright (c) 2026 Mamy Ratsimbazafy Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://opensource.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

## Chattyninja package umbrella.
## The single re-export surface of `src/`. Nothing under `src/` re-exports.
##
## | Module           | Surface                                          |
## | ---------------- | ------------------------------------------------ |
## | cnj_engine       | render engine and pull interface                 |
## | cnj_parse        | template parsing                                 |
## | cnj_types        | compiled artifact, symbol arena and render state |
## | jinja_data_model | template values and JinjaError                   |
## | jinja_serialize  | Python str/repr and tojson serialization         |
##
## `jinja_interpolation` and `jinja_builtins` stay internal, no re-export.

import ./src/cnj_engine
import ./src/cnj_types, ./src/jinja_data_model, ./src/jinja_serialize, ./src/cnj_parse

export cnj_engine, cnj_types, jinja_data_model, jinja_serialize, cnj_parse
