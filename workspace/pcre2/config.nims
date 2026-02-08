# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Per-file ENV variables configuration for PCRE2
# ---------------------------------------------------

import std/[strutils, os]
const CurDir = currentSourcePath.rsplit(DirSep, 1)[0]

const CONFIG_H =
  # Include local pcre2.h
  " -I" & CurDir/"vendor" &
  " -I" & CurDir/"vendor/pcre2/src" &
  # Platform OS/Compile specific
  " -DHAVE_ASSERT_H=true" &
  (when defined(windows):
    " -DHAVE_WINDOWS_H=true"
  else:
    " -DHAVE_UNISTD_H=true") &
  " -DHAVE_ATTRIBUTE_UNINITIALIZED=true" &
  " -DHAVE_BUILTIN_MUL_OVERFLOW=true" &
  " -DHAVE_BUILTIN_UNREACHABLE=true" &
  # PCRE2 specific
  " -DSUPPORT_PCRE2_8=true" &
  " -DSUPPORT_PCRE2_16=false" &
  " -DSUPPORT_PCRE2_32=false" &
  " -DSUPPORT_UNICODE=true" &
  " -DSUPPORT_JIT=true" &
  # config-cmake.h.in
  " -DPCRE2_EXPORT=\"\"" &
  " -DLINK_SIZE=2" &
  " -DHEAP_LIMIT=20000000" &
  " -DMATCH_LIMIT=10000000" &
  " -DMATCH_LIMIT_DEPTH=\"MATCH_LIMIT\"" &
  " -DMAX_VARLOOKBEHIND=255" &
  " -DNEWLINE_DEFAULT=2" &
  " -DPARENS_NEST_LIMIT=250" &
  " -DPCRE2GREP_BUFSIZE=20480" &
  " -DPCRE2GREP_MAX_BUFSIZE=1048576" &
  " -DMAX_NAME_SIZE=128" &
  " -DMAX_NAME_COUNT=10000" &
  # Devops
  " -UHAVE_CONFIG_H" &
  " -DPCRE2_CODE_UNIT_WIDTH=8" &
  " -DPCRE2_STATIC"

put("pcre2_chartables.always", CONFIG_H)
put("pcre2_auto_possess.always", CONFIG_H)
put("pcre2_chkdint.always", CONFIG_H)
put("pcre2_compile.always", CONFIG_H)
put("pcre2_compile_cgroup.always", CONFIG_H)
put("pcre2_compile_class.always", CONFIG_H)
put("pcre2_config.always", CONFIG_H)
put("pcre2_context.always", CONFIG_H)
put("pcre2_convert.always", CONFIG_H)
put("pcre2_dfa_match.always", CONFIG_H)
put("pcre2_error.always", CONFIG_H)
put("pcre2_extuni.always", CONFIG_H)
put("pcre2_find_bracket.always", CONFIG_H)
put("pcre2_jit_compile.always", CONFIG_H)
put("pcre2_maketables.always", CONFIG_H)
put("pcre2_match.always", CONFIG_H)
put("pcre2_match_data.always", CONFIG_H)
put("pcre2_match_next.always", CONFIG_H)
put("pcre2_newline.always", CONFIG_H)
put("pcre2_ord2utf.always", CONFIG_H)
put("pcre2_pattern_info.always", CONFIG_H)
put("pcre2_script_run.always", CONFIG_H)
put("pcre2_serialize.always", CONFIG_H)
put("pcre2_string_utils.always", CONFIG_H)
put("pcre2_study.always", CONFIG_H)
put("pcre2_substitute.always", CONFIG_H)
put("pcre2_substring.always", CONFIG_H)
put("pcre2_tables.always", CONFIG_H)
put("pcre2_ucd.always", CONFIG_H)
put("pcre2_valid_utf.always", CONFIG_H)
put("pcre2_xclass.always", CONFIG_H)