# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# This file statically compiles PCRE2 or dynamically link to system PCRE2
# and exposes its API in Nim
# We don't modify or path the original library
# so that version updates only need:
# - git checkout
# - copying vendor/pcre2/src/pcre2.h.in to ./pcre2.h·
#   and editing versioning macro
#     ```
#     #define PCRE2_MAJOR           @PCRE2_MAJOR@
#     #define PCRE2_MINOR           @PCRE2_MINOR@
#     #define PCRE2_PRERELEASE      @PCRE2_PRERELEASE@
#     #define PCRE2_DATE            @PCRE2_DATE@
#     ```
# - potentially passing new BUILD ENV variables
#
# REFERENCES
#   - workspace/pcre2/vendor/pcre2/src/config-cmake.h.in
#   - workspace/pcre2/vendor/pcre2/build.zig
#   - workspace/pcre2/vendor/pcre2/NON-AUTOTOOLS-BUILD
#
# BUILD ENV are passed through config.nims
# to restrict them on a per-file basis,
# {.localpassc: "-D...".} would only work for the current nim file
#
# We also copy workspace/pcre2/vendor/pcre2/src/pcre2_chartables.c.dist
# to workspace/pcre2/vendor/pcre2_chartables.c
# Symlinking instead might break on windows or need extra permissions
# and `git clone -c core.symlinks=true https://github.com/mratsim/tattletale`
# to enable cloning with symlink

const TTL_USE_SYSTEM_PCRE2 {.booldefine.} = true

when not TTL_USE_SYSTEM_PCRE2:
  {.compile:"vendor/pcre2_chartables.c".}
  {.compile:"vendor/pcre2/src/pcre2_auto_possess.c".}
  {.compile:"vendor/pcre2/src/pcre2_chkdint.c".}
  {.compile:"vendor/pcre2/src/pcre2_compile.c".}
  {.compile:"vendor/pcre2/src/pcre2_compile_cgroup.c".}
  {.compile:"vendor/pcre2/src/pcre2_compile_class.c".}
  {.compile:"vendor/pcre2/src/pcre2_config.c".}
  {.compile:"vendor/pcre2/src/pcre2_context.c".}
  {.compile:"vendor/pcre2/src/pcre2_convert.c".}
  {.compile:"vendor/pcre2/src/pcre2_dfa_match.c".}
  {.compile:"vendor/pcre2/src/pcre2_error.c".}
  {.compile:"vendor/pcre2/src/pcre2_extuni.c".}
  {.compile:"vendor/pcre2/src/pcre2_find_bracket.c".}
  {.compile:"vendor/pcre2/src/pcre2_jit_compile.c".}
  {.compile:"vendor/pcre2/src/pcre2_maketables.c".}
  {.compile:"vendor/pcre2/src/pcre2_match.c".}
  {.compile:"vendor/pcre2/src/pcre2_match_data.c".}
  {.compile:"vendor/pcre2/src/pcre2_match_next.c".}
  {.compile:"vendor/pcre2/src/pcre2_newline.c".}
  {.compile:"vendor/pcre2/src/pcre2_ord2utf.c".}
  {.compile:"vendor/pcre2/src/pcre2_pattern_info.c".}
  {.compile:"vendor/pcre2/src/pcre2_script_run.c".}
  {.compile:"vendor/pcre2/src/pcre2_serialize.c".}
  {.compile:"vendor/pcre2/src/pcre2_string_utils.c".}
  {.compile:"vendor/pcre2/src/pcre2_study.c".}
  {.compile:"vendor/pcre2/src/pcre2_substitute.c".}
  {.compile:"vendor/pcre2/src/pcre2_substring.c".}
  {.compile:"vendor/pcre2/src/pcre2_tables.c".}
  {.compile:"vendor/pcre2/src/pcre2_ucd.c".}
  {.compile:"vendor/pcre2/src/pcre2_valid_utf.c".}
  {.compile:"vendor/pcre2/src/pcre2_xclass.c".}

# ################################################## #
#                 PCRE2 API                          #
# ################################################## #

# Overview of general API
# ----------------------------------------------------
# man pcre2api.3
# or workspace/pcre2/vendor/pcre2/doc/pcre2api.3
#
# man pcre2demo.3
# or workspace/pcre2/vendor/pcre2/doc/pcre2demo.3
#
# This is low-level bindings that departs only slightly from
# the low-level API.
#
# Namely:
# - #define constants are enums and functions are type-checked
# - Compiler-enforced proper flag usage
# - str+len are also presented with openArray[char] overload to pass Nim strings
# - csize_t may be wrapped with int (assuming 64-bit, and addresses using less than 48 bits, int is large enough)

# Flag parameters
# ------------------------------------------------------------

type Flag*[E: enum] = distinct cint

func flag*[E: enum](e: varargs[E]): Flag[E] {.inline.} =
  ## Enum should only have power of 2 fields
  # static:
  #   for val in E:
  #     assert (ord(val) and (ord(val) - 1)) == 0, "Enum values should all be power of 2, found " &
  #                                                 $val & " with value " & $ord(val) & "."
  var flags = 0
  for val in e:
    flags = flags or ord(val)
  return Flag[E](flags)

# Constants enumified
# ----------------------------------------------------

type CompileOption* {.size: sizeof(uint32).} = enum
  # The following option bits can be passed only to pcre2_compile(). However,
  # they may affect compilation, JIT compilation, and/or interpretive execution.
  # The following tags indicate which:

  # C   alters what is compiled by pcre2_compile()
  # J   alters what is compiled by pcre2_jit_compile()
  # M   is inspected during pcre2_match() execution
  # D   is inspected during pcre2_dfa_match() execution
  ALLOW_EMPTY_CLASS   = 0x00000001'u32  # C       #
  ALT_BSUX            = 0x00000002'u32  # C       #
  AUTO_CALLOUT        = 0x00000004'u32  # C       #
  CASELESS            = 0x00000008'u32  # C       #
  DOLLAR_ENDONLY      = 0x00000010'u32  #   J M D #
  DOTALL              = 0x00000020'u32  # C       #
  DUPNAMES            = 0x00000040'u32  # C       #
  EXTENDED            = 0x00000080'u32  # C       #
  FIRSTLINE           = 0x00000100'u32  #   J M D #
  MATCH_UNSET_BACKREF = 0x00000200'u32  # C J M   #
  MULTILINE           = 0x00000400'u32  # C       #
  NEVER_UCP           = 0x00000800'u32  # C       #
  NEVER_UTF           = 0x00001000'u32  # C       #
  NO_AUTO_CAPTURE     = 0x00002000'u32  # C       #
  NO_AUTO_POSSESS     = 0x00004000'u32  # C       #
  NO_DOTSTAR_ANCHOR   = 0x00008000'u32  # C       #
  NO_START_OPTIMIZE   = 0x00010000'u32  #   J M D #
  UCP                 = 0x00020000'u32  # C J M D #
  UNGREEDY            = 0x00040000'u32  # C       #
  UTF                 = 0x00080000'u32  # C J M D #
  NEVER_BACKSLASH_C   = 0x00100000'u32  # C       #
  ALT_CIRCUMFLEX      = 0x00200000'u32  #   J M D #
  ALT_VERBNAMES       = 0x00400000'u32  # C       #
  USE_OFFSET_LIMIT    = 0x00800000'u32  #   J M D #
  EXTENDED_MORE       = 0x01000000'u32  # C       #
  LITERAL             = 0x02000000'u32  # C       #
  MATCH_INVALID_UTF   = 0x04000000'u32  #   J M D #
  ALT_EXTENDED_CLASS  = 0x08000000'u32  # C       #

  # The following option bits can be passed to pcre2_compile(), pcre2_match(),
  # or pcre2_dfa_match(). PCRE2_NO_UTF_CHECK affects only the function to which it
  # is passed. Put these bits at the most significant end of the options word so
  # others can be added next to them
  ENDANCHORED         = 0x20000000'u32
  NO_UTF_CHECK        = 0x40000000'u32
  ANCHORED            = 0x80000000'u32

type MatchOption* {.size: sizeof(uint32).} = enum
  NOTBOL                      = 0x00000001'u32
  NOTEOL                      = 0x00000002'u32
  NOTEMPTY                    = 0x00000004'u32  # ) These two must be kept
  NOTEMPTY_ATSTART            = 0x00000008'u32  # ) adjacent to each other.
  PARTIAL_SOFT                = 0x00000010'u32
  PARTIAL_HARD                = 0x00000020'u32
  DFA_RESTART                 = 0x00000040'u32  # pcre2_dfa_match() only
  DFA_SHORTEST                = 0x00000080'u32  # pcre2_dfa_match() only
  SUBSTITUTE_GLOBAL           = 0x00000100'u32  # pcre2_substitute() only
  SUBSTITUTE_EXTENDED         = 0x00000200'u32  # pcre2_substitute() only
  SUBSTITUTE_UNSET_EMPTY      = 0x00000400'u32  # pcre2_substitute() only
  SUBSTITUTE_UNKNOWN_UNSET    = 0x00000800'u32  # pcre2_substitute() only
  SUBSTITUTE_OVERFLOW_LENGTH  = 0x00001000'u32  # pcre2_substitute() only
  NO_JIT                      = 0x00002000'u32  # not for pcre2_dfa_match()
  COPY_MATCHED_SUBJECT        = 0x00004000'u32
  SUBSTITUTE_LITERAL          = 0x00008000'u32  # pcre2_substitute() only
  SUBSTITUTE_MATCHED          = 0x00010000'u32  # pcre2_substitute() only
  SUBSTITUTE_REPLACEMENT_ONLY = 0x00020000'u32  # pcre2_substitute() only
  DISABLE_RECURSELOOP_CHECK   = 0x00040000'u32  # not for pcre2_dfa_match() or pcre2_jit_match()

  # The following option bits can be passed to pcre2_compile(), pcre2_match(),
  # or pcre2_dfa_match(). PCRE2_NO_UTF_CHECK affects only the function to which it
  # is passed. Put these bits at the most significant end of the options word so
  # others can be added next to them
  ENDANCHORED         = 0x20000000'u32
  NO_UTF_CHECK        = 0x40000000'u32
  ANCHORED            = 0x80000000'u32

type JitOption* {.size: sizeof(uint32).} = enum
  JIT_COMPLETE        = 0x00000001'u32  # For full matching
  JIT_PARTIAL_SOFT    = 0x00000002'u32
  JIT_PARTIAL_HARD    = 0x00000004'u32
  JIT_INVALID_UTF     = 0x00000100'u32
  JIT_TEST_ALLOC      = 0x00000200'u32

type CompileError* {.size: sizeof(cint).} = enum
  END_BACKSLASH                  = 101
  END_BACKSLASH_C                = 102
  UNKNOWN_ESCAPE                 = 103
  QUANTIFIER_OUT_OF_ORDER        = 104
  QUANTIFIER_TOO_BIG             = 105
  MISSING_SQUARE_BRACKET         = 106
  ESCAPE_INVALID_IN_CLASS        = 107
  CLASS_RANGE_ORDER              = 108
  QUANTIFIER_INVALID             = 109
  INTERNAL_UNEXPECTED_REPEAT     = 110
  INVALID_AFTER_PARENS_QUERY     = 111
  POSIX_CLASS_NOT_IN_CLASS       = 112
  POSIX_NO_SUPPORT_COLLATING     = 113
  MISSING_CLOSING_PARENTHESIS    = 114
  BAD_SUBPATTERN_REFERENCE       = 115
  NULL_PATTERN                   = 116
  BAD_OPTIONS                    = 117
  MISSING_COMMENT_CLOSING        = 118
  PARENTHESES_NEST_TOO_DEEP      = 119
  PATTERN_TOO_LARGE              = 120
  HEAP_FAILED                    = 121
  UNMATCHED_CLOSING_PARENTHESIS  = 122
  INTERNAL_CODE_OVERFLOW         = 123
  MISSING_CONDITION_CLOSING      = 124
  LOOKBEHIND_NOT_FIXED_LENGTH    = 125
  ZERO_RELATIVE_REFERENCE        = 126
  TOO_MANY_CONDITION_BRANCHES    = 127
  CONDITION_ASSERTION_EXPECTED   = 128
  BAD_RELATIVE_REFERENCE         = 129
  UNKNOWN_POSIX_CLASS            = 130
  INTERNAL_STUDY_ERROR           = 131
  UNICODE_NOT_SUPPORTED          = 132
  PARENTHESES_STACK_CHECK        = 133
  CODE_POINT_TOO_BIG             = 134
  LOOKBEHIND_TOO_COMPLICATED     = 135
  LOOKBEHIND_INVALID_BACKSLASH_C = 136
  UNSUPPORTED_ESCAPE_SEQUENCE    = 137
  CALLOUT_NUMBER_TOO_BIG         = 138
  MISSING_CALLOUT_CLOSING        = 139
  ESCAPE_INVALID_IN_VERB         = 140
  UNRECOGNIZED_AFTER_QUERY_P     = 141
  MISSING_NAME_TERMINATOR        = 142
  DUPLICATE_SUBPATTERN_NAME      = 143
  INVALID_SUBPATTERN_NAME        = 144
  UNICODE_PROPERTIES_UNAVAILABLE = 145
  MALFORMED_UNICODE_PROPERTY     = 146
  UNKNOWN_UNICODE_PROPERTY       = 147
  SUBPATTERN_NAME_TOO_LONG       = 148
  TOO_MANY_NAMED_SUBPATTERNS     = 149
  CLASS_INVALID_RANGE            = 150
  OCTAL_BYTE_TOO_BIG             = 151
  INTERNAL_OVERRAN_WORKSPACE     = 152
  INTERNAL_MISSING_SUBPATTERN    = 153
  DEFINE_TOO_MANY_BRANCHES       = 154
  BACKSLASH_O_MISSING_BRACE      = 155
  INTERNAL_UNKNOWN_NEWLINE       = 156
  BACKSLASH_G_SYNTAX             = 157
  PARENS_QUERY_R_MISSING_CLOSING = 158
  # /* Error 159 is obsolete and should now never occur */
  VERB_ARGUMENT_NOT_ALLOWED      = 159
  VERB_UNKNOWN                   = 160
  SUBPATTERN_NUMBER_TOO_BIG      = 161
  SUBPATTERN_NAME_EXPECTED       = 162
  INTERNAL_PARSED_OVERFLOW       = 163
  INVALID_OCTAL                  = 164
  SUBPATTERN_NAMES_MISMATCH      = 165
  MARK_MISSING_ARGUMENT          = 166
  INVALID_HEXADECIMAL            = 167
  BACKSLASH_C_SYNTAX             = 168
  BACKSLASH_K_SYNTAX             = 169
  INTERNAL_BAD_CODE_LOOKBEHINDS  = 170
  BACKSLASH_N_IN_CLASS           = 171
  CALLOUT_STRING_TOO_LONG        = 172
  UNICODE_DISALLOWED_CODE_POINT  = 173
  UTF_IS_DISABLED                = 174
  UCP_IS_DISABLED                = 175
  VERB_NAME_TOO_LONG             = 176
  BACKSLASH_U_CODE_POINT_TOO_BIG = 177
  MISSING_OCTAL_OR_HEX_DIGITS    = 178
  VERSION_CONDITION_SYNTAX       = 179
  INTERNAL_BAD_CODE_AUTO_POSSESS = 180
  CALLOUT_NO_STRING_DELIMITER    = 181
  CALLOUT_BAD_STRING_DELIMITER   = 182
  BACKSLASH_C_CALLER_DISABLED    = 183
  QUERY_BARJX_NEST_TOO_DEEP      = 184
  BACKSLASH_C_LIBRARY_DISABLED   = 185
  PATTERN_TOO_COMPLICATED        = 186
  LOOKBEHIND_TOO_LONG            = 187
  PATTERN_STRING_TOO_LONG        = 188
  INTERNAL_BAD_CODE              = 189
  INTERNAL_BAD_CODE_IN_SKIP      = 190
  NO_SURROGATES_IN_UTF16         = 191
  BAD_LITERAL_OPTIONS            = 192
  SUPPORTED_ONLY_IN_UNICODE      = 193
  INVALID_HYPHEN_IN_OPTIONS      = 194
  ALPHA_ASSERTION_UNKNOWN        = 195
  SCRIPT_RUN_NOT_AVAILABLE       = 196
  TOO_MANY_CAPTURES              = 197
  MISSING_OCTAL_DIGIT            = 198
  BACKSLASH_K_IN_LOOKAROUND      = 199
  MAX_VAR_LOOKBEHIND_EXCEEDED    = 200
  PATTERN_COMPILED_SIZE_TOO_BIG  = 201
  OVERSIZE_PYTHON_OCTAL          = 202
  CALLOUT_CALLER_DISABLED        = 203
  EXTRA_CASING_REQUIRES_UNICODE  = 204
  TURKISH_CASING_REQUIRES_UTF    = 205
  EXTRA_CASING_INCOMPATIBLE      = 206
  ECLASS_NEST_TOO_DEEP           = 207
  ECLASS_INVALID_OPERATOR        = 208
  ECLASS_UNEXPECTED_OPERATOR     = 209
  ECLASS_EXPECTED_OPERAND        = 210
  ECLASS_MIXED_OPERATORS         = 211
  ECLASS_HINT_SQUARE_BRACKET     = 212
  PERL_ECLASS_UNEXPECTED_EXPR    = 213
  PERL_ECLASS_EMPTY_EXPR         = 214
  PERL_ECLASS_MISSING_CLOSE      = 215
  PERL_ECLASS_UNEXPECTED_CHAR    = 216
  EXPECTED_CAPTURE_GROUP         = 217
  MISSING_OPENING_PARENTHESIS    = 218
  MISSING_NUMBER_TERMINATOR      = 219
  NULL_ERROROFFSET               = 220

type MiscError* {.size: sizeof(cint).} = enum
    ERROR_BADDATA           = -29
    ERROR_MIXEDTABLES       = -30  # Name was changed
    ERROR_BADMAGIC          = -31
    ERROR_BADMODE           = -32
    ERROR_BADOFFSET         = -33
    ERROR_BADOPTION         = -34
    ERROR_BADREPLACEMENT    = -35
    ERROR_BADUTFOFFSET      = -36
    ERROR_CALLOUT           = -37  # Never used by PCRE2 itself
    ERROR_DFA_BADRESTART    = -38
    ERROR_DFA_RECURSE       = -39
    ERROR_DFA_UCOND         = -40
    ERROR_DFA_UFUNC         = -41
    ERROR_DFA_UITEM         = -42
    ERROR_DFA_WSSIZE        = -43
    ERROR_INTERNAL          = -44
    ERROR_JIT_BADOPTION     = -45
    ERROR_JIT_STACKLIMIT    = -46
    ERROR_MATCHLIMIT        = -47
    ERROR_NOMEMORY          = -48
    ERROR_NOSUBSTRING       = -49
    ERROR_NOUNIQUESUBSTRING = -50
    ERROR_NULL              = -51
    ERROR_RECURSELOOP       = -52
    ERROR_DEPTHLIMIT        = -53
    # ERROR_RECURSIONLIMIT    = -53  /* Obsolete synonym */
    ERROR_UNAVAILABLE       = -54
    ERROR_UNSET             = -55
    ERROR_BADOFFSETLIMIT    = -56
    ERROR_BADREPESCAPE      = -57
    ERROR_REPMISSINGBRACE   = -58
    ERROR_BADSUBSTITUTION   = -59
    ERROR_BADSUBSPATTERN    = -60
    ERROR_TOOMANYREPLACE    = -61
    ERROR_BADSERIALIZEDDATA = -62
    ERROR_HEAPLIMIT         = -63
    ERROR_CONVERT_SYNTAX    = -64
    ERROR_INTERNAL_DUPMATCH = -65
    ERROR_DFA_UINVALID_UTF  = -66
    ERROR_INVALIDOFFSET     = -67
    ERROR_JIT_UNSUPPORTED   = -68
    ERROR_REPLACECASE       = -69
    ERROR_TOOLARGEREPLACE   = -70
    ERROR_DIFFSUBSPATTERN   = -71
    ERROR_DIFFSUBSSUBJECT   = -72
    ERROR_DIFFSUBSOFFSET    = -73
    ERROR_DIFFSUBSOPTIONS   = -74
    ERROR_BAD_BACKSLASH_K   = -75

# Note: for proper resource management, we should wrap the low-level procs/types
# in high-level proc/types with destructors
type
  Code* = object
  GeneralContext* = object
  MatchData* = object

when not TTL_USE_SYSTEM_PCRE2:
  {.pragma: pcre2, importc: "pcre2_$1_8", cdecl.}
else:
  when hostOS == "windows":
    const pcre2lib = "libpcre2-8-0.dll"
  elif hostOS == "macosx":
    const pcre2lib = "libpcre2-8.0.dylib"
  else:
    const pcre2lib = "libpcre2-8.so.0"
  {.pragma: pcre2, importc: "pcre2_$1_8", cdecl, dynlib: pcre2lib.}

proc compile*(pattern: ptr char|cstring, patlen: int,
              options: Flag[CompileOption],
              errorptr: var CompileError,
              erroroffset: var csize_t,
              ccontext: pointer = nil): ptr Code {.pcre2.}

proc compile*(pattern: openArray[char],
              options: Flag[CompileOption],
              errorptr: var CompileError,
              erroroffset: var csize_t,
              ccontext: pointer = nil): ptr Code {.inline.} =
  ## Doc:
  ##    man pcre2_compile.3
  ## or workspace/pcre2/vendor/pcre2/doc/pcre2_compile.3
  ## or workspace/pcre2/vendor/pcre2/src/pcre2_compile.c#L10257
  ##
  ## Note, in Nim we pattern+patlen are passed together
  ## as openArray[char] that will match a string.
  ## And we will skip an internal strlen by passing the length explicitly
  ##
  ## ***
  ##
  ## This function reads a regular expression in the form of a string and returns
  ## a pointer to a block of store holding a compiled version of the expression.
  ##
  ## Arguments:
  ##   pattern       the regular expression
  ##   patlen        the length of the pattern, or PCRE2_ZERO_TERMINATED
  ##   options       option bits
  ##   errorptr      pointer to errorcode
  ##   erroroffset   pointer to error offset
  ##   ccontext      points to a compile context or is NULL
  ##
  ## Returns:        pointer to compiled data block, or NULL on error,
  ##                 with errorcode and erroroffset set
  return compile(
    if pattern.len == 0: nil else: pattern[0].addr,
    pattern.len,
    options,
    errorptr,
    erroroffset,
    ccontext
  )

proc match*(code: ptr Code,
           subject: openArray[char],
           startoffset: int,
           options: Flag[MatchOption],
           ovector: ptr MatchData,
           mcontext: pointer = nil): cint {.pcre2.}
  ## Doc:
  ##    man pcre2_match.3
  ## or workspace/pcre2/vendor/pcre2/doc/pcre2_match.3
  ## or workspace/pcre2/vendor/pcre2/src/pcre2_match.c#L6942
  ##
  ## Note, in Nim we subject+length are passed together
  ## as openArray[char] that will match a string.
  ## And we will skip an internal strlen by passing the length explicitly
  ##
  ## ***
  ##
  ## This function applies a compiled pattern to a subject string and picks out
  ## portions of the string if it matches. Two elements in the vector are set for
  ## each substring: the offsets to the start and end of the substring.
  ##
  ## Arguments:
  ##   code            points to the compiled expression
  ##   subject         points to the subject string
  ##   length          length of subject string (may contain binary zeros)
  ##   start_offset    where to start in the subject string
  ##   options         option bits
  ##   match_data      points to a match_data block
  ##   mcontext        points a PCRE2 context
  ##
  ## Returns:          > 0 => success; value is the number of ovector pairs filled
  ##                   = 0 => success, but ovector is not big enough
  ##                   = -1 => failed to match (PCRE2_ERROR_NOMATCH)
  ##                   = -2 => partial match (PCRE2_ERROR_PARTIAL)
  ##                   < -2 => some kind of unexpected problem

proc jit_match*(code: ptr Code,
           subject: openArray[char],
           startoffset: int,
           options: Flag[MatchOption],
           ovector: ptr MatchData,
           mcontext: pointer = nil): cint {.pcre2.}
  ## Doc:
  ##    man pcre2_match.3
  ## or workspace/pcre2/vendor/pcre2/doc/pcre2_jit_match.3
  ##
  ## This  function  matches  a compiled regular expression that has been successfully processed by the JIT compiler against a given subject string, using a matching algorithm that is similar to Perl's. It is a "fast path" interface to JIT, and it bypasses some of
  ## the sanity checks that pcre2_match() applies.
  ##
  ## In UTF mode, the subject string is not checked for UTF validity. Unless PCRE2_MATCH_INVALID_UTF was set when the pattern was compiled, passing an invalid UTF string results in undefined behaviour. Your program may crash or loop or give wrong results.  In  the
  ## absence of PCRE2_MATCH_INVALID_UTF you should only call pcre2_jit_match() in UTF mode if you are sure the subject is valid.
  ##
  ## The arguments for pcre2_jit_match() are exactly the same as for pcre2_match(), except that the subject string must be specified with a length; PCRE2_ZERO_TERMINATED is not supported.
  ##
  ## The supported options are PCRE2_NOTBOL, PCRE2_NOTEOL, PCRE2_NOTEMPTY, PCRE2_NOTEMPTY_ATSTART, PCRE2_PARTIAL_HARD, and PCRE2_PARTIAL_SOFT. Unsupported options are ignored.
  ##
  ## Comment: for Tattletale, given that the subject comes from users who may have adversarial behaviors
  ##          we should NOT use jit_match

proc match_data_create_from_pattern*(
  code: ptr Code,
  ctx: ptr GeneralContext
): ptr MatchData {.pcre2.}
  ## This  function  creates  a new match data block for holding the result of a match.  If the first argument is NULL, this function returns NULL, otherwise the first argument points to a compiled pattern. The number of capturing parentheses within the pattern is
  ## used to compute the number of pairs of offsets that are required in the match data block. These form the "output vector" (ovector) within the match data block, and are used to identify the  matched  string  and  any  captured  substrings  when  matching  with
  ## pcre2_match(). If you are using pcre2_dfa_match(), which uses the output vector in a different way, you should use pcre2_match_data_create() instead of this function.
  ##
  ## The  second argument points to a general context, for custom memory management, or is NULL to use the same memory allocator that was used for the compiled pattern. The result of the function is NULL if the memory for the block could not be obtained or if NULL
  ## was provided as the first argument.

proc match_data_free*(data: ptr MatchData) {.pcre2.}

proc get_ovector_pointer*(ovector: ptr MatchData): ptr UncheckedArray[int] {.pcre2.}
  ## This function returns a pointer to the vector of offsets that forms part of the given match data block.
  ## The number of pairs can be found by calling pcre2_get_ovector_count().

proc get_ovector_count*(ovector: ptr MatchData): uint32 {.pcre2.}
  ## This function returns the number of pairs of offsets in the ovector that forms part of the given match data block.

proc code_free*(code: ptr Code) {.pcre2.}

proc jit_compile*(code: ptr Code, options: uint32): cint {.pcre2.}
  ## This  function  requests  JIT  compilation, which, if the just-in-time compiler is available,
  ## further processes a compiled pattern into machine code that executes
  ## much faster than the pcre2_match() interpretive matching function.
  ##
  ## Otherwise, the first argument must be a pointer that was returned by a successful call to pcre2_compile(), and the second must contain one or more of the following bits:
  ##
  ##   PCRE2_JIT_COMPLETE      compile code for full matching
  ##   PCRE2_JIT_PARTIAL_SOFT  compile code for soft partial matching
  ##   PCRE2_JIT_PARTIAL_HARD  compile code for hard partial matching
  ##
  ## The  yield  of the function when called with any of the three options above is:
  ## - 0 for success,
  ## - or a negative error code otherwise.
  ## In particular, PCRE2_ERROR_JIT_BADOPTION is returned if JIT is not supported or if an unknown bit is set in options.
  ## The function can also return PCRE2_ERROR_NOMEMORY if JIT is unable to allocate executable memory for the compiler,
  ## even if it was because of a system security restriction.
  ## In a few cases, the function may return with PCRE2_ERROR_JIT_UNSUPPORTED for unsupported features.
