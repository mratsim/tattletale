# Tattletale
# Copyright (c) 2026 Mamy André-Ratsimbazafy
# Licensed and distributed under either of
#   * MIT license (license terms in the root directory or at http://opensource.org/licenses/MIT).
#   * Apache v2 license (license terms in the root directory or at http://www.apache.org/licenses/LICENSE-2.0).
# at your option. This file may not be copied, modified, or distributed except according to those terms.

# Corrected NAX mma glue for the GEMM bench.
#
# The vendored apple_gemm.metal hardcodes BaseNAXFrag's lane ->
# (row, col) formula, which encodes one offline compiler's matmul2d
# cooperative-tensor layout. The Metal spec makes that layout opaque,
# compiler/GPU-derived, so the runtime compiler legally picks a different
# one and the glue miscomputes. This module holds the spec-query-based
# replacement (AppleNaxGlueMsl) and assembles the bench's ingest source
# from the byte-identical vendored file:
#
# - the BaseNAXFrag block is spliced out and replaced
# - NAXTile's frag parameter becomes mandatory
# - the gemm_loop tiles carry their operand role through it
#
# Every splice anchor is asserted: a vendored edit that moves an anchor
# fails the build loudly at compile time.

import std/strutils

const AppleNaxGlueMsl* = """
///////////////////////////////////////////////////////////////////////////////
// NAX frag: coordinates from the cooperative-tensor layout queries.
//
// The layout of a matmul2d cooperative tensor is opaque and chosen by the
// compiling toolchain (Metal Shading Language Specification 4.1,
// sections 2.22.2 and 2.22.3.1), so a hardcoded lane -> (row, col) formula
// only matches one compiler's choice and miscomputes under another. This
// frag derives every per-lane coordinate from the spec-blessed queries
// (cooperative_tensor::get_multidimensional_index); the identity frag <->
// operand register copies are then correct for whatever layout this
// toolchain picks, and the queries fold to constants under the full
// unrolls below.
//
// Role selects the operand the frag feeds: NAXFragLeft (M x K = 16 x 16),
// NAXFragRight (K x N = 16 x 32), NAXFragDest (M x N = 16 x 32). The
// queried index is (col, row) within the operand: left (k, m), right
// (n, k), destination (n, m).
//
// The gemm_loop mma consumes two 16x16 frag columns as one 16x32 operand:
// the even frag column holds operand slots 0..7, the odd one slots 8..15,
// both addressed from the even column's tile offset. A left-operand frag
// is a whole 16x16 operand (slots 0..7, no shift).
///////////////////////////////////////////////////////////////////////////////

enum NAXFragRole {
  NAXFragLeft = 0,
  NAXFragRight = 1,
  NAXFragDest = 2,
};

template <
    typename AType,
    typename BType,
    typename CType,
    bool transpose_a,
    bool transpose_b,
    int Role>
struct BaseNAXFrag {
  STEEL_CONST short kFragRows = 16;
  STEEL_CONST short kFragCols = 16;

  STEEL_CONST short kElemsPerFrag = (kFragRows * kFragCols) / 32;

  STEEL_CONST short kElemRows = 2;
  STEEL_CONST short kElemCols = 4;

  STEEL_CONST short kElemRowsJump = 8;

  static_assert(
      kElemRows * kElemCols == kElemsPerFrag,
      "MMAFrag shape is not consistent with MMAFrag size");

  template <typename U>
  using dtype_frag_t = typename metal::vec<U, kElemsPerFrag>;

  // The cooperative tensor of this frag's operand role: the subject of the
  // layout queries. The layout is a compile-time property of the
  // descriptor, so the queried coordinates describe the same tensor the
  // mma runs on.
  METAL_FUNC static auto query_tensor() {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16,
        32,
        16,
        transpose_a,
        transpose_b,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    if constexpr (Role == NAXFragLeft) {
      return gemm_op
          .template get_left_input_cooperative_tensor<AType, BType, CType>();
    } else if constexpr (Role == NAXFragRight) {
      return gemm_op
          .template get_right_input_cooperative_tensor<AType, BType, CType>();
    } else {
      auto ct_a =
          gemm_op
              .template get_left_input_cooperative_tensor<AType, BType, CType>();
      auto ct_b =
          gemm_op
              .template get_right_input_cooperative_tensor<AType, BType, CType>();
      return gemm_op.template get_destination_cooperative_tensor<
          metal::remove_addrspace_t<decltype(ct_a)>,
          metal::remove_addrspace_t<decltype(ct_b)>,
          CType>();
    }
  }

  // Frag-pair geometry: even frag column -> operand slots 0..7 anchored
  // at its own tile offset; odd frag column -> operand slots 8..15,
  // anchored one frag width back (both columns of a pair address the same
  // 16x32 operand). Returns (slot base, column anchor).
  METAL_FUNC static constexpr short2 frag_span(short off_y) {
    if constexpr (Role == NAXFragLeft) {
      return short2{0, off_y};
    } else {
      const short hi = (off_y >> 4) & 1;
      const short anchor = off_y - (hi << 4);
      return short2{hi, anchor};
    }
  }

  // (col, row) of the operand element this frag's slot idx must hold.
  // idx spans one 16x16 frag's slots; the frag-pair slot base is the
  // caller's concern (see frag_span).
  METAL_FUNC static short2 get_coord(short idx) {
    short2 coord = short2{0, 0};
    auto ct = query_tensor();
    const auto md = ct.get_multidimensional_index(idx);
    coord = short2{short(md[0]), short(md[1])};
    return coord;
  }

  METAL_FUNC static short2 get_coord() {
    return get_coord(0);
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 span = frag_span(off_y);
    auto ct = query_tensor();
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < kElemsPerFrag; j++) {
      const auto md = ct.get_multidimensional_index(span.x * kElemsPerFrag + j);
      dst[j] = static_cast<T>(
          src[(off_x + md[1]) * str_x + (span.y + md[0]) * str_y]);
    }
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_rows(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 span = frag_span(off_y);
    auto ct = query_tensor();
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < kElemsPerFrag; j++) {
      const auto md = ct.get_multidimensional_index(span.x * kElemsPerFrag + j);
      const int row = off_x + md[1];
      const int col = span.y + md[0];
      if (row < lim_x) {
        dst[j] = static_cast<T>(src[row * str_x + col * str_y]);
      } else {
        dst[j] = T(0);
      }
    }
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_safe(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 span = frag_span(off_y);
    auto ct = query_tensor();
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < kElemsPerFrag; j++) {
      const auto md = ct.get_multidimensional_index(span.x * kElemsPerFrag + j);
      const int row = off_x + md[1];
      const int col = span.y + md[0];
      if (row < lim_x && col < lim_y) {
        dst[j] = static_cast<T>(src[row * str_x + col * str_y]);
      } else {
        dst[j] = T(0);
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 span = frag_span(off_y);
    auto ct = query_tensor();
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < kElemsPerFrag; j++) {
      const auto md = ct.get_multidimensional_index(span.x * kElemsPerFrag + j);
      dst[(off_x + md[1]) * str_x + (span.y + md[0]) * str_y] =
          static_cast<U>(src[j]);
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_rows(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 span = frag_span(off_y);
    auto ct = query_tensor();
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < kElemsPerFrag; j++) {
      const auto md = ct.get_multidimensional_index(span.x * kElemsPerFrag + j);
      const int row = off_x + md[1];
      const int col = span.y + md[0];
      if (row < lim_x) {
        dst[row * str_x + col * str_y] = static_cast<U>(src[j]);
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_safe(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 span = frag_span(off_y);
    auto ct = query_tensor();
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < kElemsPerFrag; j++) {
      const auto md = ct.get_multidimensional_index(span.x * kElemsPerFrag + j);
      const int row = off_x + md[1];
      const int col = span.y + md[0];
      if (row < lim_x && col < lim_y) {
        dst[row * str_x + col * str_y] = static_cast<U>(src[j]);
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename StartX,
      typename StopX,
      typename StartY,
      typename StopY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_slice(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      StartX start_x,
      StopX stop_x,
      StartY start_y,
      StopY stop_y,
      OffX off_x = Int<0>{},
      OffY off_y = Int<0>{}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 span = frag_span(off_y);
    auto ct = query_tensor();
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < kElemsPerFrag; j++) {
      const auto md = ct.get_multidimensional_index(span.x * kElemsPerFrag + j);
      const int row = off_x + md[1];
      const int col = span.y + md[0];
      if (row >= start_x && row < stop_x && col >= start_y && col < stop_y) {
        dst[row * str_x + col * str_y] = static_cast<U>(src[j]);
      }
    }
  }

  // Row reduction needs the layout's cross-lane row topology: which lanes
  // hold the remaining elements of each row. The per-lane layout queries do
  // not expose that, and nothing in the NAX gemm path calls these, so they
  // are deleted: any call fails to compile instead of silently reducing
  // wrong elements.
  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_reduce(
      thread const dtype_frag_t<T>& inp_vals,
      thread T* reduced_vals) = delete;

  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_bin_op(
      thread dtype_frag_t<T>& inp_vals,
      thread T* row_vals) = delete;

  // The mma overloads bind to this frag's own AType/BType/CType and
  // transpose flags: the gemm_loop instantiates every tile's frag with the
  // operand types and transposes of the gemm it drives, so the parameter
  // lists below need no template parameters of their own.
  METAL_FUNC static constexpr void mma(
      thread dtype_frag_t<CType>& Cn0,
      thread dtype_frag_t<CType>& Cn1,
      const thread dtype_frag_t<AType>& A,
      metal::bool_constant<transpose_a>,
      const thread dtype_frag_t<BType>& Bn0,
      const thread dtype_frag_t<BType>& Bn1,
      metal::bool_constant<transpose_b>) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16,
        32,
        16,
        transpose_a,
        transpose_b,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    // Create matmul op
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    // Create matmul operands in registers
    auto ct_a =
        gemm_op
            .template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b =
        gemm_op
            .template get_right_input_cooperative_tensor<AType, BType, CType>();

    // Create matmul output in register
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>,
        CType>();

    // Load A in to left operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_a[i] = A[i];
    }

    // Load B into right operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_b[i] = Bn0[i];
      ct_b[kElemsPerFrag + i] = Bn1[i];
    }

    // Load C into output registers (op handles accumulation)
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_c[i] = Cn0[i];
      ct_c[kElemsPerFrag + i] = Cn1[i];
    }

    // Do matmul
    gemm_op.run(ct_a, ct_b, ct_c);

    // Copy out results
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      Cn0[i] = ct_c[i];
      Cn1[i] = ct_c[kElemsPerFrag + i];
    }
  }

  METAL_FUNC static constexpr void mma(
      thread dtype_frag_t<CType>& Cm0,
      thread dtype_frag_t<CType>& Cm1,
      const thread dtype_frag_t<AType>& Am0,
      const thread dtype_frag_t<AType>& Am1,
      metal::bool_constant<transpose_a>,
      const thread dtype_frag_t<BType>& B,
      metal::bool_constant<transpose_b>) {
    // Create Matmul descriptor
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16,
        32,
        16,
        transpose_a,
        transpose_b,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    // Create matmul op
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    // Create matmul operands in registers
    auto ct_a =
        gemm_op
            .template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b =
        gemm_op
            .template get_right_input_cooperative_tensor<AType, BType, CType>();

    // Create matmul output in register
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>,
        CType>();

    // Load A in to left operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_a[i] = Am0[i];
      ct_a[kElemsPerFrag + i] = Am1[i];
    }

    // Load B into right operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_b[i] = B[i];
    }

    // Load C into output registers (op handles accumulation)
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_c[i] = Cm0[i];
      ct_c[kElemsPerFrag + i] = Cm1[i];
    }

    // Do matmul
    gemm_op.run(ct_a, ct_b, ct_c);

    // Copy out results
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      Cm0[i] = ct_c[i];
      Cm1[i] = ct_c[kElemsPerFrag + i];
    }
  }
};

"""

proc splice(s, old: string, new: string, want: int): string {.compileTime.} =
  let got = s.count(old)
  doAssert got == want, "splice: expected " & $want & " occurrence(s), found " &
    $got & " of: " & old
  s.replace(old, new)

proc naxGemmIngestSrc*(): string {.compileTime.} =
  ## Returns:
  ##   the NAX bench ingest source:
  ##   - the vendored apple_gemm.metal, read, never written
  ##   - the BaseNAXFrag coordinate-dependent block replaced by AppleNaxGlueMsl
  ##   - the bench wrapper appended
  ## Preconditions:
  ##   every splice anchor matches exactly once, a vendored edit that moves
  ##   an anchor fails the build loudly
  var src = staticRead("apple_gemm.metal")

  # The BaseNAXFrag struct block, up to the adjacent NAXTile template.
  let start = src.find("struct BaseNAXFrag {")
  let stop = src.find("template <\n    typename T,\n    short kTileRows_,")
  doAssert start > 0 and stop > start,
    "splice: BaseNAXFrag block boundaries not found in the vendored source"
  src = src[0 ..< start] & AppleNaxGlueMsl & src[stop ..^ 1]

  # NAXTile's frag parameter becomes mandatory: the gemm_loop tiles carry
  # their operand role through it.
  src = src.splice("    class NAXFrag_ = BaseNAXFrag>", "    class NAXFrag_>", 1)
  src = src.splice(
    "  NAXTile<AccumType, TM, TN> Dtile;",
    "  NAXTile<AccumType, TM, TN, BaseNAXFrag<T, T, AccumType, transpose_a, " &
      "transpose_b, NAXFragDest>> Dtile;",
    1)
  src = src.splice(
    "      NAXTile<T, RA, CA> Atile;\n      NAXTile<T, RB, CB> Btile;",
    "      NAXTile<T, RA, CA, BaseNAXFrag<T, T, AccumType, transpose_a, " &
      "transpose_b, NAXFragLeft>> Atile;\n" &
      "      NAXTile<T, RB, CB, BaseNAXFrag<T, T, AccumType, transpose_a, " &
      "transpose_b, NAXFragRight>> Btile;",
    2)

  result = src & staticRead("apple_nax_bench_entrypoint.metal")
