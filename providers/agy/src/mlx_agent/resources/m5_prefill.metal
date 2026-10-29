// Derived from ml-explore/mlx, commit 6e7ab39b4b1042d53860507e2e00e4854a152658.
// Copyright © 2023-2025 Apple Inc.
// Compatibility changes: a narrow fragment adapter, operand-type dequantization,
// alignment-checked activation loads, and by-value dimensions. No split-K support.
// Upstream's same-split identity claim is not a claim about this backport.
/*
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#ifndef METAL_FUNC
#define METAL_FUNC inline __attribute__((always_inline))
#endif
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define MLX_MTL_CONST static constant constexpr const
namespace mlx_agent_nax {
using namespace metal;
MLX_MTL_CONST int SIMD_SIZE=32;
template<int bits,int wsize=8> inline constexpr short get_pack_factor() {
    static_assert(bits==4 && wsize==8,"Only affine 4-bit is supported"); return 2;
}
template<int bits,int wsize=8> inline constexpr short get_bytes_per_pack() {
    static_assert(bits==4 && wsize==8,"Only affine 4-bit is supported"); return 1;
}
template<typename T> inline vec<T,4> load_four(const device T* value) {
    // Keep the upstream vector load only when its eight-byte alignment holds.
    if ((reinterpret_cast<ulong>(value)&7ul)==0)
        return *reinterpret_cast<const device vec<T,4>*>(value);
    // Contiguous views can start at a two-byte offset.
    return vec<T,4>(value[0],value[1],value[2],value[3]);
}
template<typename U,int count,int bits>
inline void dequantize(const thread uint8_t* w,U scale,U bias,thread U* result) {
    static_assert(bits==4,"Only affine 4-bit is supported");
    // Keep MLX 0.31.1's operand-type arithmetic, including its scaled high nibble.
    U s[2]={scale,scale/static_cast<U>(16.0f)};
    STEEL_PRAGMA_UNROLL
    for(int i=0;i<count/2;++i) {
        result[2*i]=s[0]*(w[i]&0x0f)+bias;
        result[2*i+1]=s[1]*(w[i]&0xf0)+bias;
    }
}
struct BaseNAXFrag {
    MLX_MTL_CONST short kElemsPerFrag=8;
    template<typename T> using dtype_frag_t=metal::vec<T,8>;
    static short2 get_coord() {
        const ushort lane=__metal_get_thread_index_in_simdgroup(ushort());
        const ushort quad=lane>>2;
        return short2(((quad&2)|(lane&1))*4,(quad&4)|((lane>>1)&3));
    }
  template <
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
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


};
template<typename T,short rows,short columns> struct NAXTile {
    static_assert(rows==1 && columns==4,"Only a 16-row by 64-column tile is supported");
    vec<T,8> values[4];
    void clear() thread {
        STEEL_PRAGMA_UNROLL
        for(short i=0;i<4;++i) values[i]=vec<T,8>(0);
    }
    thread vec<T,8>& frag_at(short,short column) thread { return values[column]; }
    template<typename U> void store_safe(device U* target,int stride,short2 limit) const thread {
        short2 coordinate=BaseNAXFrag::get_coord();
        STEEL_PRAGMA_UNROLL
        for(short frag_idx=0;frag_idx<4;++frag_idx) {
            STEEL_PRAGMA_UNROLL
            for(short halfrow=0;halfrow<2;++halfrow) {
                short row=coordinate.y+8*halfrow;
                STEEL_PRAGMA_UNROLL
                for(short item=0;item<4;++item) {
                    short column=coordinate.x+frag_idx*16+item;
                    if(row<limit.y && column<limit.x)
                        target[row*stride+column]=static_cast<U>(values[frag_idx][halfrow*4+item]);
                }
            }
        }
    }
    template<typename U> void store(device U* target,int stride) const thread {
        store_safe(target,stride,short2(64,16));
    }
};
// QuantizedBlockLoader's thread mapping split into two steps: fetch() reads the
// packed words and the group's scale and bias for one BK step into registers,
// store() dequantizes them into threadgroup memory. Fetching step k + 1 before
// the MMAs of step k hides the device reads behind the matrix work.
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short tgp_size,
    short group_size,
    short bits>
struct QuantizedBlockLoaderStaged {
  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads = (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short n_words = n_reads * bytes_per_pack / 4;

  static_assert(
      BCOLS_PACKED * BROWS >= tgp_size &&
          n_words * 4 == n_reads * bytes_per_pack,
      "Every thread loads whole words");
  static_assert(
      n_reads * pack_factor <= group_size,
      "A thread's values must lie in one quantization group");

  const short thread_idx;
  const short bi;
  const short bj;
  const device uint8_t* src;
  const device T* scales;
  const device T* biases;
  uint32_t raw[n_words];
  T scale;
  T bias;

  QuantizedBlockLoaderStaged(
      const device uint8_t* src_,
      const device T* scales_,
      const device T* biases_,
      const int src_ld,
      ushort simd_group_id,
      ushort simd_lane_id) thread
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads* thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size),
        biases(biases_ + bi * src_ld / group_size) {}

  // k: first column of the step
  void fetch(const int k) thread {
    auto p = (const device uint32_t*)(src + k * bytes_per_pack / pack_factor);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_words; i++) {
      raw[i] = p[i];
    }
    const int g = (k + bj * pack_factor) / group_size;
    scale = scales[g];
    bias = biases[g];
  }

  // Dequantize into registers, then write 8 values per store
  void store(threadgroup T* dst) const thread {
    constexpr short n_vals = n_reads * pack_factor;
    static_assert(n_vals % 8 == 0, "8-value stores");
    T v[n_vals];
    // Unpack with shifts (constant indices keep it in registers)
    uint8_t b[n_words * 4];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_words * 4; i++) {
      b[i] = uint8_t(raw[i / 4] >> (8 * (i % 4)));
    }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          b + i * bytes_per_pack, scale, bias, v + i * pack_factor);
    }
    dst += bi * dst_ld + bj * pack_factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_vals; i += 8) {
      *(threadgroup vec<T, 8>*)(dst + i) = *(thread vec<T, 8>*)(v + i);
    }
  }
};

// Transposed qmm with BM / 16 simdgroups, each computing 16 rows x all BN = 64
// columns. The weight tile is double-buffered (Ws holds two BN x BK_padded
// tiles) and each simdgroup keeps the next step's activation fragments in
// registers, so one barrier separates K steps. Rows past M read the last valid
// row and are not stored. Requires N % BN == 0 and k_size (or K) % BK == 0.
// This compatibility port retains MLX 0.31.1 dequantization arithmetic.
// Numerical equivalence must be checked for the target workload.
template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    typename U = T>
METAL_FUNC void qmm_t_nax_rows16_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device U* y,
    threadgroup T* Ws,
    const int K,
    const int N,
    const int M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    const int k_size = -1) {
  static_assert(BK == 64 && BN == 64 && BM % 16 == 0, "16-row x 64 tiles");
  static_assert(
      sizeof(T) == 2, "Two tiles fit threadgroup memory for 16-bit T");

  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int tile_size = BN * BK_padded;

  using loader_w_t = QuantizedBlockLoaderStaged<
      T,
      BN,
      BK,
      BK_padded,
      (BM / 16) * SIMD_SIZE,
      group_size,
      bits>;
  using frag_t = typename BaseNAXFrag::template dtype_frag_t<T>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  loader_w_t loader_w(
      (const device uint8_t*)w + y_col * K_w,
      scales + y_col * K_g,
      biases + y_col * K_g,
      K,
      simd_gid,
      simd_lid);

  // This simdgroup's rows and the two rows each thread holds in a fragment
  const int tile_rows = min(BM, M - y_row);
  const int sm = 16 * int(simd_gid);
  const int rows = min(16, tile_rows - sm);
  const short2 sc = BaseNAXFrag::get_coord();
  const int r0 = min(sm + int(sc.y), tile_rows - 1);
  const int r1 = min(sm + int(sc.y) + 8, tile_rows - 1);
  const device T* x0 = x + (y_row + r0) * static_cast<int64_t>(K) + sc.x;
  const device T* x1 = x + (y_row + r1) * static_cast<int64_t>(K) + sc.x;

  NAXTile<float, 1, BN / 16> Dtile;
  Dtile.clear();
  vec<T, 4> a_cur[BK / 16][2];
  vec<T, 4> a_next[BK / 16][2];

  const int k_end = k_size < 0 ? int(K) : k_size;

  loader_w.fetch(0);
  loader_w.store(Ws);
  if (rows > 0) {
    STEEL_PRAGMA_UNROLL
    for (short s = 0; s < BK / 16; s++) {
      a_cur[s][0] = load_four(x0 + s * 16);
      a_cur[s][1] = load_four(x1 + s * 16);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int k = 0, it = 0; k < k_end; k += BK, it++) {
    const bool more = k + BK < k_end;
    if (more) {
      loader_w.fetch(k + BK);
    }
    if (rows > 0) {
      const int kn = more ? k + BK : k;
      STEEL_PRAGMA_UNROLL
      for (short s = 0; s < BK / 16; s++) {
        a_next[s][0] = load_four(x0 + kn + s * 16);
        a_next[s][1] = load_four(x1 + kn + s * 16);
      }
      const threadgroup T* ws =
          Ws + (it & 1) * tile_size + sc.y * BK_padded + sc.x;
      STEEL_PRAGMA_UNROLL
      for (short s = 0; s < BK / 16; s++) {
        frag_t a;
        STEEL_PRAGMA_UNROLL
        for (short c = 0; c < 4; c++) {
          a[c] = a_cur[s][0][c];
          a[4 + c] = a_cur[s][1][c];
        }
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < BN / 16; j += 2) {
          const threadgroup T* p0 = ws + (j * 16) * BK_padded + s * 16;
          const threadgroup T* p1 = p0 + 16 * BK_padded;
          frag_t b0;
          frag_t b1;
          STEEL_PRAGMA_UNROLL
          for (short h = 0; h < 2; h++) {
            const vec<T, 4> u0 =
                *(const threadgroup vec<T, 4>*)(p0 + h * 8 * BK_padded);
            const vec<T, 4> u1 =
                *(const threadgroup vec<T, 4>*)(p1 + h * 8 * BK_padded);
            STEEL_PRAGMA_UNROLL
            for (short c = 0; c < 4; c++) {
              b0[h * 4 + c] = u0[c];
              b1[h * 4 + c] = u1[c];
            }
          }
          BaseNAXFrag::mma(
              Dtile.frag_at(0, j),
              Dtile.frag_at(0, j + 1),
              a,
              metal::bool_constant<false>{},
              b0,
              b1,
              metal::bool_constant<true>{});
        }
      }
      STEEL_PRAGMA_UNROLL
      for (short s = 0; s < BK / 16; s++) {
        a_cur[s][0] = a_next[s][0];
        a_cur[s][1] = a_next[s][1];
      }
    }
    if (more) {
      loader_w.store(Ws + ((it + 1) & 1) * tile_size);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (rows <= 0) {
    return;
  }
  y += (y_row + sm) * static_cast<int64_t>(N) + y_col;
  if (rows == 16) {
    Dtile.store(y, N);
  } else {
    Dtile.store_safe(y, N, short2(BN, rows));
  }
}

} // namespace mlx_agent_nax
