#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#include <cuda_fp16.h>
__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}
#else

typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

#define TVM_FORCE_INLINE inline __attribute__((always_inline))
#define TVM_XINLINE TVM_FORCE_INLINE __device__ __host__
#define TVM_ALIGNED(x) __attribute__ ((aligned(x)))
#define TVM_HALF_OPERATOR(RTYPE, OP)                              \
  TVM_XINLINE RTYPE operator OP (half a, half b) {                \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (half a, T b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (T a, half b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }

#define TVM_HALF_ASSIGNOP(AOP, OP)                                \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const T& a) {                    \
    return *this = half(float(*this) OP float(a));                \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const volatile T& a) volatile {  \
    return *this = half(float(*this) OP float(a));                \
  }

class TVM_ALIGNED(2) half {
 public:
  uint16_t half_;

  static TVM_XINLINE half Binary(uint16_t value) {
    half res;
    res.half_ = value;
    return res;
  }

  TVM_XINLINE half() {}

  TVM_XINLINE half(const float& value) { constructor(value); }
  TVM_XINLINE explicit half(const double& value) { constructor(value); }
  TVM_XINLINE explicit half(const int8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const int32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const long long& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint64_t& value) { constructor(value); }

  TVM_XINLINE operator float() const {                          \
    return float(half2float(half_));                            \
  }                                                             \
  TVM_XINLINE operator float() const volatile {                 \
    return float(half2float(half_));                            \
  }


  TVM_HALF_ASSIGNOP(+=, +)
  TVM_HALF_ASSIGNOP(-=, -)
  TVM_HALF_ASSIGNOP(*=, *)
  TVM_HALF_ASSIGNOP(/=, /)

  TVM_XINLINE half operator+() {
    return *this;
  }

  TVM_XINLINE half operator-() {
    return half(-float(*this));
  }

  TVM_XINLINE half operator=(const half& a) {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) {
    return *this = half(a);
  }

  TVM_XINLINE half operator=(const half& a) volatile {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) volatile {
    return *this = half(a);
  }

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };

  static int const fp16FractionBits = 10;
  static int const fp32FractionBits = 23;
  static int32_t const fp32FractionMask = ~(~0u << fp32FractionBits);   // == 0x7fffff
  static int32_t const fp32HiddenBit = 1 << fp32FractionBits;   // == 0x800000
  static int const shift = fp32FractionBits - fp16FractionBits;   // == 13
  static int const shiftSign = 16;
  static int32_t const expAdjust = 127 - 15;   // exp32-127 = exp16-15, so exp16 = exp32 - (127-15)

  static int32_t const infN = 0x7F800000;   // flt32 infinity
  static int32_t const maxN = 0x477FFFFF;   // max flt32 that's a flt16 normal after >> by shift
  static int32_t const minN = 0x38800000;   // min flt16 normal as a flt32
  static int32_t const maxZ = 0x33000000;   // max fp32 number that's still rounded to zero in fp16
  static int32_t const signN = 0x80000000;  // flt32 sign bit

  static int32_t const infC = infN >> shift;
  static int32_t const nanN = (infC + 1) << shift;   // minimum flt16 nan as a flt32
  static int32_t const maxC = maxN >> shift;
  static int32_t const minC = minN >> shift;
  static int32_t const signC = signN >> shiftSign;  // flt16 sign bit

  static int32_t const mulN = 0x52000000;  // (1 << 23) / minN
  static int32_t const mulC = 0x33800000;  // minN / (1 << (23 - shift))

  static int32_t const subC = 0x003FF;  // max flt32 subnormal down shifted
  static int32_t const norC = 0x00400;  // min flt32 normal down shifted

  static int32_t const maxD = infC - maxC - 1;
  static int32_t const minD = minC - subC - 1;

  TVM_XINLINE uint16_t float2half(const float& value) const {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  // Same as above routine, except for addition of volatile keyword
  TVM_XINLINE uint16_t float2half(
    const volatile float& value) const volatile {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  TVM_XINLINE float half2float(const uint16_t& value) const {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  TVM_XINLINE float half2float(
    const volatile uint16_t& value) const volatile {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  template<typename T>
  TVM_XINLINE void constructor(const T& value) {
    half_ = float2half(float(value));
  }
};

TVM_HALF_OPERATOR(half, +)
TVM_HALF_OPERATOR(half, -)
TVM_HALF_OPERATOR(half, *)
TVM_HALF_OPERATOR(half, /)
TVM_HALF_OPERATOR(bool, >)
TVM_HALF_OPERATOR(bool, <)
TVM_HALF_OPERATOR(bool, >=)
TVM_HALF_OPERATOR(bool, <=)

TVM_XINLINE half __float2half_rn(const float a) {
  return half(a);
}
#endif


// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Some fp16 math functions are not supported in cuda_fp16.h,
// so we define them here to make sure the generated CUDA code
// is valid.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x, half y) {   \
  float tmp_x = __half2float(x);                                          \
  float tmp_y = __half2float(y);                                          \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2half(result);                                            \
}

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x) {          \
  float tmp_x = __half2float(x);                                         \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2half(result);                                           \
}

CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY

#endif

__device__ void decode_i4s_to_f16(int *i4s, half* B_local_decode, const int N = 8) {
  uint* h = reinterpret_cast<uint*>(B_local_decode);
  
  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint BOTTOM_MASK = 0x000f000f;
  static constexpr uint TOP_MASK = 0x00f000f0;
  static constexpr uint I4s_TO_F16s_MAGIC_NUM = 0x64006400;
  static constexpr uint FP16_TOP_MAGIC_NUM = 0x64086408;
  static constexpr uint ONE_SIXTEENTH = 0x2c002c00;
  static constexpr uint NEG_72 = 0xd480d480;

  uint const top_i4s = (*i4s) >> 8;
  // Decoding operations using inline PTX
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                : "=r"(h[0])
                : "r"(*i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                : "=r"(h[1])
                : "r"(*i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                : "=r"(h[2])
                : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                : "=r"(h[3])
                : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_72));
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_72));
}
  
#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800) 
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 1
#else
#define TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using ushort = unsigned short;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif

template <typename T1, typename T2>
__device__ void decode_i1s_to_i8s(T1 *_i1s, T2 *_i8s, const int N = 32)
{
  uint *i8s = reinterpret_cast<uint *>(_i8s);
  uint const i1s = *reinterpret_cast<uint *>(_i1s);

  // First, we extract the i4s and construct an intermediate fp16 number.
  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;      // 0b11101010
  static constexpr uint BOTTOM_MASK = 0x01010101;           // 0xf -> 0b01 select 0,1
  static constexpr uint I4s_TO_I8s_MAGIC_NUM = 0x00000000; // 1024

  for (int i = 0; i < N; i++){
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                : "=r"(i8s[i])
                : "r"(i1s >> i), "n"(BOTTOM_MASK), "n"(I4s_TO_I8s_MAGIC_NUM), "n"(immLut));
  }
}

template <typename T1, typename T2>
__device__ void decode_i2s_to_i8s(T1 *_i2s, T2 *_i8s, const int N = 16)
{
  // convert 8 int2b_t to 8 int8b_t -> 2 int32
  uint *i8s = reinterpret_cast<uint *>(_i8s);

  // i2s = {e7,e6,e5,e4,e3,e2,e1,e0}
  // also require interleave {e7,e3,e6,e2,e5,e1,e4,e0}
  uint const i2s = *_i2s;

  // First, we extract the i4s and construct an intermediate fp16 number.
  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;      // 0b11101010
  static constexpr uint BOTTOM_MASK = 0x03030303;           // 0xf -> 0b11 select 0,3
  static constexpr uint I4s_TO_I8s_MAGIC_NUM = 0x00000000; // 1024
  
  #pragma unroll
  for (int i = 0; i < (N / 2); i++)
  {
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(i8s[i])
                 : "r"(i2s >> (2 * i)), "n"(BOTTOM_MASK), "n"(I4s_TO_I8s_MAGIC_NUM), "n"(immLut));
  }
}

template <typename T1, typename T2>
__device__ void decode_i4s_to_i8s(T1 *_i4s, T2 *_i8s, const int N = 8)
{
  uint *i8s = reinterpret_cast<uint *>(_i8s);
  uint const i4s = *_i4s;

  // First, we extract the i4s and construct an intermediate fp16 number.
  static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa;      // 0b11101010
  static constexpr uint BOTTOM_MASK = 0x0f0f0f0f;           // 0xf -> 0b1111 select 0,4
  static constexpr uint I4s_TO_I8s_MAGIC_NUM = 0x00000000; // 1024
  
  #pragma unroll
  for (int i = 0; i < (N / 4); i++)
  {
    // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(i8s[i])
               : "r"(i4s >> (4 * i)), "n"(BOTTOM_MASK), "n"(I4s_TO_I8s_MAGIC_NUM), "n"(immLut));
  }
}
  
  extern "C" __global__ void __launch_bounds__(128) main_kernel0(half* __restrict__ A, half* __restrict__ B, float* __restrict__ C) {
  float C_warp[128];
  __shared__ half A_shared[8192];
  __shared__ half B_shared[8192];
  half A_shared_warp[32];
  half B_shared_warp[32];
  half A_shared_warp_1[32];
  half B_shared_warp_1[32];

  const int MAX_BLOCK_N = 10;
  const auto baseBlockIdx = blockIdx.x + gridDim.x *blockIdx.y;
  const auto totalPanel = (gridDim.x * gridDim.y +MAX_BLOCK_N * gridDim.x - 1) / (MAX_BLOCK_N * gridDim.x);
  const auto totalBlock = gridDim.x * gridDim.y;
  const auto panelIdx = baseBlockIdx / (MAX_BLOCK_N *gridDim.x);
  const auto strideLd = panelIdx + 1 < totalPanel ?MAX_BLOCK_N : (totalBlock - panelIdx * (MAX_BLOCK_N *gridDim.x)) / gridDim.x;
  const auto bx = (panelIdx & 1) ? gridDim.x -(baseBlockIdx - panelIdx * MAX_BLOCK_N * gridDim.x) /strideLd - 1 : (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) / strideLd;
  const auto by = (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) % strideLd + panelIdx * MAX_BLOCK_N;
  const auto bz = blockIdx.z;
  const dim3 blockIdx(bx, by, bz);
  
  for (int ii_2_init = 0; ii_2_init < 4; ++ii_2_init) {
    for (int jj_2_init = 0; jj_2_init < 4; ++jj_2_init) {
      for (int i = 0; i < 8; ++i) {
C_warp[((ii_2_init * 32) + (jj_2_init * 8)) + i] = 0.0;}
;
    }
  }
  for (int ax0_ax1_ax2_ax3_fused_2 = 0; ax0_ax1_ax2_ax3_fused_2 < 4; ++ax0_ax1_ax2_ax3_fused_2) {
    *(uint4*)(A_shared + ((((((int)threadIdx.y) * 2048) + (((int)threadIdx.z) * 1024)) + (ax0_ax1_ax2_ax3_fused_2 * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(A + ((((((((int)blockIdx.y) * 2097152) + (((int)threadIdx.y) * 1048576)) + (((int)threadIdx.z) * 524288)) + ((ax0_ax1_ax2_ax3_fused_2 >> 1) * 262144)) + ((ax0_ax1_ax2_ax3_fused_2 & 1) * 256)) + (((int)threadIdx.x) * 8)));
  }
  for (int ax0_ax1_ax2_ax3_fused_2_1 = 0; ax0_ax1_ax2_ax3_fused_2_1 < 4; ++ax0_ax1_ax2_ax3_fused_2_1) {
    *(uint4*)(B_shared + ((((((int)threadIdx.y) * 2048) + (((int)threadIdx.z) * 1024)) + (ax0_ax1_ax2_ax3_fused_2_1 * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(B + ((((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.y) * 1048576)) + (((int)threadIdx.z) * 524288)) + ((ax0_ax1_ax2_ax3_fused_2_1 >> 1) * 262144)) + ((ax0_ax1_ax2_ax3_fused_2_1 & 1) * 256)) + (((int)threadIdx.x) * 8)));
  }
  for (int kk_0 = 0; kk_0 < 511; ++kk_0) {
    __syncthreads();
    for (int ax0_ax1_ax2_ax3_fused_2_2 = 0; ax0_ax1_ax2_ax3_fused_2_2 < 4; ++ax0_ax1_ax2_ax3_fused_2_2) {
      *(uint4*)(A_shared + (((((((kk_0 + 1) & 1) * 4096) + (((int)threadIdx.y) * 2048)) + (((int)threadIdx.z) * 1024)) + (ax0_ax1_ax2_ax3_fused_2_2 * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(A + ((((((((((int)blockIdx.y) * 2097152) + (((int)threadIdx.y) * 1048576)) + (((int)threadIdx.z) * 524288)) + ((ax0_ax1_ax2_ax3_fused_2_2 >> 1) * 262144)) + (kk_0 * 512)) + ((ax0_ax1_ax2_ax3_fused_2_2 & 1) * 256)) + (((int)threadIdx.x) * 8)) + 512));
    }
    for (int ax0_ax1_ax2_ax3_fused_2_3 = 0; ax0_ax1_ax2_ax3_fused_2_3 < 4; ++ax0_ax1_ax2_ax3_fused_2_3) {
      *(uint4*)(B_shared + (((((((kk_0 + 1) & 1) * 4096) + (((int)threadIdx.y) * 2048)) + (((int)threadIdx.z) * 1024)) + (ax0_ax1_ax2_ax3_fused_2_3 * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(B + ((((((((((int)blockIdx.x) * 2097152) + (((int)threadIdx.y) * 1048576)) + (((int)threadIdx.z) * 524288)) + ((ax0_ax1_ax2_ax3_fused_2_3 >> 1) * 262144)) + (kk_0 * 512)) + ((ax0_ax1_ax2_ax3_fused_2_3 & 1) * 256)) + (((int)threadIdx.x) * 8)) + 512));
    }
    __syncthreads();
    for (int kk_1 = 0; kk_1 < 2; ++kk_1) {
      for (int ax0 = 0; ax0 < 4; ++ax0) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(A_shared[(((((kk_0 & 1) * 4096) + (((int)threadIdx.y) * 2048)) + (ax0 * 512)) + (kk_1 * 256))])) + (((int)threadIdx.x) * 8))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_shared[(((((kk_0 & 1) * 4096) + (((int)threadIdx.y) * 2048)) + (ax0 * 512)) + (kk_1 * 256))])) + (((int)threadIdx.x) * 8)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_shared_warp + (ax0 * 8)))[0]), "=r"(((unsigned *)(A_shared_warp + (ax0 * 8)))[1]), "=r"(((unsigned *)(A_shared_warp + (ax0 * 8)))[2]), "=r"(((unsigned *)(A_shared_warp + (ax0 * 8)))[3])
      : "r"(addr)
    );
  }
      }
      for (int ax0_1 = 0; ax0_1 < 4; ++ax0_1) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(B_shared[(((((kk_0 & 1) * 4096) + (((int)threadIdx.z) * 2048)) + (ax0_1 * 512)) + (kk_1 * 256))])) + (((int)threadIdx.x) * 8))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(B_shared[(((((kk_0 & 1) * 4096) + (((int)threadIdx.z) * 2048)) + (ax0_1 * 512)) + (kk_1 * 256))])) + (((int)threadIdx.x) * 8)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(B_shared_warp + (ax0_1 * 8)))[0]), "=r"(((unsigned *)(B_shared_warp + (ax0_1 * 8)))[1]), "=r"(((unsigned *)(B_shared_warp + (ax0_1 * 8)))[2]), "=r"(((unsigned *)(B_shared_warp + (ax0_1 * 8)))[3])
      : "r"(addr)
    );
  }
      }
      for (int ii_2 = 0; ii_2 < 4; ++ii_2) {
        for (int jj_2 = 0; jj_2 < 4; ++jj_2) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(C_warp + ((ii_2 * 32) + (jj_2 * 8))))[0]), "=f"(((float *)(C_warp + ((ii_2 * 32) + (jj_2 * 8))))[1]), "=f"(((float *)(C_warp + ((ii_2 * 32) + (jj_2 * 8))))[2]), "=f"(((float *)(C_warp + ((ii_2 * 32) + (jj_2 * 8))))[3])
      : "r"(((unsigned *)(A_shared_warp + (ii_2 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (ii_2 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (ii_2 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (ii_2 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + (jj_2 * 8)))[0]), "r"(((unsigned *)(B_shared_warp + (jj_2 * 8)))[1]), "f"(((float *)(C_warp + ((ii_2 * 32) + (jj_2 * 8))))[0]), "f"(((float *)(C_warp + ((ii_2 * 32) + (jj_2 * 8))))[1]), "f"(((float *)(C_warp + ((ii_2 * 32) + (jj_2 * 8))))[2]), "f"(((float *)(C_warp + ((ii_2 * 32) + (jj_2 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(C_warp + (((ii_2 * 32) + (jj_2 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((ii_2 * 32) + (jj_2 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((ii_2 * 32) + (jj_2 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((ii_2 * 32) + (jj_2 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_shared_warp + (ii_2 * 8)))[0]), "r"(((unsigned *)(A_shared_warp + (ii_2 * 8)))[1]), "r"(((unsigned *)(A_shared_warp + (ii_2 * 8)))[2]), "r"(((unsigned *)(A_shared_warp + (ii_2 * 8)))[3]), "r"(((unsigned *)(B_shared_warp + ((jj_2 * 8) + 4)))[0]), "r"(((unsigned *)(B_shared_warp + ((jj_2 * 8) + 4)))[1]), "f"(((float *)(C_warp + (((ii_2 * 32) + (jj_2 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((ii_2 * 32) + (jj_2 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((ii_2 * 32) + (jj_2 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((ii_2 * 32) + (jj_2 * 8)) + 4)))[3]));
  }
        }
      }
    }
  }
  for (int kk_1_1 = 0; kk_1_1 < 2; ++kk_1_1) {
    for (int ax0_2 = 0; ax0_2 < 4; ++ax0_2) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(A_shared[((((((int)threadIdx.y) * 2048) + (ax0_2 * 512)) + (kk_1_1 * 256)) + 4096)])) + (((int)threadIdx.x) * 8))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(A_shared[((((((int)threadIdx.y) * 2048) + (ax0_2 * 512)) + (kk_1_1 * 256)) + 4096)])) + (((int)threadIdx.x) * 8)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(A_shared_warp_1 + (ax0_2 * 8)))[0]), "=r"(((unsigned *)(A_shared_warp_1 + (ax0_2 * 8)))[1]), "=r"(((unsigned *)(A_shared_warp_1 + (ax0_2 * 8)))[2]), "=r"(((unsigned *)(A_shared_warp_1 + (ax0_2 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ax0_3 = 0; ax0_3 < 4; ++ax0_3) {

  {
    unsigned int addr;
#if TVM_ENBALE_EFFICIENT_SMEM_PTR_CAST
    addr = static_cast<unsigned int>(__cvta_generic_to_shared((void *)((&(B_shared[((((((int)threadIdx.z) * 2048) + (ax0_3 * 512)) + (kk_1_1 * 256)) + 4096)])) + (((int)threadIdx.x) * 8))));
#else
    __asm__ __volatile__(
      "{ .reg .u64 addr; cvta.to.shared.u64 addr, %1; cvt.u32.u64 %0, addr; }\n"
      : "=r"(addr)
      : "l"((void *)((&(B_shared[((((((int)threadIdx.z) * 2048) + (ax0_3 * 512)) + (kk_1_1 * 256)) + 4096)])) + (((int)threadIdx.x) * 8)))
    );
#endif
    __asm__ __volatile__(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(((unsigned *)(B_shared_warp_1 + (ax0_3 * 8)))[0]), "=r"(((unsigned *)(B_shared_warp_1 + (ax0_3 * 8)))[1]), "=r"(((unsigned *)(B_shared_warp_1 + (ax0_3 * 8)))[2]), "=r"(((unsigned *)(B_shared_warp_1 + (ax0_3 * 8)))[3])
      : "r"(addr)
    );
  }
    }
    for (int ii_2_1 = 0; ii_2_1 < 4; ++ii_2_1) {
      for (int jj_2_1 = 0; jj_2_1 < 4; ++jj_2_1) {

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(C_warp + ((ii_2_1 * 32) + (jj_2_1 * 8))))[0]), "=f"(((float *)(C_warp + ((ii_2_1 * 32) + (jj_2_1 * 8))))[1]), "=f"(((float *)(C_warp + ((ii_2_1 * 32) + (jj_2_1 * 8))))[2]), "=f"(((float *)(C_warp + ((ii_2_1 * 32) + (jj_2_1 * 8))))[3])
      : "r"(((unsigned *)(A_shared_warp_1 + (ii_2_1 * 8)))[0]), "r"(((unsigned *)(A_shared_warp_1 + (ii_2_1 * 8)))[1]), "r"(((unsigned *)(A_shared_warp_1 + (ii_2_1 * 8)))[2]), "r"(((unsigned *)(A_shared_warp_1 + (ii_2_1 * 8)))[3]), "r"(((unsigned *)(B_shared_warp_1 + (jj_2_1 * 8)))[0]), "r"(((unsigned *)(B_shared_warp_1 + (jj_2_1 * 8)))[1]), "f"(((float *)(C_warp + ((ii_2_1 * 32) + (jj_2_1 * 8))))[0]), "f"(((float *)(C_warp + ((ii_2_1 * 32) + (jj_2_1 * 8))))[1]), "f"(((float *)(C_warp + ((ii_2_1 * 32) + (jj_2_1 * 8))))[2]), "f"(((float *)(C_warp + ((ii_2_1 * 32) + (jj_2_1 * 8))))[3]));
  }

  {
    __asm__ __volatile__(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
      :  "=f"(((float *)(C_warp + (((ii_2_1 * 32) + (jj_2_1 * 8)) + 4)))[0]), "=f"(((float *)(C_warp + (((ii_2_1 * 32) + (jj_2_1 * 8)) + 4)))[1]), "=f"(((float *)(C_warp + (((ii_2_1 * 32) + (jj_2_1 * 8)) + 4)))[2]), "=f"(((float *)(C_warp + (((ii_2_1 * 32) + (jj_2_1 * 8)) + 4)))[3])
      : "r"(((unsigned *)(A_shared_warp_1 + (ii_2_1 * 8)))[0]), "r"(((unsigned *)(A_shared_warp_1 + (ii_2_1 * 8)))[1]), "r"(((unsigned *)(A_shared_warp_1 + (ii_2_1 * 8)))[2]), "r"(((unsigned *)(A_shared_warp_1 + (ii_2_1 * 8)))[3]), "r"(((unsigned *)(B_shared_warp_1 + ((jj_2_1 * 8) + 4)))[0]), "r"(((unsigned *)(B_shared_warp_1 + ((jj_2_1 * 8) + 4)))[1]), "f"(((float *)(C_warp + (((ii_2_1 * 32) + (jj_2_1 * 8)) + 4)))[0]), "f"(((float *)(C_warp + (((ii_2_1 * 32) + (jj_2_1 * 8)) + 4)))[1]), "f"(((float *)(C_warp + (((ii_2_1 * 32) + (jj_2_1 * 8)) + 4)))[2]), "f"(((float *)(C_warp + (((ii_2_1 * 32) + (jj_2_1 * 8)) + 4)))[3]));
  }
      }
    }
  }
  for (int ax0_4 = 0; ax0_4 < 4; ++ax0_4) {
    for (int ax1 = 0; ax1 < 4; ++ax1) {
      for (int local_id = 0; local_id < 8; ++local_id) {
(&(C[((((((((int)blockIdx.y) * 2097152) + (((int)threadIdx.y) * 1048576)) + (ax0_4 * 262144)) + (((int)blockIdx.x) * 2048)) + (((int)threadIdx.z) * 1024)) + (ax1 * 256))]))[((((((local_id % 4) / 2) * 8) + (threadIdx.x / 4)) * 16) + ((((local_id / 4) * 8) + ((threadIdx.x % 4) * 2)) + (local_id % 2)))] = C_warp[((ax0_4 * 32) + (ax1 * 8)) + local_id];
}
;
    }
  }
}

