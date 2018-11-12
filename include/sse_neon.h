//
// Created by ubt on 2017/9/14.
//

#ifndef ACFDETECTION_SSE_H
#define ACFDETECTION_SSE_H


#define ENABLE_CPP_VERSION 0

#if defined(__GNUC__) || defined(__clang__)
#	pragma push_macro("FORCE_INLINE")
#	pragma push_macro("ALIGN_STRUCT")
#	define FORCE_INLINE       static inline __attribute__((always_inline))
#	define ALIGN_STRUCT(x)    __attribute__((aligned(x)))
#else
#	error "Macro name collisions may happens with unknown compiler"
#	define FORCE_INLINE       static inline
#	define ALIGN_STRUCT(x)    __declspec(align(x))
#endif

#include <stdint.h>
#include "arm_neon.h"


/*******************************************************/
/* MACRO for shuffle parameter for _mm_shuffle_ps().   */
/* Argument fp3 is a digit[0123] that represents the fp*/
/* from argument "b" of mm_shuffle_ps that will be     */
/* placed in fp3 of result. fp2 is the same for fp2 in */
/* result. fp1 is a digit[0123] that represents the fp */
/* from argument "a" of mm_shuffle_ps that will be     */
/* places in fp1 of result. fp0 is the same for fp0 of */
/* result                                              */
/*******************************************************/
#define _MM_SHUFFLE(fp3,fp2,fp1,fp0) \
	(((fp3) << 6) | ((fp2) << 4) | ((fp1) << 2) | ((fp0)))

/* indicate immediate constant argument in a given range */
#define __constrange(a,b) \
	const

typedef float32x4_t __m128;
typedef int32x4_t __m128i;

#define RETf inline __m128
#define RETi inline __m128i


// ******************************************
// type-safe casting between types
// ******************************************
#define vreinterpretq_m128_f32(x) \
	(x)

#define vreinterpretq_m128_u32(x) \
	vreinterpretq_f32_u32(x)

#define vreinterpretq_m128_s32(x) \
	vreinterpretq_f32_s32(x)

#define vreinterpretq_f32_m128(x) \
	(x)

#define vreinterpretq_s32_m128(x) \
	vreinterpretq_s32_f32(x)

#define vreinterpretq_m128i_s32(x) \
	(x)

#define vreinterpretq_m128i_u32(x) \
	vreinterpretq_s32_u32(x)

#define vreinterpretq_s32_m128i(x) \
	(x)

//第一部分，赋值---载入---存储
// Sets the four single-precision, floating-point values to w. // 第一个  正确
FORCE_INLINE __m128 _mm_set1_ps(float _w)
{
    return vreinterpretq_m128_f32(vdupq_n_f32(_w));
}
RETf SET(const float& x) { return _mm_set1_ps(x); }

// Sets the four single-precision, floating-point values to the four inputs. //第二个 正确
FORCE_INLINE __m128 _mm_set_ps(float w, float z, float y, float x)
{
    float __attribute__((aligned(16))) data[4] = { x, y, z, w };
    return vreinterpretq_m128_f32(vld1q_f32(data));
}
RETf SET(float x, float y, float z, float w) { return _mm_set_ps(x, y, z, w); }


// Sets the 4 signed 32-bit integer values to i.  //第三个  正确
FORCE_INLINE __m128i _mm_set1_epi32(int _i)
{
    return vreinterpretq_m128i_s32(vdupq_n_s32(_i));
}
RETi SET(const int& x) { return _mm_set1_epi32(x); }

// Loads four single-precision, floating-point values. //第四个 正确
FORCE_INLINE __m128 _mm_load_ps(const float * p)
{
    return vreinterpretq_m128_f32(vld1q_f32(p));
}
RETf LD(const float& x) { return _mm_load_ps(&x); }

// Loads four single-precision, floating-point values. //第五个 正确
FORCE_INLINE __m128 _mm_loadu_ps(const float * p)
{
    // for neon, alignment doesn't matter, so _mm_load_ps and _mm_loadu_ps are equivalent for neon
    return vreinterpretq_m128_f32(vld1q_f32(p));
}
RETf LDu(const float& x) { return _mm_loadu_ps(&x); }

// Stores four single-precision, floating-point values. //第六个 正确
FORCE_INLINE void _mm_store_ps(float *p, __m128 a)
{
    vst1q_f32(p, vreinterpretq_f32_m128(a));
}
RETf STR(float& x, const __m128 y)
{
    _mm_store_ps(&x, y);
    return y;
}

// Stores the lower single - precision, floating - point value. //第七个  代码中没有用到
FORCE_INLINE void _mm_store_ss(float *p, __m128 a)
{
    vst1q_lane_f32(p, vreinterpretq_f32_m128(a), 0);
}
RETf STR1(float& x, const __m128 y)
{
    _mm_store_ss(&x, y);
    return y;
}


// Stores four single-precision, floating-point values. //第八个  正确
FORCE_INLINE void _mm_storeu_ps(float *p, __m128 a)
{
    vst1q_f32(p, vreinterpretq_f32_m128(a));
}
RETf STRu(float& x, const __m128 y)
{
    _mm_storeu_ps(&x, y);
    return y;
}

RETf STR(float& x, const float y) { return STR(x, SET(y)); }



//第二部分  数学运算
// Adds the 4 signed or unsigned 32-bit integers in a to the 4 signed or unsigned 32-bit integers in b. //第九个
FORCE_INLINE __m128i _mm_add_epi32(__m128i a, __m128i b)
{
    return vreinterpretq_m128i_s32(vaddq_s32(vreinterpretq_s32_m128i(a), vreinterpretq_s32_m128i(b)));
}
RETi ADD(const __m128i x, const __m128i y) { return _mm_add_epi32(x, y); }

// Adds the four single-precision, floating-point values of a and b. //第十个
FORCE_INLINE __m128 _mm_add_ps(__m128 a, __m128 b)
{
    return vreinterpretq_m128_f32(vaddq_f32(vreinterpretq_f32_m128(a), vreinterpretq_f32_m128(b)));
}
RETf ADD(const __m128 x, const __m128 y) { return _mm_add_ps(x, y); }

RETf ADD(const __m128 x, const __m128 y, const __m128 z)
{
    return ADD(ADD(x, y), z);
}

RETf ADD(const __m128 a, const __m128 b, const __m128 c, const __m128& d)
{
    return ADD(ADD(ADD(a, b), c), d);
}

// Subtracts the four single-precision, floating-point values of a and b. //第十一个
FORCE_INLINE __m128 _mm_sub_ps(__m128 a, __m128 b)
{
    return vreinterpretq_m128_f32(vsubq_f32(vreinterpretq_f32_m128(a), vreinterpretq_f32_m128(b)));
}
RETf SUB(const __m128 x, const __m128 y) { return _mm_sub_ps(x, y); }

// Multiplies the four single-precision, floating-point values of a and b.//第十二个
FORCE_INLINE __m128 _mm_mul_ps(__m128 a, __m128 b)
{
    return vreinterpretq_m128_f32(vmulq_f32(vreinterpretq_f32_m128(a), vreinterpretq_f32_m128(b)));
}
RETf MUL(const __m128 x, const __m128 y) { return _mm_mul_ps(x, y); }

RETf MUL(const __m128 x, const float y) { return MUL(x, SET(y)); }

RETf MUL(const float x, const __m128 y) { return MUL(SET(x), y); }

RETf INC(__m128& x, const __m128 y) { return x = ADD(x, y); }

RETf INC(float& x, const __m128 y)
{
    __m128 t = ADD(LD(x), y);
    return STR(x, t);
}

RETf DEC(__m128& x, const __m128 y) { return x = SUB(x, y); }

RETf DEC(float& x, const __m128 y)
{
    __m128 t = SUB(LD(x), y);
    return STR(x, t);
}

// Computes the minima of the four single-precision, floating-point values of a and b.  //第十三个
FORCE_INLINE __m128 _mm_min_ps(__m128 a, __m128 b)
{
    return vreinterpretq_m128_f32(vminq_f32(vreinterpretq_f32_m128(a), vreinterpretq_f32_m128(b)));
}
RETf MINF(const __m128 x, const __m128 y) { return _mm_min_ps(x, y); }

// Computes the approximations of reciprocals of the four single-precision, floating-point values of a. //第十四个
FORCE_INLINE __m128 _mm_rcp_ps(__m128 in)
{
    float32x4_t recip = vrecpeq_f32(vreinterpretq_f32_m128(in));
    recip = vmulq_f32(recip, vrecpsq_f32(recip, vreinterpretq_f32_m128(in)));
    return vreinterpretq_m128_f32(recip);
}
RETf RCP(const __m128 x) { return _mm_rcp_ps(x); }

// Computes the approximations of the reciprocal square roots of the four single-precision floating point values of in. //第十五个
FORCE_INLINE __m128 _mm_rsqrt_ps(__m128 in)
{
    return vreinterpretq_m128_f32(vrsqrteq_f32(vreinterpretq_f32_m128(in)));
}
RETf RCPSQRT(const __m128 x) { return _mm_rsqrt_ps(x); }


//第三部分  逻辑运算
// Computes the bitwise AND of the four single-precision, floating-point values of a and b.  //第十六个
FORCE_INLINE __m128 _mm_and_ps(__m128 a, __m128 b)
{
    return vreinterpretq_m128_s32( vandq_s32(vreinterpretq_s32_m128(a), vreinterpretq_s32_m128(b)) );
    //return reinterpret_cast<__m128>(vandq_s32(reinterpret_cast<int32x4_t>(a), reinterpret_cast<int32x4_t>(b)));
}
RETf AND(const __m128 x, const __m128 y) { return _mm_and_ps(x, y); }

// Computes the bitwise AND of the 128-bit value in a and the 128-bit value in b.  //第十七个
FORCE_INLINE __m128i _mm_and_si128(__m128i a, __m128i b)
{
    return vreinterpretq_m128i_s32( vandq_s32(vreinterpretq_s32_m128i(a), vreinterpretq_s32_m128i(b)) );
    //return vandq_s32(a, b);
}
RETi AND(const __m128i x, const __m128i y) { return _mm_and_si128(x, y); }

// Computes the bitwise AND-NOT of the four single-precision, floating-point values of a and b. //第十八个
FORCE_INLINE __m128 _mm_andnot_ps(__m128 a, __m128 b)
{
    return vreinterpretq_m128_s32( vbicq_s32(vreinterpretq_s32_m128(b), vreinterpretq_s32_m128(a)) ); // *NOTE* argument swap
    //return reinterpret_cast<__m128>(vbicq_s32(reinterpret_cast<int32x4_t>(b), reinterpret_cast<int32x4_t>(a)));
}
RETf ANDNOT(const __m128 x, const __m128 y) { return _mm_andnot_ps(x, y); }

// Computes the bitwise OR of the four single-precision, floating-point values of a and b. //第十九个
FORCE_INLINE __m128 _mm_or_ps(__m128 a, __m128 b)
{
    return vreinterpretq_m128_s32( vorrq_s32(vreinterpretq_s32_m128(a), vreinterpretq_s32_m128(b)) );
}
RETf OR(const __m128 x, const __m128 y) { return _mm_or_ps(x, y); }

// Computes bitwise EXOR (exclusive-or) of the four single-precision, floating-point values of a and b. //第二十个
FORCE_INLINE __m128 _mm_xor_ps(__m128 a, __m128 b)
{
    return vreinterpretq_m128_s32( veorq_s32(vreinterpretq_s32_m128(a), vreinterpretq_s32_m128(b)) );
}
RETf XOR(const __m128 x, const __m128 y) { return _mm_xor_ps(x, y); }



//第四部分  大小比较运算
// Compares for greater than. //第二十一个
FORCE_INLINE __m128 _mm_cmpgt_ps(__m128 a, __m128 b)
{
    return vreinterpretq_m128_u32(vcgtq_f32(vreinterpretq_f32_m128(a), vreinterpretq_f32_m128(b)));
}
RETf CMPGT(const __m128 x, const __m128 y) { return _mm_cmpgt_ps(x, y); }

// Compares for less than //第二十二个
FORCE_INLINE __m128 _mm_cmplt_ps(__m128 a, __m128 b)
{
    return vreinterpretq_m128_u32(vcltq_f32(vreinterpretq_f32_m128(a), vreinterpretq_f32_m128(b)));
}
RETf CMPLT(const __m128 x, const __m128 y) { return _mm_cmplt_ps(x, y); }

// Compares the 4 signed 32-bit integers in a and the 4 signed 32-bit integers in b for greater than. //第二十三个
FORCE_INLINE __m128i _mm_cmpgt_epi32(__m128i a, __m128i b)
{
    return vreinterpretq_m128i_u32(vcgtq_s32(vreinterpretq_s32_m128i(a), vreinterpretq_s32_m128i(b)));
}
RETi CMPGT(const __m128i x, const __m128i y) { return _mm_cmpgt_epi32(x, y); }

// Compares the 4 signed 32-bit integers in a and the 4 signed 32-bit integers in b for less than. //第二十四个
FORCE_INLINE __m128i _mm_cmplt_epi32(__m128i a, __m128i b)
{
    return vreinterpretq_m128i_u32(vcltq_s32(vreinterpretq_s32_m128i(a), vreinterpretq_s32_m128i(b)));
}
RETi CMPLT(const __m128i x, const __m128i y) { return _mm_cmplt_epi32(x, y); }


//第五部分  转换运算
// Converts the four signed 32-bit integer values of a to single-precision, floating-point values  //第二十五个
FORCE_INLINE __m128 _mm_cvtepi32_ps(__m128i a)
{
    return vreinterpretq_m128_f32(vcvtq_f32_s32(vreinterpretq_s32_m128i(a)));
}
RETf CVT(const __m128i x) { return _mm_cvtepi32_ps(x); }

// Converts the four single-precision, floating-point values of a to signed 32-bit integer values using truncate. //第二十六个
FORCE_INLINE __m128i _mm_cvttps_epi32(__m128 a)
{
    return vreinterpretq_m128i_s32(vcvtq_s32_f32(vreinterpretq_f32_m128(a)));
}
RETi CVT(const __m128 x) { return _mm_cvttps_epi32(x); }


//第六部分  shuffle
// Takes the upper 64 bits of a and places it in the low end of the result
// Takes the lower 64 bits of b and places it into the high end of the result.
FORCE_INLINE __m128 _mm_shuffle_ps_1032(__m128 a, __m128 b)
{
    float32x2_t a32 = vget_high_f32(vreinterpretq_f32_m128(a));
    float32x2_t b10 = vget_low_f32(vreinterpretq_f32_m128(b));
    return vreinterpretq_m128_f32(vcombine_f32(a32, b10));
}

// takes the lower two 32-bit values from a and swaps them and places in high end of result
// takes the higher two 32 bit values from b and swaps them and places in low end of result.
FORCE_INLINE __m128 _mm_shuffle_ps_2301(__m128 a, __m128 b)
{
    float32x2_t a01 = vrev64_f32(vget_low_f32(vreinterpretq_f32_m128(a)));
    float32x2_t b23 = vrev64_f32(vget_high_f32(vreinterpretq_f32_m128(b)));
    return vreinterpretq_m128_f32(vcombine_f32(a01, b23));
}

FORCE_INLINE __m128 _mm_shuffle_ps_0321(__m128 a, __m128 b)
{
    float32x2_t a21 = vget_high_f32(vextq_f32(vreinterpretq_f32_m128(a), vreinterpretq_f32_m128(a), 3));
    float32x2_t b03 = vget_low_f32(vextq_f32(vreinterpretq_f32_m128(b), vreinterpretq_f32_m128(b), 3));
    return vreinterpretq_m128_f32(vcombine_f32(a21, b03));
}

FORCE_INLINE __m128 _mm_shuffle_ps_2103(__m128 a, __m128 b)
{
    float32x2_t a03 = vget_low_f32(vextq_f32(vreinterpretq_f32_m128(a), vreinterpretq_f32_m128(a), 3));
    float32x2_t b21 = vget_high_f32(vextq_f32(vreinterpretq_f32_m128(b), vreinterpretq_f32_m128(b), 3));
    return vreinterpretq_m128_f32(vcombine_f32(a03, b21));
}

FORCE_INLINE __m128 _mm_shuffle_ps_1010(__m128 a, __m128 b)
{
    float32x2_t a10 = vget_low_f32(vreinterpretq_f32_m128(a));
    float32x2_t b10 = vget_low_f32(vreinterpretq_f32_m128(b));
    return vreinterpretq_m128_f32(vcombine_f32(a10, b10));
}

FORCE_INLINE __m128 _mm_shuffle_ps_1001(__m128 a, __m128 b)
{
    float32x2_t a01 = vrev64_f32(vget_low_f32(vreinterpretq_f32_m128(a)));
    float32x2_t b10 = vget_low_f32(vreinterpretq_f32_m128(b));
    return vreinterpretq_m128_f32(vcombine_f32(a01, b10));
}

FORCE_INLINE __m128 _mm_shuffle_ps_0101(__m128 a, __m128 b)
{
    float32x2_t a01 = vrev64_f32(vget_low_f32(vreinterpretq_f32_m128(a)));
    float32x2_t b01 = vrev64_f32(vget_low_f32(vreinterpretq_f32_m128(b)));
    return vreinterpretq_m128_f32(vcombine_f32(a01, b01));
}

// keeps the low 64 bits of b in the low and puts the high 64 bits of a in the high
FORCE_INLINE __m128 _mm_shuffle_ps_3210(__m128 a, __m128 b)
{
    float32x2_t a10 = vget_low_f32(vreinterpretq_f32_m128(a));
    float32x2_t b32 = vget_high_f32(vreinterpretq_f32_m128(b));
    return vreinterpretq_m128_f32(vcombine_f32(a10, b32));
}

FORCE_INLINE __m128 _mm_shuffle_ps_0011(__m128 a, __m128 b)
{
    float32x2_t a11 = vdup_lane_f32(vget_low_f32(vreinterpretq_f32_m128(a)), 1);
    float32x2_t b00 = vdup_lane_f32(vget_low_f32(vreinterpretq_f32_m128(b)), 0);
    return vreinterpretq_m128_f32(vcombine_f32(a11, b00));
}

FORCE_INLINE __m128 _mm_shuffle_ps_0022(__m128 a, __m128 b)
{
    float32x2_t a22 = vdup_lane_f32(vget_high_f32(vreinterpretq_f32_m128(a)), 0);
    float32x2_t b00 = vdup_lane_f32(vget_low_f32(vreinterpretq_f32_m128(b)), 0);
    return vreinterpretq_m128_f32(vcombine_f32(a22, b00));
}

FORCE_INLINE __m128 _mm_shuffle_ps_2200(__m128 a, __m128 b)
{
    float32x2_t a00 = vdup_lane_f32(vget_low_f32(vreinterpretq_f32_m128(a)), 0);
    float32x2_t b22 = vdup_lane_f32(vget_high_f32(vreinterpretq_f32_m128(b)), 0);
    return vreinterpretq_m128_f32(vcombine_f32(a00, b22));
}

FORCE_INLINE __m128 _mm_shuffle_ps_3202(__m128 a, __m128 b)
{
    float32_t a0 = vgetq_lane_f32(vreinterpretq_f32_m128(a), 0);
    float32x2_t a22 = vdup_lane_f32(vget_high_f32(vreinterpretq_f32_m128(a)), 0);
    float32x2_t a02 = vset_lane_f32(a0, a22, 1); /* apoty: TODO: use vzip ?*/
    float32x2_t b32 = vget_high_f32(vreinterpretq_f32_m128(b));
    return vreinterpretq_m128_f32(vcombine_f32(a02, b32));
}

FORCE_INLINE __m128 _mm_shuffle_ps_1133(__m128 a, __m128 b)
{
    float32x2_t a33 = vdup_lane_f32(vget_high_f32(vreinterpretq_f32_m128(a)), 1);
    float32x2_t b11 = vdup_lane_f32(vget_low_f32(vreinterpretq_f32_m128(b)), 1);
    return vreinterpretq_m128_f32(vcombine_f32(a33, b11));
}

FORCE_INLINE __m128 _mm_shuffle_ps_2010(__m128 a, __m128 b)
{
    float32x2_t a10 = vget_low_f32(vreinterpretq_f32_m128(a));
    float32_t b2 = vgetq_lane_f32(vreinterpretq_f32_m128(b), 2);
    float32x2_t b00 = vdup_lane_f32(vget_low_f32(vreinterpretq_f32_m128(b)), 0);
    float32x2_t b20 = vset_lane_f32(b2, b00, 1);
    return vreinterpretq_m128_f32(vcombine_f32(a10, b20));
}

FORCE_INLINE __m128 _mm_shuffle_ps_2001(__m128 a, __m128 b)
{
    float32x2_t a01 = vrev64_f32(vget_low_f32(vreinterpretq_f32_m128(a)));
    float32_t b2 = vgetq_lane_f32(b, 2);
    float32x2_t b00 = vdup_lane_f32(vget_low_f32(vreinterpretq_f32_m128(b)), 0);
    float32x2_t b20 = vset_lane_f32(b2, b00, 1);
    return vreinterpretq_m128_f32(vcombine_f32(a01, b20));
}

FORCE_INLINE __m128 _mm_shuffle_ps_2032(__m128 a, __m128 b)
{
    float32x2_t a32 = vget_high_f32(vreinterpretq_f32_m128(a));
    float32_t b2 = vgetq_lane_f32(b, 2);
    float32x2_t b00 = vdup_lane_f32(vget_low_f32(vreinterpretq_f32_m128(b)), 0);
    float32x2_t b20 = vset_lane_f32(b2, b00, 1);
    return vreinterpretq_m128_f32(vcombine_f32(a32, b20));
}

// NEON does not support a general purpose permute intrinsic
// Currently I am not sure whether the C implementation is faster or slower than the NEON version.
// Note, this has to be expanded as a template because the shuffle value must be an immediate value.
// The same is true on SSE as well.
// Selects four specific single-precision, floating-point values from a and b, based on the mask i.
#if ENABLE_CPP_VERSION // I am not convinced that the NEON version is faster than the C version yet.
FORCE_INLINE __m128 _mm_shuffle_ps_default(__m128 a, __m128 b, __constrange(0,255) int imm)
{
	__m128 ret;
	ret[0] = a[imm & 0x3];
	ret[1] = a[(imm >> 2) & 0x3];
	ret[2] = b[(imm >> 4) & 0x03];
	ret[3] = b[(imm >> 6) & 0x03];
	return ret;
}
#else
#define _mm_shuffle_ps_default(a, b, imm) \
({ \
	float32x4_t ret; \
	ret = vmovq_n_f32(vgetq_lane_f32(vreinterpretq_f32_m128(a), (imm) & 0x3)); \
	ret = vsetq_lane_f32(vgetq_lane_f32(vreinterpretq_f32_m128(a), ((imm) >> 2) & 0x3), ret, 1); \
	ret = vsetq_lane_f32(vgetq_lane_f32(vreinterpretq_f32_m128(b), ((imm) >> 4) & 0x3), ret, 2); \
	ret = vsetq_lane_f32(vgetq_lane_f32(vreinterpretq_f32_m128(b), ((imm) >> 6) & 0x3), ret, 3); \
	vreinterpretq_m128_f32(ret); \
})
#endif

//FORCE_INLINE __m128 _mm_shuffle_ps(__m128 a, __m128 b, __constrange(0,255) int imm)
#define _mm_shuffle_ps(a, b, imm) \
({ \
	__m128 ret; \
	switch (imm) \
	{ \
		case _MM_SHUFFLE(1, 0, 3, 2): ret = _mm_shuffle_ps_1032((a), (b)); break; \
		case _MM_SHUFFLE(2, 3, 0, 1): ret = _mm_shuffle_ps_2301((a), (b)); break; \
		case _MM_SHUFFLE(0, 3, 2, 1): ret = _mm_shuffle_ps_0321((a), (b)); break; \
		case _MM_SHUFFLE(2, 1, 0, 3): ret = _mm_shuffle_ps_2103((a), (b)); break; \
		case _MM_SHUFFLE(1, 0, 1, 0): ret = _mm_shuffle_ps_1010((a), (b)); break; \
		case _MM_SHUFFLE(1, 0, 0, 1): ret = _mm_shuffle_ps_1001((a), (b)); break; \
		case _MM_SHUFFLE(0, 1, 0, 1): ret = _mm_shuffle_ps_0101((a), (b)); break; \
		case _MM_SHUFFLE(3, 2, 1, 0): ret = _mm_shuffle_ps_3210((a), (b)); break; \
		case _MM_SHUFFLE(0, 0, 1, 1): ret = _mm_shuffle_ps_0011((a), (b)); break; \
		case _MM_SHUFFLE(0, 0, 2, 2): ret = _mm_shuffle_ps_0022((a), (b)); break; \
		case _MM_SHUFFLE(2, 2, 0, 0): ret = _mm_shuffle_ps_2200((a), (b)); break; \
		case _MM_SHUFFLE(3, 2, 0, 2): ret = _mm_shuffle_ps_3202((a), (b)); break; \
		case _MM_SHUFFLE(1, 1, 3, 3): ret = _mm_shuffle_ps_1133((a), (b)); break; \
		case _MM_SHUFFLE(2, 0, 1, 0): ret = _mm_shuffle_ps_2010((a), (b)); break; \
		case _MM_SHUFFLE(2, 0, 0, 1): ret = _mm_shuffle_ps_2001((a), (b)); break; \
		case _MM_SHUFFLE(2, 0, 3, 2): ret = _mm_shuffle_ps_2032((a), (b)); break; \
		default: ret = _mm_shuffle_ps_default((a), (b), (imm)); break; \
	} \
	ret; \
})

#undef RETf
#undef RETi


#endif //ACFDETECTION_SSE_H
