// Copyright: see project license.
#include "textflag.h"

// func distL2i16_16(vecs []int16, base int, q *[16]float32) float32
//
// Argument layout (ABI0, stack-based):
//   vecs.ptr  = 0(FP),  8 bytes
//   vecs.len  = 8(FP),  8 bytes
//   vecs.cap  = 16(FP), 8 bytes
//   base      = 24(FP), 8 bytes  (element index, not byte offset)
//   q         = 32(FP), 8 bytes  (pointer to [16]float32)
//   ret       = 40(FP), 4 bytes  (float32 return value)
//   total argsize = 44 bytes
TEXT ·distL2i16_16(SB), NOSPLIT, $0-44
    MOVQ vecs+0(FP), SI        // SI = &vecs[0]
    MOVQ base+24(FP), AX       // AX = base (element index)
    MOVQ q+32(FP), DI          // DI = q (pointer to [16]float32)

    // Advance SI to &vecs[base]: each int16 is 2 bytes
    LEAQ (SI)(AX*2), SI

    // Y6 = broadcast(float32(1.0/10000.0)) = broadcast(0x38D1B717)
    MOVL $0x38D1B717, AX
    VMOVD AX, X6
    VBROADCASTSS X6, Y6

    // Dims 0–7: load 8 int16, sign-extend to int32, convert to float32, scale, subtract query, square
    VPMOVSXWD (SI), Y0          // Y0 = int32(vecs[base+0..7])
    VCVTDQ2PS Y0, Y0            // Y0 = float32(...)
    VMULPS Y6, Y0, Y0           // Y0 *= 1/10000
    VMOVUPS (DI), Y7            // Y7 = q[0..7]
    VSUBPS Y7, Y0, Y0           // Y0 = Y0 - Y7  (d = dequant - q)
    VMULPS Y0, Y0, Y0           // Y0 = d*d

    // Dims 8–15 (SI+16 bytes for 8 more int16, DI+32 bytes for 8 more float32)
    VPMOVSXWD 16(SI), Y1        // Y1 = int32(vecs[base+8..15])
    VCVTDQ2PS Y1, Y1            // Y1 = float32(...)
    VMULPS Y6, Y1, Y1           // Y1 *= 1/10000
    VMOVUPS 32(DI), Y7          // Y7 = q[8..15]
    VSUBPS Y7, Y1, Y1           // Y1 = Y1 - Y7
    VMULPS Y1, Y1, Y1           // Y1 = d*d

    // Accumulate squared differences from both halves
    VADDPS Y1, Y0, Y0           // Y0 = Y0 + Y1 (8 partial sums)

    // Horizontal reduction: 8 floats → 1 float
    VEXTRACTF128 $1, Y0, X1     // X1 = upper 128 bits of Y0 (elements 4-7)
    VADDPS X1, X0, X0           // X0 = X0 + X1 (4 sums)
    VHADDPS X0, X0, X0          // X0 = [a+b, c+d, a+b, c+d]
    VHADDPS X0, X0, X0          // X0[0] = a+b+c+d (total sum)

    VMOVSS X0, ret+40(FP)
    VZEROALL
    RET
