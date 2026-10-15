/*
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CasparCG is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CasparCG. If not, see <http://www.gnu.org/licenses/>.
 */

// SpeedHQ 4:2:2 (SHQ2) bitstream constants, shared by the CUDA kernels and the host code.
//
// The layout follows FFmpeg's speedhq encoder and decoder (libavcodec/speedhqenc.c,
// speedhqdec.c, speedhq.c), which is what NDI receivers were checked against:
//
//   byte 0      quality q; the decoder dequantizes with S = 100 - q (FFmpeg writes 100 - 2*qscale)
//   bytes 1..3  second field offset, 24-bit little-endian; 4 means one progressive picture
//   4 slices    each a 24-bit little-endian length (counting those 3 bytes) and a byte-padded
//               bitstream. Slice s holds macroblock rows s, s+4, s+8, ...
//   3 bytes     zero, the length field FFmpeg leaves for a fifth slice
//
// Bits are packed least significant first. A 4:2:2 macroblock is Y0 Y1 Y2 Y3 Cb Cr Cb Cr
// (top-left, top-right, bottom-left, bottom-right luma; then top and bottom chroma). DC is
// coded as previous - current per component (Y, Cb, Cr) and resets to 1024 at the start of
// every macroblock row. AC uses the SpeedHQ run-level table in zigzag order, then EOB.

#pragma once

#include <stdint.h>

#define SHQ_SLICES 4
#define SHQ_BLOCKS_PER_MB 8
#define SHQ_DC_PREDICTOR 1024
// Escape (24 bits) on all 63 AC coefficients, the longest DC code (20 bits) and EOB.
#define SHQ_MAX_BLOCK_BYTES 192

// clang-format off
// Natural (row-major) index of each zigzag scan position.
#define SHQ_ZIGZAG_INIT                                                                                                \
    {0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,  12, 19, 26, 33, 40, 48,                           \
     41, 34, 27, 20, 13, 6,  7,  14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23,                           \
     30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63}

// Quantizer matrix in natural order. The decoder reconstructs AC as (level * m * S) >> 4.
#define SHQ_QUANT_MATRIX_INIT                                                                                          \
    {16, 16, 19, 22, 26, 27, 29, 34, 16, 16, 22, 24, 27, 29, 34, 37, 19, 22, 26, 27, 29, 34,                           \
     34, 38, 22, 22, 26, 27, 29, 34, 37, 40, 22, 26, 27, 29, 32, 35, 40, 48, 26, 27, 29, 32,                           \
     35, 40, 48, 58, 26, 27, 29, 34, 38, 46, 56, 69, 27, 29, 35, 38, 46, 56, 69, 83}

// MPEG-2 DC size codes, bit-reversed for the little-endian writer.
#define SHQ_DC_LUM_BITS_INIT {3, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9}
#define SHQ_DC_LUM_CODE_INIT {0x1, 0x0, 0x2, 0x5, 0x3, 0x7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF, 0x1FF}
#define SHQ_DC_CHROMA_BITS_INIT {2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10}
#define SHQ_DC_CHROMA_CODE_INIT {0x0, 0x2, 0x1, 0x3, 0x7, 0xF, 0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF}

// ff_speedhq_vlc_table: {code, length} for the 121 run-level pairs. The sign bit follows the code.
#define SHQ_AC_TABLE_INIT                                                                                              \
    {{0x0001, 2},  {0x0003, 3},  {0x000E, 4},  {0x0007, 5},  {0x0017, 5},  {0x0028, 6},  {0x0008, 6},                  \
     {0x006F, 7},  {0x001F, 7},  {0x00C4, 8},  {0x0044, 8},  {0x005F, 8},  {0x00DF, 8},  {0x007F, 8},                  \
     {0x00FF, 8},  {0x3E00, 14}, {0x1E00, 14}, {0x2E00, 14}, {0x0E00, 14}, {0x3600, 14}, {0x1600, 14},                 \
     {0x2600, 14}, {0x0600, 14}, {0x3A00, 14}, {0x1A00, 14}, {0x2A00, 14}, {0x0A00, 14}, {0x3200, 14},                 \
     {0x1200, 14}, {0x2200, 14}, {0x0200, 14}, {0x0C00, 15}, {0x7400, 15}, {0x3400, 15}, {0x5400, 15},                 \
     {0x1400, 15}, {0x6400, 15}, {0x2400, 15}, {0x4400, 15}, {0x0400, 15}, {0x0002, 3},  {0x000C, 5},                  \
     {0x004F, 7},  {0x00E4, 8},  {0x0004, 8},  {0x0D00, 13}, {0x1500, 13}, {0x7C00, 15}, {0x3C00, 15},                 \
     {0x5C00, 15}, {0x1C00, 15}, {0x6C00, 15}, {0x2C00, 15}, {0x4C00, 15}, {0xC800, 16}, {0x4800, 16},                 \
     {0x8800, 16}, {0x0800, 16}, {0x0300, 13}, {0x1D00, 13}, {0x0014, 5},  {0x0070, 7},  {0x003F, 8},                  \
     {0x00C0, 10}, {0x0500, 13}, {0x0180, 12}, {0x0280, 12}, {0x0C80, 12}, {0x0080, 12}, {0x0B00, 13},                 \
     {0x1300, 13}, {0x001C, 5},  {0x0064, 8},  {0x0380, 12}, {0x1900, 13}, {0x0D80, 12}, {0x0018, 6},                  \
     {0x00BF, 8},  {0x0480, 12}, {0x0B80, 12}, {0x0038, 6},  {0x0040, 9},  {0x0900, 13}, {0x0030, 7},                  \
     {0x0780, 12}, {0x2800, 16}, {0x0010, 7},  {0x0A80, 12}, {0x0050, 7},  {0x0880, 12}, {0x000F, 7},                  \
     {0x1100, 13}, {0x002F, 7},  {0x0100, 13}, {0x0084, 8},  {0x5800, 16}, {0x00A4, 8},  {0x9800, 16},                 \
     {0x0024, 8},  {0x1800, 16}, {0x0140, 9},  {0xE800, 16}, {0x01C0, 9},  {0x6800, 16}, {0x02C0, 10},                 \
     {0xA800, 16}, {0x0F80, 12}, {0x0580, 12}, {0x0980, 12}, {0x0E80, 12}, {0x0680, 12}, {0x1F00, 13},                 \
     {0x0F00, 13}, {0x1700, 13}, {0x0700, 13}, {0x1B00, 13}, {0xF800, 16}, {0x7800, 16}, {0xB800, 16},                 \
     {0x3800, 16}, {0xD800, 16}}

// Largest level with its own code for each run, and the table index of (run, level 1).
// Derived from ff_speedhq_run and ff_speedhq_level; runs from 32 always escape.
#define SHQ_MAX_LEVEL_INIT                                                                                             \
    {40, 20, 11, 5, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,                \
     0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
#define SHQ_INDEX_RUN_INIT                                                                                             \
    {0,   40,  60,  71,  76,  80,  83,  86,  88,  90,  92,  94,  96,  98,  100, 102,                                   \
     104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,                                   \
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,                                     \
     0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0}

#define SHQ_ESCAPE_CODE 0x20 // 6 bits, then 6 bits of run and 12 bits of level + 2048
#define SHQ_EOB_CODE 0x6     // 4 bits
// clang-format on

// Everything a kernel needs, passed by value as the only kernel argument.
struct shq_params
{
    uint64_t bgra;       // device BGRA picture, pitch bytes per row
    uint64_t coef;       // int16[blocks][64]: quantized coefficients in scan order, DC (not a diff) at [0]
    uint64_t last;       // uint8[blocks]: last non-zero scan position, 0 for none
    uint64_t acbits;     // uint16[blocks]: AC codes plus EOB, in bits
    uint64_t dcdiff;     // int16[blocks]: previous DC minus this DC
    uint64_t offset;     // uint32[blocks]: bit offset inside the block's slice
    uint64_t slice_bits; // uint32[SHQ_SLICES]
    uint64_t out;        // uint32 words of output bitstream
    uint64_t planes;     // optional uint8 Y, Cb, Cr planes (4:2:2, unpadded) for testing; 0 for none
    uint64_t host_out;   // page-locked, device-mapped: the finished bitstream
    uint64_t host_meta;  // page-locked, device-mapped uint32[1]: its size in bytes

    int32_t pitch;
    int32_t width;
    int32_t height;
    int32_t mb_width;
    int32_t mb_height;
    int32_t blocks;
    int32_t scale; // S, 1..99; the quality byte is 100 - S
    int32_t out_words;

    // Y' = ky.x * R + ky.y * G + ky.z * B + 16, already scaled to limited range; same for Cb, Cr around 128.
    float ky[3];
    float kcb[3];
    float kcr[3];
};

#ifdef __CUDACC__
#define SHQ_HD __host__ __device__
#else
#define SHQ_HD
#endif

// Rows of macroblocks in slice s: FFmpeg's ff_speedhq_mb_rows_in_slice.
static inline SHQ_HD int shq_rows_in_slice(int slice, int mb_height)
{
    return mb_height / 4 + (slice < (mb_height % 4));
}

// First block of slice s. Blocks are numbered in bitstream order: slice, row in slice, macroblock, block.
static inline SHQ_HD int shq_slice_base(int slice, int mb_width, int mb_height)
{
    int base = 0;
    for (int s = 0; s < slice; ++s)
        base += shq_rows_in_slice(s, mb_height) * mb_width * SHQ_BLOCKS_PER_MB;
    return base;
}

// Byte offset of each slice's length field, and the total size, from the slices' bit counts.
static inline SHQ_HD uint32_t shq_layout(const uint32_t* slice_bits, uint32_t* slice_start, uint32_t* slice_len)
{
    uint32_t pos = 4;
    for (int s = 0; s < SHQ_SLICES; ++s) {
        slice_start[s] = pos;
        slice_len[s]   = 3 + (slice_bits[s] + 7) / 8;
        pos += slice_len[s];
    }
    return pos + 3;
}
