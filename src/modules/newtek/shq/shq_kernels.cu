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

// SpeedHQ 4:2:2 encoder kernels. Built into a fatbin by build_kernels.sh and loaded with the
// driver API, so the server needs libcuda.so.1 at run time and no CUDA toolkit to build.
//
// One frame is five launches on one stream:
//   shq_transform  one CUDA block per macroblock: BGRA -> Y'CbCr 4:2:2, 8x8 DCT, quantize, AC bit count
//   shq_scan       one CUDA block per slice: DC differences, then each block's bit offset in its slice
//   shq_zero       clears the words the frame will occupy
//   shq_emit       one thread per 8x8 block writes its codes at its offset; boundary words use atomicOr
//   shq_copy_out   writes the frame, and its size, straight into mapped host memory. The size is
//                  only known on the GPU, so this saves the host a wait for it before the copy.

#include "shq_format.h"

__constant__ uint8_t  c_zigzag[64]        = SHQ_ZIGZAG_INIT;
__constant__ uint8_t  c_quant[64]         = SHQ_QUANT_MATRIX_INIT;
__constant__ uint8_t  c_dc_lum_bits[12]   = SHQ_DC_LUM_BITS_INIT;
__constant__ uint16_t c_dc_lum_code[12]   = SHQ_DC_LUM_CODE_INIT;
__constant__ uint8_t  c_dc_chr_bits[12]   = SHQ_DC_CHROMA_BITS_INIT;
__constant__ uint16_t c_dc_chr_code[12]   = SHQ_DC_CHROMA_CODE_INIT;
__constant__ uint16_t c_ac[121][2]        = SHQ_AC_TABLE_INIT;
__constant__ uint8_t  c_max_level[64]     = SHQ_MAX_LEVEL_INIT;
__constant__ uint8_t  c_index_run[64]     = SHQ_INDEX_RUN_INIT;

__device__ __forceinline__ int block_index(const shq_params& p, int mbx, int mby, int b)
{
    const int slice = mby & 3;
    return shq_slice_base(slice, p.mb_width, p.mb_height) + ((mby >> 2) * p.mb_width + mbx) * SHQ_BLOCKS_PER_MB + b;
}

__device__ __forceinline__ int clamp_byte(float v)
{
    return min(max(__float2int_rd(v + 0.5f), 0), 255);
}

extern "C" __global__ void __launch_bounds__(64) shq_transform(shq_params p)
{
    __shared__ float   basis[64];
    __shared__ float   samples[SHQ_BLOCKS_PER_MB][64];
    __shared__ float   rows[SHQ_BLOCKS_PER_MB][64];
    __shared__ int16_t levels[SHQ_BLOCKS_PER_MB][64];

    const int t   = threadIdx.x;
    const int mbx = blockIdx.x;
    const int mby = blockIdx.y;

    {
        // Orthonormal DCT-II basis: basis[u][x] = c(u) / 2 * cos((2x + 1) u pi / 16).
        const int u = t >> 3;
        const int x = t & 7;
        basis[t]    = (u == 0 ? 0.35355339059327373f : 0.5f) * cospif((2 * x + 1) * u / 16.0f);
    }

    {
        // Thread t converts 4 pixels of macroblock row t / 4. Rows past the picture repeat the last one.
        const int r  = t >> 2;
        const int c0 = (t & 3) * 4;
        const int py = mby * 16 + r;
        const int sy = min(py, p.height - 1);
        const int px = mbx * 16 + c0;

        const uint4 quad = *reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(p.bgra) +
                                                           static_cast<size_t>(sy) * p.pitch + px * 4);
        const uint32_t words[4] = {quad.x, quad.y, quad.z, quad.w};

        float red[4], green[4], blue[4];
        for (int i = 0; i < 4; ++i) {
            blue[i]  = static_cast<float>(words[i] & 0xff);
            green[i] = static_cast<float>((words[i] >> 8) & 0xff);
            red[i]   = static_cast<float>((words[i] >> 16) & 0xff);
        }

        uint8_t* planes = reinterpret_cast<uint8_t*>(p.planes);
        const bool write_planes = planes && py < p.height;

        const int luma_block = (r >= 8 ? 2 : 0) + (c0 >= 8 ? 1 : 0);
        for (int i = 0; i < 4; ++i) {
            const int y = clamp_byte(p.ky[0] * red[i] + p.ky[1] * green[i] + p.ky[2] * blue[i] + 16.0f);
            samples[luma_block][(r & 7) * 8 + ((c0 + i) & 7)] = static_cast<float>(y);
            if (write_planes)
                planes[static_cast<size_t>(py) * p.width + px + i] = static_cast<uint8_t>(y);
        }

        // Chroma sits between each pair of luma samples, so a pair's average is its chroma.
        for (int j = 0; j < 2; ++j) {
            const float rr = 0.5f * (red[2 * j] + red[2 * j + 1]);
            const float gg = 0.5f * (green[2 * j] + green[2 * j + 1]);
            const float bb = 0.5f * (blue[2 * j] + blue[2 * j + 1]);
            const int   cb = clamp_byte(p.kcb[0] * rr + p.kcb[1] * gg + p.kcb[2] * bb + 128.0f);
            const int   cr = clamp_byte(p.kcr[0] * rr + p.kcr[1] * gg + p.kcr[2] * bb + 128.0f);
            const int   cc = (c0 >> 1) + j;
            const int   at = (r & 7) * 8 + cc;
            samples[r >= 8 ? 6 : 4][at] = static_cast<float>(cb);
            samples[r >= 8 ? 7 : 5][at] = static_cast<float>(cr);
            if (write_planes) {
                const size_t luma   = static_cast<size_t>(p.width) * p.height;
                const size_t chroma = static_cast<size_t>(py) * (p.width / 2) + (px >> 1) + j;
                planes[luma + chroma]              = static_cast<uint8_t>(cb);
                planes[luma + luma / 2 + chroma]   = static_cast<uint8_t>(cr);
            }
        }
    }
    __syncthreads();

    const int b    = t >> 3; // 8x8 block this thread works on
    const int lane = t & 7;

    {
        // Horizontal pass: row `lane` of block b.
        float s[8];
        for (int x = 0; x < 8; ++x)
            s[x] = samples[b][lane * 8 + x];
        for (int u = 0; u < 8; ++u) {
            float acc = 0.0f;
            for (int x = 0; x < 8; ++x)
                acc = fmaf(s[x], basis[u * 8 + x], acc);
            rows[b][lane * 8 + u] = acc;
        }
    }
    __syncthreads();

    {
        // Vertical pass: column `lane` of block b, back into samples.
        float s[8];
        for (int y = 0; y < 8; ++y)
            s[y] = rows[b][y * 8 + lane];
        for (int v = 0; v < 8; ++v) {
            float acc = 0.0f;
            for (int y = 0; y < 8; ++y)
                acc = fmaf(s[y], basis[v * 8 + y], acc);
            samples[b][v * 8 + lane] = acc;
        }
    }
    __syncthreads();

    const int idx  = block_index(p, mbx, mby, b);
    int16_t*  coef = reinterpret_cast<int16_t*>(p.coef) + static_cast<size_t>(idx) * 64;
    {
        // Quantize scan positions lane*8 .. lane*8+7. DC is sent unscaled (8 x the mean);
        // AC rounds with FFmpeg's intra bias of 3/8.
        const float inv_scale = 16.0f / static_cast<float>(p.scale);
        for (int j = 0; j < 8; ++j) {
            const int   i = lane * 8 + j;
            const int   k = c_zigzag[i];
            const float f = samples[b][k];
            int         level;
            if (i == 0) {
                level = min(max(__float2int_rn(f), 0), 2047);
            } else {
                const int l = min(__float2int_rd(fabsf(f) * inv_scale / c_quant[k] + 0.375f), 2047);
                level       = f < 0.0f ? -l : l;
            }
            levels[b][i] = static_cast<int16_t>(level);
            coef[i]      = static_cast<int16_t>(level);
        }
    }
    __syncthreads();

    if (t < SHQ_BLOCKS_PER_MB) {
        const int bb   = t;
        const int id   = block_index(p, mbx, mby, bb);
        int       bits = 4; // EOB
        int       last = 0;
        for (int i = 1; i < 64; ++i) {
            const int level = levels[bb][i];
            if (level) {
                const int run   = i - last - 1;
                const int alevel = abs(level);
                bits += alevel <= c_max_level[run] ? c_ac[c_index_run[run] + alevel - 1][1] + 1 : 24;
                last = i;
            }
        }
        reinterpret_cast<uint16_t*>(p.acbits)[id] = static_cast<uint16_t>(bits);
        reinterpret_cast<uint8_t*>(p.last)[id]    = static_cast<uint8_t>(last);
    }
}

__device__ __forceinline__ int dc_size(int diff)
{
    const int a = abs(diff);
    return a ? 32 - __clz(a) : 0;
}

extern "C" __global__ void __launch_bounds__(1024) shq_scan(shq_params p)
{
    __shared__ uint32_t warp_sums[32];
    __shared__ uint32_t carry;

    const int slice = blockIdx.x;
    const int base  = shq_slice_base(slice, p.mb_width, p.mb_height);
    const int n     = shq_rows_in_slice(slice, p.mb_height) * p.mb_width * SHQ_BLOCKS_PER_MB;
    const int t     = threadIdx.x;
    const int lane  = t & 31;
    const int warp  = t >> 5;

    const int16_t*  coef   = reinterpret_cast<const int16_t*>(p.coef);
    const uint16_t* acbits = reinterpret_cast<const uint16_t*>(p.acbits);
    int16_t*        dcdiff = reinterpret_cast<int16_t*>(p.dcdiff);
    uint32_t*       offset = reinterpret_cast<uint32_t*>(p.offset);

    if (t == 0)
        carry = 0;
    __syncthreads();

    for (int start = 0; start < n; start += blockDim.x) {
        const int i   = start + t;
        const int idx = base + i;
        uint32_t  v   = 0;
        if (i < n) {
            // The DC predictor is the previous block of the same component in this macroblock row:
            // Y0..Y3 chain across macroblocks, as do Cb top/bottom and Cr top/bottom.
            const int b    = i & 7;
            const int mbx  = (i >> 3) % p.mb_width;
            int       prev = -1;
            if (b == 1 || b == 2 || b == 3)
                prev = idx - 1;
            else if (b == 6 || b == 7)
                prev = idx - 2;
            else if (mbx > 0)
                prev = idx - b - 8 + (b == 0 ? 3 : b + 2); // previous macroblock's Y3, bottom Cb or bottom Cr
            const int dc   = coef[static_cast<size_t>(idx) * 64];
            const int pred = prev >= 0 ? coef[static_cast<size_t>(prev) * 64] : SHQ_DC_PREDICTOR;
            const int diff = pred - dc;
            const int size = dc_size(diff);
            dcdiff[idx]    = static_cast<int16_t>(diff);
            v = acbits[idx] + size + (b < 4 ? c_dc_lum_bits[size] : c_dc_chr_bits[size]);
        }

        uint32_t x = v;
        for (int d = 1; d < 32; d <<= 1) {
            const uint32_t y = __shfl_up_sync(0xffffffffu, x, d);
            if (lane >= d)
                x += y;
        }
        if (lane == 31)
            warp_sums[warp] = x;
        __syncthreads();
        if (warp == 0) {
            uint32_t w = lane < static_cast<int>(blockDim.x >> 5) ? warp_sums[lane] : 0;
            for (int d = 1; d < 32; d <<= 1) {
                const uint32_t y = __shfl_up_sync(0xffffffffu, w, d);
                if (lane >= d)
                    w += y;
            }
            warp_sums[lane] = w;
        }
        __syncthreads();
        if (i < n)
            offset[idx] = carry + x - v + (warp > 0 ? warp_sums[warp - 1] : 0);
        __syncthreads();
        if (t == 0)
            carry += warp_sums[31];
        __syncthreads();
    }

    if (t == 0)
        reinterpret_cast<uint32_t*>(p.slice_bits)[slice] = carry;
}

extern "C" __global__ void shq_zero(shq_params p)
{
    uint32_t start[SHQ_SLICES], len[SHQ_SLICES];
    const uint32_t total = shq_layout(reinterpret_cast<const uint32_t*>(p.slice_bits), start, len);
    const uint32_t words = min((total + 3) / 4 + 1, static_cast<uint32_t>(p.out_words));
    uint32_t*      out   = reinterpret_cast<uint32_t*>(p.out);
    for (uint32_t w = blockIdx.x * blockDim.x + threadIdx.x; w < words; w += gridDim.x * blockDim.x)
        out[w] = 0;
}

// Little-endian bit writer starting at an arbitrary bit. The first and last words can be
// shared with the neighbouring block, so they are ORed in; the words between are this block's.
struct bit_writer
{
    uint32_t* out;
    uint32_t  word;
    uint64_t  acc;
    int       bits;
    bool      first;

    __device__ bit_writer(uint32_t* o, uint32_t bit)
        : out(o)
        , word(bit >> 5)
        , acc(0)
        , bits(bit & 31)
        , first(true)
    {
    }

    __device__ __forceinline__ void put(uint32_t code, int len)
    {
        acc |= static_cast<uint64_t>(code) << bits;
        bits += len;
        if (bits >= 32) {
            if (first) {
                atomicOr(out + word, static_cast<uint32_t>(acc));
                first = false;
            } else {
                out[word] = static_cast<uint32_t>(acc);
            }
            ++word;
            acc >>= 32;
            bits -= 32;
        }
    }

    __device__ __forceinline__ void finish()
    {
        if (bits > 0)
            atomicOr(out + word, static_cast<uint32_t>(acc));
    }
};

__device__ __forceinline__ void or_byte(uint32_t* out, uint32_t pos, uint32_t value)
{
    atomicOr(out + (pos >> 2), (value & 0xff) << ((pos & 3) * 8));
}

extern "C" __global__ void shq_emit(shq_params p)
{
    uint32_t start[SHQ_SLICES], len[SHQ_SLICES];
    shq_layout(reinterpret_cast<const uint32_t*>(p.slice_bits), start, len);
    uint32_t* out = reinterpret_cast<uint32_t*>(p.out);

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        or_byte(out, 0, 100 - p.scale);
        or_byte(out, 1, 4); // second field offset 4: one progressive picture
        for (int s = 0; s < SHQ_SLICES; ++s)
            for (int k = 0; k < 3; ++k)
                or_byte(out, start[s] + k, len[s] >> (8 * k));
    }
    if (idx >= p.blocks)
        return;

    int slice = 0;
    while (slice + 1 < SHQ_SLICES && idx >= shq_slice_base(slice + 1, p.mb_width, p.mb_height))
        ++slice;

    const uint32_t bit = (start[slice] + 3) * 8 + reinterpret_cast<const uint32_t*>(p.offset)[idx];
    bit_writer     w(out, bit);

    const int b    = (idx - shq_slice_base(slice, p.mb_width, p.mb_height)) & 7;
    const int diff = reinterpret_cast<const int16_t*>(p.dcdiff)[idx];
    const int size = dc_size(diff);
    const int dd   = diff < 0 ? diff - 1 : diff;
    const uint32_t extra = static_cast<uint32_t>(dd) & ((1u << size) - 1);
    if (b < 4)
        w.put(c_dc_lum_code[size] | (extra << c_dc_lum_bits[size]), c_dc_lum_bits[size] + size);
    else
        w.put(c_dc_chr_code[size] | (extra << c_dc_chr_bits[size]), c_dc_chr_bits[size] + size);

    const int16_t* coef = reinterpret_cast<const int16_t*>(p.coef) + static_cast<size_t>(idx) * 64;
    const int      last = reinterpret_cast<const uint8_t*>(p.last)[idx];
    int            prev = 0;
    for (int i = 1; i <= last; ++i) {
        const int level = coef[i];
        if (!level)
            continue;
        const int run    = i - prev - 1;
        const int alevel = abs(level);
        if (alevel <= c_max_level[run]) {
            const int code = c_index_run[run] + alevel - 1;
            const int n    = c_ac[code][1];
            w.put(c_ac[code][0] | (static_cast<uint32_t>(level < 0) << n), n + 1);
        } else {
            w.put(SHQ_ESCAPE_CODE | (run << 6) | (static_cast<uint32_t>(level + 2048) << 12), 24);
        }
        prev = i;
    }
    w.put(SHQ_EOB_CODE, 4);
    w.finish();
}

extern "C" __global__ void shq_copy_out(shq_params p)
{
    uint32_t start[SHQ_SLICES], len[SHQ_SLICES];
    const uint32_t total = shq_layout(reinterpret_cast<const uint32_t*>(p.slice_bits), start, len);
    const uint32_t words = (total + 3) / 4;
    const uint32_t* in   = reinterpret_cast<const uint32_t*>(p.out);
    uint32_t*       out  = reinterpret_cast<uint32_t*>(p.host_out);
    for (uint32_t w = blockIdx.x * blockDim.x + threadIdx.x; w < words; w += gridDim.x * blockDim.x)
        out[w] = in[w];
    if (blockIdx.x == 0 && threadIdx.x == 0)
        reinterpret_cast<uint32_t*>(p.host_meta)[0] = total;
}
