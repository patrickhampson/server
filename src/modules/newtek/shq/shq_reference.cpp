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

#include "shq_reference.h"

#include "shq_format.h"

#include <cstdlib>

namespace caspar { namespace newtek { namespace shq {

namespace {

const uint8_t  dc_lum_bits[12] = SHQ_DC_LUM_BITS_INIT;
const uint16_t dc_lum_code[12] = SHQ_DC_LUM_CODE_INIT;
const uint8_t  dc_chr_bits[12] = SHQ_DC_CHROMA_BITS_INIT;
const uint16_t dc_chr_code[12] = SHQ_DC_CHROMA_CODE_INIT;
const uint16_t ac[121][2]      = SHQ_AC_TABLE_INIT;
const uint8_t  max_level[64]   = SHQ_MAX_LEVEL_INIT;
const uint8_t  index_run[64]   = SHQ_INDEX_RUN_INIT;

// put_bits_le: least significant bit first.
struct writer
{
    std::vector<uint8_t>& out;
    uint64_t              acc  = 0;
    int                   bits = 0;

    void put(uint32_t code, int len)
    {
        acc |= static_cast<uint64_t>(code) << bits;
        bits += len;
        while (bits >= 8) {
            out.push_back(static_cast<uint8_t>(acc));
            acc >>= 8;
            bits -= 8;
        }
    }

    void flush()
    {
        if (bits > 0)
            out.push_back(static_cast<uint8_t>(acc));
        acc  = 0;
        bits = 0;
    }
};

void put_u24(std::vector<uint8_t>& out, size_t at, uint32_t v)
{
    out[at]     = static_cast<uint8_t>(v);
    out[at + 1] = static_cast<uint8_t>(v >> 8);
    out[at + 2] = static_cast<uint8_t>(v >> 16);
}

void encode_dc(writer& w, int diff, bool luma)
{
    int a     = std::abs(diff);
    int index = 0;
    while (a) {
        ++index;
        a >>= 1;
    }
    const int      d     = diff < 0 ? diff - 1 : diff;
    const uint32_t extra = static_cast<uint32_t>(d) & ((1u << index) - 1);
    if (luma)
        w.put(dc_lum_code[index] | (extra << dc_lum_bits[index]), dc_lum_bits[index] + index);
    else
        w.put(dc_chr_code[index] | (extra << dc_chr_bits[index]), dc_chr_bits[index] + index);
}

void encode_block(writer& w, const int16_t* block, int last_dc[3], int component)
{
    const int dc = block[0];
    encode_dc(w, last_dc[component] - dc, component == 0);
    last_dc[component] = dc;

    int last_non_zero = 0;
    for (int i = 1; i < 64; ++i) {
        const int level = block[i];
        if (!level)
            continue;
        const int run    = i - last_non_zero - 1;
        const int alevel = std::abs(level);
        if (alevel <= max_level[run]) {
            const int code = index_run[run] + alevel - 1;
            w.put(ac[code][0] | (static_cast<uint32_t>(level < 0) << ac[code][1]), ac[code][1] + 1);
        } else {
            w.put(SHQ_ESCAPE_CODE | (run << 6) | (static_cast<uint32_t>(level + 2048) << 12), 24);
        }
        last_non_zero = i;
    }
    w.put(SHQ_EOB_CODE, 4);
}

} // namespace

std::vector<uint8_t> pack_reference(int width, int height, int scale, const int16_t* coef)
{
    const int mb_width  = width / 16;
    const int mb_height = (height + 15) / 16;

    std::vector<uint8_t> out;
    out.push_back(static_cast<uint8_t>(100 - scale));
    out.resize(4);
    put_u24(out, 1, 4);

    const int16_t* block = coef;
    for (int slice = 0; slice < SHQ_SLICES; ++slice) {
        const size_t start = out.size();
        out.resize(start + 3);
        writer w{out};
        for (int row = 0; row < shq_rows_in_slice(slice, mb_height); ++row) {
            int last_dc[3] = {SHQ_DC_PREDICTOR, SHQ_DC_PREDICTOR, SHQ_DC_PREDICTOR};
            for (int mbx = 0; mbx < mb_width; ++mbx) {
                // Y0 Y1 Y2 Y3 Cb Cr Cb Cr
                static const int component[SHQ_BLOCKS_PER_MB] = {0, 0, 0, 0, 1, 2, 1, 2};
                for (int b = 0; b < SHQ_BLOCKS_PER_MB; ++b, block += 64)
                    encode_block(w, block, last_dc, component[b]);
            }
        }
        w.flush();
        put_u24(out, start, static_cast<uint32_t>(out.size() - start));
    }
    out.resize(out.size() + 3);
    return out;
}

}}} // namespace caspar::newtek::shq
