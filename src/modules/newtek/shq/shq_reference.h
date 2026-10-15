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

#pragma once

#include <cstdint>
#include <vector>

namespace caspar { namespace newtek { namespace shq {

// Serial SpeedHQ packer, written the way FFmpeg's speedhqenc.c walks a frame: slice by slice,
// row by row, one bit writer. It takes coefficients in the encoder's layout (bitstream block
// order, scan order, DC unscaled at [0]) and is the check on the parallel packing on the GPU.
std::vector<uint8_t> pack_reference(int width, int height, int scale, const int16_t* coef);

}}} // namespace caspar::newtek::shq
