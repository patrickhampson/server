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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace caspar { namespace newtek { namespace shq {

enum class matrix
{
    bt709,
    bt601,
};

struct config
{
    int    width  = 1920; // multiple of 16
    int    height = 1080;
    matrix colour = matrix::bt709;
    int    device = 0;
    int    slots  = 3; // frames that can be in flight at once
};

// What a test needs to check a frame: the quantized coefficients (bitstream block order,
// scan order, DC unscaled at [0]) and the Y'CbCr 4:2:2 planes the encoder transformed.
struct debug_output
{
    std::vector<int16_t> coef;
    std::vector<uint8_t> planes;
};

enum class status
{
    failed, // first, so a default-constructed status is a failure
    ready,
    timeout, // still on the GPU; the slot stays busy
};

// SpeedHQ 4:2:2 (SHQ2) encoder on an NVIDIA GPU, through libcuda.so.1 loaded at run time.
//
// Each slot is one frame in flight: fill input(slot), submit(slot), and collect(slot) later.
// Work runs in submission order on one stream, so collect slots in the order they were submitted.
// Only one thread may use an encoder.
class cuda_encoder
{
  public:
    static std::unique_ptr<cuda_encoder> create(const config& cfg, std::string& error);
    ~cuda_encoder();

    cuda_encoder(const cuda_encoder&)            = delete;
    cuda_encoder& operator=(const cuda_encoder&) = delete;

    int slots() const;

    // Page-locked BGRA picture, width * 4 bytes per row.
    uint8_t* input(int slot) const;

    // Queues an encode of input(slot) with quantizer scale 1..99 (quality byte 100 - scale;
    // FFmpeg's qscale q is scale 2q) and returns at once.
    bool submit(int slot, int scale, std::string& error);

    // Waits up to timeout_ms for the slot's bitstream, which stays valid until the slot is
    // submitted again. After a timeout the GPU may still write to the slot.
    status collect(int slot, int timeout_ms, const uint8_t*& data, size_t& size, std::string& error);

    // submit and collect on slot 0, for tests. With debug, also downloads coefficients and planes.
    bool encode(int scale, const uint8_t*& data, size_t& size, std::string& error, debug_output* debug = nullptr);

    int   blocks() const;
    float gpu_ms(int slot) const; // upload to the end of the bitstream, timed on the GPU

  private:
    struct impl;
    explicit cuda_encoder(std::unique_ptr<impl> p);
    std::unique_ptr<impl> impl_;
};

}}} // namespace caspar::newtek::shq
