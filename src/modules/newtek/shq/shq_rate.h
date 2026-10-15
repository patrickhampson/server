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

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace caspar { namespace newtek { namespace shq {

// Picks the next frame's quantizer scale from the size of the last finished one.
//
// SpeedHQ has one scale per frame and no way to change it mid-frame, so this steers between
// frames. Frame size falls slowly with scale (roughly with its square root on the test
// material, less on flat graphics, where DC and end-of-block codes dominate), so a size error
// asks for a much larger scale change. Steps are damped and capped so a single odd frame does
// not swing the picture.
//
// The scale stays in 2..40. Below 2 costs bits for nothing visible. Above 40 (quality byte
// under 60), libndi and FFmpeg begin to decode the same packet differently.
class rate_control
{
  public:
    static constexpr int min_scale = 2;
    static constexpr int max_scale = 40;

    explicit rate_control(double target_bytes, int start_scale = 8)
        : target_(target_bytes)
        , log_scale_(std::log(static_cast<double>(std::clamp(start_scale, min_scale, max_scale))))
    {
    }

    int scale() const { return std::clamp(static_cast<int>(std::lround(std::exp(log_scale_))), min_scale, max_scale); }

    // `scale` is the one the measured frame was encoded with; frames in flight lag the controller.
    void update(int scale, size_t bytes)
    {
        if (bytes == 0 || target_ <= 0)
            return;
        const double error = std::log(static_cast<double>(bytes) / target_);
        // Size ~ scale^-0.5, so the scale that would have hit the target is scale * ratio^2.
        double wanted = std::log(static_cast<double>(scale)) + 2.0 * error;
        // Over target by more than a third (a cut to busy content) moves at once; anything
        // else moves halfway, and never more than 1.5x in one frame.
        const double step = error > std::log(1.33) ? 1.0 : 0.5;
        wanted            = log_scale_ + step * (wanted - log_scale_);
        wanted            = std::clamp(wanted, log_scale_ - std::log(1.5), log_scale_ + std::log(1.5));
        log_scale_ =
            std::clamp(wanted, std::log(static_cast<double>(min_scale)), std::log(static_cast<double>(max_scale)));
    }

    double target_bytes() const { return target_; }

  private:
    double target_;
    double log_scale_;
};

}}} // namespace caspar::newtek::shq
