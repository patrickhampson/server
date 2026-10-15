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

// A plain NDI receiver for testing a sender from another machine. Not part of the server build.
//
//   ndi_watch LIBNDI NAME SECONDS [OUT.ppm]
//
// Connects to the first source whose name contains NAME, receives UYVY for SECONDS, and reports
// how long the first picture took, the frame rate, the longest gap between frames, and the
// receiver's dropped count. OUT.ppm gets the last frame, converted with BT.709.

#include <cstddef>

#include "../../interop/Processing.NDI.Lib.h"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

void write_ppm(const char* path, const NDIlib_video_frame_v2_t& f)
{
    FILE* out = std::fopen(path, "wb");
    if (!out)
        return;
    std::fprintf(out, "P6\n%d %d\n255\n", f.xres, f.yres);
    auto clamp = [](double v) { return static_cast<unsigned char>(std::min(255.0, std::max(0.0, v + 0.5))); };
    for (int y = 0; y < f.yres; ++y) {
        const uint8_t* row = f.p_data + static_cast<size_t>(y) * f.line_stride_in_bytes;
        for (int x = 0; x < f.xres; ++x) {
            const double        yy     = (row[(x / 2) * 4 + 1 + (x & 1) * 2] - 16) * 255.0 / 219;
            const double        cb     = (row[(x / 2) * 4 + 0] - 128) * 255.0 / 224;
            const double        cr     = (row[(x / 2) * 4 + 2] - 128) * 255.0 / 224;
            const unsigned char rgb[3] = {
                clamp(yy + 1.5748 * cr), clamp(yy - 0.1873 * cb - 0.4681 * cr), clamp(yy + 1.8556 * cb)};
            std::fwrite(rgb, 1, 3, out);
        }
    }
    std::fclose(out);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::fprintf(stderr, "usage: ndi_watch LIBNDI NAME SECONDS [OUT.ppm]\n");
        return 1;
    }
    void*            lib  = dlopen(argv[1], RTLD_LOCAL | RTLD_LAZY);
    auto             load = lib ? reinterpret_cast<const NDIlib_v6* (*)()>(dlsym(lib, "NDIlib_v6_load")) : nullptr;
    const NDIlib_v6* ndi  = load ? load() : nullptr;
    if (!ndi || !ndi->initialize()) {
        std::fprintf(stderr, "cannot load NDI from %s\n", argv[1]);
        return 1;
    }

    NDIlib_find_create_t find_settings;
    auto                 finder = ndi->find_create_v2(&find_settings);
    NDIlib_source_t      source;
    bool                 found = false;
    for (int tries = 0; tries < 30 && !found; ++tries) {
        ndi->find_wait_for_sources(finder, 500);
        uint32_t count   = 0;
        auto*    sources = ndi->find_get_current_sources(finder, &count);
        for (uint32_t i = 0; i < count && !found; ++i) {
            if (std::strstr(sources[i].p_ndi_name, argv[2])) {
                source = sources[i];
                found  = true;
                std::printf("source: %s\n", source.p_ndi_name);
            }
        }
    }
    if (!found) {
        std::fprintf(stderr, "no source matching %s\n", argv[2]);
        return 1;
    }

    NDIlib_recv_create_v3_t settings;
    settings.source_to_connect_to = source;
    settings.color_format         = NDIlib_recv_color_format_UYVY_BGRA;
    settings.bandwidth            = NDIlib_recv_bandwidth_highest;
    settings.allow_video_fields   = false;
    settings.p_ndi_recv_name      = "ndi_watch";
    const double connected        = now();
    auto         recv             = ndi->recv_create_v3(&settings);
    ndi->find_destroy(finder);

    const double            seconds = std::atof(argv[3]);
    double                  first = 0, last = 0, longest_gap = 0;
    int                     frames = 0, audio = 0;
    NDIlib_video_frame_v2_t keep;
    bool                    have_keep = false;
    while (now() - connected < seconds) {
        NDIlib_video_frame_v2_t video;
        NDIlib_audio_frame_v3_t sound;
        switch (ndi->recv_capture_v3(recv, &video, &sound, nullptr, 200)) {
            case NDIlib_frame_type_video: {
                const double t = now();
                if (!frames) {
                    first = t;
                    std::printf("first picture after %.0f ms: %dx%d, %d/%d fps, FourCC %.4s\n",
                                (t - connected) * 1000,
                                video.xres,
                                video.yres,
                                video.frame_rate_N,
                                video.frame_rate_D,
                                reinterpret_cast<const char*>(&video.FourCC));
                } else {
                    longest_gap = std::max(longest_gap, t - last);
                }
                last = t;
                ++frames;
                if (have_keep)
                    ndi->recv_free_video_v2(recv, &keep);
                keep      = video;
                have_keep = true;
                break;
            }
            case NDIlib_frame_type_audio:
                ++audio;
                ndi->recv_free_audio_v3(recv, &sound);
                break;
            default:
                break;
        }
    }
    NDIlib_recv_performance_t total, dropped;
    ndi->recv_get_performance(recv, &total, &dropped);
    std::printf("%d video frames, %.2f fps after the first, longest gap %.1f ms, %d audio frames\n",
                frames,
                frames > 1 ? (frames - 1) / (last - first) : 0.0,
                longest_gap * 1000,
                audio);
    std::printf("receiver: %lld video frames, %lld dropped\n",
                static_cast<long long>(total.video_frames),
                static_cast<long long>(dropped.video_frames));
    if (have_keep) {
        if (argc > 4)
            write_ppm(argv[4], keep);
        ndi->recv_free_video_v2(recv, &keep);
    }
    ndi->recv_destroy(recv);
    return frames > 0 ? 0 : 1;
}
