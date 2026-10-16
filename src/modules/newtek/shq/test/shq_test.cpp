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

// Standalone checks for the CUDA SpeedHQ encoder. Not part of the server build.
//
//   shq_test encode W H in.bgra SCALE out.avi planes.yuv [601]
//       Encodes every frame of a raw BGRA file. Each bitstream is compared byte for byte with
//       the serial reference packer run on the GPU's own coefficients. Writes the bitstreams as
//       an SHQ2 AVI for FFmpeg to decode, and the encoder's Y'CbCr 4:2:2 input as yuv422p.
//   shq_test bench W H in.bgra SCALE SECONDS [MBPS]
//       Encodes at 50 fps the way the NDI consumer does: each tick collects the frame submitted
//       on the last one, then copies and submits the next. Reports GPU time, process CPU and
//       bytes. SCALE 0 runs the rate controller at MBPS instead of a fixed scale.
//   shq_test ndi W H in.bgra SCALE SECONDS
//       Sends the CUDA bitstreams through libndi as SHQ2, receives them on this machine as UYVY
//       and compares what arrives with what was encoded.
//   shq_test ndi-matrix W H in.bgra
//       Sends frame 0 through libndi's own encoder as BGRX, receives UYVY, and compares it with
//       BT.709 and BT.601 conversions of the same picture.

#include "../shq_cuda.h"
#include "../shq_rate.h"
#include "../shq_reference.h"

#include "../../interop/Processing.NDI.Lib.h"

#include <dlfcn.h>
#include <sys/resource.h>
#include <time.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace caspar::newtek;

namespace {

double now_seconds()
{
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

double cpu_seconds()
{
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
}

std::vector<std::vector<uint8_t>> read_frames(const char* path, int w, int h)
{
    std::ifstream                     in(path, std::ios::binary);
    const size_t                      bytes = static_cast<size_t>(w) * h * 4;
    std::vector<std::vector<uint8_t>> frames;
    for (;;) {
        std::vector<uint8_t> f(bytes);
        if (!in.read(reinterpret_cast<char*>(f.data()), bytes))
            break;
        frames.push_back(std::move(f));
    }
    return frames;
}

std::unique_ptr<shq::cuda_encoder> open_encoder(int w, int h, shq::matrix m)
{
    shq::config cfg;
    cfg.width  = w;
    cfg.height = h;
    cfg.colour = m;
    std::string error;
    auto        enc = shq::cuda_encoder::create(cfg, error);
    if (!enc)
        std::fprintf(stderr, "encoder: %s\n", error.c_str());
    return enc;
}

// Just enough AVI for FFmpeg: one SHQ2 video stream at 50 fps, with an index.
class avi_writer
{
  public:
    avi_writer(const char* path, int w, int h)
        : f_(std::fopen(path, "wb"))
        , w_(w)
        , h_(h)
    {
        write_header(0);
        tag("LIST");
        movi_size_at_ = std::ftell(f_);
        u32(0);
        movi_at_ = std::ftell(f_);
        tag("movi");
    }

    void add(const uint8_t* data, size_t size)
    {
        index_.push_back({static_cast<uint32_t>(std::ftell(f_) - movi_at_), static_cast<uint32_t>(size)});
        tag("00dc");
        u32(static_cast<uint32_t>(size));
        std::fwrite(data, 1, size, f_);
        if (size & 1)
            std::fputc(0, f_);
    }

    ~avi_writer()
    {
        const long movi_end = std::ftell(f_);
        tag("idx1");
        u32(static_cast<uint32_t>(index_.size() * 16));
        for (auto& e : index_) {
            tag("00dc");
            u32(0x10); // AVIIF_KEYFRAME
            u32(e.first);
            u32(e.second);
        }
        const long end = std::ftell(f_);
        std::fseek(f_, movi_size_at_, SEEK_SET);
        u32(static_cast<uint32_t>(movi_end - movi_at_));
        std::fseek(f_, 0, SEEK_SET);
        write_header(static_cast<uint32_t>(end - 8));
        std::fclose(f_);
    }

  private:
    void tag(const char* t) { std::fwrite(t, 1, 4, f_); }
    void u32(uint32_t v) { std::fwrite(&v, 4, 1, f_); }
    void u16(uint16_t v) { std::fwrite(&v, 2, 1, f_); }

    void write_header(uint32_t riff_size)
    {
        const uint32_t frames = static_cast<uint32_t>(index_.size());
        tag("RIFF");
        u32(riff_size);
        tag("AVI ");
        tag("LIST");
        u32(4 + 8 + 56 + 8 + 4 + 8 + 56 + 8 + 40);
        tag("hdrl");
        tag("avih");
        u32(56);
        u32(20000); // microseconds per frame
        u32(0);
        u32(0);
        u32(0x10); // AVIF_HASINDEX
        u32(frames);
        u32(0);
        u32(1);
        u32(0);
        u32(w_);
        u32(h_);
        for (int i = 0; i < 4; ++i)
            u32(0);
        tag("LIST");
        u32(4 + 8 + 56 + 8 + 40);
        tag("strl");
        tag("strh");
        u32(56);
        tag("vids");
        tag("SHQ2");
        u32(0);
        u16(0);
        u16(0);
        u32(0);
        u32(1);  // scale
        u32(50); // rate
        u32(0);
        u32(frames);
        u32(0);
        u32(0xffffffffu);
        u32(0);
        u16(0);
        u16(0);
        u16(static_cast<uint16_t>(w_));
        u16(static_cast<uint16_t>(h_));
        tag("strf");
        u32(40);
        u32(40);
        u32(w_);
        u32(h_);
        u16(1);
        u16(16);
        tag("SHQ2");
        u32(w_ * h_ * 2);
        for (int i = 0; i < 4; ++i)
            u32(0);
    }

    FILE*                                      f_;
    int                                        w_, h_;
    long                                       movi_size_at_ = 0;
    long                                       movi_at_      = 0;
    std::vector<std::pair<uint32_t, uint32_t>> index_;
};

int run_encode(int w, int h, const char* in, int scale, const char* avi, const char* planes_path, shq::matrix m)
{
    const auto frames = read_frames(in, w, h);
    if (frames.empty()) {
        std::fprintf(stderr, "no frames in %s\n", in);
        return 1;
    }
    auto enc = open_encoder(w, h, m);
    if (!enc)
        return 1;

    avi_writer    out(avi, w, h);
    std::ofstream planes(planes_path, std::ios::binary);
    size_t        total = 0;
    int           bad   = 0;
    for (size_t n = 0; n < frames.size(); ++n) {
        std::memcpy(enc->input(0), frames[n].data(), frames[n].size());
        const uint8_t*    data = nullptr;
        size_t            size = 0;
        std::string       error;
        shq::debug_output debug;
        if (!enc->encode(scale, data, size, error, &debug)) {
            std::fprintf(stderr, "frame %zu: %s\n", n, error.c_str());
            return 1;
        }
        const auto ref = shq::pack_reference(w, h, scale, debug.coef.data());
        if (ref.size() != size || std::memcmp(ref.data(), data, size) != 0) {
            size_t at = 0;
            while (at < std::min(ref.size(), size) && ref[at] == data[at])
                ++at;
            std::printf("frame %zu: MISMATCH gpu %zu bytes, reference %zu bytes, first difference at byte %zu\n",
                        n,
                        size,
                        ref.size(),
                        at);
            ++bad;
        }
        // SHQ_WRITE_REFERENCE=1 writes the serial packer's bitstream instead, to tell which side is wrong.
        if (std::getenv("SHQ_WRITE_REFERENCE"))
            out.add(ref.data(), ref.size());
        else
            out.add(data, size);
        planes.write(reinterpret_cast<const char*>(debug.planes.data()), debug.planes.size());
        total += size;
        if (n < 3 || n + 1 == frames.size())
            std::printf("frame %zu: %zu bytes, gpu %.3f ms\n", n, size, enc->gpu_ms(0));
    }
    std::printf("%zu frames, scale %d (quality byte %d), %.0f bytes/frame average, %.1f Mbit/s at 50p, "
                "%d mismatches with the reference packer\n",
                frames.size(),
                scale,
                100 - scale,
                double(total) / frames.size(),
                double(total) / frames.size() * 8 * 50 / 1e6,
                bad);
    return bad ? 2 : 0;
}

int run_bench(int w, int h, const char* in, int fixed_scale, double seconds, double mbps)
{
    const auto frames = read_frames(in, w, h);
    auto       enc    = open_encoder(w, h, shq::matrix::bt709);
    if (frames.empty() || !enc)
        return 1;

    shq::rate_control rate(mbps * 1e6 / 8 / 50);
    // SHQ_NO_COPY=1 leaves the pinned inputs alone, to separate the frame copy from the encode.
    const bool copy = !std::getenv("SHQ_NO_COPY");
    for (int slot = 0; slot < enc->slots(); ++slot)
        std::memcpy(enc->input(slot), frames[0].data(), frames[0].size());

    struct pending
    {
        int slot;
        int scale;
    };
    std::vector<pending> queue;
    std::vector<double>  gpu, wait;
    std::vector<int>     scales;
    size_t               bytes = 0, sent = 0;
    int                  next_slot = 0;
    const double         cpu0      = cpu_seconds();
    const double         t0        = now_seconds();
    timespec             tick{};
    clock_gettime(CLOCK_MONOTONIC, &tick);
    for (size_t n = 0; now_seconds() - t0 < seconds; ++n) {
        if (!queue.empty()) {
            const double   c0   = now_seconds();
            const uint8_t* data = nullptr;
            size_t         size = 0;
            std::string    error;
            if (enc->collect(queue.front().slot, 100, data, size, error) != shq::status::ready) {
                std::fprintf(stderr, "collect: %s\n", error.empty() ? "timed out" : error.c_str());
                return 1;
            }
            wait.push_back((now_seconds() - c0) * 1000);
            gpu.push_back(enc->gpu_ms(queue.front().slot));
            scales.push_back(queue.front().scale);
            rate.update(queue.front().scale, size);
            bytes += size;
            ++sent;
            queue.erase(queue.begin());
        }
        const int slot = next_slot;
        next_slot      = (slot + 1) % enc->slots();
        if (copy)
            std::memcpy(enc->input(slot), frames[n % frames.size()].data(), frames[0].size());
        const int   scale = fixed_scale > 0 ? fixed_scale : rate.scale();
        std::string error;
        if (!enc->submit(slot, scale, error)) {
            std::fprintf(stderr, "submit: %s\n", error.c_str());
            return 1;
        }
        queue.push_back({slot, scale});

        tick.tv_nsec += 20000000;
        if (tick.tv_nsec >= 1000000000) {
            tick.tv_nsec -= 1000000000;
            ++tick.tv_sec;
        }
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &tick, nullptr);
    }
    const double elapsed = now_seconds() - t0;
    const double cpu     = cpu_seconds() - cpu0;
    auto         stats   = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        double sum = 0;
        for (double x : v)
            sum += x;
        std::printf("mean %.3f ms, p50 %.3f, p99 %.3f, max %.3f\n",
                    sum / v.size(),
                    v[v.size() / 2],
                    v[v.size() * 99 / 100],
                    v.back());
    };
    std::printf("%zu frames in %.1f s, %.0f bytes/frame, %.1f Mbit/s\n",
                sent,
                elapsed,
                double(bytes) / sent,
                double(bytes) * 8 / elapsed / 1e6);
    std::sort(scales.begin(), scales.end());
    std::printf("scale min %d, p50 %d, max %d\n", scales.front(), scales[scales.size() / 2], scales.back());
    std::printf("gpu (upload to bitstream in host memory): ");
    stats(gpu);
    std::printf("wait in collect: ");
    stats(wait);
    std::printf("process cpu, %s the 8 MB frame copy into pinned memory: %.1f%% of one core\n",
                copy ? "including" : "without",
                cpu / elapsed * 100);
    return 0;
}

const NDIlib_v6* load_ndi()
{
    void* lib = dlopen(NDILIB_LIBRARY_NAME, RTLD_LOCAL | RTLD_LAZY);
    if (!lib)
        lib = dlopen("/opt/ndi-lib/" NDILIB_LIBRARY_NAME, RTLD_LOCAL | RTLD_LAZY);
    if (!lib) {
        std::fprintf(stderr, "cannot load %s\n", NDILIB_LIBRARY_NAME);
        return nullptr;
    }
    auto             load = reinterpret_cast<const NDIlib_v6* (*)()>(dlsym(lib, "NDIlib_v6_load"));
    const NDIlib_v6* ndi  = load ? load() : nullptr;
    if (!ndi || !ndi->initialize()) {
        std::fprintf(stderr, "NDI did not initialize\n");
        return nullptr;
    }
    return ndi;
}

// Receives from the first source whose name contains `name`.
NDIlib_recv_instance_t connect(const NDIlib_v6* ndi, const char* name)
{
    NDIlib_find_create_t find_settings;
    find_settings.show_local_sources = true;
    auto finder                      = ndi->find_create_v2(&find_settings);
    for (int tries = 0; tries < 20; ++tries) {
        ndi->find_wait_for_sources(finder, 500);
        uint32_t count   = 0;
        auto*    sources = ndi->find_get_current_sources(finder, &count);
        for (uint32_t i = 0; i < count; ++i) {
            if (std::strstr(sources[i].p_ndi_name, name)) {
                NDIlib_recv_create_v3_t settings;
                settings.source_to_connect_to = sources[i];
                settings.color_format         = NDIlib_recv_color_format_UYVY_BGRA;
                settings.bandwidth            = NDIlib_recv_bandwidth_highest;
                settings.allow_video_fields   = false;
                settings.p_ndi_recv_name      = "SHQ CUDA TEST RECV";
                auto recv                     = ndi->recv_create_v3(&settings);
                ndi->find_destroy(finder);
                return recv;
            }
        }
    }
    ndi->find_destroy(finder);
    std::fprintf(stderr, "no NDI source named %s\n", name);
    return nullptr;
}

struct plane_error
{
    double y_psnr;
    double c_psnr;
    int    y_max;
    int    c_max;
};

// UYVY from NDI against planar Y, Cb, Cr.
plane_error compare_uyvy(const NDIlib_video_frame_v2_t& f, const uint8_t* planes, int w, int h)
{
    const uint8_t* cb = planes + static_cast<size_t>(w) * h;
    const uint8_t* cr = cb + static_cast<size_t>(w) * h / 2;
    double         ey = 0, ec = 0;
    int            my = 0, mc = 0;
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = f.p_data + static_cast<size_t>(y) * f.line_stride_in_bytes;
        for (int x = 0; x < w / 2; ++x) {
            const int du  = row[x * 4 + 0] - cb[y * (w / 2) + x];
            const int dy0 = row[x * 4 + 1] - planes[y * w + 2 * x];
            const int dv  = row[x * 4 + 2] - cr[y * (w / 2) + x];
            const int dy1 = row[x * 4 + 3] - planes[y * w + 2 * x + 1];
            ey += dy0 * dy0 + dy1 * dy1;
            ec += du * du + dv * dv;
            my = std::max({my, std::abs(dy0), std::abs(dy1)});
            mc = std::max({mc, std::abs(du), std::abs(dv)});
        }
    }
    auto psnr = [](double sse, double n) { return sse == 0 ? 99.0 : 10 * std::log10(255.0 * 255.0 * n / sse); };
    return {psnr(ey, double(w) * h), psnr(ec, double(w) * h), my, mc};
}

NDIlib_video_frame_v2_t video_frame(int w, int h)
{
    NDIlib_video_frame_v2_t f;
    f.xres                 = w;
    f.yres                 = h;
    f.frame_rate_N         = 50;
    f.frame_rate_D         = 1;
    f.picture_aspect_ratio = static_cast<float>(w) / h;
    f.frame_format_type    = NDIlib_frame_format_type_progressive;
    f.timecode             = NDIlib_send_timecode_synthesize;
    return f;
}

bool capture(const NDIlib_v6* ndi, NDIlib_recv_instance_t recv, NDIlib_video_frame_v2_t& out, int timeout_ms)
{
    const double deadline = now_seconds() + timeout_ms / 1000.0;
    while (now_seconds() < deadline) {
        if (ndi->recv_capture_v3(recv, &out, nullptr, nullptr, 200) == NDIlib_frame_type_video)
            return true;
    }
    return false;
}

int run_ndi(int w, int h, const char* in, int scale, double seconds)
{
    const auto frames = read_frames(in, w, h);
    auto       enc    = open_encoder(w, h, shq::matrix::bt709);
    auto*      ndi    = load_ndi();
    if (frames.empty() || !enc || !ndi)
        return 1;

    // Encode everything first, so the send loop only sends.
    std::vector<std::vector<uint8_t>> packets, planes;
    for (auto& f : frames) {
        std::memcpy(enc->input(0), f.data(), f.size());
        const uint8_t*    data = nullptr;
        size_t            size = 0;
        std::string       error;
        shq::debug_output debug;
        if (!enc->encode(scale, data, size, error, &debug)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        packets.emplace_back(data, data + size);
        planes.push_back(std::move(debug.planes));
    }

    NDIlib_send_create_t send_settings;
    send_settings.p_ndi_name  = "SHQ CUDA TEST";
    send_settings.clock_video = true;
    auto send                 = ndi->send_create(&send_settings);
    auto recv                 = connect(ndi, "SHQ CUDA TEST");
    if (!send || !recv)
        return 1;

    // The packet index rides in the timecode, so each received frame is compared with its own source.
    auto frame            = video_frame(w, h);
    frame.FourCC          = static_cast<NDIlib_FourCC_video_type_e>(NDI_LIB_FOURCC('S', 'H', 'Q', '2'));
    int          received = 0;
    double       worst_y = 99, worst_c = 99;
    int          max_y = 0, max_c = 0;
    const double t0 = now_seconds();
    size_t       n  = 0;
    while (now_seconds() - t0 < seconds) {
        const size_t i           = n % packets.size();
        frame.p_data             = const_cast<uint8_t*>(packets[i].data());
        frame.data_size_in_bytes = static_cast<int>(packets[i].size());
        frame.timecode           = static_cast<int64_t>(i);
        ndi->send_send_video_v2(send, &frame);
        ++n;

        NDIlib_video_frame_v2_t got;
        while (ndi->recv_capture_v3(recv, &got, nullptr, nullptr, 0) == NDIlib_frame_type_video) {
            if (got.xres == w && got.yres == h && got.FourCC == NDIlib_FourCC_video_type_UYVY && got.timecode >= 0 &&
                got.timecode < static_cast<int64_t>(planes.size())) {
                const auto e = compare_uyvy(got, planes[got.timecode].data(), w, h);
                worst_y      = std::min(worst_y, e.y_psnr);
                worst_c      = std::min(worst_c, e.c_psnr);
                max_y        = std::max(max_y, e.y_max);
                max_c        = std::max(max_c, e.c_max);
                ++received;
            }
            ndi->recv_free_video_v2(recv, &got);
        }
    }
    NDIlib_recv_performance_t total, dropped;
    ndi->recv_get_performance(recv, &total, &dropped);
    std::printf("sent %zu, compared %d, receiver total %lld video, dropped %lld\n",
                n,
                received,
                static_cast<long long>(total.video_frames),
                static_cast<long long>(dropped.video_frames));
    std::printf("libndi decode against the encoder's input: worst Y %.2f dB (max error %d), worst CbCr %.2f dB "
                "(max error %d)\n",
                worst_y,
                max_y,
                worst_c,
                max_c);
    ndi->recv_destroy(recv);
    ndi->send_destroy(send);
    return received > 0 ? 0 : 1;
}

void cpu_convert(const uint8_t* bgra, int w, int h, shq::matrix m, std::vector<uint8_t>& planes)
{
    const double kr = m == shq::matrix::bt709 ? 0.2126 : 0.299;
    const double kb = m == shq::matrix::bt709 ? 0.0722 : 0.114;
    const double kg = 1 - kr - kb;
    planes.assign(static_cast<size_t>(w) * h * 2, 0);
    uint8_t* cb    = planes.data() + static_cast<size_t>(w) * h;
    uint8_t* cr    = cb + static_cast<size_t>(w) * h / 2;
    auto     clamp = [](double v) { return static_cast<uint8_t>(std::min(255.0, std::max(0.0, std::floor(v + 0.5)))); };
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; x += 2) {
            const uint8_t* p = bgra + (static_cast<size_t>(y) * w + x) * 4;
            for (int i = 0; i < 2; ++i) {
                const double e        = kr * p[i * 4 + 2] + kg * p[i * 4 + 1] + kb * p[i * 4];
                planes[y * w + x + i] = clamp(16 + e * 219 / 255);
            }
            const double r = (p[2] + p[6]) / 2.0, g = (p[1] + p[5]) / 2.0, b = (p[0] + p[4]) / 2.0;
            const double e          = kr * r + kg * g + kb * b;
            cb[y * (w / 2) + x / 2] = clamp(128 + (b - e) / (2 * (1 - kb)) * 224 / 255);
            cr[y * (w / 2) + x / 2] = clamp(128 + (r - e) / (2 * (1 - kr)) * 224 / 255);
        }
    }
}

int run_ndi_matrix(int w, int h, const char* in)
{
    const auto frames = read_frames(in, w, h);
    auto*      ndi    = load_ndi();
    if (frames.empty() || !ndi)
        return 1;

    NDIlib_send_create_t send_settings;
    send_settings.p_ndi_name  = "SHQ CUDA MATRIX";
    send_settings.clock_video = true;
    auto send                 = ndi->send_create(&send_settings);
    auto recv                 = connect(ndi, "SHQ CUDA MATRIX");
    if (!send || !recv)
        return 1;

    auto frame                 = video_frame(w, h);
    frame.FourCC               = NDIlib_FourCC_type_BGRX;
    frame.p_data               = const_cast<uint8_t*>(frames[0].data());
    frame.line_stride_in_bytes = w * 4;

    NDIlib_video_frame_v2_t got;
    bool                    have = false;
    for (int i = 0; i < 100 && !have; ++i) {
        ndi->send_send_video_v2(send, &frame);
        while (ndi->recv_capture_v3(recv, &got, nullptr, nullptr, 0) == NDIlib_frame_type_video) {
            if (got.FourCC == NDIlib_FourCC_video_type_UYVY && got.xres == w && got.yres == h) {
                have = true;
                break;
            }
            ndi->recv_free_video_v2(recv, &got);
        }
    }
    if (!have && capture(ndi, recv, got, 2000))
        have = true;
    if (!have) {
        std::fprintf(stderr, "no UYVY frame came back\n");
        return 1;
    }
    for (auto m : {shq::matrix::bt709, shq::matrix::bt601}) {
        std::vector<uint8_t> planes;
        cpu_convert(frames[0].data(), w, h, m, planes);
        const auto e = compare_uyvy(got, planes.data(), w, h);
        std::printf("libndi BGRX -> UYVY against %s: Y %.2f dB (max %d), CbCr %.2f dB (max %d)\n",
                    m == shq::matrix::bt709 ? "BT.709" : "BT.601",
                    e.y_psnr,
                    e.y_max,
                    e.c_psnr,
                    e.c_max);
    }
    ndi->recv_free_video_v2(recv, &got);
    ndi->recv_destroy(recv);
    ndi->send_destroy(send);
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 5) {
        std::fprintf(stderr, "usage: see the comment at the top of shq_test.cpp\n");
        return 1;
    }
    const std::string cmd = argv[1];
    const int         w   = std::atoi(argv[2]);
    const int         h   = std::atoi(argv[3]);
    const char*       in  = argv[4];
    if (cmd == "encode" && argc >= 8)
        return run_encode(w,
                          h,
                          in,
                          std::atoi(argv[5]),
                          argv[6],
                          argv[7],
                          argc > 8 && std::string(argv[8]) == "601" ? shq::matrix::bt601 : shq::matrix::bt709);
    if (cmd == "bench" && argc >= 7)
        return run_bench(w, h, in, std::atoi(argv[5]), std::atof(argv[6]), argc > 7 ? std::atof(argv[7]) : 124.0);
    if (cmd == "ndi" && argc >= 7)
        return run_ndi(w, h, in, std::atoi(argv[5]), std::atof(argv[6]));
    if (cmd == "ndi-matrix")
        return run_ndi_matrix(w, h, in);
    std::fprintf(stderr, "usage: see the comment at the top of shq_test.cpp\n");
    return 1;
}
