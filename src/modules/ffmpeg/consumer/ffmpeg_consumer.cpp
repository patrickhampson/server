/*
 * Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
 *
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
 *
 * Author: Robert Nagy, ronag89@gmail.com
 */

#include "ffmpeg_consumer.h"

#include "../util/av_assert.h"
#include "../util/av_util.h"

#include <common/bit_depth.h>
#include <common/diagnostics/graph.h>
#include <common/env.h>
#include <common/executor.h>
#include <common/future.h>
#include <common/memory.h>
#include <common/scope_exit.h>
#include <common/timer.h>

#include <core/consumer/channel_info.h>
#include <core/frame/frame.h>
#include <core/video_format.h>

#if defined(__GNUC__) && __GNUC__ == 14
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/regex.hpp>
#if defined(__GNUC__) && __GNUC__ == 14
#pragma GCC diagnostic pop
#endif

#include <boost/crc.hpp>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavutil/buffer.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libavutil/pixfmt.h>
#include <libavutil/samplefmt.h>
}

#include <tbb/concurrent_queue.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include <cstring>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace caspar { namespace ffmpeg {

// TODO multiple output streams
// TODO multiple output files
// TODO run video filter, video encoder, audio filter, audio encoder in separate threads.
// TODO realtime with smaller buffer?

#ifndef _WIN32
// Page-locked frames for hwupload_cuda. cuMemcpy2DAsync DMAs from page-locked memory; from a pageable AVFrame the
// driver first copies the whole frame on the CPU.
struct pinned_frames
{
    void* lib                 = nullptr;
    void* ctx                 = nullptr;
    int (*ctx_set)(void*)     = nullptr;
    int (*ctx_destroy)(void*) = nullptr;
    int (*host_free)(void*)   = nullptr;
    std::vector<void*>                    blocks;
    std::vector<std::shared_ptr<AVFrame>> frames;
    size_t                                next = 0;

    pinned_frames()                                = default;
    pinned_frames(const pinned_frames&)            = delete;
    pinned_frames& operator=(const pinned_frames&) = delete;

    ~pinned_frames()
    {
        frames.clear();
        if (ctx) {
            ctx_set(ctx);
            for (auto* block : blocks) {
                host_free(block);
            }
            ctx_destroy(ctx);
        }
        if (lib) {
            dlclose(lib);
        }
    }

    bool prepare(int width, int height, AVRational sar)
    {
        lib = dlopen("libcuda.so.1", RTLD_LAZY);
        if (!lib) {
            return false;
        }
        auto init       = reinterpret_cast<int (*)(unsigned)>(dlsym(lib, "cuInit"));
        auto device_get = reinterpret_cast<int (*)(int*, int)>(dlsym(lib, "cuDeviceGet"));
        auto ctx_create = reinterpret_cast<int (*)(void**, unsigned, int)>(dlsym(lib, "cuCtxCreate_v2"));
        auto ctx_pop    = reinterpret_cast<int (*)(void**)>(dlsym(lib, "cuCtxPopCurrent_v2"));
        auto host_alloc = reinterpret_cast<int (*)(void**, size_t, unsigned)>(dlsym(lib, "cuMemHostAlloc"));
        ctx_set         = reinterpret_cast<int (*)(void*)>(dlsym(lib, "cuCtxSetCurrent"));
        ctx_destroy     = reinterpret_cast<int (*)(void*)>(dlsym(lib, "cuCtxDestroy_v2"));
        host_free       = reinterpret_cast<int (*)(void*)>(dlsym(lib, "cuMemFreeHost"));
        if (!init || !device_get || !ctx_create || !ctx_pop || !host_alloc || !ctx_set || !ctx_destroy || !host_free) {
            return false;
        }
        int device = 0;
        if (init(0) != 0 || device_get(&device, 0) != 0 || ctx_create(&ctx, 0, device) != 0) {
            ctx = nullptr;
            return false;
        }

        const auto bytes = static_cast<size_t>(width) * height * 4;
        for (int n = 0; n < 3; ++n) {
            void* host = nullptr;
            // CU_MEMHOSTALLOC_PORTABLE: the context hwupload_cuda creates can DMA from it too.
            if (host_alloc(&host, bytes, 0x01) != 0) {
                break;
            }
            blocks.push_back(host);
            // The memory is freed by the destructor, after the filter graph has released the frames.
            auto buf = av_buffer_create(static_cast<uint8_t*>(host), bytes, [](void*, uint8_t*) {}, nullptr, 0);
            if (!buf) {
                break;
            }
            auto frame                 = alloc_frame();
            frame->format              = AV_PIX_FMT_BGR0;
            frame->width               = width;
            frame->height              = height;
            frame->sample_aspect_ratio = sar;
            frame->buf[0]              = buf;
            frame->data[0]             = buf->data;
            frame->linesize[0]         = width * 4;
            frames.push_back(std::move(frame));
        }
        void* popped = nullptr;
        ctx_pop(&popped);
        return frames.size() == 3;
    }

    // hwupload_cuda releases its input when the copy is queued, not when the DMA has finished. Frames are therefore
    // reused in order, so a frame is only rewritten after the two that follow it were sent. Returns null when the
    // next frame is still referenced, or the source is not a tightly packed BGRA frame of the same size.
    std::shared_ptr<AVFrame> acquire(const core::const_frame& src)
    {
        auto&      frame = frames[next];
        const auto bytes = static_cast<size_t>(frame->linesize[0]) * frame->height;
        if (src.pixel_format_desc().format != core::pixel_format::bgra ||
            src.pixel_format_desc().planes.at(0).linesize != frame->linesize[0] || src.image_data(0).size() < bytes ||
            av_buffer_get_ref_count(frame->buf[0]) != 1) {
            return nullptr;
        }
        next = (next + 1) % frames.size();
        std::memcpy(frame->data[0], src.image_data(0).data(), bytes);
        return frame;
    }
};
#endif

struct Stream
{
#ifndef _WIN32
    // Declared first so that it outlives the filter graph, which may still reference its frames.
    std::unique_ptr<pinned_frames> pinned;
#endif

    std::shared_ptr<AVFilterGraph> graph  = nullptr;
    AVFilterContext*               sink   = nullptr;
    AVFilterContext*               source = nullptr;

    std::shared_ptr<AVCodecContext> enc = nullptr;
    AVStream*                       st  = nullptr;

    Stream(AVFormatContext*                    oc,
           std::string                         suffix,
           AVCodecID                           codec_id,
           const core::video_format_desc&      format_desc,
           bool                                realtime,
           common::bit_depth                   depth,
           std::map<std::string, std::string>& options)
    {
        std::map<std::string, std::string> stream_options;

        {
            auto tmp = std::move(options);
            for (auto& p : tmp) {
                if (boost::algorithm::ends_with(p.first, suffix)) {
                    const auto key = p.first.substr(0, p.first.size() - suffix.size());
                    stream_options.emplace(key, std::move(p.second));
                } else {
                    options.insert(std::move(p));
                }
            }
        }

        std::string filter_spec = "";
        {
            const auto it = stream_options.find("filter");
            if (it != stream_options.end()) {
                filter_spec = std::move(it->second);
                stream_options.erase(it);
            }
        }

        auto codec = avcodec_find_encoder(codec_id);
        {
            const auto it = stream_options.find("codec");
            if (it != stream_options.end()) {
                codec = avcodec_find_encoder_by_name(it->second.c_str());
                stream_options.erase(it);
            }
        }

        if (!codec) {
            FF_RET(AVERROR(EINVAL), "avcodec_find_encoder");
        }

        AVFilterInOut* outputs = nullptr;
        AVFilterInOut* inputs  = nullptr;

        CASPAR_SCOPE_EXIT
        {
            avfilter_inout_free(&inputs);
            avfilter_inout_free(&outputs);
        };

        graph = std::shared_ptr<AVFilterGraph>(avfilter_graph_alloc(),
                                               [](AVFilterGraph* ptr) { avfilter_graph_free(&ptr); });

        if (!graph) {
            FF_RET(AVERROR(ENOMEM), "avfilter_graph_alloc");
        }

        if (codec->type == AVMEDIA_TYPE_VIDEO) {
            if (filter_spec.empty()) {
                filter_spec = "null";
            }
        } else {
            if (filter_spec.empty()) {
                filter_spec = "anull";
            }
        }

        FF(avfilter_graph_parse2(graph.get(), filter_spec.c_str(), &inputs, &outputs));

        {
            auto cur = inputs;

            if (!cur || cur->next) {
                CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_errno(EINVAL)
                                                        << msg_info_t("invalid filter graph input count"));
            }

            if (codec->type == AVMEDIA_TYPE_VIDEO) {
                const auto sar = boost::rational<int>(format_desc.square_width, format_desc.square_height) /
                                 boost::rational<int>(format_desc.width, format_desc.height);

                auto pix_fmt = (depth == common::bit_depth::bit8) ? AV_PIX_FMT_BGRA : AV_PIX_FMT_BGRA64LE;
#ifndef _WIN32
                // BGRA and bgr0 are the same bytes. hwupload_cuda accepts bgr0 in every FFmpeg version (BGRA only
                // from 8.0), so the engine's frames are labelled bgr0 and uploaded from page-locked memory.
                if (depth == common::bit_depth::bit8 && filter_spec.find("hwupload_cuda") != std::string::npos) {
                    auto frames = std::make_unique<pinned_frames>();
                    if (frames->prepare(format_desc.width, format_desc.height, {sar.numerator(), sar.denominator()})) {
                        pinned  = std::move(frames);
                        pix_fmt = AV_PIX_FMT_BGR0;
                        CASPAR_LOG(info) << L"ffmpeg consumer: pinned CUDA upload " << format_desc.width << L"x"
                                         << format_desc.height;
                    } else {
                        CASPAR_LOG(warning) << L"ffmpeg consumer: pinned CUDA upload unavailable";
                    }
                }
#endif

                auto args = (boost::format("video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:sar=%d/%d:frame_rate=%d/%d") %
                             format_desc.width % format_desc.height % pix_fmt % format_desc.duration %
                             (format_desc.time_scale * format_desc.field_count) % sar.numerator() % sar.denominator() %
                             (format_desc.framerate.numerator() * format_desc.field_count) %
                             format_desc.framerate.denominator())
                                .str();
                auto name = (boost::format("in_%d") % 0).str();

                FF(avfilter_graph_create_filter(
                    &source, avfilter_get_by_name("buffer"), name.c_str(), args.c_str(), nullptr, graph.get()));
                FF(avfilter_link(source, 0, cur->filter_ctx, cur->pad_idx));
            } else if (codec->type == AVMEDIA_TYPE_AUDIO) {
                auto args = (boost::format("time_base=%d/%d:sample_rate=%d:sample_fmt=%s:channel_layout=%#x") % 1 %
                             format_desc.audio_sample_rate % format_desc.audio_sample_rate % AV_SAMPLE_FMT_S32 %
                             get_channel_layout_mask_for_channels(format_desc.audio_channels))
                                .str();
                auto name = (boost::format("in_%d") % 0).str();

                FF(avfilter_graph_create_filter(
                    &source, avfilter_get_by_name("abuffer"), name.c_str(), args.c_str(), nullptr, graph.get()));
                FF(avfilter_link(source, 0, cur->filter_ctx, cur->pad_idx));
            } else {
                CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_errno(EINVAL)
                                                        << msg_info_t("invalid filter input media type"));
            }
        }

        if (codec->type == AVMEDIA_TYPE_VIDEO) {
            sink = FFMEM(avfilter_graph_alloc_filter(graph.get(), avfilter_get_by_name("buffersink"), "out"));

            // TODO codec->profiles
            // TODO FF(av_opt_set_int_list(sink, "framerates", codec->supported_framerates, { 0, 0 },
            // AV_OPT_SEARCH_CHILDREN));
#if LIBAVUTIL_VERSION_MAJOR >= 60 // FFmpeg 8
            const void* pix_fmts;
            int         nb_pix_fmts = 0;
            FF(avcodec_get_supported_config(nullptr, codec, AV_CODEC_CONFIG_PIX_FORMAT, 0, &pix_fmts, &nb_pix_fmts));

            FF(av_opt_set_array(sink,
                                "pixel_formats",
                                AV_OPT_SEARCH_CHILDREN | AV_OPT_ARRAY_REPLACE,
                                0,
                                nb_pix_fmts,
                                AV_OPT_TYPE_PIXEL_FMT,
                                pix_fmts));
#else
            FF(av_opt_set_int_list(sink, "pix_fmts", codec->pix_fmts, -1, AV_OPT_SEARCH_CHILDREN));
#endif

        } else if (codec->type == AVMEDIA_TYPE_AUDIO) {
            sink = FFMEM(avfilter_graph_alloc_filter(graph.get(), avfilter_get_by_name("abuffersink"), "out"));
            // TODO codec->profiles

#if LIBAVUTIL_VERSION_MAJOR >= 60 // FFmpeg 8
            const void* sample_fmts;
            int         nb_sample_fmts = 0;
            FF(avcodec_get_supported_config(
                nullptr, codec, AV_CODEC_CONFIG_SAMPLE_FORMAT, 0, &sample_fmts, &nb_sample_fmts));

            FF(av_opt_set_array(sink,
                                "sample_formats",
                                AV_OPT_SEARCH_CHILDREN | AV_OPT_ARRAY_REPLACE,
                                0,
                                nb_sample_fmts,
                                AV_OPT_TYPE_SAMPLE_FMT,
                                sample_fmts));

            const void* sample_rates;
            int         nb_sample_rates = 0;
            FF(avcodec_get_supported_config(
                nullptr, codec, AV_CODEC_CONFIG_SAMPLE_RATE, 0, &sample_rates, &nb_sample_rates));

            FF(av_opt_set_array(sink,
                                "samplerates",
                                AV_OPT_SEARCH_CHILDREN | AV_OPT_ARRAY_REPLACE,
                                0,
                                nb_sample_rates,
                                AV_OPT_TYPE_INT,
                                sample_rates));
#else
            FF(av_opt_set_int_list(sink, "sample_fmts", codec->sample_fmts, -1, AV_OPT_SEARCH_CHILDREN));
            FF(av_opt_set_int_list(sink, "sample_rates", codec->supported_samplerates, 0, AV_OPT_SEARCH_CHILDREN));
#endif

            // TODO: need to translate codec->ch_layouts into something that can be passed via av_opt_set_*
            // FF(av_opt_set_chlayout(sink, "ch_layouts", codec->ch_layouts, AV_OPT_SEARCH_CHILDREN));

        } else {
            CASPAR_THROW_EXCEPTION(ffmpeg_error_t()
                                   << boost::errinfo_errno(EINVAL) << msg_info_t("invalid output media type"));
        }

        FF(avfilter_init_str(sink, nullptr));

        {
            const auto cur = outputs;

            if (!cur || cur->next) {
                CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_errno(EINVAL)
                                                        << msg_info_t("invalid filter graph output count"));
            }

            if (avfilter_pad_get_type(cur->filter_ctx->output_pads, cur->pad_idx) != codec->type) {
                CASPAR_THROW_EXCEPTION(ffmpeg_error_t() << boost::errinfo_errno(EINVAL)
                                                        << msg_info_t("invalid filter output media type"));
            }

            FF(avfilter_link(cur->filter_ctx, cur->pad_idx, sink, 0));
        }

        FF(avfilter_graph_config(graph.get(), nullptr));

        st = avformat_new_stream(oc, nullptr);
        if (!st) {
            FF_RET(AVERROR(ENOMEM), "avformat_new_stream");
        }

        enc = std::shared_ptr<AVCodecContext>(avcodec_alloc_context3(codec),
                                              [](AVCodecContext* ptr) { avcodec_free_context(&ptr); });

        if (!enc) {
            FF_RET(AVERROR(ENOMEM), "avcodec_alloc_context3")
        }

        if (codec->type == AVMEDIA_TYPE_VIDEO) {
            st->time_base = av_inv_q(av_buffersink_get_frame_rate(sink));

            // Ensure the frame_rate is set in a way that rtmp will find it
            st->avg_frame_rate = av_buffersink_get_frame_rate(sink);

            enc->width               = av_buffersink_get_w(sink);
            enc->height              = av_buffersink_get_h(sink);
            enc->framerate           = av_buffersink_get_frame_rate(sink);
            enc->sample_aspect_ratio = av_buffersink_get_sample_aspect_ratio(sink);
            enc->time_base           = st->time_base;
            enc->pix_fmt             = static_cast<AVPixelFormat>(av_buffersink_get_format(sink));
            // Hardware frames from the filter graph (e.g. hwupload_cuda) need the frames context on the encoder.
            // The sink returns the graph's reference, and the codec context unrefs its own.
            if (auto hw_frames = av_buffersink_get_hw_frames_ctx(sink)) {
                enc->hw_frames_ctx = av_buffer_ref(hw_frames);
                if (!enc->hw_frames_ctx) {
                    FF_RET(AVERROR(ENOMEM), "av_buffer_ref");
                }
            }
        } else if (codec->type == AVMEDIA_TYPE_AUDIO) {
            st->time_base = {1, av_buffersink_get_sample_rate(sink)};

            enc->sample_fmt  = static_cast<AVSampleFormat>(av_buffersink_get_format(sink));
            enc->sample_rate = av_buffersink_get_sample_rate(sink);
            enc->time_base   = st->time_base;

            FF(av_buffersink_get_ch_layout(sink, &enc->ch_layout));

        } else {
            // TODO
        }

        if (realtime && codec->capabilities & AV_CODEC_CAP_SLICE_THREADS) {
            enc->thread_type = FF_THREAD_SLICE;
        }

        if (oc->oformat->flags & AVFMT_GLOBALHEADER) {
            enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }

        auto dict = to_dict(std::move(stream_options));
        CASPAR_SCOPE_EXIT { av_dict_free(&dict); };
        FF(avcodec_open2(enc.get(), codec, &dict));
        for (auto& p : to_map(&dict)) {
            options[p.first] = p.second + suffix;
        }

        FF(avcodec_parameters_from_context(st->codecpar, enc.get()));

        if (codec->type == AVMEDIA_TYPE_AUDIO && !(codec->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE)) {
            av_buffersink_set_frame_size(sink, enc->frame_size);
        }
    }

    void send(std::tuple<core::const_frame, std::int64_t, std::int64_t>& data,
              const core::video_format_desc&                             format_desc,
              std::function<void(std::shared_ptr<AVPacket>)>             cb)
    {
        std::shared_ptr<AVFrame>  frame;
        std::shared_ptr<AVPacket> pkt;

        const auto [in_frame, video_pts, audio_pts] = data;

        if (in_frame) {
            if (enc->codec_type == AVMEDIA_TYPE_VIDEO) {
#ifndef _WIN32
                if (pinned) {
                    frame = pinned->acquire(in_frame);
                }
#endif
                if (!frame) {
                    frame = make_av_video_frame(in_frame, format_desc);
#ifndef _WIN32
                    if (pinned && frame->format == AV_PIX_FMT_BGRA) {
                        frame->format = AV_PIX_FMT_BGR0;
                    }
#endif
                }
                frame->pts = video_pts;
            } else if (enc->codec_type == AVMEDIA_TYPE_AUDIO) {
                frame      = make_av_audio_frame(in_frame, format_desc);
                frame->pts = audio_pts;
            } else {
                // TODO
            }
            FF(av_buffersrc_write_frame(source, frame.get()));
        } else {
            FF(av_buffersrc_close(source, AV_NOPTS_VALUE, 0));
        }

        while (true) {
            pkt     = alloc_packet();
            int ret = avcodec_receive_packet(enc.get(), pkt.get());

            if (ret == AVERROR(EAGAIN)) {
                frame = alloc_frame();
                ret   = av_buffersink_get_frame(sink, frame.get());
                if (ret == AVERROR(EAGAIN)) {
                    return;
                }
                if (ret == AVERROR_EOF) {
                    FF(avcodec_send_frame(enc.get(), nullptr));
                } else {
                    FF_RET(ret, "av_buffersink_get_frame");
                    FF(avcodec_send_frame(enc.get(), frame.get()));
                }
            } else if (ret == AVERROR_EOF) {
                return;
            } else {
                FF_RET(ret, "avcodec_receive_packet");
                pkt->stream_index = st->index;
                av_packet_rescale_ts(pkt.get(), enc->time_base, st->time_base);
                cb(std::move(pkt));
            }
        }
    }
};

struct ffmpeg_consumer : public core::frame_consumer
{
    core::monitor::state    state_;
    mutable std::mutex      state_mutex_;
    int                     channel_index_ = -1;
    core::video_format_desc format_desc_;
    bool                    realtime_ = false;
    std::int64_t            video_pts = 0;
    std::int64_t            audio_pts = 0;

    spl::shared_ptr<diagnostics::graph> graph_;

    std::string path_;
    std::string args_;

    std::exception_ptr exception_;
    std::mutex         exception_mutex_;

    tbb::concurrent_bounded_queue<std::tuple<core::const_frame, std::int64_t, std::int64_t>> frame_buffer_;
    std::thread                                                                              frame_thread_;

    common::bit_depth depth_;

  public:
    ffmpeg_consumer(std::string path, std::string args, bool realtime, common::bit_depth depth)
        : channel_index_([&] {
            boost::crc_16_type result;
            result.process_bytes(path.data(), path.length());
            return result.checksum();
        }())
        , realtime_(realtime)
        , path_(std::move(path))
        , args_(std::move(args))
        , depth_(depth)
    {
        state_["file/path"] = u8(path_);

        frame_buffer_.set_capacity(realtime_ ? 1 : 64);

        diagnostics::register_graph(graph_);
        graph_->set_color("frame-time", diagnostics::color(0.1f, 1.0f, 0.1f));
        graph_->set_color("dropped-frame", diagnostics::color(0.3f, 0.6f, 0.3f));
        graph_->set_color("input", diagnostics::color(0.7f, 0.4f, 0.4f));
    }

    ~ffmpeg_consumer()
    {
        if (frame_thread_.joinable()) {
            frame_buffer_.push({core::const_frame{}, -1, -1});
            frame_thread_.join();
        }
    }

    // frame consumer

    void initialize(const core::video_format_desc& format_desc,
                    const core::channel_info&      channel_info,
                    int                            port_index) override
    {
        if (frame_thread_.joinable()) {
            CASPAR_THROW_EXCEPTION(invalid_operation() << msg_info("Cannot reinitialize ffmpeg-consumer."));
        }

        format_desc_   = format_desc;
        channel_index_ = channel_info.index;

        graph_->set_text(print());

        frame_thread_ = std::thread([=, this] {
            try {
                std::map<std::string, std::string> options;
                {
                    static boost::regex opt_exp("-(?<NAME>[^\\s]+)(\\s+(?<VALUE>[^\\s]+))?");
                    for (auto it = boost::sregex_iterator(args_.begin(), args_.end(), opt_exp);
                         it != boost::sregex_iterator();
                         ++it) {
                        options[(*it)["NAME"].str().c_str()] =
                            (*it)["VALUE"].matched ? (*it)["VALUE"].str().c_str() : "";
                    }
                }

                boost::filesystem::path full_path = path_;

                static boost::regex prot_exp("^.+:.*");
                if (!boost::regex_match(path_, prot_exp)) {
                    if (!full_path.is_absolute()) {
                        full_path = u8(env::media_folder()) + path_;
                    }

                    // TODO -y?
                    if (boost::filesystem::exists(full_path)) {
                        boost::filesystem::remove(full_path);
                    }

                    boost::filesystem::create_directories(full_path.parent_path());
                }

                AVFormatContext* oc = nullptr;

                {
                    std::string format;
                    {
                        const auto format_it = options.find("format");
                        if (format_it != options.end()) {
                            format = std::move(format_it->second);
                            options.erase(format_it);
                        }
                    }

                    FF(avformat_alloc_output_context2(
                        &oc, nullptr, !format.empty() ? format.c_str() : nullptr, path_.c_str()));
                }

                CASPAR_SCOPE_EXIT { avformat_free_context(oc); };

                std::optional<Stream> video_stream;
                if (oc->oformat->video_codec != AV_CODEC_ID_NONE) {
                    if (oc->oformat->video_codec == AV_CODEC_ID_H264 && options.find("preset:v") == options.end()) {
                        options["preset:v"] = "veryfast";
                    }
                    video_stream.emplace(oc, ":v", oc->oformat->video_codec, format_desc, realtime_, depth_, options);

                    {
                        std::lock_guard<std::mutex> lock(state_mutex_);
                        state_["file/fps"] = av_q2d(av_buffersink_get_frame_rate(video_stream->sink));
                    }
                }

                std::optional<Stream> audio_stream;
                if (oc->oformat->audio_codec != AV_CODEC_ID_NONE) {
                    audio_stream.emplace(oc, ":a", oc->oformat->audio_codec, format_desc, realtime_, depth_, options);
                }

                if (!(oc->oformat->flags & AVFMT_NOFILE)) {
                    // TODO (fix) interrupt_cb
                    auto dict = to_dict(std::move(options));
                    CASPAR_SCOPE_EXIT { av_dict_free(&dict); };
                    FF(avio_open2(&oc->pb, full_path.string().c_str(), AVIO_FLAG_WRITE, nullptr, &dict));
                    options = to_map(&dict);
                }

                {
                    auto dict = to_dict(std::move(options));
                    CASPAR_SCOPE_EXIT { av_dict_free(&dict); };
                    FF(avformat_write_header(oc, &dict));
                    options = to_map(&dict);
                }

                {
                    for (auto& p : options) {
                        CASPAR_LOG(warning) << print() << " Unused option " << p.first << "=" << p.second;
                    }
                }

                tbb::concurrent_bounded_queue<std::shared_ptr<AVPacket>> packet_buffer;
                packet_buffer.set_capacity(realtime_ ? 1 : 128);
                auto packet_thread = std::thread([&] {
                    try {
                        CASPAR_SCOPE_EXIT
                        {
                            if (!(oc->oformat->flags & AVFMT_NOFILE)) {
                                FF(avio_closep(&oc->pb));
                            }
                        };

                        std::map<int, int64_t> count;

                        std::shared_ptr<AVPacket> pkt;
                        while (true) {
                            packet_buffer.pop(pkt);
                            if (!pkt) {
                                break;
                            }
                            count[pkt->stream_index] += 1;
                            FF(av_interleaved_write_frame(oc, pkt.get()));
                        }

                        auto video_st = video_stream ? video_stream->st : nullptr;
                        auto audio_st = audio_stream ? audio_stream->st : nullptr;

                        if ((!video_st || count[video_st->index]) && (!audio_st || count[audio_st->index])) {
                            FF(av_write_trailer(oc));
                        }

                    } catch (...) {
                        CASPAR_LOG_CURRENT_EXCEPTION();
                        // TODO
                        packet_buffer.abort();
                    }
                });
                CASPAR_SCOPE_EXIT
                {
                    if (packet_thread.joinable()) {
                        // TODO Is nullptr needed?
                        packet_buffer.push(nullptr);
                        packet_buffer.abort();
                        packet_thread.join();
                    }
                };

                auto packet_cb = [&](std::shared_ptr<AVPacket>&& pkt) { packet_buffer.push(std::move(pkt)); };

                std::int64_t frame_number = 0;
                while (true) {
                    {
                        std::lock_guard<std::mutex> lock(state_mutex_);
                        state_["file/frame"] = frame_number++;
                    }

                    std::tuple<core::const_frame, std::int64_t, std::int64_t> data;
                    frame_buffer_.pop(data);
                    graph_->set_value("input",
                                      static_cast<double>(frame_buffer_.size() + 0.001) / frame_buffer_.capacity());

                    caspar::timer frame_timer;
                    tbb::parallel_invoke(
                        [&] {
                            if (video_stream) {
                                video_stream->send(data, format_desc, packet_cb);
                            }
                        },
                        [&] {
                            if (audio_stream) {
                                audio_stream->send(data, format_desc, packet_cb);
                            }
                        });
                    graph_->set_value("frame-time", frame_timer.elapsed() * format_desc.fps * 0.5);

                    if (!std::get<0>(data)) {
                        packet_buffer.push(nullptr);
                        break;
                    }
                }

                packet_thread.join();
            } catch (...) {
                std::lock_guard<std::mutex> lock(exception_mutex_);
                exception_ = std::current_exception();
            }
        });
    }

    std::future<bool> send(core::video_field field, core::const_frame frame) override
    {
        // TODO - field alignment

        {
            std::lock_guard<std::mutex> lock(exception_mutex_);
            if (exception_ != nullptr) {
                std::rethrow_exception(exception_);
            }
        }

        if (!frame_buffer_.try_push({frame, video_pts, audio_pts})) {
            graph_->set_tag(diagnostics::tag_severity::WARNING, "dropped-frame");
        }

        video_pts += 1;
        audio_pts += frame.audio_data().size() / format_desc_.audio_channels;

        graph_->set_value("input", static_cast<double>(frame_buffer_.size() + 0.001) / frame_buffer_.capacity());

        return make_ready_future(true);
    }

    std::wstring print() const override { return L"ffmpeg[" + u16(path_) + L"]"; }

    std::wstring name() const override { return L"ffmpeg"; }

    bool has_synchronization_clock() const override { return false; }

    int index() const override { return 100000 + channel_index_; }

    core::monitor::state state() const override
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return state_;
    }
};

spl::shared_ptr<core::frame_consumer> create_consumer(const std::vector<std::wstring>&     params,
                                                      const core::video_format_repository& format_repository,
                                                      const std::vector<spl::shared_ptr<core::video_channel>>& channels,
                                                      const core::channel_info& channel_info)
{
    if (params.size() < 2 || (!boost::iequals(params.at(0), L"STREAM") && !boost::iequals(params.at(0), L"FILE")))
        return core::frame_consumer::empty();

    auto                     path = u8(params.at(1));
    std::vector<std::string> args;
    for (auto n = 2; n < params.size(); ++n) {
        args.emplace_back(u8(params[n]));
    }
    return spl::make_shared<ffmpeg_consumer>(
        path, boost::join(args, " "), boost::iequals(params.at(0), L"STREAM"), channel_info.depth);
}

spl::shared_ptr<core::frame_consumer>
create_preconfigured_consumer(const boost::property_tree::wptree&                      ptree,
                              const core::video_format_repository&                     format_repository,
                              const std::vector<spl::shared_ptr<core::video_channel>>& channels,
                              const core::channel_info&                                channel_info)
{
    return spl::make_shared<ffmpeg_consumer>(u8(ptree.get<std::wstring>(L"path", L"")),
                                             u8(ptree.get<std::wstring>(L"args", L"")),
                                             ptree.get(L"realtime", false),
                                             channel_info.depth);
}
}} // namespace caspar::ffmpeg
