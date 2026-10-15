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

#include "shq_cuda.h"

#include "shq_format.h"
#include "shq_kernels_fatbin.h"

#include <dlfcn.h>

#include <chrono>
#include <cstring>
#include <thread>

namespace caspar { namespace newtek { namespace shq {

namespace {

using CUresult    = int;
using CUdeviceptr = unsigned long long;

const CUresult cuda_not_ready = 600;

// The handful of driver API calls the encoder makes, resolved from libcuda.so.1.
struct driver
{
    void* lib = nullptr;

    CUresult (*init)(unsigned)                                      = nullptr;
    CUresult (*device_get)(int*, int)                               = nullptr;
    CUresult (*primary_retain)(void**, int)                         = nullptr;
    CUresult (*primary_release)(int)                                = nullptr;
    CUresult (*ctx_push)(void*)                                     = nullptr;
    CUresult (*ctx_pop)(void**)                                     = nullptr;
    CUresult (*module_load)(void**, const void*)                    = nullptr;
    CUresult (*module_unload)(void*)                                = nullptr;
    CUresult (*module_function)(void**, void*, const char*)         = nullptr;
    CUresult (*mem_alloc)(CUdeviceptr*, size_t)                     = nullptr;
    CUresult (*mem_free)(CUdeviceptr)                               = nullptr;
    CUresult (*host_alloc)(void**, size_t, unsigned)                = nullptr;
    CUresult (*host_free)(void*)                                    = nullptr;
    CUresult (*host_device_pointer)(CUdeviceptr*, void*, unsigned)  = nullptr;
    CUresult (*htod_async)(CUdeviceptr, const void*, size_t, void*) = nullptr;
    CUresult (*dtoh_async)(void*, CUdeviceptr, size_t, void*)       = nullptr;
    CUresult (
        *launch)(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void*, void**, void**) =
        nullptr;
    CUresult (*stream_create)(void**, unsigned)      = nullptr;
    CUresult (*stream_destroy)(void*)                = nullptr;
    CUresult (*stream_sync)(void*)                   = nullptr;
    CUresult (*event_create)(void**, unsigned)       = nullptr;
    CUresult (*event_destroy)(void*)                 = nullptr;
    CUresult (*event_record)(void*, void*)           = nullptr;
    CUresult (*event_query)(void*)                   = nullptr;
    CUresult (*event_elapsed)(float*, void*, void*)  = nullptr;
    CUresult (*error_string)(CUresult, const char**) = nullptr;

    template <typename T>
    bool bind(T& fn, const char* name, const char* fallback = nullptr)
    {
        auto* found = dlsym(lib, name);
        if (!found && fallback)
            found = dlsym(lib, fallback);
        fn = reinterpret_cast<T>(found);
        return found != nullptr;
    }

    bool load()
    {
        lib = dlopen("libcuda.so.1", RTLD_LAZY);
        if (!lib)
            return false;
        return bind(init, "cuInit") && bind(device_get, "cuDeviceGet") &&
               bind(primary_retain, "cuDevicePrimaryCtxRetain") &&
               bind(primary_release, "cuDevicePrimaryCtxRelease_v2", "cuDevicePrimaryCtxRelease") &&
               bind(ctx_push, "cuCtxPushCurrent_v2", "cuCtxPushCurrent") &&
               bind(ctx_pop, "cuCtxPopCurrent_v2", "cuCtxPopCurrent") && bind(module_load, "cuModuleLoadData") &&
               bind(module_unload, "cuModuleUnload") && bind(module_function, "cuModuleGetFunction") &&
               bind(mem_alloc, "cuMemAlloc_v2") && bind(mem_free, "cuMemFree_v2") &&
               bind(host_alloc, "cuMemHostAlloc") && bind(host_free, "cuMemFreeHost") &&
               bind(host_device_pointer, "cuMemHostGetDevicePointer_v2", "cuMemHostGetDevicePointer") &&
               bind(htod_async, "cuMemcpyHtoDAsync_v2") && bind(dtoh_async, "cuMemcpyDtoHAsync_v2") &&
               bind(launch, "cuLaunchKernel") && bind(stream_create, "cuStreamCreate") &&
               bind(stream_destroy, "cuStreamDestroy_v2", "cuStreamDestroy") &&
               bind(stream_sync, "cuStreamSynchronize") && bind(event_create, "cuEventCreate") &&
               bind(event_destroy, "cuEventDestroy_v2", "cuEventDestroy") && bind(event_record, "cuEventRecord") &&
               bind(event_query, "cuEventQuery") && bind(event_elapsed, "cuEventElapsedTime") &&
               bind(error_string, "cuGetErrorString");
    }

    std::string describe(CUresult r, const char* what) const
    {
        const char* text = nullptr;
        if (error_string)
            error_string(r, &text);
        return std::string(what) + ": " + (text ? text : "CUDA error " + std::to_string(r));
    }
};

void colour_matrix(matrix m, shq_params& p)
{
    const float kr = m == matrix::bt709 ? 0.2126f : 0.299f;
    const float kb = m == matrix::bt709 ? 0.0722f : 0.114f;
    const float kg = 1.0f - kr - kb;
    const float ys = 219.0f / 255.0f;
    const float cs = 224.0f / 255.0f;
    p.ky[0]        = ys * kr;
    p.ky[1]        = ys * kg;
    p.ky[2]        = ys * kb;
    p.kcb[0]       = cs * -kr / (2.0f * (1.0f - kb));
    p.kcb[1]       = cs * -kg / (2.0f * (1.0f - kb));
    p.kcb[2]       = cs * 0.5f;
    p.kcr[0]       = cs * 0.5f;
    p.kcr[1]       = cs * -kg / (2.0f * (1.0f - kr));
    p.kcr[2]       = cs * -kb / (2.0f * (1.0f - kr));
}

// CU_MEMHOSTALLOC_PORTABLE and CU_MEMHOSTALLOC_DEVICEMAP.
const unsigned host_portable = 0x01;
const unsigned host_mapped   = 0x02;

} // namespace

struct cuda_encoder::impl
{
    struct slot
    {
        uint8_t*    input    = nullptr; // page-locked BGRA
        uint8_t*    out      = nullptr; // page-locked, mapped; written by shq_copy_out
        uint32_t*   meta     = nullptr; // page-locked, mapped; the bitstream size
        CUdeviceptr out_dev  = 0;
        CUdeviceptr meta_dev = 0;
        void*       started  = nullptr;
        void*       done     = nullptr;
        float       gpu_ms   = 0.0f;
    };

    driver     cu;
    config     cfg;
    int        device      = 0;
    void*      ctx         = nullptr;
    void*      module      = nullptr;
    void*      stream      = nullptr;
    void*      k_transform = nullptr;
    void*      k_scan      = nullptr;
    void*      k_zero      = nullptr;
    void*      k_emit      = nullptr;
    void*      k_copy_out  = nullptr;
    shq_params params{};
    size_t     out_bytes = 0;

    std::vector<slot>        slots;
    std::vector<CUdeviceptr> device_memory;

    ~impl()
    {
        if (!cu.lib)
            return;
        if (ctx) {
            cu.ctx_push(ctx);
            if (stream)
                cu.stream_sync(stream);
            for (auto ptr : device_memory)
                cu.mem_free(ptr);
            for (auto& s : slots) {
                for (void* host : {static_cast<void*>(s.input), static_cast<void*>(s.out), static_cast<void*>(s.meta)})
                    if (host)
                        cu.host_free(host);
                for (void* event : {s.started, s.done})
                    if (event)
                        cu.event_destroy(event);
            }
            if (stream)
                cu.stream_destroy(stream);
            if (module)
                cu.module_unload(module);
            void* popped = nullptr;
            cu.ctx_pop(&popped);
            cu.primary_release(device);
        }
        dlclose(cu.lib);
    }

    bool check(CUresult r, const char* what, std::string& error) const
    {
        if (r == 0)
            return true;
        error = cu.describe(r, what);
        return false;
    }

    bool alloc(uint64_t& out, size_t bytes, std::string& error)
    {
        CUdeviceptr ptr = 0;
        if (!check(cu.mem_alloc(&ptr, bytes), "cuMemAlloc", error))
            return false;
        device_memory.push_back(ptr);
        out = ptr;
        return true;
    }

    template <typename T>
    bool alloc_host(T*& out, size_t bytes, unsigned flags, std::string& error)
    {
        void* host = nullptr;
        if (!check(cu.host_alloc(&host, bytes, flags), "cuMemHostAlloc", error))
            return false;
        out = static_cast<T*>(host);
        return true;
    }

    // Runs f with the encoder's context current on this thread.
    template <typename F>
    auto in_context(std::string& error, F&& f) -> decltype(f())
    {
        if (!check(cu.ctx_push(ctx), "cuCtxPushCurrent", error))
            return decltype(f())();
        auto  result = f();
        void* popped = nullptr;
        cu.ctx_pop(&popped);
        return result;
    }

    bool open(std::string& error)
    {
        if (cfg.width <= 0 || cfg.height <= 0 || cfg.width % 16 != 0) {
            error = "SpeedHQ CUDA encoder needs a width that is a multiple of 16";
            return false;
        }
        if (cfg.slots < 1) {
            error = "SpeedHQ CUDA encoder needs at least one slot";
            return false;
        }
        if (!cu.load()) {
            error = "libcuda.so.1 is missing or too old";
            return false;
        }
        if (!check(cu.init(0), "cuInit", error) || !check(cu.device_get(&device, cfg.device), "cuDeviceGet", error) ||
            !check(cu.primary_retain(&ctx, device), "cuDevicePrimaryCtxRetain", error))
            return false;
        return in_context(error, [&] { return setup(error); });
    }

    bool setup(std::string& error)
    {
        if (!check(cu.module_load(&module, shq_kernels_fatbin), "cuModuleLoadData", error) ||
            !check(cu.module_function(&k_transform, module, "shq_transform"), "shq_transform", error) ||
            !check(cu.module_function(&k_scan, module, "shq_scan"), "shq_scan", error) ||
            !check(cu.module_function(&k_zero, module, "shq_zero"), "shq_zero", error) ||
            !check(cu.module_function(&k_emit, module, "shq_emit"), "shq_emit", error) ||
            !check(cu.module_function(&k_copy_out, module, "shq_copy_out"), "shq_copy_out", error))
            return false;
        // CU_STREAM_NON_BLOCKING: never wait on work other code puts on the legacy stream.
        if (!check(cu.stream_create(&stream, 1), "cuStreamCreate", error))
            return false;

        auto& p     = params;
        p.width     = cfg.width;
        p.height    = cfg.height;
        p.pitch     = cfg.width * 4;
        p.mb_width  = cfg.width / 16;
        p.mb_height = (cfg.height + 15) / 16;
        p.blocks    = p.mb_width * p.mb_height * SHQ_BLOCKS_PER_MB;
        colour_matrix(cfg.colour, p);

        const size_t frame = static_cast<size_t>(p.pitch) * p.height;
        const size_t n     = p.blocks;
        out_bytes          = n * SHQ_MAX_BLOCK_BYTES + 64;
        p.out_words        = static_cast<int32_t>(out_bytes / 4);

        // One set of device buffers is enough: the stream runs one frame at a time.
        if (!alloc(p.bgra, frame, error) || !alloc(p.coef, n * 64 * sizeof(int16_t), error) ||
            !alloc(p.last, n, error) || !alloc(p.acbits, n * sizeof(uint16_t), error) ||
            !alloc(p.dcdiff, n * sizeof(int16_t), error) || !alloc(p.offset, n * sizeof(uint32_t), error) ||
            !alloc(p.slice_bits, SHQ_SLICES * sizeof(uint32_t), error) || !alloc(p.out, out_bytes, error))
            return false;

        slots.resize(cfg.slots);
        for (auto& s : slots) {
            // Events time the GPU work; collect polls them, so the waiting thread never spins.
            if (!alloc_host(s.input, frame, host_portable, error) ||
                !alloc_host(s.out, out_bytes, host_portable | host_mapped, error) ||
                !alloc_host(s.meta, sizeof(uint32_t), host_portable | host_mapped, error) ||
                !check(cu.host_device_pointer(&s.out_dev, s.out, 0), "cuMemHostGetDevicePointer", error) ||
                !check(cu.host_device_pointer(&s.meta_dev, s.meta, 0), "cuMemHostGetDevicePointer", error) ||
                !check(cu.event_create(&s.started, 0), "cuEventCreate", error) ||
                !check(cu.event_create(&s.done, 0), "cuEventCreate", error))
                return false;
            std::memset(s.input, 0, frame);
        }
        return true;
    }

    bool
    run(void* kernel, unsigned gx, unsigned gy, unsigned threads, shq_params& p, const char* what, std::string& error)
    {
        void* args[] = {&p};
        return check(cu.launch(kernel, gx, gy, 1, threads, 1, 1, 0, stream, args, nullptr), what, error);
    }

    bool submit(int index, int scale, uint64_t planes, std::string& error)
    {
        if (index < 0 || index >= static_cast<int>(slots.size())) {
            error = "SpeedHQ slot out of range";
            return false;
        }
        if (scale < 1 || scale > 99) {
            error = "SpeedHQ scale must be 1..99";
            return false;
        }
        auto&      s = slots[index];
        shq_params p = params;
        p.scale      = scale;
        p.planes     = planes;
        p.host_out   = s.out_dev;
        p.host_meta  = s.meta_dev;

        const size_t   frame       = static_cast<size_t>(p.pitch) * p.height;
        const unsigned emit_blocks = (p.blocks + 127) / 128;
        return check(cu.event_record(s.started, stream), "cuEventRecord", error) &&
               check(cu.htod_async(p.bgra, s.input, frame, stream), "upload", error) &&
               run(k_transform, p.mb_width, p.mb_height, 64, p, "shq_transform", error) &&
               run(k_scan, SHQ_SLICES, 1, 1024, p, "shq_scan", error) &&
               run(k_zero, 128, 1, 256, p, "shq_zero", error) &&
               run(k_emit, emit_blocks, 1, 128, p, "shq_emit", error) &&
               run(k_copy_out, 64, 1, 256, p, "shq_copy_out", error) &&
               check(cu.event_record(s.done, stream), "cuEventRecord", error);
    }

    status collect(int index, int timeout_ms, const uint8_t*& data, size_t& size, std::string& error)
    {
        if (index < 0 || index >= static_cast<int>(slots.size())) {
            error = "SpeedHQ slot out of range";
            return status::failed;
        }
        auto&      s        = slots[index];
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        for (;;) {
            const CUresult r = cu.event_query(s.done);
            if (r == 0)
                break;
            if (r != cuda_not_ready) {
                check(r, "cuEventQuery", error);
                return status::failed;
            }
            if (std::chrono::steady_clock::now() >= deadline)
                return status::timeout;
            // Usually the frame finished during the previous tick and this never runs. Sleeping
            // rather than spinning keeps the waiting core idle.
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
        if (!check(cu.event_elapsed(&s.gpu_ms, s.started, s.done), "cuEventElapsedTime", error))
            return status::failed;
        size = s.meta[0];
        if (size > out_bytes) {
            error = "SpeedHQ bitstream overflowed its buffer";
            return status::failed;
        }
        data = s.out;
        return status::ready;
    }

    bool encode(int scale, const uint8_t*& data, size_t& size, std::string& error, debug_output* debug)
    {
        uint64_t planes = 0;
        if (debug && !alloc(planes, static_cast<size_t>(params.width) * params.height * 2, error))
            return false;
        bool ok = submit(0, scale, planes, error) && collect(0, 10000, data, size, error) == status::ready;
        if (!ok && error.empty())
            error = "SpeedHQ encode timed out";
        if (ok && debug) {
            debug->coef.resize(static_cast<size_t>(params.blocks) * 64);
            debug->planes.resize(static_cast<size_t>(params.width) * params.height * 2);
            ok = check(cu.dtoh_async(debug->coef.data(), params.coef, debug->coef.size() * sizeof(int16_t), stream),
                       "download",
                       error) &&
                 check(cu.dtoh_async(debug->planes.data(), planes, debug->planes.size(), stream), "download", error) &&
                 check(cu.stream_sync(stream), "cuStreamSynchronize", error);
        }
        if (planes) {
            cu.stream_sync(stream);
            cu.mem_free(planes);
            device_memory.pop_back();
        }
        return ok;
    }
};

std::unique_ptr<cuda_encoder> cuda_encoder::create(const config& cfg, std::string& error)
{
    auto p = std::make_unique<impl>();
    p->cfg = cfg;
    if (!p->open(error))
        return nullptr;
    return std::unique_ptr<cuda_encoder>(new cuda_encoder(std::move(p)));
}

cuda_encoder::cuda_encoder(std::unique_ptr<impl> p)
    : impl_(std::move(p))
{
}

cuda_encoder::~cuda_encoder() = default;

int cuda_encoder::slots() const { return static_cast<int>(impl_->slots.size()); }

uint8_t* cuda_encoder::input(int slot) const { return impl_->slots.at(slot).input; }

bool cuda_encoder::submit(int slot, int scale, std::string& error)
{
    return impl_->in_context(error, [&] { return impl_->submit(slot, scale, 0, error); });
}

status cuda_encoder::collect(int slot, int timeout_ms, const uint8_t*& data, size_t& size, std::string& error)
{
    return impl_->in_context(error, [&] { return impl_->collect(slot, timeout_ms, data, size, error); });
}

bool cuda_encoder::encode(int scale, const uint8_t*& data, size_t& size, std::string& error, debug_output* debug)
{
    return impl_->in_context(error, [&] { return impl_->encode(scale, data, size, error, debug); });
}

int cuda_encoder::blocks() const { return impl_->params.blocks; }

float cuda_encoder::gpu_ms(int slot) const { return impl_->slots.at(slot).gpu_ms; }

}}} // namespace caspar::newtek::shq
