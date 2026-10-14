#include "edge/model_engine.h"
#include "edge/logger.h"
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
// ORT 在 Windows 上的路径类型是 wchar_t（ORTCHAR_T），需 UTF-8 → 宽字符转换
static std::wstring to_wide(const std::string& s) {
    if (s.empty()) return std::wstring();
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0);
    std::wstring w(n, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), &w[0], n);
    return w;
}
#endif

namespace edge {

Ort::Env& ModelEngine::env() {
    static Ort::Env e(ORT_LOGGING_LEVEL_WARNING, "edge-infer");
    return e;
}

std::shared_ptr<ModelEngine> ModelEngine::load(const ModelOptions& opts) {
    auto eng = std::shared_ptr<ModelEngine>(new ModelEngine);
    eng->opts_ = opts;
    try {
        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(
            static_cast<GraphOptimizationLevel>(opts.graph_opt_level));
        so.SetIntraOpNumThreads(opts.intra_op_threads);
        if (opts.enable_mem_arena) so.EnableCpuMemArena();
        else so.DisableCpuMemArena();
        // 平台适配点：按需追加 Execution Provider。
        // 本机 ORT 为 CPU 构建；CUDA/TensorRT（Jetson）、RKNN（RK3588）、
        // CoreML（Apple）等 EP 需要对应构建的 ORT（见 README「平台移植路径」）
        if (opts.provider != "cpu") {
            EDGE_LOG_WARN("provider '%s' requested but this ORT build has "
                          "CPU EP only, fallback to CPU", opts.provider.c_str());
        }

#ifdef _WIN32
        auto wpath = to_wide(opts.path);
        eng->session_ = std::make_unique<Ort::Session>(env(), wpath.c_str(), so);
#else
        eng->session_ = std::make_unique<Ort::Session>(env(), opts.path.c_str(), so);
#endif

        Ort::AllocatorWithDefaultOptions alloc;
        auto collect = [&](bool is_input) -> std::vector<TensorInfo> {
            size_t n = is_input ? eng->session_->GetInputCount()
                                : eng->session_->GetOutputCount();
            std::vector<TensorInfo> infos;
            infos.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                auto name = is_input
                    ? eng->session_->GetInputNameAllocated(i, alloc)
                    : eng->session_->GetOutputNameAllocated(i, alloc);
                auto type_info = is_input
                    ? eng->session_->GetInputTypeInfo(i)
                    : eng->session_->GetOutputTypeInfo(i);
                auto shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
                size_t elems = 1;
                bool dynamic = false;
                for (auto& d : shape) {
                    if (d <= 0) { dynamic = true; d = 1; }  // 动态维（如 batch）固定为 1
                    elems *= static_cast<size_t>(d);
                }
                if (dynamic)
                    EDGE_LOG_DEBUG("tensor '%s' has dynamic shape, resolved to static batch=1",
                                   name.get());
                infos.push_back({name.get(), std::move(shape), elems});
            }
            return infos;
        };
        eng->inputs_ = collect(true);
        eng->outputs_ = collect(false);
        eng->model_bytes_ = std::filesystem::file_size(opts.path);

        EDGE_LOG_INFO("model loaded: %s (%zu MB, %zu inputs, %zu outputs)",
                      opts.path.c_str(), eng->model_bytes_ / 1024 / 1024,
                      eng->inputs_.size(), eng->outputs_.size());
        return eng;
    } catch (const Ort::Exception& e) {
        EDGE_LOG_ERROR("load model failed: %s : %s", opts.path.c_str(), e.what());
        return nullptr;
    }
}

bool ModelEngine::run(const std::vector<Ort::Value>& inputs,
                      std::vector<Ort::Value>& outputs) {
    try {
        Ort::IoBinding binding(*session_);
        for (size_t i = 0; i < inputs.size(); ++i)
            binding.BindInput(inputs_[i].name.c_str(), inputs[i]);
        for (size_t i = 0; i < outputs.size(); ++i)
            binding.BindOutput(outputs_[i].name.c_str(), outputs[i]);
        Ort::RunOptions run_opts;
        session_->Run(run_opts, binding);
        return true;
    } catch (const Ort::Exception& e) {
        EDGE_LOG_ERROR("inference failed for %s: %s", opts_.path.c_str(), e.what());
        return false;
    }
}

Ort::Value ModelEngine::create_input_tensor(size_t idx, void* buffer) {
    const auto& info = inputs_[idx];
    Ort::MemoryInfo mi =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor(mi, static_cast<float*>(buffer),
                                    info.elem_count,
                                    info.shape.data(), info.shape.size());
}

Ort::Value ModelEngine::create_output_tensor(size_t idx, void* buffer) {
    const auto& info = outputs_[idx];
    Ort::MemoryInfo mi =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor(mi, static_cast<float*>(buffer),
                                    info.elem_count,
                                    info.shape.data(), info.shape.size());
}

} // namespace edge
