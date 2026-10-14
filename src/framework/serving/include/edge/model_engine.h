#pragma once
#include <memory>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

namespace edge {

struct ModelOptions {
    std::string path;
    std::string provider = "cpu";    // cpu / coreml / cuda / rknn（平台适配点）
    int intra_op_threads = 0;        // 0 = 交给 ORT 默认
    int graph_opt_level = ORT_ENABLE_ALL;  // 图优化等级
    bool enable_mem_arena = true;    // ORT 内部内存池，减少运行期分配
};

// 模型推理引擎：ONNX Runtime Session + IoBinding 零拷贝封装。
// 设计要点：
// - load 失败返回 nullptr 而非抛异常，上层做降级/恢复（对应"异常恢复"）
// - Session 线程安全，可多线程并发 Run（ORT 内部管理 intra-op 线程池）
// - 每次 Run 使用独立 IoBinding，输入/输出绑定外部池化内存，
//   推理全程不产生数据拷贝与运行期大块分配（对应"数据重复拷贝"治理）
class ModelEngine {
public:
    struct TensorInfo {
        std::string name;
        std::vector<int64_t> shape;
        size_t elem_count = 0;  // 动态 shape 时为 0
    };

    static std::shared_ptr<ModelEngine> load(const ModelOptions& opts);

    bool run(const std::vector<Ort::Value>& inputs, std::vector<Ort::Value>& outputs);

    const std::vector<TensorInfo>& inputs_info() const { return inputs_; }
    const std::vector<TensorInfo>& outputs_info() const { return outputs_; }
    size_t model_file_bytes() const { return model_bytes_; }
    const ModelOptions& options() const { return opts_; }

    // 用外部 buffer（池化内存）创建输入/输出 Tensor —— 零拷贝入口
    Ort::Value create_input_tensor(size_t idx, void* buffer);
    Ort::Value create_output_tensor(size_t idx, void* buffer);

private:
    static Ort::Env& env();  // 全局唯一 Env（进程内共享，避免重复初始化）

    ModelOptions opts_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<TensorInfo> inputs_, outputs_;
    size_t model_bytes_ = 0;
};

} // namespace edge
