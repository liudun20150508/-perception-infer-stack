#pragma once
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "edge/buffer_pool.h"
#include "edge/model_manager.h"
#include "edge/thread_pool.h"

namespace edge {

struct InferRequest {
    int64_t id = 0;
    std::string model;
    std::shared_ptr<const std::vector<uint8_t>> image;  // 编码后的图像
    int topk = 5;
    int priority = 0;                      // 越大越优先
    std::chrono::milliseconds deadline{2000};  // 端到端超时
    // 前处理参数（多模型可配不同 mean/std/尺寸，引擎保持通用）
    int input_w = 224;
    int input_h = 224;
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float stdv[3] = {0.229f, 0.224f, 0.225f};
    bool use_mean_std = true;
};

struct InferResponse {
    int64_t id = 0;
    bool ok = false;
    std::string error;
    std::vector<std::pair<int, float>> topk;  // (class_id, score)
    int64_t pre_us = 0, infer_us = 0, post_us = 0, queue_us = 0;
};

struct ServerOptions {
    size_t pre_threads = 2;
    size_t infer_threads = 4;
    size_t post_threads = 1;
    size_t queue_capacity = 128;    // 每级有界队列容量（背压上限）
    size_t buffer_blocks = 64;      // 输入/输出缓冲池块数
    size_t max_inflight = 4096;     // 在途请求上限（超限快速失败）
    size_t input_elems = 3 * 224 * 224;  // 按部署模型的最大输入尺寸配置
    size_t output_elems = 1000;
};

// 端侧多模型推理服务：preprocess → infer → postprocess 三段流水线。
// 关键设计（逐条对应岗位职责）：
// 1) 有界队列背压：任何一级队列满，上游 submit 阻塞 → 内存占用有上界
// 2) 零拷贝：前处理直接写入池化输入缓冲，推理经 IoBinding 原地读写，
//    后处理原地读 —— 全链路仅一次数据拷贝（图像解码）
// 3) 优先级调度：infer 级按请求优先级出队（交互请求优先于批量任务）
// 4) 任务级异常隔离 + 协作式超时取消：超时请求在阶段边界丢弃，
//    不占用推理资源（ORT Run 无法安全中断，故不强行 kill）
// 5) watchdog 线程扫描在途请求，超时未完成定位卡点并告警
// 6) 优雅停机：drain 排空后退出；立即停机时所有悬挂 promise 返回错误
class InferenceServer {
public:
    InferenceServer(ServerOptions opts, std::shared_ptr<ModelManager> models);
    ~InferenceServer();

    InferenceServer(const InferenceServer&) = delete;
    InferenceServer& operator=(const InferenceServer&) = delete;

    void start();
    void stop(bool drain = true);

    // 提交请求；背压时阻塞；停机中抛异常；超在途上限快速失败
    std::future<InferResponse> submit(InferRequest req);

    struct Snapshot {
        bool running = false;
        size_t inflight = 0;
        size_t pre_queue = 0, infer_queue = 0, post_queue = 0;
        size_t active_pre = 0, active_infer = 0, active_post = 0;
        uint64_t done = 0, failed = 0, timed_out = 0;
        double qps = 0;
        int64_t avg_us = 0, p50_us = 0, p99_us = 0;
        BufferPool<float>::Stats input_pool, output_pool;
    };
    Snapshot snapshot();

private:
    struct Task {
        InferRequest req;
        std::promise<InferResponse> resp;
        BufferPool<float>::Handle input_buf;   // 池化输入（前处理写入）
        BufferPool<float>::Handle output_buf;  // 池化输出（推理写入）
        std::chrono::steady_clock::time_point enqueue_at;
        int64_t pre_us = 0, infer_us = 0, post_us = 0;
        size_t out_elems = 0;
        int stage = 0;  // 0=pre队列 1=前处理 2=推理 3=后处理（watchdog 定位卡点）
        bool wd_reported = false;
        std::mutex mtx;
        bool finished = false;
        // 只允许成功设置一次（并发安全：worker 与停机路径竞争时防止重复/悬挂）
        bool set_response(InferResponse r) {
            std::lock_guard<std::mutex> lk(mtx);
            if (finished) return false;
            finished = true;
            resp.set_value(std::move(r));
            return true;
        }
    };
    using TaskPtr = std::shared_ptr<Task>;

    void preprocess_worker(TaskPtr t);
    void infer_worker(TaskPtr t);
    void postprocess_worker(TaskPtr t);
    void finalize(TaskPtr t, InferResponse r);
    void watchdog_loop();
    static InferResponse make_error(const TaskPtr& t, const std::string& err);

    ServerOptions opts_;
    std::shared_ptr<ModelManager> models_;

    ThreadPool pre_pool_, infer_pool_, post_pool_;
    std::unique_ptr<BufferPool<float>> input_pool_, output_pool_;

    std::atomic<bool> running_{false};
    std::atomic<uint64_t> inflight_{0};
    std::atomic<int64_t> next_id_{0};

    std::thread watchdog_;
    std::mutex wd_mu_;
    std::unordered_map<int64_t, TaskPtr> inflight_map_;  // 在途任务表（watchdog 扫描用）

    std::atomic<uint64_t> done_{0}, failed_{0}, timed_out_{0}, total_lat_us_{0};
    std::mutex hist_mu_;
    std::vector<int64_t> lat_hist_;  // 环形，容量 4096
    size_t hist_pos_ = 0;

    std::chrono::steady_clock::time_point start_at_;
};

} // namespace edge
