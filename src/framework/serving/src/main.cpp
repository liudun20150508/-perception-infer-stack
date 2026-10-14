// edge-infer demo：六个场景演示端侧推理服务的核心能力
//  1. 单请求：结果 + 各阶段耗时拆解（排队/前处理/推理/后处理）
//  2. 吞吐压测：3 轮各 N 请求，观察峰值内存收敛（缓冲池复用，无泄漏）
//  3. 优先级调度：300 低优 + 1 高优，高优插队完成
//  4. 超时保护：deadline=1ms 的请求在阶段边界被协作式取消
//  5. 热更新：reload 期间在途请求零失败
//  6. 多模型 LRU：16MB 内存预算下两模型互相淘汰
#include "edge/inference_server.h"
#include "edge/logger.h"
#include "edge/model_manager.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

using namespace edge;
using namespace std::chrono;

// 压测类场景的端到端 deadline。默认 2s 与生产口径一致（InferRequest 默认值）；
// 机器被别的活儿压着时排队会超过 2s，失败数会全部来自超时而不是服务本身出问题 ——
// 这时用 `--deadline 20000` 放宽，就能在负载机上拿到与空载机同一口径的吞吐/内存数，
// 而"超时"列会如实报出被放宽前的排队情况。
static milliseconds g_deadline{2000};

static std::vector<std::string> load_labels(const std::string& path) {
    std::vector<std::string> labels;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line))
        if (!line.empty()) labels.push_back(line);
    return labels;
}

static std::shared_ptr<const std::vector<uint8_t>> load_image_bytes(
    const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return nullptr;
    return std::make_shared<std::vector<uint8_t>>(
        (std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static long rss_mb() {  // 进程峰值内存（macOS: ru_maxrss；Windows: 峰值工作集）
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return (long)(pmc.PeakWorkingSetSize / (1024 * 1024));
    return 0;
#else
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss / (1024 * 1024);
#endif
}

static void print_snapshot(const InferenceServer::Snapshot& s) {
    printf("  [指标] 吞吐 %.0f req/s | 完成 %llu | 失败 %llu | 超时 %llu\n",
           s.qps, (unsigned long long)s.done, (unsigned long long)s.failed,
           (unsigned long long)s.timed_out);
    printf("  [延迟] avg %lld us | P50 %lld us | P99 %lld us\n",
           (long long)s.avg_us, (long long)s.p50_us, (long long)s.p99_us);
    printf("  [队列] pre %zu / infer %zu / post %zu | 在途 %zu | 活跃线程 %zu/%zu/%zu\n",
           s.pre_queue, s.infer_queue, s.post_queue, s.inflight,
           s.active_pre, s.active_infer, s.active_post);
    printf("  [缓冲池] input %zu/%zu 使用中 | output %zu/%zu 使用中（全归还=无泄漏）\n",
           s.input_pool.in_use, s.input_pool.total_blocks,
           s.output_pool.in_use, s.output_pool.total_blocks);
}

// ---- 场景 1：单请求 ----
static void scene_single(InferenceServer& srv,
    const std::shared_ptr<const std::vector<uint8_t>>& img,
    const std::vector<std::string>& labels) {
    printf("\n===== 场景 1：单请求（耗时拆解）=====\n");
    InferRequest req;
    req.model = "mobilenet";
    req.image = img;
    req.topk = 5;
    auto r = srv.submit(req).get();
    if (!r.ok) {
        printf("  失败: %s\n", r.error.c_str());
        return;
    }
    printf("  Top1: %s (score=%.4f)\n",
           labels[r.topk[0].first].c_str(), r.topk[0].second);
    for (size_t i = 1; i < r.topk.size(); ++i)
        printf("  Top%zu: %s (%.4f)\n", i + 1,
               labels[r.topk[i].first].c_str(), r.topk[i].second);
    printf("  耗时: 排队 %lld us | 前处理 %lld us | 推理 %lld us | 后处理 %lld us\n",
           (long long)r.queue_us, (long long)r.pre_us,
           (long long)r.infer_us, (long long)r.post_us);
}

// ---- 场景 2：吞吐压测 + 内存稳定性 ----
static void scene_throughput(InferenceServer& srv,
    const std::shared_ptr<const std::vector<uint8_t>>& img, int n) {
    printf("\n===== 场景 2：吞吐压测（3 轮 x %d 请求，deadline=%lldms，观察内存稳定性）=====\n",
           n, (long long)g_deadline.count());
    for (int round = 1; round <= 3; ++round) {
        long mem_before = rss_mb();
        auto t0 = steady_clock::now();
        std::vector<std::future<InferResponse>> futs;
        futs.reserve(n);
        for (int i = 0; i < n; ++i) {
            InferRequest req;
            req.model = "mobilenet";
            req.image = img;
            req.topk = 1;
            req.deadline = g_deadline;
            futs.push_back(srv.submit(req));
        }
        size_t ok = 0, fail = 0;
        for (auto& f : futs) {
            auto r = f.get();
            r.ok ? ok++ : fail++;
        }
        auto t1 = steady_clock::now();
        double sec = duration<double>(t1 - t0).count();
        long mem_after = rss_mb();
        printf("  第%d轮: %zu 成功 / %zu 失败 | %.0f req/s | 峰值内存 %ld MB -> %ld MB\n",
               round, ok, fail, n / sec, mem_before, mem_after);
    }
    print_snapshot(srv.snapshot());
}

// ---- 场景 3：优先级调度 ----
static void scene_priority(InferenceServer& srv,
    const std::shared_ptr<const std::vector<uint8_t>>& img) {
    printf("\n===== 场景 3：优先级调度（300 低优 + 1 高优，deadline=%lldms）=====\n",
           (long long)g_deadline.count());
    const int n_low = 300;
    std::atomic<int> low_done{0};
    std::atomic<int> low_done_at_hi{0};
    InferResponse hi_r;

    std::vector<std::future<InferResponse>> futs;
    futs.reserve(n_low);
    for (int i = 0; i < n_low; ++i) {
        InferRequest req;
        req.model = "mobilenet";
        req.image = img;
        req.topk = 1;
        req.priority = 0;  // 低优先级批量任务
        req.deadline = g_deadline;
        futs.push_back(srv.submit(req));
    }
    InferRequest hi;
    hi.model = "mobilenet";
    hi.image = img;
    hi.topk = 1;
    hi.priority = 10;  // 高优先级交互请求
    hi.deadline = g_deadline;
    auto hi_fut = srv.submit(hi);

    std::thread hi_waiter([&] {
        hi_r = hi_fut.get();
        low_done_at_hi = low_done.load();  // 高优完成时低优完成了多少
    });
    for (auto& f : futs) {
        f.get();
        low_done++;
    }
    hi_waiter.join();

    long long hi_total = hi_r.pre_us + hi_r.infer_us + hi_r.post_us + hi_r.queue_us;
    printf("  高优先级请求总耗时 %lld us；它完成时，先提交的低优先级请求仅完成 %d/%d\n",
           hi_total, low_done_at_hi.load(), n_low);
    printf("  => 高优先级插队生效（交互请求不被批量任务饿死）\n");
}

// ---- 场景 4：超时保护 ----
static void scene_timeout(InferenceServer& srv,
    const std::shared_ptr<const std::vector<uint8_t>>& img) {
    printf("\n===== 场景 4：超时保护（deadline=1ms）=====\n");
    InferRequest req;
    req.model = "mobilenet";
    req.image = img;
    req.topk = 1;
    req.deadline = std::chrono::milliseconds(1);
    auto r = srv.submit(req).get();
    printf("  响应: ok=%d error=\"%s\"\n", r.ok, r.error.c_str());
    printf("  => 超时请求在阶段边界被协作式取消，不占用推理算力\n");
}

// ---- 场景 5：模型热更新 ----
static void scene_hotreload(InferenceServer& srv,
    const std::shared_ptr<const std::vector<uint8_t>>& img,
    std::shared_ptr<ModelManager> mgr) {
    printf("\n===== 场景 5：模型热更新（在途请求零失败，deadline=%lldms）=====\n",
           (long long)g_deadline.count());
    std::vector<std::future<InferResponse>> futs;
    for (int i = 0; i < 30; ++i) {
        InferRequest req;
        req.model = "mobilenet";
        req.image = img;
        req.topk = 1;
        req.deadline = g_deadline;
        futs.push_back(srv.submit(req));
    }
    // 请求在途时触发热更新：新 session 加载成功后原子切换，
    // 在途请求持有的旧 session 通过 shared_ptr 存活到请求结束
    mgr->reload("mobilenet");
    size_t ok = 0;
    for (auto& f : futs) ok += f.get().ok ? 1 : 0;
    printf("  热更新期间 %zu/30 请求成功，服务无中断\n", ok);
}

// ---- 场景 6：多模型 LRU 资源管理 ----
static void scene_lru(const std::shared_ptr<const std::vector<uint8_t>>& img,
                      const std::vector<std::string>& labels) {
    printf("\n===== 场景 6：多模型 LRU 资源管理（内存预算 16MB）=====\n");
    // 16MB 预算 < mobilenet(13.9MB) + squeezenet(4.7MB)，强制触发 LRU 淘汰
    auto mgr = std::make_shared<ModelManager>(
        ModelManagerOptions{16u * 1024 * 1024});
    ModelOptions mo;
    mo.path = "data/mobilenetv2-12.onnx";
    mgr->register_model("mobilenet", mo);
    mo.path = "data/squeezenet1.1-7.onnx";
    mgr->register_model("squeezenet", mo);

    ServerOptions so;
    so.pre_threads = 1;
    so.infer_threads = 2;
    so.post_threads = 1;
    so.queue_capacity = 32;
    so.buffer_blocks = 8;
    so.max_inflight = 64;
    InferenceServer srv(so, mgr);
    srv.start();

    auto run = [&](const char* model) {
        InferRequest req;
        req.model = model;
        req.image = img;
        req.topk = 1;
        // 不同模型的前处理契约不同：squeezenet 仅 [0,1] 归一化，不做 mean/std
        req.use_mean_std = (std::strcmp(model, "squeezenet") != 0);
        auto r = srv.submit(req).get();
        printf("  请求 %-12s -> %s (%.4f)\n", model,
               r.ok ? labels[r.topk[0].first].c_str() : "FAIL",
               r.ok ? r.topk[0].second : 0.f);
    };
    auto print_resident = [&] {
        printf("  常驻模型: ");
        for (auto& r : mgr->residents())
            if (r.loaded)
                printf("[%s %.1fMB] ", r.name.c_str(), r.model_bytes / 1048576.0);
        printf("\n");
    };

    run("mobilenet");
    print_resident();  // 只加载 mobilenet（13.9MB）
    run("squeezenet");
    print_resident();  // 超预算 → LRU 淘汰 mobilenet
    run("mobilenet");
    print_resident();  // 再请求 → 重新加载，淘汰 squeezenet

    srv.stop(true);
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);  // 中文输出正常显示（cmd 默认代码页非 UTF-8）
#endif
    std::string image_path = "data/dog.jpg";
    int throughput_n = 500;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--image") && i + 1 < argc)
            image_path = argv[++i];
        else if (!strcmp(argv[i], "--requests") && i + 1 < argc)
            throughput_n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--deadline") && i + 1 < argc)
            g_deadline = milliseconds(atoll(argv[++i]));
        else if (!strcmp(argv[i], "--debug"))
            Logger::instance().set_level(LogLevel::Debug);
    }

    auto labels = load_labels("data/imagenet_classes.txt");
    auto img = load_image_bytes(image_path);
    if (!img || labels.empty()) {
        fprintf(stderr, "资源加载失败：请确认 data/ 目录下有模型与图片\n");
        return 1;
    }
    printf("图像: %s (%zu KB) | ImageNet 标签 %zu 类\n\n",
           image_path.c_str(), img->size() / 1024, labels.size());
    printf("压测 deadline: %lld ms（--deadline 可调；负载机上放宽它才能测出服务本身的吞吐）\n\n",
           (long long)g_deadline.count());

    // 主服务：256MB 模型预算
    auto mgr = std::make_shared<ModelManager>(
        ModelManagerOptions{256u * 1024 * 1024});
    ModelOptions mo;
    mo.path = "data/mobilenetv2-12.onnx";
    mgr->register_model("mobilenet", mo);
    mo.path = "data/squeezenet1.1-7.onnx";
    mgr->register_model("squeezenet", mo);

    ServerOptions so;
    so.pre_threads = 2;
    so.infer_threads = 4;
    so.post_threads = 1;
    so.queue_capacity = 128;
    so.buffer_blocks = 64;
    so.max_inflight = 4096;
    so.input_elems = 3 * 224 * 224;
    so.output_elems = 1000;
    InferenceServer srv(so, mgr);
    srv.start();

    scene_single(srv, img, labels);
    scene_throughput(srv, img, throughput_n);
    scene_priority(srv, img);
    scene_timeout(srv, img);
    scene_hotreload(srv, img, mgr);

    srv.stop(true);
    printf("\n最终指标:\n");
    print_snapshot(srv.snapshot());

    scene_lru(img, labels);
    printf("\n全部场景完成 ✔\n");
    return 0;
}
