#pragma once
#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include "edge/model_engine.h"

namespace edge {

struct ModelManagerOptions {
    size_t memory_budget_bytes = 256 * 1024 * 1024;  // 常驻模型内存总预算
};

// 多模型管理与资源控制（能力点"多模型协同及资源管理"）：
// - 惰性加载：注册不占内存，首次 acquire 才加载
// - LRU 淘汰：新模型加载超预算时，淘汰最久未使用且无在途请求的模型
// - 并发加载保护：同一模型同时只有一个线程加载（loading 标志 + 条件变量），
//   其余线程等待复用结果；加载在锁外执行，不阻塞其他模型的服务
// - 热更新：reload() 先加载新 session，成功后原子切换；在途请求持有的
//   旧 session 通过 shared_ptr 自然存活到请求结束（优雅切换，零中断）
class ModelManager {
public:
    explicit ModelManager(ModelManagerOptions opts) : opts_(opts) {}

    void register_model(const std::string& name, const ModelOptions& opts);
    void unregister_model(const std::string& name);

    // 获取引擎引用；未加载则触发加载（可能 LRU 淘汰）；失败返回 nullptr
    std::shared_ptr<ModelEngine> acquire(const std::string& name);

    // 热更新：新模型加载成功后原子替换旧 session，失败保留旧模型
    bool reload(const std::string& name);

    struct Resident {
        std::string name;
        size_t model_bytes = 0;
        bool loaded = false;
    };
    std::vector<Resident> residents();

private:
    struct Entry {
        ModelOptions opts;
        std::shared_ptr<ModelEngine> engine;  // nullptr = 未加载
        bool loading = false;
        bool registered = true;
        std::list<std::string>::iterator lru_it;
    };

    void evict_lru_locked(size_t need_bytes);
    size_t resident_bytes_locked();
    void touch_lru_locked(Entry& e, const std::string& name);

    ModelManagerOptions opts_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::unordered_map<std::string, Entry> models_;
    std::list<std::string> lru_;  // 头部 = 最近使用
};

} // namespace edge
