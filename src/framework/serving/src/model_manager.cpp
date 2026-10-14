#include "edge/model_manager.h"
#include "edge/logger.h"
#include <filesystem>

namespace edge {

void ModelManager::register_model(const std::string& name,
                                  const ModelOptions& opts) {
    std::lock_guard<std::mutex> lk(mu_);
    if (models_.count(name)) {
        EDGE_LOG_WARN("model already registered: %s", name.c_str());
        return;
    }
    auto it = lru_.insert(lru_.end(), name);
    models_.emplace(name, Entry{opts, nullptr, false, true, it});
    EDGE_LOG_INFO("model registered: %s (%zu MB, lazy-load)",
                  name.c_str(),
                  std::filesystem::file_size(opts.path) / 1024 / 1024);
}

void ModelManager::unregister_model(const std::string& name) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = models_.find(name);
    if (it == models_.end()) return;
    if (it->second.loading) {
        // 加载中：不能擦除 map 槽位（等待线程持有引用），标记后由 acquire 清理
        it->second.registered = false;
        return;
    }
    lru_.remove(name);
    models_.erase(it);
}

void ModelManager::touch_lru_locked(Entry& e, const std::string& name) {
    lru_.remove(name);
    lru_.push_front(name);
    e.lru_it = lru_.begin();
}

std::shared_ptr<ModelEngine> ModelManager::acquire(const std::string& name) {
    ModelOptions opts;
    {
        std::unique_lock<std::mutex> lk(mu_);
        for (;;) {
            auto it = models_.find(name);
            if (it == models_.end()) {
                EDGE_LOG_ERROR("acquire unknown model: %s", name.c_str());
                return nullptr;
            }
            auto& e = it->second;
            if (e.engine) {
                touch_lru_locked(e, name);
                return e.engine;
            }
            if (e.loading) {  // 其他线程正在加载：等待复用结果
                cv_.wait(lk);
                continue;     // 醒来重新查表（防槽位被清理）
            }
            if (!e.registered) return nullptr;
            e.loading = true;
            opts = e.opts;
            break;
        }
    }

    // 锁外加载：模型加载可达数百 ms，不能阻塞其他模型的服务
    auto t0 = std::chrono::steady_clock::now();
    auto engine = ModelEngine::load(opts);
    auto cost_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    if (engine) EDGE_LOG_INFO("model %s loaded in %lld ms", name.c_str(), (long long)cost_ms);
    else EDGE_LOG_ERROR("model %s load failed after %lld ms", name.c_str(), (long long)cost_ms);

    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = models_.find(name);
        if (it != models_.end()) {
            auto& e = it->second;
            e.loading = false;
            if (!e.registered) {  // 加载期间被注销
                lru_.remove(name);
                models_.erase(it);
            } else if (engine) {
                e.engine = engine;
                touch_lru_locked(e, name);
                size_t resident = resident_bytes_locked();
                if (resident > opts_.memory_budget_bytes)
                    evict_lru_locked(resident - opts_.memory_budget_bytes);
            }
        }
    }
    cv_.notify_all();
    return engine;
}

bool ModelManager::reload(const std::string& name) {
    ModelOptions opts;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = models_.find(name);
        if (it == models_.end() || !it->second.registered) return false;
        opts = it->second.opts;
    }
    // 关键：先在锁外加载新 session，失败不影响旧 session 继续服务
    auto fresh = ModelEngine::load(opts);
    if (!fresh) return false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = models_.find(name);
        if (it == models_.end()) return false;
        auto& e = it->second;
        e.engine = fresh;  // 原子切换；旧 session 由在途请求的 shared_ptr 自然释放
        touch_lru_locked(e, name);
    }
    EDGE_LOG_INFO("model hot-reloaded: %s", name.c_str());
    return true;
}

size_t ModelManager::resident_bytes_locked() {
    size_t total = 0;
    for (auto& [name, e] : models_)
        if (e.engine) total += e.engine->model_file_bytes();
    return total;
}

void ModelManager::evict_lru_locked(size_t need_bytes) {
    size_t freed = 0;
    std::vector<std::string> names(lru_.rbegin(), lru_.rend());  // 从最久未用开始
    for (const auto& name : names) {
        if (freed >= need_bytes) break;
        auto it = models_.find(name);
        if (it == models_.end()) continue;
        auto& e = it->second;
        // use_count == 1 表示只有 manager 持有，无在途请求 → 可安全淘汰
        if (e.engine && e.engine.use_count() == 1) {
            EDGE_LOG_INFO("LRU evict: %s (free %zu MB)", name.c_str(),
                          e.engine->model_file_bytes() / 1024 / 1024);
            freed += e.engine->model_file_bytes();
            e.engine.reset();
            lru_.erase(e.lru_it);
        }
    }
    if (freed < need_bytes)
        EDGE_LOG_WARN("memory budget exceeded: still need %zu MB (all models busy?)",
                      (need_bytes - freed) / 1024 / 1024);
}

std::vector<ModelManager::Resident> ModelManager::residents() {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<Resident> out;
    for (auto& [name, e] : models_)
        out.push_back({name,
                       e.engine ? e.engine->model_file_bytes() : 0,
                       e.engine != nullptr});
    return out;
}

} // namespace edge
