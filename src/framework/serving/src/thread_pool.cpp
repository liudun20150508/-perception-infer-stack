#include "edge/thread_pool.h"
#include "edge/logger.h"
#include <stdexcept>

namespace edge {

ThreadPool::ThreadPool(const std::string& name, size_t n_threads,
                       size_t max_queue_size)
    : name_(name), max_queue_size_(max_queue_size) {
    workers_.reserve(n_threads);
    for (size_t i = 0; i < n_threads; ++i)
        workers_.emplace_back([this] { worker_loop(); });
}

ThreadPool::~ThreadPool() {
    stop(false);
}

void ThreadPool::submit(Task task, int priority) {
    {
        std::unique_lock<std::mutex> lk(mu_);
        if (stopping_.load())
            throw std::runtime_error("submit to stopped pool: " + name_);
        // 有界队列：满则等待 worker 消费（背压传导到上游）
        if (max_queue_size_ > 0)
            not_full_.wait(lk, [this] {
                return stopping_.load() || queue_.size() < max_queue_size_;
            });
        if (stopping_.load())
            throw std::runtime_error("submit to stopped pool: " + name_);
        queue_.push(Item{std::move(task), priority, seq_++});
    }
    cv_.notify_one();
}

size_t ThreadPool::queue_depth() const {
    std::lock_guard<std::mutex> lk(mu_);
    return queue_.size();
}

void ThreadPool::stop(bool drain) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (stopping_.exchange(true)) return;
        if (!drain) {
            while (!queue_.empty()) queue_.pop();
        }
    }
    cv_.notify_all();
    not_full_.notify_all();
    for (auto& t : workers_)
        if (t.joinable()) t.join();
}

void ThreadPool::worker_loop() {
    for (;;) {
        Task task;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] { return stopping_.load() || !queue_.empty(); });
            if (queue_.empty()) {
                if (stopping_.load()) return;  // drain 完成或立即停止
                continue;
            }
            task = queue_.top().task;
            queue_.pop();
            not_full_.notify_one();
        }
        active_++;
        try {
            task();  // 任务级异常隔离：不影响其他任务与 worker 存活
        } catch (const std::exception& e) {
            EDGE_LOG_ERROR("[%s] task exception: %s", name_.c_str(), e.what());
        } catch (...) {
            EDGE_LOG_ERROR("[%s] task unknown exception", name_.c_str());
        }
        active_--;
    }
}

} // namespace edge
