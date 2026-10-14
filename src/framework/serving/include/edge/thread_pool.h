#pragma once
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace edge {

// 支持优先级与有界队列的线程池。
// 关键设计：
// 1) 优先级队列：priority 大的先出队，同优先级按 seq 保 FIFO 顺序
// 2) 有界队列背压：队列满时 submit 阻塞，防止请求洪峰导致内存无限增长
// 3) 任务级异常隔离：单个任务抛异常只记日志，worker 不退出
// 4) 两级停机策略：drain=true 排空队列后退出（优雅停机），false 立即丢弃
class ThreadPool {
public:
    using Task = std::function<void()>;

    ThreadPool(const std::string& name, size_t n_threads, size_t max_queue_size = 0);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // 提交任务；有界队列满时阻塞等待（天然背压）
    void submit(Task task, int priority = 0);
    size_t queue_depth() const;
    size_t active_workers() const { return active_.load(); }

    // drain=true: 处理完剩余任务后退出；false: 丢弃未执行任务立即退出
    void stop(bool drain = true);

private:
    struct Item {
        Task task;
        int priority;
        uint64_t seq;
        bool operator<(const Item& o) const {
            if (priority != o.priority) return priority < o.priority;
            return seq > o.seq;  // seq 小的先出（同优先级 FIFO）
        }
    };

    void worker_loop();

    std::string name_;
    size_t max_queue_size_;
    std::vector<std::thread> workers_;
    std::priority_queue<Item> queue_;
    mutable std::mutex mu_;
    std::condition_variable cv_;       // 通知 worker 有任务
    std::condition_variable not_full_; // 通知 submit 队列有空间
    std::atomic<bool> stopping_{false};
    std::atomic<uint64_t> seq_{0};
    std::atomic<size_t> active_{0};
};

} // namespace edge
