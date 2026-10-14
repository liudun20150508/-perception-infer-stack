#pragma once
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>

namespace edge {

// 定长内存块池：构造时一次性预分配，运行期只借/还，不 malloc/free。
// 目的：
// 1) 消除高并发推理场景下频繁分配造成的碎片与开销（对应"内存泄漏/重复拷贝"治理）
// 2) 池化 buffer 直接绑定给 ORT IoBinding 作为输入/输出内存 → 全链路零拷贝
// Handle 为 RAII：析构自动归还，异常路径也不泄漏，可经 in_use 统计验证
template <typename T>
class BufferPool {
public:
    struct Stats {
        size_t total_blocks = 0;
        size_t in_use = 0;
        size_t block_elems = 0;
    };

    BufferPool(size_t n_blocks, size_t elems_per_block)
        : block_elems_(elems_per_block), total_blocks_(n_blocks) {
        for (size_t i = 0; i < n_blocks; ++i)
            free_.push_back(std::unique_ptr<T[]>(new T[elems_per_block]));
    }

    class Handle {
    public:
        Handle() = default;
        Handle(BufferPool* pool, std::unique_ptr<T[]> block)
            : pool_(pool), block_(std::move(block)) {}
        Handle(Handle&&) noexcept = default;
        Handle& operator=(Handle&&) noexcept = default;
        Handle(const Handle&) = delete;
        Handle& operator=(const Handle&) = delete;
        ~Handle() { if (block_) pool_->give_back(std::move(block_)); }

        T* data() { return block_.get(); }
        const T* data() const { return block_.get(); }
        bool valid() const { return block_ != nullptr; }
        size_t elems() const { return block_ ? pool_->block_elems_ : 0; }

    private:
        BufferPool* pool_ = nullptr;
        std::unique_ptr<T[]> block_;
    };

    // 借一块缓冲；池空时最多等待 timeout（超时返回空 Handle，
    // 由上层决定拒绝请求还是重试 —— 避免无限阻塞）
    Handle borrow(std::chrono::milliseconds timeout = std::chrono::milliseconds(100)) {
        std::unique_lock<std::mutex> lk(mu_);
        if (!cv_.wait_for(lk, timeout, [this] { return !free_.empty(); }))
            return {};
        auto block = std::move(free_.front());
        free_.pop_front();
        in_use_++;
        return Handle(this, std::move(block));
    }

    Stats stats() const {
        std::lock_guard<std::mutex> lk(mu_);
        return {total_blocks_, in_use_, block_elems_};
    }

private:
    void give_back(std::unique_ptr<T[]> block) {
        std::lock_guard<std::mutex> lk(mu_);
        free_.push_back(std::move(block));
        in_use_--;
        cv_.notify_one();
    }

    size_t block_elems_;
    size_t total_blocks_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::unique_ptr<T[]>> free_;
    size_t in_use_ = 0;
};

} // namespace edge
