#pragma once
#include <cstdio>
#include <mutex>
#include <string>

namespace edge {

enum class LogLevel { Debug = 0, Info, Warn, Error };

// 轻量级线程安全日志。端侧环境刻意不引入 spdlog 等重依赖，
// 生产环境可无缝替换为异步日志（写文件 + 滚动）。
class Logger {
public:
    static Logger& instance();
    void set_level(LogLevel lv) { level_ = lv; }
    LogLevel level() const { return level_; }
    void log(LogLevel lv, const char* file, int line, const std::string& msg);

private:
    std::mutex mu_;
    LogLevel level_ = LogLevel::Info;
};

#define EDGE_LOG(lv, ...) do { \
    auto& _l = ::edge::Logger::instance(); \
    if (lv >= _l.level()) { \
        char _buf[1024]; \
        snprintf(_buf, sizeof(_buf), __VA_ARGS__); \
        _l.log(lv, __FILE__, __LINE__, _buf); \
    } \
} while (0)

#define EDGE_LOG_DEBUG(...) EDGE_LOG(::edge::LogLevel::Debug, __VA_ARGS__)
#define EDGE_LOG_INFO(...)  EDGE_LOG(::edge::LogLevel::Info,  __VA_ARGS__)
#define EDGE_LOG_WARN(...)  EDGE_LOG(::edge::LogLevel::Warn,  __VA_ARGS__)
#define EDGE_LOG_ERROR(...) EDGE_LOG(::edge::LogLevel::Error, __VA_ARGS__)

} // namespace edge
