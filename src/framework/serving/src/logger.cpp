#include "edge/logger.h"
#include <chrono>
#include <ctime>

namespace edge {

Logger& Logger::instance() {
    static Logger inst;
    return inst;
}

void Logger::log(LogLevel lv, const char* file, int line, const std::string& msg) {
    static const char* names[] = {"DEBUG", "INFO ", "WARN ", "ERROR"};
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()).count();
    auto tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    char ts[32];
    snprintf(ts, sizeof(ts), "%02d:%02d:%02d.%03d",
             tm.tm_hour, tm.tm_min, tm.tm_sec, (int)(ms % 1000));
    // 只保留文件名（去掉路径），日志更紧凑
    const char* base = file;
    for (const char* p = file; *p; ++p)
        if (*p == '/') base = p + 1;
    std::lock_guard<std::mutex> lk(mu_);
    fprintf(stderr, "[%s][%s][%s:%d] %s\n",
            ts, names[(int)lv], base, line, msg.c_str());
}

} // namespace edge
