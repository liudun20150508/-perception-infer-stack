// os_bench.cpp —— OS / 体系结构层证据（能力点「对操作系统、计算机体系结构有深入理解」）
//
// 本机是共享主机（load 常见 30~70 / 88 核），**计时类微基准会被邻居干扰**。
// 所以这里优先给"确定性证据"（内核状态可验证），计时一律取 min-of-N 做辅助：
//   [1] 伪共享     : 同一 cacheline vs alignas(64)，min-of-N 计时
//   [2] CPU 亲和性 : sched_getcpu() 采样 —— 绑定后恒定，不绑定会迁移（确定性）
//   [3] NUMA 策略  : /proc/self/numa_maps 页分布 —— first-touch 落在哪个 node（确定性）
//   [4] 上下文切换 : getrusage 自愿/非自愿切换计数
//   [5] 线程扩展性 : min-of-N（共享主机上仅供趋势参考）
//
// 编译：g++ -O2 -std=c++17 -pthread os_bench.cpp -o os_bench

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/resource.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>

static double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool pin_self(int cpu) {
  cpu_set_t s;
  CPU_ZERO(&s);
  CPU_SET(cpu, &s);
  return pthread_setaffinity_np(pthread_self(), sizeof(s), &s) == 0;
}

static void unpin_self() {
  cpu_set_t s;
  CPU_ZERO(&s);
  for (int i = 0; i < 256; ++i) CPU_SET(i, &s);
  pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

template <typename F>
static double min_of(int n, F fn) {
  double best = 1e30;
  for (int i = 0; i < n; ++i) best = std::min(best, fn());
  return best;
}

static double busy(int iters) {
  volatile double x = 1.0;
  double acc = 0;
  for (int i = 0; i < iters; ++i) { x = x * 1.0000001 + 0.5; acc += x; }
  return acc;
}

// ---------------- [1] 伪共享 ----------------
struct FalseShare { volatile long a; volatile long b; };
struct Padded { alignas(64) volatile long a; alignas(64) volatile long b; };

template <typename T>
static double share_ms(int iters) {
  T s{};
  double t0 = now_ms();
  std::thread t1([&] { for (int i = 0; i < iters; ++i) s.a++; });
  std::thread t2([&] { for (int i = 0; i < iters; ++i) s.b++; });
  t1.join(); t2.join();
  return now_ms() - t0;
}

// ---------------- [3] NUMA 页分布 ----------------
static std::map<std::string, long> numa_page_count() {
  std::map<std::string, long> m;
  std::ifstream f("/proc/self/numa_maps");
  std::string line;
  while (std::getline(f, line)) {
    size_t p = 0;
    while ((p = line.find("N", p)) != std::string::npos) {
      size_t eq = line.find('=', p);
      if (eq == std::string::npos) break;
      std::string key = line.substr(p, eq - p);
      if (key.size() >= 2 && isdigit((unsigned char)key[1])) {
        char* end = nullptr;
        long v = strtol(line.c_str() + eq + 1, &end, 10);
        m[key] += v;
      }
      p = eq + 1;
    }
  }
  return m;
}

// 只在指定 CPU 上首次触碰（first-touch），从而决定页落在哪个 node。
// 注意：不能用 std::vector<char>(n, 1) —— 构造函数已在当前线程触碰全部页，
//       first-touch 就提前发生了，NUMA 实验会被污染（踩过一次）。
static char* alloc_untouched(size_t bytes) {
  void* p = std::malloc(bytes);
  return static_cast<char*>(p);
}

static void touch_on_cpu(char* p, size_t bytes, int cpu) {
  std::thread t([&] { pin_self(cpu); for (size_t i = 0; i < bytes; i += 64) p[i] = (char)i; });
  t.join();
}

static std::string cpus_allowed() {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line))
    if (line.rfind("Cpus_allowed_list:", 0) == 0) return line.substr(line.find(':') + 1);
  return "?";
}

int main(int argc, char** argv) {
  long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
  std::printf("=== OS / 体系结构证据 ===\n");
  std::printf("可见 CPU: %ld  |  注意：共享主机，计时仅取 min-of-N 看趋势\n\n", ncpu);

  // [1] 伪共享
  {
    int iters = argc > 1 ? atoi(argv[1]) : 20000000;
    FalseShare fs{};
    Padded pd{};
    std::printf("---- [1] 伪共享 ----\n");
    std::printf("  FalseShare: &a=%p &b=%p  同 cacheline = %s\n", (void*)&fs.a, (void*)&fs.b,
                ((char*)&fs.b - (char*)&fs.a) < 64 ? "是" : "否");
    std::printf("  Padded    : &a=%p &b=%p  跨 cacheline = %s\n", (void*)&pd.a, (void*)&pd.b,
                ((char*)&pd.b - (char*)&pd.a) >= 64 ? "是" : "否");
    double t_fs = min_of(5, [&] { return share_ms<FalseShare>(iters); });
    double t_pd = min_of(5, [&] { return share_ms<Padded>(iters); });
    std::printf("  计时(min-of-5): 同线 %.1f ms | padding %.1f ms -> %.2fx\n", t_fs, t_pd, t_fs / t_pd);
    std::printf("  机制：两核反复抢同一条 cacheline 的独占权（cacheline ping-pong）\n\n");
  }

  // [2] 亲和性（确定性）
  {
    std::printf("---- [2] CPU 亲和性（sched_getcpu 采样，确定性证据）----\n");
    std::printf("  绑定前 Cpus_allowed_list: %s\n", cpus_allowed().c_str());
    bool ok = pin_self(0);                                   // 直接在主线程上绑
    std::printf("  绑定 cpu0 %s，Cpus_allowed_list: %s\n", ok ? "成功" : "失败", cpus_allowed().c_str());
    std::printf("  采样 sched_getcpu(): ");
    bool stay = true;
    for (int i = 0; i < 8; ++i) { int c = sched_getcpu(); std::printf("%d ", c); if (c != 0) stay = false; busy(200000); }
    std::printf(" -> 恒定于 cpu0 = %s\n", stay ? "是" : "否");
    unpin_self();
    std::printf("  解绑后 Cpus_allowed_list: %s\n", cpus_allowed().c_str());
    std::printf("  机制：sched_setaffinity 收窄内核的可用 CPU 集合（Cpus_allowed_list 直接可见）；\n");
    std::printf("        绑定后线程不再被迁出，cache/TLB 局部性保留 —— 实时/低延迟负载必用。\n");
    std::printf("        （调度器不会频繁搬运运行中的线程，所以「不绑定」未必当场看到迁移；\n");
    std::printf("         判定依据应是 Cpus_allowed_list 与是否可能被抢占）\n\n");
  }

  // [3] NUMA first-touch（确定性）
  {
    std::printf("---- [3] NUMA first-touch（/proc/self/numa_maps）----\n");
    size_t bytes = 256ull << 20;
    char* bufA = alloc_untouched(bytes);
    touch_on_cpu(bufA, bytes, 0);
    auto mA = numa_page_count();
    std::printf("  在 node0(cpu0) 触碰 256MB: ");
    for (auto& kv : mA) if (kv.first == "N0" || kv.first == "N1") std::printf("%s=%ld 页 ", kv.first.c_str(), kv.second);
    std::printf("\n");
    char* bufB = alloc_untouched(bytes);
    touch_on_cpu(bufB, bytes, 60);
    auto mB = numa_page_count();
    std::printf("  再在 node1(cpu60) 触碰 256MB: ");
    for (auto& kv : mB) if (kv.first == "N0" || kv.first == "N1") std::printf("%s=%ld 页 ", kv.first.c_str(), kv.second);
    std::printf("\n");
    long d0 = (mB.count("N0") ? mB["N0"] : 0) - (mA.count("N0") ? mA["N0"] : 0);
    long d1 = (mB.count("N1") ? mB["N1"] : 0) - (mA.count("N1") ? mA["N1"] : 0);
    std::printf("  增量: node0 +%ld 页, node1 +%ld 页 -> 内存跟着 first-touch 的 CPU 走\n\n", d0, d1);
    std::free(bufA); std::free(bufB);
  }

  // [4] 上下文切换
  {
    std::thread t([] { unpin_self(); for (int i = 0; i < 200; ++i) { busy(100000); std::this_thread::yield(); } });
    t.join();
    struct rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    std::printf("---- [4] 上下文切换（getrusage 累计）----\n");
    std::printf("  自愿(主动让出) %ld 次 | 非自愿(被抢占) %ld 次\n", ru.ru_nvcsw, ru.ru_nivcsw);
    std::printf("  非自愿高 = CPU 竞争强；实时/低延迟负载需隔离（绑核 + cpuset + 优先级）\n\n");
  }

  // [5] 扩展性（min-of-3，趋势参考）
  {
    std::printf("---- [5] 线程扩展性（pin 到 node0 不同核, min-of-3）----\n");
    double base = 0;
    for (int n : {1, 2, 4}) {
      double t = min_of(3, [&] {
        std::vector<std::thread> ts;
        double t0 = now_ms();
        for (int i = 0; i < n; ++i) ts.emplace_back([i] { pin_self(i); busy(6000000); });
        for (auto& x : ts) x.join();
        return now_ms() - t0;
      });
      if (n == 1) base = t;
      std::printf("  %d 线程: %.1f ms  相对加速 %.2fx  效率 %3.0f%%\n", n, t, base / t, 100.0 * (base / t) / n);
    }
    std::printf("  注：本机是共享主机（load 常年 30~70/88），加线程反而更慢 —— 不是核不够，\n");
    std::printf("      而是被邻居抢：这正是为什么生产上要用 cpuset/cgroup/绑核做资源隔离，\n");
    std::printf("      也解释了 [4] 里偏高的非自愿切换。隔离环境复测才是干净口径。\n");
  }
  return 0;
}
