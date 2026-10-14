// dag_exec.cpp —— 一个最小但真实的「算子 DAG 执行器」（能力点「搭建高效可靠的计算框架」）
//
// 组件：
//   · 依赖图：节点间的 data 依赖（邻接 + 入度），拓扑调度
//   · 线程池 + 就绪队列：入度归零的节点立刻可执行，天然重叠
//   · 内存池：按"最大并发节点数"预分配缓冲并复用，避免每节点 malloc/free
//   · 监控：墙钟、有效并行度、内存池命中率
//
// 用感知流水线的形状做负载：preprocess(width 宽而浅) -> backbone(串行深) -> heads(分叉) -> nms(汇聚)
//
// 编译：g++ -O2 -std=c++17 -pthread dag_exec.cpp -o dag_exec

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

static double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

static volatile double g_sink;   // 防止计算结果被优化掉（比 printf("") 干净）

// 模拟一段确定性计算（防止被优化掉）
static double compute(int units) {
  volatile double x = 1.0;
  double acc = 0;
  for (int i = 0; i < units; ++i) { x = x * 1.0000001 + 0.25; acc += x; }
  return acc;
}

struct Node {
  std::string name;
  int units = 0;                    // 计算量（模拟算子的工作量）
  std::vector<int> deps;            // 依赖的节点
  std::vector<int> outs;            // 依赖它的节点
};

// ------------------------------ 内存池 ------------------------------
class BufferPool {
 public:
  explicit BufferPool(size_t block_bytes, int blocks)
      : block_(block_bytes), free_(blocks) {
    for (int i = 0; i < blocks; ++i) free_[i] = i;
    store_.resize(blocks, std::vector<char>(block_bytes));
  }
  // 借一块缓冲；池空则返回 -1（由调用方决定是否报错/阻塞）
  int acquire() {
    std::lock_guard<std::mutex> lk(mu_);
    if (free_.empty()) { miss_++; return -1; }
    int id = free_.back(); free_.pop_back(); hit_++; borrowed_++; peak_ = std::max(peak_, borrowed_);
    return id;
  }
  void release(int id) {
    std::lock_guard<std::mutex> lk(mu_);
    free_.push_back(id); borrowed_--;
  }
  long hits() const { return hit_; }
  long misses() const { return miss_; }
  long peak() const { return peak_; }

 private:
  std::vector<std::vector<char>> store_;
  std::vector<int> free_;
  size_t block_;
  mutable std::mutex mu_;
  long hit_ = 0, miss_ = 0, borrowed_ = 0, peak_ = 0;
};

// ------------------------------ 执行器 ------------------------------
class DagExecutor {
 public:
  DagExecutor(std::vector<Node> nodes, int nthreads, int pool_blocks)
      : nodes_(std::move(nodes)), indeg_(nodes_.size(), 0), pool_(1 << 20, pool_blocks) {
    for (size_t i = 0; i < nodes_.size(); ++i)
      for (int d : nodes_[i].deps) { nodes_[d].outs.push_back((int)i); indeg_[i]++; }
    for (size_t i = 0; i < nodes_.size(); ++i) if (indeg_[i] == 0) ready_.push((int)i);
    for (int t = 0; t < nthreads; ++t) workers_.emplace_back([this] { worker(); });
  }
  ~DagExecutor() {
    { std::lock_guard<std::mutex> lk(mu_); stop_ = true; cv_.notify_all(); }
    for (auto& w : workers_) w.join();
  }

  struct Stats { double wall_ms; long peak_concurrent; double max_parallel; long pool_hits, pool_misses, pool_peak; };

  Stats run() {
    double t0 = now_ms();
    { std::lock_guard<std::mutex> lk(mu_); cv_.notify_all(); }
    {
      std::unique_lock<std::mutex> lk(done_mu_);
      done_cv_.wait(lk, [this] { return finished_ == (int)nodes_.size(); });
    }
    double wall = now_ms() - t0;
    Stats s{wall, peak_conc_.load(), (double)busy_us_.load() / (wall * 1000.0 * workers_.size()),
            pool_.hits(), pool_.misses(), pool_.peak()};
    return s;
  }

  // 串行参考：按拓扑序单线程跑
  double run_serial() {
    std::vector<int> order; std::vector<int> ind = indeg_;
    std::queue<int> q;
    for (size_t i = 0; i < ind.size(); ++i) if (ind[i] == 0) q.push((int)i);
    while (!q.empty()) {
      int u = q.front(); q.pop(); order.push_back(u);
      for (int v : nodes_[u].outs) if (--ind[v] == 0) q.push(v);
    }
    double t0 = now_ms();
    double acc = 0;
    for (int u : order) acc += compute(nodes_[u].units);
    double t = now_ms() - t0;
    g_sink = acc;                        // 防优化
    return t;
  }

 private:
  void worker() {
    for (;;) {
      int id = -1;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this] { return stop_ || !ready_.empty(); });
        if (stop_ && ready_.empty()) return;
        id = ready_.front(); ready_.pop();
        int cur = ++running_;
        peak_conc_.store(std::max(peak_conc_.load(), (long)cur));
      }
      double t0 = now_ms();
      int buf = pool_.acquire();                 // 借用一块输出缓冲（内存池复用）
      double acc = compute(nodes_[id].units);
      if (buf >= 0) pool_.release(buf);
      g_sink = acc;
      busy_us_.fetch_add((long)((now_ms() - t0) * 1000));
      {
        std::lock_guard<std::mutex> lk(mu_);
        running_--;
        for (int v : nodes_[id].outs) if (--indeg_[v] == 0) { ready_.push(v); cv_.notify_one(); }
        {
          std::lock_guard<std::mutex> dk(done_mu_);
          if (++finished_ == (int)nodes_.size()) done_cv_.notify_all();
        }
      }
    }
  }

  std::vector<Node> nodes_;
  std::vector<int> indeg_;
  std::queue<int> ready_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::mutex done_mu_;
  std::condition_variable done_cv_;
  int finished_ = 0, running_ = 0;
  bool stop_ = false;
  std::atomic<long> peak_conc_{0};
  std::atomic<long> busy_us_{0};
  BufferPool pool_;
  std::vector<std::thread> workers_;
};

// 造一张"感知流水线"形状的图：4 路并行预处理 -> 3 层主干(串行) -> 2 个分叉头 -> 1 个 NMS 汇聚
static std::vector<Node> make_graph(int streams) {
  std::vector<Node> g;
  std::vector<int> pre_ids;
  for (int s = 0; s < streams; ++s) pre_ids.push_back((int)g.size()), g.push_back({"pre" + std::to_string(s), 300000, {}, {}});
  int prev = -1;
  std::vector<int> deps = pre_ids;
  for (int l = 0; l < 3; ++l) {
    int id = (int)g.size();
    g.push_back({"backbone" + std::to_string(l), 1500000, deps, {}});
    deps = {id};
    prev = id;
  }
  int h1 = (int)g.size(); g.push_back({"cls_head", 500000, {prev}, {}});
  int h2 = (int)g.size(); g.push_back({"box_head", 500000, {prev}, {}});
  g.push_back({"nms", 200000, {h1, h2}, {}});
  return g;
}

int main() {
  int streams = 8, threads = (int)std::thread::hardware_concurrency();
  if (threads <= 0) threads = 4;
  auto g = make_graph(streams);
  int max_conc = streams + 2;                       // 最大并发节点数（估计）
  std::printf("=== 算子 DAG 执行器 ===\n");
  std::printf("图: %zu 节点, %d 路并行预处理 + 3 层主干 + 2 头 + NMS  线程池 %d\n\n",
              g.size(), streams, threads);

  DagExecutor ex(g, threads, max_conc);
  double serial = ex.run_serial();
  auto st = ex.run();

  std::printf("---- 结果 ----\n");
  std::printf("  串行(拓扑序单线程) : %8.1f ms\n", serial);
  std::printf("  DAG 调度并行       : %8.1f ms   加速 %.2fx\n", st.wall_ms, serial / st.wall_ms);
  std::printf("  峰值并发节点数     : %ld   线程池 %d  (有效并行度 %.2f)\n", st.peak_concurrent, threads, st.max_parallel);
  std::printf("  内存池: 命中 %ld 次 / 未命中 %ld 次, 池内峰值占用 %ld 块 (预分配 %d 块)\n",
              st.pool_hits, st.pool_misses, st.pool_peak, max_conc);
  std::printf("\n要点：\n");
  std::printf("  1. 依赖调度只在「入度归零」时放行 -> 不依赖人工排流水线，改图不改调度器\n");
  std::printf("  2. 串行链路(backbone)决定下界，宽阶段(pre/*)决定并行上限 -> 图结构即性能上限\n");
  std::printf("  3. 内存池按「最大并发」预分配 -> 池内峰值不超上限，避免每节点 malloc 抖动\n");
  return 0;
}
