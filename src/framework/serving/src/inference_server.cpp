#include "edge/inference_server.h"
#include "edge/image_utils.h"
#include "edge/logger.h"
#include <algorithm>
#include <stdexcept>

namespace edge {

InferenceServer::InferenceServer(ServerOptions opts,
                                 std::shared_ptr<ModelManager> models)
    : opts_(opts),
      models_(std::move(models)),
      pre_pool_("pre", opts_.pre_threads, opts_.queue_capacity),
      infer_pool_("infer", opts_.infer_threads, opts_.queue_capacity),
      post_pool_("post", opts_.post_threads, opts_.queue_capacity),
      input_pool_(std::make_unique<BufferPool<float>>(opts_.buffer_blocks,
                                                      opts_.input_elems)),
      output_pool_(std::make_unique<BufferPool<float>>(opts_.buffer_blocks,
                                                       opts_.output_elems)) {
    lat_hist_.reserve(4096);
}

InferenceServer::~InferenceServer() {
    if (running_.load()) stop(false);
}

void InferenceServer::start() {
    running_ = true;
    start_at_ = std::chrono::steady_clock::now();
    watchdog_ = std::thread([this] { watchdog_loop(); });
    EDGE_LOG_INFO("inference server started: pre %zu / infer %zu / post %zu threads, "
                  "queue cap %zu, buffer blocks %zu",
                  opts_.pre_threads, opts_.infer_threads, opts_.post_threads,
                  opts_.queue_capacity, opts_.buffer_blocks);
}

void InferenceServer::stop(bool drain) {
    if (!running_.exchange(false)) return;
    if (watchdog_.joinable()) watchdog_.join();

    if (!drain) {
        // 立即停机：所有在途/排队请求以错误返回，避免 future 悬挂
        std::vector<TaskPtr> pending;
        {
            std::lock_guard<std::mutex> lk(wd_mu_);
            for (auto& [id, t] : inflight_map_) pending.push_back(t);
        }
        for (auto& t : pending)
            finalize(t, make_error(t, "server stopped"));
    }

    // 顺序停机：pre → infer → post。上游先排空，保证下游在 drain 期间仍可消费
    pre_pool_.stop(drain);
    infer_pool_.stop(drain);
    post_pool_.stop(drain);
    EDGE_LOG_INFO("inference server stopped (drain=%d)", (int)drain);
}

std::future<InferResponse> InferenceServer::submit(InferRequest req) {
    if (!running_.load())
        throw std::runtime_error("server not running");

    // 在途上限：快速失败而非无限堆积（保护系统，对应"资源管理"）
    if (inflight_.load() >= opts_.max_inflight) {
        InferResponse r;
        r.id = req.id;
        r.ok = false;
        r.error = "server overloaded (inflight >= max)";
        std::promise<InferResponse> p;
        p.set_value(std::move(r));
        return p.get_future();
    }

    if (req.id == 0) req.id = next_id_++;
    auto t = std::make_shared<Task>();
    t->req = std::move(req);
    t->enqueue_at = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lk(wd_mu_);
        inflight_map_[t->req.id] = t;
    }
    inflight_++;
    auto fut = t->resp.get_future();
    try {
        // 前处理级入队（有界队列满则在此阻塞 → 背压传导给调用方）
        pre_pool_.submit([this, t] { preprocess_worker(t); }, t->req.priority);
    } catch (...) {
        finalize(t, make_error(t, "server stopping"));
    }
    return fut;
}

InferenceServer::Snapshot InferenceServer::snapshot() {
    Snapshot s;
    s.running = running_.load();
    s.inflight = inflight_.load();
    s.pre_queue = pre_pool_.queue_depth();
    s.infer_queue = infer_pool_.queue_depth();
    s.post_queue = post_pool_.queue_depth();
    s.active_pre = pre_pool_.active_workers();
    s.active_infer = infer_pool_.active_workers();
    s.active_post = post_pool_.active_workers();
    s.done = done_.load();
    s.failed = failed_.load();
    s.timed_out = timed_out_.load();
    double sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_at_).count();
    s.qps = sec > 0 ? (double)s.done / sec : 0;
    if (s.done > 0) {
        s.avg_us = (int64_t)(total_lat_us_.load() / s.done);
        std::vector<int64_t> lat;
        {
            std::lock_guard<std::mutex> lk(hist_mu_);
            for (auto v : lat_hist_)
                if (v > 0) lat.push_back(v);
        }
        if (!lat.empty()) {
            std::sort(lat.begin(), lat.end());
            s.p50_us = lat[lat.size() / 2];
            s.p99_us = lat[lat.size() * 99 / 100];
        }
    }
    s.input_pool = input_pool_->stats();
    s.output_pool = output_pool_->stats();
    return s;
}

InferResponse InferenceServer::make_error(const TaskPtr& t,
                                           const std::string& err) {
    InferResponse r;
    r.id = t->req.id;
    r.ok = false;
    r.error = err;
    r.pre_us = t->pre_us;
    r.infer_us = t->infer_us;
    r.post_us = t->post_us;
    return r;
}

void InferenceServer::finalize(TaskPtr t, InferResponse r) {
    // 注意：下面的 set_response 会 move 走 r，之后再读 r.error 只会读到空串
    //（实测后果：782 次 timeout 失败全部被计成"非超时"，指标里"超时 0"是假数）。
    // 所以先把分类读出来，再 move。
    const bool ok = r.ok;
    const bool is_timeout =
        !ok && r.error.find("timeout") != std::string::npos;
    if (!t->set_response(std::move(r))) return;  // 已处理过（例如停机时）
    if (ok) done_++;
    else {
        failed_++;
        if (is_timeout) timed_out_++;
    }
    int64_t total = t->pre_us + t->infer_us + t->post_us;
    total_lat_us_ += total;
    {
        std::lock_guard<std::mutex> lk(hist_mu_);
        if (lat_hist_.size() < 4096) lat_hist_.push_back(total);
        else {
            lat_hist_[hist_pos_] = total;
            hist_pos_ = (hist_pos_ + 1) % 4096;
        }
    }
    {
        std::lock_guard<std::mutex> lk(wd_mu_);
        inflight_map_.erase(t->req.id);
    }
    inflight_--;
}

void InferenceServer::preprocess_worker(TaskPtr t) {
    try {
        t->stage = 1;
        auto now = std::chrono::steady_clock::now();
        if (now - t->enqueue_at > t->req.deadline) {
            // 协作式超时取消：排队期间已超时，直接丢弃不浪费算力
            finalize(t, make_error(t, "timeout in pre queue"));
            return;
        }
        auto in = input_pool_->borrow(std::chrono::milliseconds(200));
        if (!in.valid()) {
            finalize(t, make_error(t, "input buffer pool exhausted"));
            return;
        }
        auto t0 = std::chrono::steady_clock::now();
        bool ok = preprocess_image(t->req.image->data(), t->req.image->size(),
                                   t->req.input_w, t->req.input_h,
                                   t->req.mean, t->req.stdv, t->req.use_mean_std,
                                   in.data());
        auto t1 = std::chrono::steady_clock::now();
        t->pre_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        if (!ok) {
            finalize(t, make_error(t, "image decode failed"));
            return;
        }
        t->input_buf = std::move(in);  // 池化缓冲随任务传递，推理完自动归还
        // 推理级按优先级调度（ThreadPool 优先级队列）
        infer_pool_.submit([this, t] { infer_worker(t); }, t->req.priority);
    } catch (const std::exception& e) {
        // 兜底：任何异常都必须设置 promise，避免调用方永久悬挂
        finalize(t, make_error(t, std::string("preprocess exception: ") + e.what()));
    } catch (...) {
        finalize(t, make_error(t, "preprocess unknown exception"));
    }
}

void InferenceServer::infer_worker(TaskPtr t) {
    try {
        t->stage = 2;
        auto now = std::chrono::steady_clock::now();
        if (now - t->enqueue_at > t->req.deadline) {
            finalize(t, make_error(t, "timeout before inference"));
            return;
        }
        auto engine = models_->acquire(t->req.model);
        if (!engine) {
            finalize(t, make_error(t, "model unavailable: " + t->req.model));
            return;
        }
        const auto& out_info = engine->outputs_info()[0];
        if (engine->inputs_info().empty() || engine->outputs_info().empty() ||
            out_info.elem_count > opts_.output_elems) {
            finalize(t, make_error(t, "bad model tensor info"));
            return;
        }
        auto out = output_pool_->borrow(std::chrono::milliseconds(200));
        if (!out.valid()) {
            finalize(t, make_error(t, "output buffer pool exhausted"));
            return;
        }
        // 零拷贝：输入/输出都绑定池化内存，ORT 原地读写，无中间拷贝
        auto in_tensor = engine->create_input_tensor(0, t->input_buf.data());
        auto out_tensor = engine->create_output_tensor(0, out.data());
        std::vector<Ort::Value> inputs, outputs;
        inputs.push_back(std::move(in_tensor));
        outputs.push_back(std::move(out_tensor));

        auto t0 = std::chrono::steady_clock::now();
        bool ok = engine->run(inputs, outputs);
        auto t1 = std::chrono::steady_clock::now();
        t->infer_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        if (!ok) {
            finalize(t, make_error(t, "inference failed"));
            return;
        }
        t->output_buf = std::move(out);
        t->out_elems = out_info.elem_count;
        t->stage = 3;
        post_pool_.submit([this, t] { postprocess_worker(t); });
    } catch (const std::exception& e) {
        finalize(t, make_error(t, std::string("infer exception: ") + e.what()));
    } catch (...) {
        finalize(t, make_error(t, "infer unknown exception"));
    }
}

void InferenceServer::postprocess_worker(TaskPtr t) {
    try {
        auto t0 = std::chrono::steady_clock::now();
        // 通用后处理：softmax + top-k（检测类模型可在此挂 decode+NMS）
        auto topk = softmax_topk(t->output_buf.data(), t->out_elems, t->req.topk);
        auto t1 = std::chrono::steady_clock::now();
        t->post_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        InferResponse r;
        r.id = t->req.id;
        r.ok = true;
        r.topk = std::move(topk);
        r.pre_us = t->pre_us;
        r.infer_us = t->infer_us;
        r.post_us = t->post_us;
        r.queue_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         t1 - t->enqueue_at).count() - t->pre_us - t->infer_us - t->post_us;
        finalize(t, std::move(r));
    } catch (const std::exception& e) {
        finalize(t, make_error(t, std::string("postprocess exception: ") + e.what()));
    } catch (...) {
        finalize(t, make_error(t, "postprocess unknown exception"));
    }
}

void InferenceServer::watchdog_loop() {
    static const char* stages[] = {"pre-queue", "preprocess", "inference", "postprocess"};
    while (running_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        auto now = std::chrono::steady_clock::now();
        std::vector<std::string> msgs;
        {
            std::lock_guard<std::mutex> lk(wd_mu_);
            for (const auto& [id, t] : inflight_map_) {
                if (t->wd_reported) continue;
                if (now - t->enqueue_at > t->req.deadline) {
                    t->wd_reported = true;
                    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  now - t->enqueue_at).count();
                    msgs.push_back("watchdog: request " + std::to_string(id) +
                                   " (model " + t->req.model + ") stuck in stage " +
                                   stages[t->stage] + " for " +
                                   std::to_string(ms) + " ms");
                }
            }
        }
        for (auto& m : msgs) EDGE_LOG_WARN("%s", m.c_str());
    }
}

} // namespace edge
