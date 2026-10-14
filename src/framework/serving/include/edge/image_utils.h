#pragma once
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace edge {

// 图像前处理：解码(JPEG/PNG) → 双线性缩放 → 归一化 → HWC→CHW，
// 结果直接写入 out（池化内存，零拷贝目标）。
// 生产环境端侧可替换为硬件加速（RK 平台用 RGA，Intel 用 VA-API/OpenVINO preprocess）。
bool preprocess_image(const uint8_t* data, size_t len,
                      int w, int h,
                      const float* mean, const float* stdv, bool use_mean_std,
                      float* out);

// 通用后处理：softmax + top-k（分类模型）。
// 检测类模型（YOLO 等）可在此挂 decode + NMS，接口保持一致。
std::vector<std::pair<int, float>> softmax_topk(const float* logits,
                                                size_t n, int k);

} // namespace edge
