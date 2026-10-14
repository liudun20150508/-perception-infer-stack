#include "edge/image_utils.h"
#include <algorithm>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#include "stb_image.h"

namespace edge {

bool preprocess_image(const uint8_t* data, size_t len,
                      int w, int h,
                      const float* mean, const float* stdv, bool use_mean_std,
                      float* out) {
    int iw = 0, ih = 0, ch = 0;
    stbi_uc* img = stbi_load_from_memory(data, (int)len, &iw, &ih, &ch, 3);
    if (!img) return false;

    const float sx = (float)iw / (float)w;
    const float sy = (float)ih / (float)h;
    const int C = 3;
    // 双线性插值 + 归一化 + HWC→CHW 一步完成（只写一遍内存）
    for (int y = 0; y < h; ++y) {
        float fy = ((float)y + 0.5f) * sy - 0.5f;
        int y0 = (int)floorf(fy);
        float wy = fy - (float)y0;
        int y1 = y0 + 1;
        if (y0 < 0) y0 = 0;
        if (y1 > ih - 1) y1 = ih - 1;
        for (int x = 0; x < w; ++x) {
            float fx = ((float)x + 0.5f) * sx - 0.5f;
            int x0 = (int)floorf(fx);
            float wx = fx - (float)x0;
            int x1 = x0 + 1;
            if (x0 < 0) x0 = 0;
            if (x1 > iw - 1) x1 = iw - 1;
            for (int c = 0; c < C; ++c) {
                const uint8_t* p = img + (size_t)(y0 * iw + x0) * C + c;
                float v = ((float)p[0] * (1.f - wx) + (float)p[C] * wx) * (1.f - wy) +
                          ((float)p[iw * C] * (1.f - wx) +
                           (float)p[iw * C + C] * wx) * wy;
                v /= 255.0f;
                if (use_mean_std) v = (v - mean[c]) / stdv[c];
                out[(size_t)c * w * h + (size_t)y * w + x] = v;
            }
        }
    }
    stbi_image_free(img);
    return true;
}

std::vector<std::pair<int, float>> softmax_topk(const float* logits,
                                                size_t n, int k) {
    std::vector<std::pair<int, float>> scores;
    scores.reserve(n);
    // 减去最大值保证数值稳定（防 exp 溢出）
    float mx = -INFINITY;
    for (size_t i = 0; i < n; ++i) mx = std::max(mx, logits[i]);
    float sum = 0.f;
    for (size_t i = 0; i < n; ++i) {
        float v = expf(logits[i] - mx);
        scores.emplace_back((int)i, v);
        sum += v;
    }
    for (auto& s : scores) s.second /= sum;
    if ((size_t)k > n) k = (int)n;
    std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    scores.resize(k);
    return scores;
}

} // namespace edge
