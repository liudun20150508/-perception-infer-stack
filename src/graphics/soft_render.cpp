// soft_render.cpp —— 零依赖 CPU 软件光栅化器（能力点「游戏、图形引擎或仿真系统」）
//
// 为什么自研：本机没有任何图形栈（无 libGL/X11/EGL），也没装浏览器，
// 但"图形引擎"这件事本身不依赖 GPU —— 光栅化的每一步都可以在 CPU 上实现：
//   顶点变换(模型->世界->观察->裁剪->屏幕) -> 背面剔除 -> 三角形光栅化(z-buffer,
//   重心坐标插值) -> 兰伯特+环境光着色 -> gamma -> PPM 输出
// 用它把布料仿真引擎导出的真实网格 wedding_dress.obj（26386 顶点 / 51928 三角面）
// 无头渲染成图，证明"图形管线"这一格有实物。
//
// 编译：g++ -O3 -std=c++17 soft_render.cpp -o soft_render
// 用法：./soft_render <in.obj> <out.ppm> [width] [views]

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct V3 { float x = 0, y = 0, z = 0; };
static V3 sub(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 norm(V3 a) { float l = std::sqrt(dot(a, a)); l = l > 1e-9f ? l : 1; return {a.x / l, a.y / l, a.z / l}; }

// ------------------------------ OBJ 解析 ------------------------------
struct Mesh {
  std::vector<V3> pos, nrm;
  std::vector<std::array<int, 3>> tri_pos, tri_nrm;   // 0-based 索引
};

static Mesh load_obj(const char* path) {
  Mesh m;
  FILE* f = std::fopen(path, "r");
  if (!f) { std::fprintf(stderr, "打不开 %s\n", path); std::exit(1); }
  char line[512];
  while (std::fgets(line, sizeof(line), f)) {
    if (line[0] == 'v' && line[1] == ' ') {
      V3 p; std::sscanf(line + 2, "%f %f %f", &p.x, &p.y, &p.z); m.pos.push_back(p);
    } else if (line[0] == 'v' && line[1] == 'n') {
      V3 n; std::sscanf(line + 3, "%f %f %f", &n.x, &n.y, &n.z); m.nrm.push_back(n);
    } else if (line[0] == 'f' && line[1] == ' ') {
      int vi[3], ni[3], got = 0;
      const char* p = line + 2;
      for (int k = 0; k < 3; ++k) {
        while (*p == ' ') ++p;
        int v = 0, t = 0, n = 0;
        if (std::sscanf(p, "%d/%d/%d", &v, &t, &n) == 3) { vi[k] = v; ni[k] = n; }
        else if (std::sscanf(p, "%d//%d", &v, &n) == 2) { vi[k] = v; ni[k] = n; }
        else if (std::sscanf(p, "%d", &v) == 1) { vi[k] = v; ni[k] = 0; }
        else break;
        ++got;
        while (*p && *p != ' ' && *p != '\n') ++p;
      }
      if (got == 3) {
        auto idx = [](int v, int n) { return std::array<int, 3>{v > 0 ? v - 1 : 0, n > 0 ? n - 1 : 0, 0}; };
        auto a = idx(vi[0], ni[0]), b = idx(vi[1], ni[1]), c = idx(vi[2], ni[2]);
        m.tri_pos.push_back({a[0], b[0], c[0]});
        m.tri_nrm.push_back({a[1], b[1], c[1]});
      }
    }
  }
  std::fclose(f);
  return m;
}

// ------------------------------ 渲染 ------------------------------
struct Framebuffer {
  int w, h;
  std::vector<float> zbuf;
  std::vector<std::array<unsigned char, 3>> color;
  Framebuffer(int W, int H) : w(W), h(H), zbuf(W * H, -1e30f), color(W * H, {12, 18, 26}) {}
};

struct Camera {
  V3 eye, at, up;
  float fovy;
};

static void render_view(const Mesh& m, Framebuffer& fb, const Camera& cam, V3 light) {
  V3 fwd = norm(sub(cam.at, cam.eye));
  V3 right = norm(cross(fwd, cam.up));
  V3 up = cross(right, fwd);
  float aspect = (float)fb.w / fb.h;
  float t = std::tan(cam.fovy * 0.5f);
  light = norm(light);

  for (size_t i = 0; i < m.tri_pos.size(); ++i) {
    V3 wp[3], wn[3];
    V3 sp[3]; float sz[3];
    bool ok = true;
    for (int k = 0; k < 3; ++k) {
      wp[k] = m.pos[m.tri_pos[i][k]];
      wn[k] = m.nrm.empty() ? V3{0, 0, 1} : m.nrm[m.tri_nrm[i][k]];
      V3 d = sub(wp[k], cam.eye);
      float zc = dot(d, fwd);                       // 观察空间深度（前方为正）
      if (zc <= 0.01f) { ok = false; break; }
      float xc = dot(d, right), yc = dot(d, up);
      sp[k] = { (xc / (zc * t * aspect) * 0.5f + 0.5f) * fb.w,
                (1.0f - (yc / (zc * t) * 0.5f + 0.5f)) * fb.h, 0 };
      sz[k] = zc;
    }
    if (!ok) continue;

    // 屏幕空间有向面积：只用来判退化；是否可见交给 z-buffer（不做背面剔除，
    // 因为 OBJ 的绕序与屏幕 Y 翻转叠加后方向不确定，剔除会整片剔空 —— 踩过）。
    float area = (sp[1].x - sp[0].x) * (sp[2].y - sp[0].y) - (sp[2].x - sp[0].x) * (sp[1].y - sp[0].y);
    if (std::fabs(area) < 1e-6f) continue;

    int minx = std::max(0, (int)std::floor(std::min({sp[0].x, sp[1].x, sp[2].x})));
    int maxx = std::min(fb.w - 1, (int)std::ceil(std::max({sp[0].x, sp[1].x, sp[2].x})));
    int miny = std::max(0, (int)std::floor(std::min({sp[0].y, sp[1].y, sp[2].y})));
    int maxy = std::min(fb.h - 1, (int)std::ceil(std::max({sp[0].y, sp[1].y, sp[2].y})));
    if (minx > maxx || miny > maxy) continue;

    for (int y = miny; y <= maxy; ++y) {
      for (int x = minx; x <= maxx; ++x) {
        float px = x + 0.5f, py = y + 0.5f;
        float w0 = (sp[1].x - sp[0].x) * (py - sp[0].y) - (px - sp[0].x) * (sp[1].y - sp[0].y);
        float w1 = (sp[2].x - sp[1].x) * (py - sp[1].y) - (px - sp[1].x) * (sp[2].y - sp[1].y);
        float w2 = (sp[0].x - sp[2].x) * (py - sp[2].y) - (px - sp[2].x) * (sp[0].y - sp[2].y);
        // 归一化重心坐标（与 area 同号即为内部）
        float l0 = w0 / area, l1 = w1 / area, l2 = w2 / area;
        if (l0 < 0 || l1 < 0 || l2 < 0) continue;
        // 透视校正深度：1/z 在屏幕空间线性
        float zview = 1.0f / (l0 / sz[0] + l1 / sz[1] + l2 / sz[2]);
        if (zview <= fb.zbuf[y * fb.w + x]) continue;
        // 透视校正的属性插值
        float a0 = l0 / sz[0], a1 = l1 / sz[1], a2 = l2 / sz[2];
        float inv = a0 + a1 + a2;
        float b0 = a0 / inv, b1 = a1 / inv, b2 = a2 / inv;

        V3 n = norm({ wn[0].x * b0 + wn[1].x * b1 + wn[2].x * b2,
                      wn[0].y * b0 + wn[1].y * b1 + wn[2].y * b2,
                      wn[0].z * b0 + wn[1].z * b1 + wn[2].z * b2 });
        V3 nf = dot(n, fwd) > 0 ? n : V3{-n.x, -n.y, -n.z};   // 面向观察者
        float diff = std::max(0.0f, dot(nf, light));
        float rim = std::pow(1.0f - std::max(0.0f, dot(nf, V3{-fwd.x, -fwd.y, -fwd.z})), 3.0f);
        float amb = 0.30f + 0.25f * (nf.y * 0.5f + 0.5f);      // 半球环境光
        float lum = std::min(1.5f, amb + 0.85f * diff + 0.5f * rim);
        // 暖白布料（婚礼婚纱），带一点天光反射
        float cr = lum * (0.95f + 0.15f * nf.y), cg = lum * 0.95f, cb = lum * (0.98f + 0.10f * (1 - nf.y));
        auto gam = [](float c) { c = std::min(1.0f, std::max(0.0f, c)); return (unsigned char)(255.0f * std::pow(c, 1.0f / 2.2f)); };
        fb.zbuf[y * fb.w + x] = zview;
        fb.color[y * fb.w + x] = {gam(cr), gam(cg), gam(cb)};
      }
    }
  }
}

static void write_ppm(const Framebuffer& fb, const char* path) {
  FILE* f = std::fopen(path, "wb");
  std::fprintf(f, "P6\n%d %d\n255\n", fb.w, fb.h);
  for (auto& c : fb.color) std::fwrite(c.data(), 1, 3, f);
  std::fclose(f);
}

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: %s in.obj out.ppm [W] [views]\n", argv[0]); return 1; }
  const char* in = argv[1];
  const char* out = argv[2];
  int W = argc > 3 ? atoi(argv[3]) : 900;
  int views = argc > 4 ? atoi(argv[4]) : 4;

  Mesh m = load_obj(in);
  std::printf("网格: %zu 顶点, %zu 三角面, %zu 法线\n", m.pos.size(), m.tri_pos.size(), m.nrm.size());

  // 求包围盒，把模型居中并缩放到相机合适的尺度
  V3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
  for (auto& p : m.pos) {
    lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
    hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
  }
  V3 c{(lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2};
  float radius = 0.5f * std::sqrt(dot(sub(hi, lo), sub(hi, lo)));
  std::printf("包围盒: [%.2f %.2f %.2f] - [%.2f %.2f %.2f]  中心 y=%.2f 半径 %.2f\n",
              lo.x, lo.y, lo.z, hi.x, hi.y, hi.z, c.y, radius);
  for (auto& p : m.pos) { p.x -= c.x; p.y -= c.y; p.z -= c.z; }

  float dist = radius * 2.15f;
  V3 light = norm(V3{0.4f, 0.85f, 0.35f});

  // 单张主图
  Framebuffer hero(W, W);
  Camera cam{{dist * 0.62f, radius * 0.42f, dist * 0.62f}, {0, radius * 0.05f, 0}, {0, 1, 0}, 0.62f};
  render_view(m, hero, cam, light);
  write_ppm(hero, out);
  std::printf("写出 %s (%dx%d)\n", out, W, W);

  // 转台多视图（横向拼接）
  if (views > 1) {
    int vw = W / views;
    Framebuffer sheet(vw * views, vw);
    for (int v = 0; v < views; ++v) {
      float ang = 2.0f * 3.14159265f * v / views;
      Framebuffer fb(vw, vw);
      Camera c2{{std::cos(ang) * dist, radius * 0.34f, std::sin(ang) * dist}, {0, radius * 0.05f, 0}, {0, 1, 0}, 0.62f};
      render_view(m, fb, c2, light);
      for (int y = 0; y < vw; ++y)
        for (int x = 0; x < vw; ++x)
          sheet.color[y * sheet.w + v * vw + x] = fb.color[y * vw + x];
    }
    std::string sp = std::string(out);
    sp = sp.substr(0, sp.find_last_of('.')) + "_turntable.ppm";
    write_ppm(sheet, sp.c_str());
    std::printf("写出 %s (%dx%d, %d 视角)\n", sp.c_str(), sheet.w, sheet.h, views);
  }
  return 0;
}
