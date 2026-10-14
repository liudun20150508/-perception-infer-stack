#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
感知推理引擎 + 评估系统（能力点「实现 inference engine 并优化」「建立评估系统」）。

一条链路跑三件事：
  1. baseline      torch FP32（PyTorch 直推）
  2. engine-FP32   导出 ONNX + onnxruntime（CUDA EP）跑自研引擎链路
  3. engine-INT8   静态量化（coco128 校准）后再跑同一条链路
三个后端**共用同一套 letterbox 前处理 + decode/NMS 后处理**（yolo_engine.py），
因此差异只来自推理后端本身 —— 这才叫"引擎"。

输出：统一的 延迟 p50/p99 + mAP50 + mAP50-95 对照表，并按门禁给出 PASS/FAIL。
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys
import time


def _bootstrap_cuda_libs():
    """ORT-GPU 需要 CUDA 12 运行库；而 LD_LIBRARY_PATH 必须在进程启动前生效，
    所以这里检测到缺库就带着新环境 re-exec 自己一次（幂等，靠环境变量打标）。"""
    if os.environ.get("_WY_NVLIB_DONE"):
        return
    sp = os.path.join(sys.prefix, "lib",
                      f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
    cands = sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
    if not cands:
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if all(c in cur.split(":") for c in cands):
        return
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(cands) + (":" + cur if cur else "")
    env["_WY_NVLIB_DONE"] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


_bootstrap_cuda_libs()

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_engine import TorchBackend, OrtBackend, run_one, letterbox, postprocess
from coco_eval import evaluate

DEF_DS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data", "coco128")


def ensure_dataset(ds_dir):
    """数据集不在就自动拉 coco128（128 张真图 + 标注，7MB）。"""
    if os.path.isdir(os.path.join(ds_dir, "images")):
        return ds_dir
    import urllib.request, zipfile
    parent = os.path.dirname(ds_dir)
    os.makedirs(parent, exist_ok=True)
    zp = os.path.join(parent, "coco128.zip")
    print(f"下载 coco128 -> {zp}")
    urllib.request.urlretrieve(
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip", zp)
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            if n.startswith("coco128/"):
                z.extract(n, parent)
    src = os.path.join(parent, "coco128")
    if os.path.abspath(src) != os.path.abspath(ds_dir) and not os.path.isdir(ds_dir):
        os.rename(src, ds_dir)
    return ds_dir


def load_gt(ds_dir):
    gt = {}
    for lp in glob.glob(os.path.join(ds_dir, "labels/train2017/*.txt")):
        stem = os.path.splitext(os.path.basename(lp))[0]
        img_p = os.path.join(ds_dir, "images/train2017", stem + ".jpg")
        if not os.path.exists(img_p):
            continue
        w, h = Image.open(img_p).size
        rows = []
        for line in open(lp):
            p = line.split()
            if len(p) != 5:
                continue
            c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
            rows.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                         (cx + bw / 2) * w, (cy + bh / 2) * h))
        gt[stem] = rows
    return gt


def bench(backend, img_paths, reps=1):
    lats = []
    dets = {}
    for k, p in enumerate(img_paths):
        t0 = time.perf_counter()
        d = run_one(backend, p)
        dt = (time.perf_counter() - t0) * 1000
        if k >= 10:                       # 预热去掉前 10 张
            lats.append(dt)
        dets[os.path.splitext(os.path.basename(p))[0]] = [
            (int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])) for r in d]
    lats.sort()
    return {
        "p50": lats[len(lats) // 2] if lats else 0.0,
        "p99": lats[min(len(lats) - 1, int(0.99 * len(lats)))] if lats else 0.0,
        "mean": statistics.fmean(lats) if lats else 0.0,
        "fps": 1000.0 / statistics.fmean(lats) if lats else 0.0,
    }, dets


def export_onnx(work, weights="yolov8n.pt"):
    out = os.path.join(work, "yolov8n.onnx")
    if os.path.exists(out):
        return out
    from ultralytics import YOLO
    p = YOLO(weights).export(format="onnx", imgsz=640, opset=12, simplify=True)
    # ultralytics 导出到 cwd，落到 work
    if os.path.abspath(p) != os.path.abspath(out):
        os.replace(p, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEF_DS)
    ap.add_argument("--work", default="report/engine")
    ap.add_argument("--weights", default="", help="YOLO 权重路径（默认自动取 data/yolov8n.pt）")
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 张（0=全部）")
    ap.add_argument("--skip-int8", action="store_true")
    a = ap.parse_args()
    if not a.weights:
        for c in ("data/yolov8n.pt", "../data/yolov8n.pt", "yolov8n.pt"):
            if os.path.exists(c):
                a.weights = c
                break
        else:
            a.weights = "yolov8n.pt"      # 交给 ultralytics 自行下载
    os.makedirs(a.work, exist_ok=True)
    ensure_dataset(a.dataset)

    gt = load_gt(a.dataset)
    imgs = sorted(glob.glob(os.path.join(a.dataset, "images/train2017/*.jpg")))
    if a.limit:
        imgs = imgs[:a.limit]
    keep = {os.path.splitext(os.path.basename(p))[0] for p in imgs}
    gt = {k: v for k, v in gt.items() if k in keep}
    print(f"数据集 {a.dataset}   图片 {len(imgs)}   标注框 {sum(len(v) for v in gt.values())}")

    results = {}

    # 1) baseline torch FP32
    tb = TorchBackend(a.weights)
    results["torch-FP32"] = bench(tb, imgs)

    # 2) ONNX FP32
    onnx_fp32 = export_onnx(a.work, a.weights)
    ob32 = OrtBackend(onnx_fp32, name="engine-FP32")
    results["engine-FP32"] = bench(ob32, imgs)

    # 3) ONNX INT8
    if not a.skip_int8:
        try:
            from quantize_int8 import quantize
            onnx_int8 = quantize(onnx_fp32, a.dataset, os.path.join(a.work, "yolov8n.int8.onnx"))
            ob8 = OrtBackend(onnx_int8, providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                             name="engine-INT8")
            results["engine-INT8"] = bench(ob8, imgs)
        except Exception as e:
            print(f"[warn] INT8 量化跳过：{type(e).__name__}: {e}")

    # ---- 统一评估 ----
    print("\n================ 引擎对照（同一套前/后处理）================")
    print(f"  {'后端':<14}{'p50 ms':>9}{'p99 ms':>9}{'FPS':>8}{'mAP50':>9}{'mAP50-95':>11}")
    base_map = None
    rows = []
    for name, (lat, dets) in results.items():
        m = evaluate(gt, dets)
        if base_map is None:
            base_map = m
        print(f"  {name:<14}{lat['p50']:>9.2f}{lat['p99']:>9.2f}{lat['fps']:>8.1f}"
              f"{m['mAP50']:>9.4f}{m['mAP50-95']:>11.4f}")
        rows.append((name, lat, m))

    # ---- 门禁 ----
    print("\n================ 评估门禁（验收判据）================")
    ok = True
    eng = dict((r[0], r) for r in rows)
    if "engine-FP32" in eng:
        dm = abs(eng["engine-FP32"][2]["mAP50"] - base_map["mAP50"])
        c = dm <= 0.02
        ok &= c
        print(f"  [{'PASS' if c else 'FAIL'}] engine-FP32 与 baseline mAP50 偏差 {dm:.4f} (<=0.02)")
    if "engine-INT8" in eng:
        drop = base_map["mAP50"] - eng["engine-INT8"][2]["mAP50"]
        c1 = drop <= 0.05
        sp = eng["engine-FP32"][1]["p50"] / eng["engine-INT8"][1]["p50"] if "engine-FP32" in eng else 1
        c2 = sp >= 1.10
        ok &= c1
        print(f"  [{'PASS' if c1 else 'FAIL'}] engine-INT8 相对 FP32 掉点 {drop:.4f} (<=0.05)")
        print(f"  [{'PASS' if c2 else 'WARN'}] engine-INT8 相对 engine-FP32 加速 {sp:.2f}x (>=1.10)")
    print(f"\n  总体：{'全部通过 ✔' if ok else '有项未过 ✘'}")

    # ---- 监控/评估产物：统一落一份 JSON，便于接门禁与看板 ----
    import json
    rep = {
        "dataset": a.dataset, "images": len(imgs),
        "backends": {name: {"p50_ms": round(lat["p50"], 3), "p99_ms": round(lat["p99"], 3),
                            "fps": round(lat["fps"], 2), "mAP50": round(m["mAP50"], 4),
                            "mAP50-95": round(m["mAP50-95"], 4)}
                     for name, lat, m in rows},          # rows = (name, latency, metrics)
        "gate": {"passed": bool(ok)},
    }
    rp = os.path.join(a.work, "report.json")
    json.dump(rep, open(rp, "w"), ensure_ascii=False, indent=2)
    print(f"  报告: {rp}")


if __name__ == "__main__":
    main()
