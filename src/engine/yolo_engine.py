# -*- coding: utf-8 -*-
"""
自研感知推理引擎链路（YOLOv8）：统一 预处理(letterbox) -> 推理后端 -> 后处理(decode+NMS)。

设计要点：
  · 三个后端（torch-FP32 / onnxruntime-FP32 / onnxruntime-INT8）**共用同一套前/后处理**，
    横向比较才公平，也才说得清"引擎链路"与"后端"各自的职责。
  · 前后处理全部自己实现（numpy），不依赖 ultralytics 的 predict 包装 —— 对应
    能力点「实现模型的 inference engine」；与 src/kernels 的 CUDA letterbox/NMS 是同链路两落地。
"""
from __future__ import annotations

import os
import numpy as np
from PIL import Image

IMG_SIZE = 640
CONF_THR = 0.25
IOU_THR = 0.45
NMS_MAX = 300


# ----------------------------- 预处理：letterbox -----------------------------
def letterbox(img: np.ndarray, size: int = IMG_SIZE, pad: int = 114):
    """保持长宽比缩放到 size×size 并居中填充。返回 (nchw_float, gain, (dw,dh), orig_hw)。"""
    h0, w0 = img.shape[:2]
    r = min(size / h0, size / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    resized = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR), dtype=np.float32)
    canvas = np.full((size, size, 3), pad, dtype=np.float32)
    dw, dh = (size - nw) // 2, (size - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    chw = canvas.transpose(2, 0, 1)[None] / 255.0                 # RGB, NCHW, 0-1
    return np.ascontiguousarray(chw, dtype=np.float32), r, (dw, dh), (h0, w0)


# ----------------------------- 后处理：decode + NMS -----------------------------
def xywh2xyxy(x):
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms_np(boxes, scores, iou_thr):
    """按分数贪心抑制（numpy 参考实现；CUDA 版见 src/kernels/perception_ops）。"""
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(1e-9, area_i + area_r - inter)
        order = rest[iou <= iou_thr]
    return np.array(keep, dtype=np.int64)


def postprocess(out: np.ndarray, gain, pad, orig_hw, conf_thr=CONF_THR, iou_thr=IOU_THR):
    """YOLOv8 原始输出 [1,84,8400] -> [(cls, score, x1,y1,x2,y2)...]（已还原到原图坐标）"""
    pred = out[0].T                       # [8400, 84]
    cls_scores = pred[:, 4:]
    cls = cls_scores.argmax(1)
    conf = cls_scores.max(1)
    m = conf > conf_thr
    if not m.any():
        return np.zeros((0, 6), dtype=np.float32)
    boxes = xywh2xyxy(pred[m, :4])
    cls = cls[m]
    conf = conf[m]

    dw, dh = pad
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes /= gain
    h0, w0 = orig_hw
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)

    rows = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        for k in nms_np(boxes[idx], conf[idx], iou_thr):
            j = idx[k]
            rows.append([c, conf[j], boxes[j, 0], boxes[j, 1], boxes[j, 2], boxes[j, 3]])
    if not rows:
        return np.zeros((0, 6), dtype=np.float32)
    rows = np.array(rows, dtype=np.float32)
    return rows[np.argsort(-rows[:, 1])][:NMS_MAX]


# ----------------------------- 推理后端 -----------------------------
class TorchBackend:
    name = "torch-FP32"

    def __init__(self, weights="yolov8n.pt"):
        import torch
        from ultralytics import YOLO
        self.torch = torch
        self.m = YOLO(weights).model.eval().cuda()

    def infer(self, nchw):
        with self.torch.no_grad():
            t = self.torch.from_numpy(nchw).cuda()
            o = self.m(t)
            o = o[0] if isinstance(o, (list, tuple)) else o
            return o.float().cpu().numpy()


class OrtBackend:
    def __init__(self, onnx_path, providers=None, name=None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            onnx_path, so,
            providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        self.name = name or f"onnx-{os.path.basename(onnx_path)}"

    def infer(self, nchw):
        return self.sess.run(None, {self.iname: nchw})[0]


# ----------------------------- 统一执行 -----------------------------
def run_one(backend, img_path, conf_thr=CONF_THR):
    img = np.asarray(Image.open(img_path).convert("RGB"))
    nchw, gain, pad, orig = letterbox(img)
    out = backend.infer(nchw)
    return postprocess(np.asarray(out), gain, pad, orig, conf_thr=conf_thr)
