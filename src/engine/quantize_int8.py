# -*- coding: utf-8 -*-
"""INT8 静态量化（ORT），用 coco128 做校准 —— 能力点「优化模型/降低资源使用率」。"""
from __future__ import annotations

import glob
import os
import numpy as np
from PIL import Image
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
from onnxruntime.quantization.shape_inference import quant_pre_process


class Reader(CalibrationDataReader):
    def __init__(self, ds_dir, n=64):
        self.items = []
        for p in sorted(glob.glob(os.path.join(ds_dir, "images/train2017/*.jpg")))[:n]:
            img = np.asarray(Image.open(p).convert("RGB"))
            h0, w0 = img.shape[:2]
            r = min(640 / h0, 640 / w0)
            nh, nw = int(round(h0 * r)), int(round(w0 * r))
            rs = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR), dtype=np.float32)
            c = np.full((640, 640, 3), 114, dtype=np.float32)
            dw, dh = (640 - nw) // 2, (640 - nh) // 2
            c[dh:dh + nh, dw:dw + nw] = rs
            self.items.append(np.ascontiguousarray(
                c.transpose(2, 0, 1)[None] / 255.0, dtype=np.float32))
        self.i = 0
        self.name = "images"

    def get_next(self):
        if self.i >= len(self.items):
            return None
        x = self.items[self.i]
        self.i += 1
        return {self.name: x}

    def rewind(self):
        self.i = 0


def quantize(fp32_path, ds_dir, out_path):
    prep = out_path.replace(".onnx", ".prep.onnx")
    try:
        quant_pre_process(fp32_path, prep, skip_symbolic_shape=True)
    except Exception:
        prep = fp32_path
    # 关键：只量化 Conv。整图量化（含 Concat/Sigmoid 等）会把分类头打崩 ——
    # 实测 max class score 从 0.x 直接变 0.0，mAP 归零。见 README「负结果」。
    quantize_static(
        prep, out_path, Reader(ds_dir),
        quant_format=QuantFormat.QDQ, per_channel=False,
        weight_type=QuantType.QUInt8, activation_type=QuantType.QUInt8,
        op_types_to_quantize=["Conv"],
   )
    return out_path
