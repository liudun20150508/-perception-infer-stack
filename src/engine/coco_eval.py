# -*- coding: utf-8 -*-
"""
自研 COCO 风格 mAP 评估器（101 点插值）—— 能力点「建立精确稳定高效的评估系统」。

简化说明：不含 crowd / area-range / ignore 标记（coco128 也没有这些字段）；
AP@0.5:0.95 取 10 个 IoU 阈值平均，与官方口径同构。
"""
from __future__ import annotations

import numpy as np

IOU_THRS = np.arange(0.5, 1.0, 0.05)


def _iou_mat(gt, dt):
    if len(gt) == 0 or len(dt) == 0:
        return np.zeros((len(dt), len(gt)), dtype=np.float32)
    d = dt[:, None, :]
    g = gt[None, :, :]
    x1 = np.maximum(d[..., 0], g[..., 0]); y1 = np.maximum(d[..., 1], g[..., 1])
    x2 = np.minimum(d[..., 2], g[..., 2]); y2 = np.minimum(d[..., 3], g[..., 3])
    w = np.maximum(0, x2 - x1); h = np.maximum(0, y2 - y1)
    inter = w * h
    ad = (d[..., 2] - d[..., 0]) * (d[..., 3] - d[..., 1])
    ag = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
    return inter / np.maximum(1e-9, ad + ag - inter)


def _voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    grid = np.linspace(0, 1, 101)
    return float(np.mean(np.interp(grid, mrec, mpre)))


def evaluate(gt_by_img, dt_by_img, n_classes=80):
    """gt_by_img: {img: [(cls,x1,y1,x2,y2)...]}; dt_by_img: {img: [(cls,score,x1,y1,x2,y2)...]}"""
    img_ids = sorted(gt_by_img.keys())
    aps = {float(t): [] for t in IOU_THRS}
    for thr in IOU_THRS:
        thr = float(thr)
        for c in range(n_classes):
            dets = []
            n_gt = 0
            gts = {}
            for iid in img_ids:
                g = np.array([b for (gc, *b) in gt_by_img[iid] if gc == c], dtype=np.float32)
                gts[iid] = g
                n_gt += len(g)
                for (dc, score, *b) in dt_by_img.get(iid, []):
                    if dc == c:
                        dets.append((iid, float(score), np.array(b, dtype=np.float32)))
            if n_gt == 0:
                continue
            if not dets:
                aps[thr].append(0.0)
                continue
            dets.sort(key=lambda x: -x[1])
            used = {iid: np.zeros(len(gts[iid]), dtype=bool) for iid in img_ids}
            tp = np.zeros(len(dets)); fp = np.zeros(len(dets))
            for i, (iid, score, box) in enumerate(dets):
                g = gts[iid]
                if len(g) == 0:
                    fp[i] = 1
                    continue
                ious = _iou_mat(g, box[None, :])[0]
                j = int(np.argmax(ious))
                if ious[j] >= thr and not used[iid][j]:
                    tp[i] = 1; used[iid][j] = True
                else:
                    fp[i] = 1
            ctp = np.cumsum(tp); cfp = np.cumsum(fp)
            rec = ctp / n_gt
            prec = ctp / np.maximum(1e-9, ctp + cfp)
            aps[thr].append(_voc_ap(rec, prec))
    ap50 = float(np.mean(aps[0.5])) if aps[0.5] else 0.0
    all_ap = [v for t in aps for v in aps[t]]
    ap5095 = float(np.mean(all_ap)) if all_ap else 0.0
    return {"mAP50": ap50, "mAP50-95": ap5095}
