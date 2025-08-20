# -*- coding: utf-8 -*-
"""
YOLOv8 ONNX Runtime 推理（适配常见导出）
- 预处理：letterbox 到固定尺寸、BGR->NCHW、[0,1] 归一化
- 推理：ONNXRuntime (CUDA / CPU)
- 后处理：支持两类常见输出
    (A) 原始预测 (N, 5+K) 或 (N, 4+K)  [xywh(+obj)+cls_logits]
    (B) 已经NMS的 (N, 6)                 [x1,y1,x2,y2,score,cls]
- NMS：Python IoU-NMS（若模型已NMS则跳过）
- 坐标：自动反 letterbox 映射回原图尺寸

用法（单帧）：
    det = YOLOv8ONNX('model/xxx.onnx', img_size=(640,640), class_names=[...])
    boxes, scores, labels = det(frame_bgr)  # xyxy, conf, cls_id
"""

from __future__ import annotations
from typing import Tuple, List
import numpy as np
import onnxruntime as ort
import cv2


# ---------------------
# 工具函数
# ---------------------
def letterbox(img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    """将图像按比例缩放后，在两侧填充到 new_shape，返回 (padded_img, ratio, (pad_w_left, pad_h_top))"""
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)  # 缩放比
    nh, nw = int(round(h * r)), int(round(w * r))
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = new_shape[1] - nw, new_shape[0] - nh
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    out = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                             borderType=cv2.BORDER_CONSTANT, value=color)
    return out, r, (left, top)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.65) -> np.ndarray:
    """标准 IoU-NMS，返回保留的索引（降序按分数）"""
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int32)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int32)


# ---------------------
# 推理封装
# ---------------------
class YOLOv8ONNX:
    def __init__(self,
                 onnx_path: str,
                 img_size: Tuple[int, int] = (640, 640),
                 class_names: List[str] | None = None,
                 score_thr: float = 0.25,
                 iou_thr: float = 0.65,
                 providers: List[str] | None = None):
        """
        :param onnx_path: ONNX 模型路径（例如 fridge_demo/model/xxx.onnx）
        :param img_size:  网络输入尺寸 (H, W)
        :param class_names: 类名列表（用于取 K=类数）
        :param score_thr:  置信度阈值
        :param iou_thr:    NMS 阈值
        :param providers:  ORT providers，默认 ['CUDAExecutionProvider','CPUExecutionProvider']
        """
        self.onnx_path = onnx_path
        self.img_size = tuple(img_size)
        self.class_names = class_names or []
        self.num_classes = len(self.class_names) if self.class_names else None
        self.score_thr = float(score_thr)
        self.iou_thr = float(iou_thr)

        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(self.onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.out_names = [o.name for o in self.session.get_outputs()]

    # -------- 预处理 / 后处理 --------
    def _preprocess(self, img_bgr: np.ndarray):
        lb, ratio, pad = letterbox(img_bgr, self.img_size)
        x = lb.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None, ...]  # NCHW
        return x, ratio, pad, img_bgr.shape[:2]

    def _postprocess(self, out0: np.ndarray, ratio: float, pad: Tuple[int, int], orig_shape: Tuple[int, int]):
        """
        将模型输出转为 (boxes_xyxy, scores, labels)，并映射回原图坐标。
        兼容：
          - 原始输出 (N, 5+K) or (N, 4+K)（xywh(+obj)+cls_logits）
          - 已NMS输出 (N, 6): [x1,y1,x2,y2,score,cls]
        """
        pred = np.array(out0)

        # 若是 (1,N,no) -> (N,no)； 若是 (no,N) -> (N,no)
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred[0]
        if pred.shape[0] in (4 + (self.num_classes or 0), 5 + (self.num_classes or 0)):
            pred = pred.T

        # 已经是 NMS 结果： [x1,y1,x2,y2,score,cls]
        if pred.ndim == 2 and pred.shape[1] == 6:
            boxes = pred[:, :4].astype(np.float32)
            scores = pred[:, 4].astype(np.float32)
            labels = pred[:, 5].astype(np.int32)
            return boxes, scores, labels

        # 原始预测：xywh + cls(logits) [+ obj]
        no = pred.shape[1]
        if self.num_classes is None:
            # 无 class_names 时尝试推断 K（支持有/无 obj 两个分支）
            # 优先尝试无 obj：no-4
            k_noobj = no - 4
            k_obj = no - 5
            if k_noobj > 0:
                self.num_classes = k_noobj
            elif k_obj > 0:
                self.num_classes = k_obj
            else:
                raise RuntimeError(f'Cannot infer num_classes from output shape: {pred.shape}')

        has_obj = (no == 5 + self.num_classes)

        xywh = pred[:, :4].astype(np.float32)
        if has_obj:
            obj = pred[:, 4:5].astype(np.float32)
            cls = pred[:, 5:].astype(np.float32)
        else:
            obj = None
            cls = pred[:, 4:].astype(np.float32)

        # 安全 sigmoid（有些导出已 sigmoid，不会出错）
        cls_p = sigmoid(cls)
        if obj is not None:
            obj_p = sigmoid(obj)
            conf_mat = cls_p * obj_p  # (N,K)
        else:
            conf_mat = cls_p

        labels = conf_mat.argmax(1).astype(np.int32)
        scores = conf_mat.max(1).astype(np.float32)

        # 置信度阈值
        keep = scores >= self.score_thr
        if not np.any(keep):
            return (np.zeros((0, 4), np.float32),
                    np.zeros((0,), np.float32),
                    np.zeros((0,), np.int32))

        xywh = xywh[keep]
        scores = scores[keep]
        labels = labels[keep]

        # xywh -> xyxy（网络输入尺度）
        x, y, w, h = xywh.T
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        # 反 letterbox：去 padding，再除以 ratio
        left, top = pad
        boxes[:, [0, 2]] -= left
        boxes[:, [1, 3]] -= top
        boxes /= (ratio + 1e-12)

        # 裁剪到原图
        H, W = orig_shape
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, W - 1)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, H - 1)

        # NMS
        keep_idx = nms(boxes, scores, self.iou_thr)
        return boxes[keep_idx], scores[keep_idx], labels[keep_idx]

    # -------- 对外接口 --------
    def __call__(self, img_bgr: np.ndarray):
        """对单帧做推理，返回 (boxes_xyxy[N,4], scores[N], labels[N])"""
        x, ratio, pad, orig_shape = self._preprocess(img_bgr)
        out_list = self.session.run(self.out_names, {self.input_name: x})

        # 取首个输出（大多数导出只有一个主输出）
        out0 = out_list[0]
        boxes, scores, labels = self._postprocess(out0, ratio, pad, orig_shape)
        return boxes, scores, labels

    def infer_batch(self, imgs_bgr: List[np.ndarray]):
        """简单的批量接口（会统一 letterbox 尺寸并拼 batch）"""
        x_list, metas = [], []
        for im in imgs_bgr:
            x, r, pad, orig = self._preprocess(im)
            x_list.append(x)
            metas.append((r, pad, orig))
        x_batch = np.concatenate(x_list, axis=0)  # (B,3,H,W)
        out_list = self.session.run(self.out_names, {self.input_name: x_batch})
        out0 = out_list[0]
        # 若模型不支持 batch，这里需要遍历；多数导出支持 (B, N, no)
        outs = []
        if out0.ndim == 3 and out0.shape[0] == len(imgs_bgr):
            for i in range(len(imgs_bgr)):
                boxes, scores, labels = self._postprocess(out0[i], metas[i][0], metas[i][1], metas[i][2])
                outs.append((boxes, scores, labels))
        else:
            # 不支持 batch，回退单帧
            outs = [self(img) for img in imgs_bgr]
        return outs
