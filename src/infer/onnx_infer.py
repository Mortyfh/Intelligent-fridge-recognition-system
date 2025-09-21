# -*- coding: utf-8 -*-
# src/infer/onnx_raw.py
from __future__ import annotations
import numpy as np, cv2, onnxruntime as ort

def _pick_providers(flag: str):
    flag = (flag or 'auto').lower()
    avail = ort.get_available_providers()
    if flag == 'cpu': return ['CPUExecutionProvider']
    if flag == 'cuda' and 'CUDAExecutionProvider' in avail: return ['CUDAExecutionProvider']
    return ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in avail else ['CPUExecutionProvider']

class RawONNXDetector:
    def __init__(self, onnx_path: str, provider: str = "auto", imgsz: int = 640, conf: float = 0.25):
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.session = ort.InferenceSession(onnx_path, providers=_pick_providers(provider))
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, image):
        img = cv2.resize(image, (self.imgsz, self.imgsz))
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def _postprocess(self, outputs, orig_shape):
        preds = outputs[0][0]  # (N,6) 假设
        if preds.ndim == 1:
            preds = preds.reshape(-1, preds.shape[-1])

        H, W = orig_shape[:2]
        x_scale = W / float(self.imgsz)
        y_scale = H / float(self.imgsz)

        boxes, scores, class_ids = [], [], []
        for det in preds:
            if det.shape[0] < 6:
                continue
            if det[4] < self.conf:
                continue
            x1, y1, x2, y2 = det[:4]
            boxes.append([int(x1 * x_scale), int(y1 * y_scale), int(x2 * x_scale), int(y2 * y_scale)])
            scores.append(float(det[4]))
            class_ids.append(int(det[5]))
        return (np.array(boxes, dtype=int),
                np.array(scores, dtype=float),
                np.array(class_ids, dtype=int))

    def __call__(self, image):
        blob = self._preprocess(image)
        outputs = self.session.run(None, {self.input_name: blob})
        return self._postprocess(outputs, image.shape)
