# -*- coding: utf-8 -*-
"""
run_det.py
Purpose: Decode a video frame by frame → save frames as images → run ONNX inference on each image → save results with bounding boxes.
Dependencies: onnxruntime, opencv-python, numpy
"""

import os
import cv2
import time
import argparse
import numpy as np
import onnxruntime as ort


# ======= Class/Color =======
CLASSES = [
    'apple', 'banana', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'sandwich',
]
COLORS = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in CLASSES]


# ======= ONNX provider choice =======
def pick_providers(flag: str):
    flag = (flag or 'auto').lower()
    avail = ort.get_available_providers()
    if flag == 'cpu':
        return ['CPUExecutionProvider']
    if flag == 'cuda' and 'CUDAExecutionProvider' in avail:
        return ['CUDAExecutionProvider']
    return ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in avail else ['CPUExecutionProvider']


# ======= Preprocessing & Postprocessing =======
def preprocess(image, input_size=(640, 640)):
    img = cv2.resize(image, input_size)
    img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(outputs, orig_shape, input_shape=(640, 640), conf=0.25):
    preds = outputs[0][0]  
    if preds.ndim == 1:
        preds = preds.reshape(-1, preds.shape[-1])

    boxes, scores, class_ids = [], [], []
    for det in preds:
        if det.shape[0] < 6:  # Prevent shape/dimension mismatches
            continue
        if det[4] < conf:
            continue
        x1, y1, x2, y2 = det[:4]
        x_scale = orig_shape[1] / input_shape[0]
        y_scale = orig_shape[0] / input_shape[1]
        boxes.append([int(x1 * x_scale), int(y1 * y_scale),
                      int(x2 * x_scale), int(y2 * y_scale)])
        scores.append(float(det[4]))
        class_ids.append(int(det[5]))
    return boxes, scores, class_ids


def draw(image, boxes, scores, class_ids):
    for box, score, cid in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        color = COLORS[cid % len(COLORS)]
        name = CLASSES[cid] if 0 <= cid < len(CLASSES) else f"id{cid}"
        label = f"{name} {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return image


# ======= 主流程 =======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, required=True, help="Path to the ONNX model")
    ap.add_argument("--source", type=str, required=True, help="Path to the input video")
    ap.add_argument("--out-dir", type=str, default="runs/frames", help="Output directory for images")
    ap.add_argument("--provider", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Execution provider")
    ap.add_argument("--imgsz", type=int, default=640, help="Input size")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    providers = pick_providers(args.provider)
    session = ort.InferenceSession(args.onnx, providers=providers)
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise FileNotFoundError(f"unable to open video: {args.source}")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t1 = time.time()
        blob = preprocess(frame, (args.imgsz, args.imgsz))
        outputs = session.run(None, {input_name: blob})
        boxes, scores, class_ids = postprocess(outputs, frame.shape, (args.imgsz, args.imgsz), args.conf)
        result = draw(frame, boxes, scores, class_ids)
        out_path = os.path.join(args.out_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(out_path, result)
        frame_idx += 1
        print(f"frame {frame_idx} finish, {(time.time()-t1):.3f}s")
    cap.release()
    print(f"All frames have been saved to {args.out_dir}")


if __name__ == "__main__":
    main()
