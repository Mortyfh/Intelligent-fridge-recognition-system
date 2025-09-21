# FRIDGE_DEMO

Lightweight, on-device pipeline for smart-fridge perception:

- **Detection** (YOLOv8 ONNX via ONNX Runtime)
- **Tracking** (Kalman filter)
- **Decision** (Put-In / Take-Out) + event logging

The model is **trained/fine-tuned with MMYOLO** and **exported to ONNX**, then run here with a minimal NumPy/OpenCV/ORT stack.

---

## 1) Environment

```bash
pip install -r requirements.txt   # Python 3.9–3.11 recommended
```

> Uses **PyAV** for decoding if available; falls back to **OpenCV VideoCapture**.

---

## 2) Model training & export (MMYOLO)

- Train / transfer-train in **MMYOLO** (configs inherit `_base_` COCO recipe, then adapt to **4 classes**: apple/banana/broccoli/donut).
- Export to **ONNX** using MMYOLO’s export tooling.
- Place the exported file under `model/` (e.g., `model/yolov8_nano_lite_640x640.onnx`).

---

## 3) Project layout

```
FRIDGE_DEMO/
├─ data/                      # input videos
├─ model/                     # exported .onnx
├─ runs/                      # outputs (frames, videos, logs)
└─ src/
   ├─ decode/av_reader.py     # video/RTSP reader (PyAV → OpenCV fallback)
   ├─ infer/onnx_infer.py     # ORT inference (YOLOv8 ONNX)
   ├─ tracking/
   │  ├─ simple_kf.py         # Kalman filter core
   │  └─ tracker_kf.py        # tracker utilities
   └─ runners/
      ├─ run_det.py           # per-frame detection & save
      └─ run_track.py         # tracking + decision + logging
```

---

## 4) Usage

### A) Per-frame detection (`run_det.py`)
Decode a video → save frames → run ONNX on each image → save images with boxes.

**CUDA (one-liner):**
```bash
python -m src.runners.run_det --onnx "model/yolov8_nano_lite_640x640.onnx" --source "data/1.mov" --out-dir "runs/frames" --provider cuda --imgsz 640 --conf 0.25
```

**CPU:**
```bash
python -m src.runners.run_det --onnx "model/yolov8_nano_lite_640x640.onnx" --source "data/1.mov" --out-dir "runs/frames" --provider cpu --imgsz 640 --conf 0.25
```

**Key args**
- `--onnx` path to ONNX model  
- `--source` input video  
- `--out-dir` output folder for frames/overlays  
- `--provider` `auto|cpu|cuda`  
- `--imgsz` input size (e.g., 640)  
- `--conf` confidence threshold

---

### B) Tracking + decision (`run_track.py`)
Single-object Kalman tracking with optional cross-line trigger; outputs an overlay video and **appends** events to a global TXT log.

**CUDA (one-liner):**
```bash
python -m src.runners.run_track --video "data/1.mov" --onnx "model/yolov8_nano_lite_640x640.onnx" --provider cuda --imgsz 640 --conf 0.25 --prefer-class --gate-iou 0.15 --reacq-after 1 --max-miss 30 --event-close-miss 3 --q-pos 1e-2 --q-vel 2e-1 --lookahead --out "runs/out_track.mp4" --log "runs/inout_log.txt"
```

**CPU:**
```bash
python -m src.runners.run_track --video "data/1.mov" --onnx "model/yolov8_nano_lite_640x640.onnx" --provider cpu --imgsz 640 --conf 0.25 --out "runs/out_track.mp4" --log "runs/inout_log.txt"
```

**Useful flags**
- **Association & robustness**:  
  `--gate-iou`, `--prefer-class`, `--reacq-after`, `--max-miss`, `--event-close-miss`
- **Zones / trigger**:  
  `--trigger-on-cross` (instant log on first midline crossing), `--zone-margin` (e.g., `0.05` = ±5%H buffer)
- **Responsiveness / display**:  
  `--q-pos`, `--q-vel`, `--lookahead`, `--target-fps`, `--out-fps`
- **Outputs**:  
  `--out` overlay video (`.mp4`), `--log` append-only TXT with lines like  
  `YYYY-MM-DD HH:MM:SS  <item>  Put-In/Take-Out  (video_t=..s, start_t=..s)`

---

## 5) Notes

- Boxes render **green** with **class name only**.
- The log file is **shared across runs** unless you pass a new `--log` path.
- Works with normal and reversed videos; supports **instant cross-line trigger** or **segment-end** flush.

---
