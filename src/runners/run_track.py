# -*- coding: utf-8 -*-
# src/runners/run_track_kf_raw.py
from __future__ import annotations
import argparse, os, cv2, numpy as np
from typing import Optional, Tuple, Dict
from datetime import datetime

from src.decode.av_reader import frames_from_source
from infer.onnx_infer import RawONNXDetector
from tracking.tracker_kf import SimpleKF

# Model class names (edit/extend as needed)
CLASSES = ['apple', 'banana', 'broccoli', 'donut']
GREEN = (0, 255, 0)  

# ---------- Utilities ----------
def ensure_uint8_bgr(img):
    if img.dtype != np.uint8: img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def iou(a,b):
    xx1,yy1 = max(a[0],b[0]), max(a[1],b[1])
    xx2,yy2 = min(a[2],b[2]), min(a[3],b[3])
    w,h = max(0,xx2-xx1), max(0,yy2-yy1)
    inter = w*h
    A = max(0,a[2]-a[0])*max(0,a[3]-a[1])
    B = max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter / (A+B-inter+1e-6)

def pick_candidate(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
    prev_box: Optional[np.ndarray], last_cls: Optional[int],
    gate_iou: float, prefer_class: bool, reacq: bool
) -> Optional[Tuple[np.ndarray, float, int]]:
    """Return (box, score, cls) or None"""
    if boxes is None or len(boxes) == 0:
        return None
    idx_all = np.arange(len(boxes))
    # 1) Prefer same class
    if prefer_class and last_cls is not None and np.any(labels == last_cls):
        idx_pool = idx_all[labels == last_cls]
    else:
        idx_pool = idx_all
    # 2) IoU gating
    if prev_box is not None and len(idx_pool) > 0:
        pool_boxes = boxes[idx_pool]
        ious = np.array([iou(b, prev_box) for b in pool_boxes], dtype=float)
        j = int(np.argmax(ious))
        di = idx_pool[j]
        if ious[j] >= gate_iou:
            return boxes[di], float(scores[di]), int(labels[di])
        if reacq:
            # 3) Reacquire: highest score within same class, otherwise global highest
            if prefer_class and last_cls is not None and np.any(labels == last_cls):
                sub = idx_all[labels == last_cls]
                di = int(sub[np.argmax(scores[sub])])
            else:
                di = int(np.argmax(scores))
            return boxes[di], float(scores[di]), int(labels[di])
        return None
    # 3) No previous box: pick the highest score in the pool
    di = int(idx_pool[np.argmax(scores[idx_pool])])
    return boxes[di], float(scores[di]), int(labels[di])

def one_step_ahead_box(kf: SimpleKF, dt: float) -> Optional[np.ndarray]:
    if not kf.inited or kf.wh is None: return None
    cx = float(kf.x[0,0]) + float(kf.x[2,0]) * dt
    cy = float(kf.x[1,0]) + float(kf.x[3,0]) * dt
    w, h = float(kf.wh[0]), float(kf.wh[1])
    return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2], float)

def draw(im, box_xyxy, idx, cls_name):
    vis = im.copy()
    cv2.circle(vis, (20,20), 8, GREEN, -1)
    cv2.putText(vis, f"F:{idx}", (40,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
    if box_xyxy is not None:
        x1,y1,x2,y2 = [int(v) for v in box_xyxy]
        cv2.rectangle(vis, (x1,y1), (x2,y2), GREEN, 2)
        if cls_name:
            cv2.putText(vis, f"{cls_name}", (x1, max(0,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
    return vis

# ---------- Zones / Events ----------
def zone_of_center(box_xyxy: np.ndarray, H: int, margin_frac: float) -> Optional[str]:
    """
    Split by the box center into top/bottom, with a middle buffer zone (mid).
    mmargin_frac=0.05 means a ±5% of image height buffer around the midline.
    """
    if box_xyxy is None: return None
    cy = 0.5 * (box_xyxy[1] + box_xyxy[3])
    margin = float(margin_frac) * H
    mid_low  = H * 0.5 - margin
    mid_high = H * 0.5 + margin
    if cy < mid_low: return "top"
    if cy > mid_high: return "bottom"
    return "mid"

def append_log(log_path: str, line: str):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")

def decide_action(start_zone: Optional[str], end_zone: Optional[str]) -> Optional[str]:
    if start_zone == "bottom" and end_zone == "top":   return "takein"
    if start_zone == "top"    and end_zone == "bottom":return "takeout"
    return None

def main():
    ap = argparse.ArgumentParser("Single-object KF tracker (fixed color + events + cross-trigger)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", default="runs/out_track_kf_raw.mp4")
    ap.add_argument("--log", default="runs/inout_log.txt", help="Event log TXT (append-only, shared across runs)")
    ap.add_argument("--provider", default="cuda", choices=["auto","cpu","cuda"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)

    # Association & anti-drop
    ap.add_argument("--gate-iou", type=float, default=0.20, help="IoU gating threshold")
    ap.add_argument("--prefer-class", action="store_true", help="Prefer same class when associating")
    ap.add_argument("--max-miss", type=int, default=15, help="Force reset after this many consecutive misses")
    ap.add_argument("--reacq-after", type=int, default=3, help="Allow re-acquisition after this many consecutive misses")
    ap.add_argument("--event-close-miss", type=int, default=3, help="Close a segment after this many misses (for end-of-segment decision)")

    # Zones / cross-line trigger
    ap.add_argument("--trigger-on-cross", action="store_true",
                help="Enable: log an event immediately on the first midline crossing")
    ap.add_argument("--zone-margin", type=float, default=0.05,
                help="Top/bottom buffer as a fraction of image height (default 5%)")

    # Responsiveness / display
    ap.add_argument("--q-pos", type=float, default=1e-2)
    ap.add_argument("--q-vel", type=float, default=2e-1)
    ap.add_argument("--lookahead", action="store_true")
    ap.add_argument("--target-fps", type=float, default=None)
    ap.add_argument("--out-fps", type=float, default=None)
    args = ap.parse_args()

    det = RawONNXDetector(args.onnx, provider=args.provider, imgsz=args.imgsz, conf=args.conf)
    kf  = SimpleKF(q_pos=args.q_pos, q_vel=args.q_vel, r_meas=1.0)
    base_Q = kf.Q_base.copy()

    gen = frames_from_source(args.video, target_fps=args.target_fps)
    first, ts0, idx0 = next(gen)
    first = ensure_uint8_bgr(first)
    H, W = first.shape[:2]

    out_fps = args.out_fps if (args.out_fps and args.out_fps>0) else 25.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W,H))
    if not writer.isOpened(): raise RuntimeError(f"Unable to open the video writer: {args.out}")

    # Event state (single-object)
    last_ts = ts0 if (ts0 and ts0>0) else None
    prev_est = None
    last_cls = None
    missed   = 0

    # Segment info (majority vote + start/last zone)
    segment_active: bool = False
    seg_start_zone: Optional[str] = None
    seg_last_zone:  Optional[str] = None
    seg_start_ts:   Optional[float] = None
    seg_class_counts: Dict[int, float] = {} 

    def add_class_vote(cid: Optional[int], weight: float):
        nonlocal seg_class_counts
        if cid is None: return
        if not np.isfinite(weight): weight = 1.0
        seg_class_counts[cid] = seg_class_counts.get(cid, 0.0) + max(1e-3, float(weight))

    def majority_class_name() -> str:
        if len(seg_class_counts) == 0:
            return "-"
        best_cid = max(seg_class_counts.items(), key=lambda kv: kv[1])[0]
        return CLASSES[best_cid] if 0 <= best_cid < len(CLASSES) else "-"

    def log_event(now_ts: float, action: str):
        item = majority_class_name()
        wall = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        t_video = f"{now_ts:.2f}s" if now_ts is not None else "-"
        line = f"[{wall}] {item} {action} (video_t={t_video}, start_t={seg_start_ts:.2f}s)"
        append_log(args.log, line)

    def reset_segment():
        nonlocal segment_active, seg_start_zone, seg_last_zone, seg_start_ts, seg_class_counts
        segment_active = False
        seg_start_zone = None
        seg_last_zone  = None
        seg_start_ts   = None
        seg_class_counts = {}

    def maybe_close_segment(now_ts: float):
        nonlocal segment_active, seg_start_zone, seg_last_zone
        if not segment_active or seg_start_zone is None or seg_last_zone is None:
            return
        action = decide_action(seg_start_zone, seg_last_zone)
        if action is not None:
            log_event(now_ts, action)
        reset_segment()

    append_log(args.log, f"=== START video='{args.video}' at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

    boxes, scores, labels = det(first)
    if len(boxes) > 0:
        sel = pick_candidate(boxes, scores, labels, None, None, args.gate_iou, args.prefer_class, reacq=True)
        if sel is not None:
            b, sc, cid = sel
            cx, cy = (b[0]+b[2])/2.0, (b[1]+b[3])/2.0
            w,  h  = (b[2]-b[0]), (b[3]-b[1])
            kf.init(cx, cy, w, h)
            last_cls = cid
            missed = 0
            kf.Q_base = base_Q
            prev_est = kf.box_xyxy()

            # 段开始（若在 top/bottom）
            z0 = zone_of_center(prev_est, H, args.zone_margin)
            if z0 in ("top", "bottom"):
                segment_active = True
                seg_start_zone = z0
                seg_last_zone  = z0
                seg_start_ts   = ts0 if ts0 is not None else 0.0
                seg_class_counts = {}
                add_class_vote(last_cls, float(sc) if 'sc' in locals() else 1.0)

    est = prev_est
    est_draw = one_step_ahead_box(kf, dt=1.0/25.0) if (args.lookahead and est is not None) else est
    writer.write(draw(first, est_draw, idx0, (CLASSES[last_cls] if last_cls is not None and 0 <= last_cls < len(CLASSES) else None)))

    for frame, ts, idx in gen:
        frame = ensure_uint8_bgr(frame)
        # dt
        dt = None
        if last_ts is not None and ts and ts>0:
            dt = max(1e-3, float(ts - last_ts))
        last_ts = ts if (ts and ts>0) else last_ts

        if kf.inited:
            kf.predict(dt if dt else 1/25)

        boxes, scores, labels = det(frame)
        sel = pick_candidate(
            boxes, scores, labels,
            prev_est, last_cls,
            gate_iou=args.gate_iou,
            prefer_class=args.prefer_class,
            reacq=(missed >= args.reacq_after)
        )

        updated_this_frame = False
        if sel is not None:
            b, sc, cid = sel
            cx, cy = (b[0]+b[2])/2.0, (b[1]+b[3])/2.0
            w,  h  = (b[2]-b[0]), (b[3]-b[1])
            kf.update(cx, cy, w, h)
            last_cls = cid
            missed = 0
            kf.Q_base = base_Q
            updated_this_frame = True

            # 段处理
            cur_zone = zone_of_center(kf.box_xyxy(), H, args.zone_margin)
            add_class_vote(last_cls, float(sc))

            if not segment_active:
                if cur_zone in ("top", "bottom"):
                    segment_active = True
                    seg_start_zone = cur_zone
                    seg_last_zone  = cur_zone
                    seg_start_ts   = ts if ts is not None else 0.0
            else:
                if cur_zone in ("top", "bottom"):
                    if args.trigger_on_cross and seg_last_zone in ("top","bottom") and cur_zone != seg_last_zone:
                        action = decide_action(seg_last_zone, cur_zone)  
                        if action is not None:
                            log_event(ts if ts is not None else 0.0, action)
                        seg_start_zone = cur_zone
                        seg_last_zone  = cur_zone
                        seg_start_ts   = ts if ts is not None else 0.0
                        seg_class_counts = {}
                        add_class_vote(last_cls, float(sc))
                    else:
                        seg_last_zone = cur_zone

        else:
            missed = min(missed + 1, 9999)
            scale = 1.0 + 0.6 * min(missed, 5) 
            kf.Q_base = base_Q * scale
            if len(boxes) > 0 and missed >= args.max_miss:
                sel2 = pick_candidate(boxes, scores, labels, None, last_cls, args.gate_iou, args.prefer_class, reacq=True)
                if sel2 is not None:
                    b2, sc2, cid2 = sel2
                    cx, cy = (b2[0]+b2[2])/2.0, (b2[1]+b2[3])/2.0
                    w,  h  = (b2[2]-b2[0]), (b2[3]-b2[1])
                    kf.init(cx, cy, w, h)
                    last_cls = cid2
                    missed = 0
                    kf.Q_base = base_Q
                    updated_this_frame = True
                    cur_zone = zone_of_center(kf.box_xyxy(), H, args.zone_margin)
                    if cur_zone in ("top","bottom"):
                        segment_active = True
                        seg_start_zone = cur_zone
                        seg_last_zone  = cur_zone
                        seg_start_ts   = ts if ts is not None else 0.0
                        seg_class_counts = {}
                        add_class_vote(last_cls, float(sc2))

        prev_est = kf.box_xyxy()

        if not args.trigger_on_cross:
            if segment_active and not updated_this_frame and missed >= args.event_close_miss:
                if prev_est is not None:
                    seg_last_zone = zone_of_center(prev_est, H, args.zone_margin)
                maybe_close_segment(ts if ts is not None else 0.0)

        est = prev_est
        est_draw = one_step_ahead_box(kf, dt=(dt if dt else 1.0/25.0)) if (args.lookahead and est is not None) else est
        cls_name = (CLASSES[last_cls] if last_cls is not None and 0 <= last_cls < len(CLASSES) else None)
        writer.write(draw(frame, est_draw, idx, cls_name))

    if segment_active:
        if prev_est is not None:
            seg_last_zone = zone_of_center(prev_est, H, args.zone_margin)
        final_ts = last_ts if last_ts is not None else 0.0
        if args.trigger_on_cross:
            action = decide_action(seg_start_zone, seg_last_zone)
            if action is not None:
                log_event(final_ts, action)
        else:
            maybe_close_segment(final_ts)

    writer.release()
    append_log(args.log, f"=== END   video='{args.video}' at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"Output video: {args.out}\nEvent log appended to: {args.log}")

if __name__ == "__main__":
    main()
