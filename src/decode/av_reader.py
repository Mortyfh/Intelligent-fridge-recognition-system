# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import Generator, Optional, Tuple

# Try PyAV(FFmpeg) first
HAVE_AV = False
try:
    import av  # type: ignore
    HAVE_AV = True
except Exception:
    pass

import cv2  

def frames_from_source(
    src: str,
    target_fps: Optional[float] = None,
    *,
    rtsp_tcp: bool = True,
    timeout_ms: Optional[int] = None,
) -> Generator[Tuple[np.ndarray, float, int], None, None]:
    """
    Decode any video/RTSP source and yield (frame_bgr, pts_seconds, frame_index) per frame.
    Prefers PyAV; automatically falls back to OpenCV VideoCapture if PyAV is unavailable.
    """
    if HAVE_AV:
        # ---------- PyAV  ----------
        open_opts = {}
        if str(src).lower().startswith('rtsp') and rtsp_tcp:
            open_opts['rtsp_transport'] = 'tcp'
        if timeout_ms is not None:
            open_opts['stimeout'] = str(int(timeout_ms) * 1000)

        container = av.open(src, options=open_opts)
        vstream = next((s for s in container.streams if s.type == 'video'), None)
        if vstream is None:
            raise RuntimeError('No video stream found.')

        try:
            vstream.thread_type = 'AUTO'
        except Exception:
            pass

        avg_rate = None
        try:
            if vstream.average_rate is not None:
                avg_rate = float(vstream.average_rate)
        except Exception:
            avg_rate = None

        next_emit_ts = None
        emit_interval = (1.0 / target_fps) if (target_fps and target_fps > 0) else None

        frame_idx = 0
        logical_idx = 0
        for frame in container.decode(vstream):
            img_bgr = frame.to_ndarray(format='bgr24')
            if frame.pts is not None and vstream.time_base is not None:
                ts = float(frame.pts * vstream.time_base)
            elif frame.time is not None:
                ts = float(frame.time)
            else:
                ts = (frame_idx / avg_rate) if (avg_rate and avg_rate > 0) else (frame_idx / 30.0)

            if emit_interval is not None:
                if next_emit_ts is None:
                    next_emit_ts = ts
                if ts + 1e-9 < next_emit_ts:
                    frame_idx += 1
                    continue
                while next_emit_ts <= ts:
                    next_emit_ts += emit_interval

            yield img_bgr, ts, logical_idx
            logical_idx += 1
            frame_idx += 1

        container.close()
    else:
        # ---------- OpenCV  ----------
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f'OpenCV cannot open: {src}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1e-3:
            fps = 30.0
        dt = 1.0 / fps
        emit_interval = (1.0 / target_fps) if (target_fps and target_fps > 0) else None
        next_emit_ts = None

        idx = 0
        ts = 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if emit_interval is not None:
                if next_emit_ts is None:
                    next_emit_ts = ts
                if ts + 1e-9 < next_emit_ts:
                    ts += dt
                    idx += 1
                    continue
                while next_emit_ts <= ts:
                    next_emit_ts += emit_interval
            yield frame, ts, idx
            ts += dt
            idx += 1
        cap.release()
