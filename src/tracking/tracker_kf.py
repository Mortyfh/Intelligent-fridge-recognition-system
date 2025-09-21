# -*- coding: utf-8 -*-
# src/tracking/simple_kf.py
from __future__ import annotations
import numpy as np

class SimpleKF:
    # State [cx, cy, vx, vy]; measurement [cx, cy]; supports non-straight motion via continuous velocity updates
    def __init__(self, q_pos=1e-2, q_vel=1e-1, r_meas=1.0):
        self.x = np.zeros((4,1), float)
        self.P = np.eye(4) * 100.0
        self.H = np.array([[1,0,0,0],[0,1,0,0]], float)
        self.R = np.eye(2) * r_meas
        self.Q_base = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(float)
        self.dt_last = 1/25
        self.wh = None
        self.alpha_wh = 0.8
        self.inited = False

    def _F(self, dt):
        return np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], float)
    def _Q(self, dt):
        return self.Q_base * max(1e-6, dt)

    def init(self, cx, cy, w, h):
        self.x[:] = 0
        self.x[0,0], self.x[1,0] = cx, cy
        self.P = np.eye(4) * 100.0
        self.wh = np.array([w,h], float)
        self.inited = True

    def predict(self, dt):
        if not self.inited: return
        if dt<=0 or dt>1.0: dt = self.dt_last
        F = self._F(dt); Q = self._Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.dt_last = dt

    def update(self, cx, cy, w, h):
        if not self.inited:
            self.init(cx, cy, w, h); return
        z = np.array([[cx],[cy]], float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4); KH = K @ self.H
        self.P = (I-KH) @ self.P @ (I-KH).T + K @ self.R @ K.T
        self.wh = self.alpha_wh*np.array([w,h],float) + (1-self.alpha_wh)*self.wh

    def box_xyxy(self):
        if not self.inited or self.wh is None: return None
        cx, cy = float(self.x[0,0]), float(self.x[1,0])
        w, h = float(self.wh[0]), float(self.wh[1])
        return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2], float)
