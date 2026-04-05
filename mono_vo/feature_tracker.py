"""
Feature detection, matching, and optical-flow tracking for monocular VO.

Two modes (select via FeatureConfig.method):
  'orb'  – ORB detect + BF Hamming match  (fast, good for indoor)
  'sift' – SIFT detect + BF L2 match       (slower, more robust)

Typical call sequence per frame pair:
    tracker = FeatureTracker(FeatureConfig())
    kp0, kp1, pts0, pts1 = tracker.detect_and_match(img0, img1)
    # or use optical flow tracking across a short window:
    pts1_tracked, mask = tracker.track_optical_flow(img0, img1, pts0)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class FeatureConfig:
    method:           str   = "orb"    # "orb" | "sift"
    max_features:     int   = 2000
    ratio_thresh:     float = 0.75     # Lowe ratio test
    min_matches:      int   = 50       # abort frame if fewer survive
    # Optical-flow params (Lucas-Kanade)
    lk_win_size:      int   = 21
    lk_max_level:     int   = 3
    lk_max_iter:      int   = 30
    lk_eps:           float = 0.01
    # FAST grid detector (used to seed optical flow)
    fast_threshold:   int   = 20
    grid_rows:        int   = 4        # divide image into grid for uniform kp spread
    grid_cols:        int   = 4


class FeatureTracker:

    def __init__(self, cfg: FeatureConfig = FeatureConfig()):
        self.cfg = cfg
        self._build_detector()

    def _build_detector(self):
        m = self.cfg.method.lower()
        if m == "orb":
            self._det  = cv2.ORB_create(self.cfg.max_features)
            self._norm = cv2.NORM_HAMMING
        elif m == "sift":
            self._det  = cv2.SIFT_create(self.cfg.max_features)
            self._norm = cv2.NORM_L2
        else:
            raise ValueError(f"Unknown feature method: {self.cfg.method}")

        self._matcher = cv2.BFMatcher(self._norm, crossCheck=False)

    # ── descriptor-based matching ─────────────────────────────────────────

    def detect_and_match(
        self,
        img0: np.ndarray,
        img1: np.ndarray,
        mask0: Optional[np.ndarray] = None,
        mask1: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect features in both images, match with ratio test.

        Returns
        -------
        pts0, pts1 : np.ndarray shape (N, 2)  float32
            Matched pixel coordinates in img0 and img1.
        """
        kp0, des0 = self._det.detectAndCompute(img0, mask0)
        kp1, des1 = self._det.detectAndCompute(img1, mask1)

        if des0 is None or des1 is None or len(des0) < 2 or len(des1) < 2:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

        # kNN match k=2, then Lowe ratio test
        raw = self._matcher.knnMatch(des0, des1, k=2)
        good = []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.cfg.ratio_thresh * n.distance:
                    good.append(m)

        if len(good) < self.cfg.min_matches:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

        pts0 = np.array([kp0[m.queryIdx].pt for m in good], dtype=np.float32)
        pts1 = np.array([kp1[m.trainIdx].pt for m in good], dtype=np.float32)
        return pts0, pts1

    # ── optical-flow tracking ─────────────────────────────────────────────

    def detect_grid(self, img: np.ndarray) -> np.ndarray:
        """
        Detect FAST keypoints on a grid so features are spread across
        the image rather than clustered in one texture-rich region.

        Returns pts : (N, 2) float32
        """
        h, w  = img.shape[:2]
        rh    = h // self.cfg.grid_rows
        rw    = w // self.cfg.grid_cols
        fast  = cv2.FastFeatureDetector_create(self.cfg.fast_threshold)
        pts   = []

        for r in range(self.cfg.grid_rows):
            for c in range(self.cfg.grid_cols):
                y0, y1 = r * rh, (r + 1) * rh
                x0, x1 = c * rw, (c + 1) * rw
                cell   = img[y0:y1, x0:x1]
                kps    = fast.detect(cell, None)
                if not kps:
                    continue
                # keep best N per cell
                kps = sorted(kps, key=lambda k: -k.response)[:10]
                for kp in kps:
                    pts.append([kp.pt[0] + x0, kp.pt[1] + y0])

        if not pts:
            return np.empty((0, 2), np.float32)
        return np.array(pts, dtype=np.float32)

    def track_optical_flow(
        self,
        img0:  np.ndarray,
        img1:  np.ndarray,
        pts0:  np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track pts0 from img0 into img1 using Lucas-Kanade optical flow.

        Returns
        -------
        pts0_good, pts1_good : (N, 2) float32  – inlier correspondences
        """
        if len(pts0) == 0:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

        lk_params = dict(
            winSize  = (self.cfg.lk_win_size, self.cfg.lk_win_size),
            maxLevel = self.cfg.lk_max_level,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        self.cfg.lk_max_iter, self.cfg.lk_eps),
        )

        pts0_in = pts0.reshape(-1, 1, 2)
        pts1, st, _ = cv2.calcOpticalFlowPyrLK(img0, img1, pts0_in, None,
                                                **lk_params)
        # back-track for forward-backward consistency check
        pts0_back, st_back, _ = cv2.calcOpticalFlowPyrLK(img1, img0, pts1, None,
                                                          **lk_params)
        fb_err  = np.linalg.norm(pts0_in - pts0_back, axis=2).squeeze()
        valid   = (st.squeeze() == 1) & (st_back.squeeze() == 1) & (fb_err < 1.0)

        return pts0[valid], pts1.reshape(-1, 2)[valid]