import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class FeatureConfig:
    method:           str   = "orb"
    max_features:     int   = 2000
    ratio_thresh:     float = 0.75
    use_cross_check:  bool  = False
    min_matches:      int   = 50
    lk_win_size:      int   = 21
    lk_max_level:     int   = 3
    lk_max_iter:      int   = 30
    lk_eps:           float = 0.01
    fast_threshold:   int   = 20
    grid_rows:        int   = 4
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

    def detect_and_match(
        self,
        img0: np.ndarray,
        img1: np.ndarray,
        mask0: Optional[np.ndarray] = None,
        mask1: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect + match with Lowe ratio test. Returns (pts0, pts1) float32."""
        kp0, des0 = self._det.detectAndCompute(img0, mask0)
        kp1, des1 = self._det.detectAndCompute(img1, mask1)

        if des0 is None or des1 is None or len(des0) < 2 or len(des1) < 2:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

        raw = self._matcher.knnMatch(des0, des1, k=2)
        good = [pair[0] for pair in raw
                if len(pair) == 2
                and pair[0].distance < self.cfg.ratio_thresh * pair[1].distance]

        if self.cfg.use_cross_check and good:
            raw_rev = self._matcher.knnMatch(des1, des0, k=2)
            rev_pairs = {(pair[0].queryIdx, pair[0].trainIdx)
                         for pair in raw_rev
                         if len(pair) == 2
                         and pair[0].distance < self.cfg.ratio_thresh * pair[1].distance}
            good = [m for m in good if (m.trainIdx, m.queryIdx) in rev_pairs]

        if len(good) < self.cfg.min_matches:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

        pts0 = np.array([kp0[m.queryIdx].pt for m in good], dtype=np.float32)
        pts1 = np.array([kp1[m.trainIdx].pt for m in good], dtype=np.float32)
        return pts0, pts1

    def detect_grid(self, img: np.ndarray) -> np.ndarray:
        """Detect FAST keypoints on a uniform grid for even spatial coverage."""
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
                kps = sorted(kps, key=lambda k: -k.response)[:10]  # best per cell
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
        """LK optical flow with forward-backward consistency check."""
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
        pts0_back, st_back, _ = cv2.calcOpticalFlowPyrLK(img1, img0, pts1, None,
                                                          **lk_params)
        fb_err  = np.linalg.norm(pts0_in - pts0_back, axis=2).squeeze()
        valid   = (st.squeeze() == 1) & (st_back.squeeze() == 1) & (fb_err < 1.0)

        return pts0[valid], pts1.reshape(-1, 2)[valid]