"""
Disparity computation and 3-D reconstruction for stereo VO.
Tuned for TUM VI 512×512 global-shutter stereo cameras.

Z = f*B / d   (metric depth from disparity)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from data.data_loader import StereoPair


@dataclass
class DisparityConfig:
    method:          str   = "sgbm"
    num_disparities: int   = 64      # max disparity range (must be /16)
    block_size:      int   = 11      # larger = smoother but less detail
    p1_coeff:        int   = 8
    p2_coeff:        int   = 32
    disp12_diff:     int   = 1
    uniqueness:      int   = 15      # higher = fewer false matches
    speckle_window:  int   = 200     # larger = more noise removed
    speckle_range:   int   = 2
    pre_filter_cap:  int   = 63
    mode:            int   = cv2.STEREO_SGBM_MODE_SGBM_3WAY
    # Depth filtering — conservative for TUM VI
    min_depth:       float = 0.3     # TUM VI room: nothing closer than 30cm
    max_depth:       float = 6.0     # room2 is ~6m diameter
    min_disparity:   float = 2.0     # reject near-zero disparity
    # Patch size for robust depth at keypoints
    patch_radius:    int   = 2       # median over (2r+1)^2 patch


class DisparityComputer:

    def __init__(self, calib: StereoPair,
                 cfg: DisparityConfig = DisparityConfig()):
        self.calib = calib
        self.cfg   = cfg
        self._build_matcher()

        # Rectified intrinsics from P_left  [fx 0 cx 0 / 0 fy cy 0 / 0 0 1 0]
        self.f  = float(calib.P_left[0, 0])
        self.cx = float(calib.P_left[0, 2])
        self.cy = float(calib.P_left[1, 2])
        self.B  = calib.baseline          # metres

        print(f"[Disparity] f={self.f:.2f}  B={self.B*100:.2f}cm  "
              f"fB={self.f*self.B:.4f}")

    def _build_matcher(self) -> None:
        c  = self.cfg
        bs = c.block_size
        self._matcher = cv2.StereoSGBM_create(
            minDisparity      = 0,
            numDisparities    = c.num_disparities,
            blockSize         = bs,
            P1                = c.p1_coeff  * bs * bs,
            P2                = c.p2_coeff  * bs * bs,
            disp12MaxDiff     = c.disp12_diff,
            uniquenessRatio   = c.uniqueness,
            speckleWindowSize = c.speckle_window,
            speckleRange      = c.speckle_range,
            preFilterCap      = c.pre_filter_cap,
            mode              = c.mode,
        )
        # WLS filter for smoother disparity (optional — uses right matcher)
        self._wls     = None
        self._right_m = None
        try:
            self._right_m = cv2.ximgproc.createRightMatcher(self._matcher)
            self._wls     = cv2.ximgproc.createDisparityWLSFilter(
                self._matcher)
            self._wls.setLambda(8000)
            self._wls.setSigmaColor(1.5)
            self._use_wls = True
        except AttributeError:
            self._use_wls = False

    def compute(
        self,
        img_left:  np.ndarray,
        img_right: np.ndarray,
        rectified: bool = False,
    ) -> np.ndarray:
        """
        Returns float32 disparity map (pixels). Invalid = 0.
        """
        if not rectified:
            img_left, img_right = self.calib.rectify(img_left, img_right)

        if self._use_wls:
            disp_l = self._matcher.compute(img_left, img_right)
            disp_r = self._right_m.compute(img_right, img_left)
            disp_f = self._wls.filter(disp_l, img_left,
                                       disparity_map_right=disp_r)
            disp   = disp_f.astype(np.float32) / 16.0
        else:
            disp_raw = self._matcher.compute(img_left, img_right)
            disp     = disp_raw.astype(np.float32) / 16.0

        disp[disp < self.cfg.min_disparity] = 0
        return disp

    def disparity_to_depth(self, disp: np.ndarray) -> np.ndarray:
        valid = disp > self.cfg.min_disparity
        depth = np.zeros_like(disp)
        depth[valid] = (self.f * self.B) / disp[valid]
        depth[(depth < self.cfg.min_depth) |
              (depth > self.cfg.max_depth)] = 0
        return depth

    def unproject_points(
        self,
        pts2d:  np.ndarray,    # (N,2) pixel coords in rectified left image
        disp:   np.ndarray,    # (H,W) disparity map
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Lift 2-D keypoints to 3-D using disparity.
        Uses a median patch for robustness against disparity noise.

        Returns
        -------
        pts3d      : (M,3) valid 3-D points in left-camera frame
        pts2d_v    : (M,2) corresponding pixel coords
        valid_mask : (N,) bool
        """
        r  = self.cfg.patch_radius
        h, w = disp.shape
        N  = len(pts2d)
        pts3d = np.zeros((N, 3), dtype=np.float64)
        valid = np.zeros(N, dtype=bool)

        for i, (u, v) in enumerate(pts2d.astype(int)):
            u = int(np.clip(u, r, w - r - 1))
            v = int(np.clip(v, r, h - r - 1))

            patch = disp[v-r:v+r+1, u-r:u+r+1]
            good  = patch[patch > self.cfg.min_disparity]
            if len(good) < 3:
                continue
            d = float(np.median(good))

            Z = self.f * self.B / d
            if Z < self.cfg.min_depth or Z > self.cfg.max_depth:
                continue

            X = (u - self.cx) * Z / self.f
            Y = (v - self.cy) * Z / self.f
            pts3d[i] = [X, Y, Z]
            valid[i] = True

        return pts3d[valid], pts2d[valid].astype(np.float32), valid
