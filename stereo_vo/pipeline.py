"""
Stereo Visual Odometry — clean classical implementation.
Follows the project spec exactly:
  1. Rectify stereo pair
  2. SGBM disparity
  3. 3D point cloud via Z = fB/d
  4. Track 3D→2D via LK + PnP
  5. Add new stereo points when map runs low
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from mono_vo.feature_tracker import FeatureTracker, FeatureConfig
from mono_vo.epipolar        import pnp_ransac, refine_pose_ba
from stereo_vo.disparity     import DisparityComputer, DisparityConfig
from data.data_loader        import StereoPair, save_trajectory_tum
from utils.math_utils        import Rt_to_T, invert_T, cam_from_world
from utils.print_utils import print_map_init, print_stereo_reproj_error


@dataclass
class StereoVOConfig:
    feature:          FeatureConfig   = field(default_factory=FeatureConfig)
    disparity:        DisparityConfig = field(default_factory=DisparityConfig)
    min_tracked_pts:  int   = 120
    max_map_pts:      int   = 500
    pnp_min_inliers:  int   = 15
    pnp_ransac_th:    float = 4.0
    reproj_thresh:    float = 4.0
    use_ba:           bool  = True
    max_velocity:     float = 0.5
    use_depth_update: bool  = True   # disable for low-texture scenes (corridor)
    use_clahe:        bool  = False  # CLAHE contrast enhancement before SGBM+LK
    verbose:          bool  = True


class StereoVO:

    def __init__(self, calib: StereoPair,
                 cfg: StereoVOConfig = StereoVOConfig()):
        self.calib    = calib
        self.cfg      = cfg
        self.disp_cmp = DisparityComputer(calib, cfg.disparity)
        self.tracker  = FeatureTracker(cfg.feature)

        self.K    = calib.P_left[:3, :3].copy()
        self.dist = np.zeros(5, dtype=np.float64)

        self._initialised  = False
        self._frame_id     = 0
        self.n_failures    = 0
        self.n_new_pts     = 0

        self._map3d: Optional[np.ndarray] = None
        self._map2d: Optional[np.ndarray] = None

        self._prev_left: Optional[np.ndarray] = None
        self._rect_r_current: Optional[np.ndarray] = None

        self._T_cur  = np.eye(4, dtype=np.float64)
        self._T_prev = np.eye(4, dtype=np.float64)

        self._timestamps: List[float]      = []
        self._poses:      List[np.ndarray] = []

        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ── public ────────────────────────────────────────────────────────────

    def process(self, img_left: np.ndarray, img_right: np.ndarray,
                timestamp: float = 0.0) -> Optional[np.ndarray]:
        self._frame_id += 1
        rect_l, rect_r = self.calib.rectify(img_left, img_right)
        self._rect_r_current = rect_r  # store original for display
        if self.cfg.use_clahe:
            rect_l = self._clahe.apply(rect_l)
            rect_r = self._clahe.apply(rect_r)
        disp = self.disp_cmp.compute(rect_l, rect_r, rectified=True)

        if not self._initialised:
            return self._init(rect_l, disp, timestamp)
        return self._track(rect_l, disp, timestamp)

    def save_trajectory(self, path: str) -> None:
        save_trajectory_tum(path, self._timestamps, self._poses)

    @property
    def trajectory(self):
        return self._timestamps, self._poses

    @property
    def current_pose(self) -> np.ndarray:
        return self._T_cur.copy()

    # Properties used by the live visualizer (map2d = current tracked pts)
    @property
    def cur_pts(self) -> Optional[np.ndarray]:
        return self._map2d

    @property
    def cur_pts_right(self) -> Optional[np.ndarray]:
        return None

    @property
    def cur_right_img(self) -> Optional[np.ndarray]:
        return self._rect_r_current

    # ── init ──────────────────────────────────────────────────────────────

    def _init(self, rect_l: np.ndarray, disp: np.ndarray,
               ts: float) -> Optional[np.ndarray]:
        pts2d = self.tracker.detect_grid(rect_l)
        if len(pts2d) < 30:
            return None

        pts3d_cam, pts2d_v, _ = self.disp_cmp.unproject_points(pts2d, disp)
        if len(pts3d_cam) < 20:
            self._log(f"Init: {len(pts3d_cam)} pts – retry")
            return None

        if self.cfg.verbose:
            print_map_init(pts3d_cam, pts2d_v,
                           frame_i=0, frame_j=0)
            print_stereo_reproj_error(
                pts3d_cam, pts2d_v,
                self.K,
                label="stereo init  frame 0",
            )

        # World = camera frame at t=0  (T = identity)
        self._map3d     = pts3d_cam.copy()
        self._map2d     = pts2d_v.copy()
        self._prev_left = rect_l
        self._T_cur     = np.eye(4, dtype=np.float64)
        self._T_prev    = np.eye(4, dtype=np.float64)
        self._initialised = True

        self._log(f"Init OK: {len(pts3d_cam)} pts  "
                  f"med_Z={np.median(pts3d_cam[:,2]):.3f}m  "
                  f"B={self.calib.baseline*100:.1f}cm")
        self._record(ts)
        return self._T_cur.copy()

    # ── tracking ──────────────────────────────────────────────────────────

    def _track(self, rect_l: np.ndarray, disp: np.ndarray,
                ts: float) -> Optional[np.ndarray]:

        # 1. LK tracking
        map2d_cur, ok = self._lk_track(self._prev_left, rect_l, self._map2d)
        map3d_t       = self._map3d[ok]
        n_tracked     = ok.sum()
        self._log(f"[{self._frame_id:04d}] tracked {n_tracked}/{len(self._map2d)}")

        if n_tracked < self.cfg.pnp_min_inliers:
            map2d_cur, ok = self._lk_track(
                self._prev_left, rect_l, self._map2d, extra_levels=2)
            map3d_t   = self._map3d[ok]
            n_tracked = ok.sum()

        if n_tracked < self.cfg.pnp_min_inliers:
            self._log(f"[{self._frame_id:04d}] too few – hold")
            self.n_failures += 1
            if ok.sum() > 5:
                self._map3d = map3d_t
                self._map2d = map2d_cur
            self._add_points(rect_l, disp)
            self._prev_left = rect_l
            self._record(ts)
            return self._T_cur.copy()

        # 2. PnP
        R, t, inlier_mask = pnp_ransac(
            map3d_t, map2d_cur,
            self.K, self.dist,
            min_inliers=self.cfg.pnp_min_inliers,
            ransac_th=self.cfg.pnp_ransac_th,
        )
        if R is None:
            self._log(f"[{self._frame_id:04d}] PnP failed – hold")
            self.n_failures += 1
            self._map3d = map3d_t
            self._map2d = map2d_cur
            self._add_points(rect_l, disp)
            self._prev_left = rect_l
            self._record(ts)
            return self._T_cur.copy()

        # 3. BA
        if self.cfg.use_ba and inlier_mask.sum() >= 6:
            R, t = refine_pose_ba(
                R, t,
                map3d_t[inlier_mask],
                map2d_cur[inlier_mask],
                self.K, self.dist,
            )

        # 4. Velocity check
        T_cw = Rt_to_T(R, t)
        T_wc = invert_T(T_cw)
        vel  = np.linalg.norm(T_wc[:3, 3] - self._T_cur[:3, 3])

        if vel > self.cfg.max_velocity:
            self._log(f"[{self._frame_id:04d}] vel={vel:.3f} – hold")
            self.n_failures += 1
            self._map3d = map3d_t[inlier_mask]
            self._map2d = map2d_cur[inlier_mask]
            self._add_points(rect_l, disp)
            self._prev_left = rect_l
            self._record(ts)
            return self._T_cur.copy()

        # 5. Accept pose
        self._T_prev = self._T_cur.copy()
        self._T_cur  = T_wc
        n_in = inlier_mask.sum()
        self._log(f"[{self._frame_id:04d}] OK in={n_in}/{n_tracked} "
                  f"vel={vel:.4f}m pos={np.round(T_wc[:3,3], 3)}")

        # 6. Reprojection filter on PnP inliers
        pts3d_in = map3d_t[inlier_mask]
        pts2d_in = map2d_cur[inlier_mask]
        pts3d_f, pts2d_f = self._filter_reproj(
            pts3d_in, pts2d_in, R, t,
            thresh=self.cfg.reproj_thresh,
        )
        self._map3d = pts3d_f if len(pts3d_f) > 10 else pts3d_in
        self._map2d = pts2d_f if len(pts2d_f) > 10 else pts2d_in

        # 6b. Refresh map2d: reproject map3d to exact pixel positions.
        # Eliminates LK drift that accumulates when map2d is only
        # updated via optical flow (without this, rotation errors grow
        # monotonically because 2D correspondences slowly diverge from
        # the true projected positions).
        self._refresh_map2d(rect_l)

        # 6c. Re-measure existing map point depths from current disparity.
        # Disabled for low-texture scenes (e.g. corridors) where SGBM
        # produces noisy disparity on walls — depth noise corrupts PnP.
        if self.cfg.use_depth_update:
            self._update_depths_from_disp(disp)

        # 7. Add new stereo points if map is thin
        if len(self._map3d) < self.cfg.min_tracked_pts:
            self._add_points(rect_l, disp)

        # 8. Cap map
        if len(self._map3d) > self.cfg.max_map_pts:
            idx = np.random.choice(len(self._map3d),
                                   self.cfg.max_map_pts, replace=False)
            self._map3d = self._map3d[idx]
            self._map2d = self._map2d[idx]

        self._prev_left = rect_l
        self._record(ts)
        return self._T_cur.copy()

    # ── add new stereo points ─────────────────────────────────────────────

    def _add_points(self, rect_l: np.ndarray, disp: np.ndarray) -> None:
        pts2d = self.tracker.detect_grid(rect_l)
        if len(pts2d) == 0:
            return

        pts3d_cam, pts2d_v, _ = self.disp_cmp.unproject_points(pts2d, disp)
        if len(pts3d_cam) == 0:
            return

        R_wc = self._T_cur[:3, :3]
        t_wc = self._T_cur[:3,  3]
        pts3d_w = (R_wc @ pts3d_cam.T).T + t_wc

        d    = np.linalg.norm(pts3d_w - t_wc, axis=1)
        good = (d > 0.1) & (d < self.cfg.disparity.max_depth * 1.5) & np.all(np.isfinite(pts3d_w), 1)
        if good.sum() == 0:
            return

        if self._map3d is not None and len(self._map3d) > 0:
            self._map3d = np.vstack([self._map3d, pts3d_w[good]])
            self._map2d = np.vstack([self._map2d, pts2d_v[good]])
        else:
            self._map3d = pts3d_w[good]
            self._map2d = pts2d_v[good]

        self.n_new_pts += good.sum()
        self._log(f"  +{good.sum()} pts → {len(self._map3d)}")

    # ── helpers ───────────────────────────────────────────────────────────

    def _update_depths_from_disp(self, disp: np.ndarray) -> None:
        """Re-measure depth of each tracked map point from the current
        disparity map and update its world-frame 3D position accordingly.
        Uses the same patch-median approach as point addition."""
        if self._map3d is None or self._map2d is None or len(self._map2d) == 0:
            return
        r     = self.disp_cmp.cfg.patch_radius
        h, w  = disp.shape
        R_wc  = self._T_cur[:3, :3]
        t_wc  = self._T_cur[:3, 3]
        cx    = self.disp_cmp.cx
        cy    = self.disp_cmp.cy
        f     = self.disp_cmp.f
        B     = self.disp_cmp.B
        min_d = self.disp_cmp.cfg.min_disparity
        min_z = self.disp_cmp.cfg.min_depth
        max_z = self.disp_cmp.cfg.max_depth

        for i, (u_f, v_f) in enumerate(self._map2d):
            u = int(np.clip(round(u_f), r, w - r - 1))
            v = int(np.clip(round(v_f), r, h - r - 1))
            patch = disp[v-r:v+r+1, u-r:u+r+1].ravel()
            good  = patch[patch > min_d]
            if len(good) < 12:
                continue
            Z = f * B / float(np.median(good))
            if not (min_z <= Z <= max_z):
                continue
            X = (u_f - cx) * Z / f
            Y = (v_f - cy) * Z / f
            self._map3d[i] = R_wc @ np.array([X, Y, Z]) + t_wc

    def _refresh_map2d(self, rect_l: np.ndarray) -> None:
        """Reproject map3d to current pose → replace map2d with exact pixel
        positions and drop points that have left the image.  Called after
        every accepted pose so LK tracking always starts from the true
        projection rather than the accumulated optical-flow position."""
        if self._map3d is None or len(self._map3d) == 0:
            return
        R_cw, t_cw = cam_from_world(self._T_cur)
        rvec, _ = cv2.Rodrigues(R_cw)
        proj, _ = cv2.projectPoints(
            self._map3d.astype(np.float64),
            rvec, t_cw.reshape(3, 1).astype(np.float64),
            self.K, self.dist,
        )
        proj    = proj.reshape(-1, 2).astype(np.float32)
        pts_cam = (R_cw @ self._map3d.T).T + t_cw
        h, w    = rect_l.shape[:2]
        in_img  = (
            (proj[:, 0] >= 1) & (proj[:, 0] < w - 1) &
            (proj[:, 1] >= 1) & (proj[:, 1] < h - 1) &
            (pts_cam[:, 2] > 0.01)
        )
        if in_img.sum() < self.cfg.pnp_min_inliers:
            return
        self._map3d = self._map3d[in_img]
        self._map2d = proj[in_img]

    def _filter_reproj(
        self,
        pts3d: np.ndarray, pts2d: np.ndarray,
        R: np.ndarray, t: np.ndarray,
        thresh: float = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        thresh = thresh or self.cfg.reproj_thresh
        if len(pts3d) == 0:
            return pts3d, pts2d
        rvec, _ = cv2.Rodrigues(R)
        proj, _ = cv2.projectPoints(
            pts3d.astype(np.float64),
            rvec, t.reshape(3, 1).astype(np.float64),
            self.K, self.dist,
        )
        err  = np.linalg.norm(
            proj.reshape(-1, 2) - pts2d.astype(np.float64), axis=1)
        keep = err < thresh
        return pts3d[keep], pts2d[keep]

    def _lk_track(
        self, img0: np.ndarray, img1: np.ndarray,
        pts0: np.ndarray, extra_levels: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(pts0) == 0:
            return np.empty((0, 2), np.float32), np.zeros(0, bool)
        lk = dict(
            winSize  = (self.cfg.feature.lk_win_size,) * 2,
            maxLevel = self.cfg.feature.lk_max_level + extra_levels,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        self.cfg.feature.lk_max_iter,
                        self.cfg.feature.lk_eps),
        )
        p0         = pts0.reshape(-1, 1, 2).astype(np.float32)
        p1, s1, _  = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk)
        p0b, s0, _ = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk)
        fb  = np.linalg.norm(p0 - p0b, axis=2).ravel()
        ok  = (s1.ravel() == 1) & (s0.ravel() == 1) & (fb < 2.0)
        return p1.reshape(-1, 2)[ok], ok

    def _record(self, ts: float) -> None:
        self._timestamps.append(ts)
        self._poses.append(self._T_cur.copy())

    def _log(self, msg: str) -> None:
        if self.cfg.verbose:
            print(f"[StereoVO] {msg}")
