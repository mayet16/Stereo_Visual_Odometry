"""
Monocular VO — final tuned version for TUM VI room2.

Design:
  - Single init, never resets coordinate frame
  - ORB detect+match for init (large baseline needed for good triangulation)
  - LK optical flow for tracking (fast, sub-pixel accurate)
  - Motion-only BA (solvePnPRefineLM) after every PnP
  - Loose reprojection filter (keep more points, trust RANSAC)
  - Uniform grid re-detection when map runs low
  - Auto-scale: waits for enough parallax before initialising
"""


import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from mono_vo.feature_tracker import FeatureTracker, FeatureConfig
from mono_vo.epipolar        import (estimate_essential, recover_pose,
                                      triangulate_points, pnp_ransac,
                                      refine_pose_ba)
from data.data_loader        import CameraIntrinsics, save_trajectory_tum
from utils.math_utils        import Rt_to_T, invert_T, cam_from_world
from utils.print_utils import print_map_init, print_mono_reproj_error

@dataclass
class MonoVOConfig:
    feature:          FeatureConfig = field(default_factory=FeatureConfig)
    min_tracked_pts:  int   = 120
    max_map_pts:      int   = 800
    min_parallax_px:  float = 8.0
    max_parallax_px:  float = 40.0
    init_scale:       Optional[float] = 0.02
    expected_depth:   float = 2.0
    pnp_min_inliers:  int   = 15
    pnp_ransac_th:    float = 6.0
    reproj_thresh:    float = 4.0
    use_ba:           bool  = True
    use_local_ba:     bool  = True
    local_ba_window:  int   = 7
    local_ba_every:   int   = 5
    max_velocity:     float = 0.5
    verbose:          bool  = True


class MonoVO:

    def __init__(self, cam: CameraIntrinsics,
                 cfg: MonoVOConfig = MonoVOConfig()):
        self.cam  = cam
        self.cfg  = cfg
        self.K    = cam.K
        self.dist = cam.dist_coeffs

        self.tracker = FeatureTracker(cfg.feature)

        self._initialised  = False
        self._frame_id     = 0
        self.n_failures    = 0
        self.n_map_extends = 0
        self.n_local_ba    = 0
        self.cur_pts: Optional[np.ndarray] = None

        self._map3d: Optional[np.ndarray] = None
        self._map2d: Optional[np.ndarray] = None

        self._init_img0: Optional[np.ndarray] = None
        self._init_pts0: Optional[np.ndarray] = None
        self._prev_img:  Optional[np.ndarray] = None

        # ── using Rt_to_T for clean pose construction ─────────────────────
        self._T_cur  = np.eye(4, dtype=np.float64)
        self._T_prev = np.eye(4, dtype=np.float64)

        # Sliding window for local BA
        self._window_poses: List[np.ndarray] = []
        self._window_pts3d: List[np.ndarray] = []
        self._window_pts2d: List[np.ndarray] = []

        self._timestamps: List[float]      = []
        self._poses:      List[np.ndarray] = []

    # ── public ────────────────────────────────────────────────────────────

    def process(self, img: np.ndarray,
                timestamp: float = 0.0) -> Optional[np.ndarray]:
        self._frame_id += 1
        if not self._initialised:
            return self._try_init(img, timestamp)
        return self._track(img, timestamp)

    def save_trajectory(self, path: str) -> None:
        save_trajectory_tum(path, self._timestamps, self._poses)

    @property
    def trajectory(self):
        return self._timestamps, self._poses

    @property
    def current_pose(self) -> np.ndarray:
        return self._T_cur.copy()

    # ── initialisation ────────────────────────────────────────────────────

    def _try_init(self, img: np.ndarray, ts: float) -> Optional[np.ndarray]:
        if self._init_img0 is None:
            self._init_img0 = img
            self._init_pts0 = self.tracker.detect_grid(img)
            self._prev_img  = img
            self._log(f"Init: stored frame 0 ({len(self._init_pts0)} kp)")
            return None

        # Track frame-0 features into current frame
        pts0_t, ok = self._lk_track(self._init_img0, img, self._init_pts0)
        pts0_v = self._init_pts0[ok]

        if len(pts0_t) < 20:
            self._log("Init: too few tracks – reset ref")
            self._init_img0 = img
            self._init_pts0 = self.tracker.detect_grid(img)
            self._prev_img  = img
            return None

        # Parallax gate
        flow     = np.linalg.norm(pts0_t - pts0_v, axis=1)
        med_flow = float(np.median(flow))
        self._log(f"Init: flow={med_flow:.1f}px")

        if med_flow < self.cfg.min_parallax_px:
            self._prev_img = img
            return None

        if med_flow > self.cfg.max_parallax_px:
            self._log(f"Init: flow too large ({med_flow:.1f}) – reset ref")
            self._init_img0 = img
            self._init_pts0 = self.tracker.detect_grid(img)
            self._prev_img  = img
            return None

        # Essential matrix — validation now inside estimate_essential
        E, mask = estimate_essential(pts0_v, pts0_t, self.K)
        if E is None:
            self._prev_img = img
            return None

        # recover_pose — det(R) check now inside recover_pose
        R, t, n_in = recover_pose(E, pts0_v, pts0_t, self.K, mask)
        if n_in < 30:
            self._log(f"Init: only {n_in} inliers")
            self._prev_img = img
            return None

        # Triangulate
        pts3d_unit = triangulate_points(
            np.eye(3), np.zeros(3), R, t,
            pts0_v[mask], pts0_t[mask], self.K,
        )
        valid = (pts3d_unit[:, 2] > 0) & np.all(np.isfinite(pts3d_unit), axis=1)
        if valid.sum() < 20:
            self._log(f"Init: only {valid.sum()} valid 3D pts")
            self._prev_img = img
            return None

        pts3d_v = pts3d_unit[valid]
        pts2d_v = pts0_t[mask][valid]

        # Scale
        if self.cfg.init_scale is not None:
            scale = self.cfg.init_scale
        else:
            med_depth = float(np.median(pts3d_v[:, 2]))
            scale     = self.cfg.expected_depth / max(med_depth, 1e-6)

        pts3d_v  = pts3d_v * scale
        t_scaled = t       * scale

        # ── Rt_to_T replaces manual np.eye(4) construction ───────────────
        self._T_prev = np.eye(4, dtype=np.float64)
        self._T_cur  = Rt_to_T(R, t_scaled)

        # ── print map init summary (for report) ──────────────────────────────
        if self.cfg.verbose:
            print_map_init(
                pts3d=pts3d_v,  pts2d=pts2d_v,
                frame_i=0, frame_j=self._frame_id,  # current frame when init triggers
                reproj_thresh=3.0,
            )
            # R, t here is cam1-from-cam0 (the init pair pose)
            R_cw, t_cw = cam_from_world(Rt_to_T(R, t_scaled))
            # print mono reporjection error
            print_mono_reproj_error(
                pts3d_v, pts2d_v, R_cw, t_cw,
                self.K, self.dist,
                label=f"mono init  frame 0→{self._frame_id}",
            )

        self._map3d    = pts3d_v
        self._map2d    = pts2d_v
        self._prev_img = img
        self._initialised = True
        self._log(f"Init OK: scale={scale:.4f}  n={valid.sum()}  "
                  f"pos={self._T_cur[:3,3]}")
        self._record(ts)
        return self._T_cur.copy()

    # ── per-frame tracking ────────────────────────────────────────────────

    def _track(self, img: np.ndarray, ts: float) -> Optional[np.ndarray]:

        # 1. LK tracking
        map2d_t, ok = self._lk_track(self._prev_img, img, self._map2d)
        map3d_t     = self._map3d[ok]
        n_tracked   = ok.sum()
        self._log(f"[{self._frame_id:04d}] tracked {n_tracked}/{len(self._map2d)}")

        if n_tracked < self.cfg.pnp_min_inliers:
            map2d_t, ok = self._lk_track(
                self._prev_img, img, self._map2d, extra_levels=2)
            map3d_t   = self._map3d[ok]
            n_tracked = ok.sum()

        if n_tracked < self.cfg.pnp_min_inliers:
            self._log(f"[{self._frame_id:04d}] too few – hold")
            self.n_failures += 1
            self._prev_img = img
            if ok.sum() > 5:
                self._map3d = map3d_t
                self._map2d = map2d_t
                if len(self._map3d) < self.cfg.min_tracked_pts:
                    self._extend_map(img)
            self._record(ts)
            return self._T_cur.copy()

        # 2. PnP — det(R) check now inside pnp_ransac
        R, t, inlier_mask = pnp_ransac(
            map3d_t, map2d_t, self.K, self.dist,
            min_inliers=self.cfg.pnp_min_inliers,
            ransac_th=self.cfg.pnp_ransac_th,
        )
        if R is None:
            self._log(f"[{self._frame_id:04d}] PnP failed – hold")
            self.n_failures += 1
            self._map3d    = map3d_t
            self._map2d    = map2d_t
            self._prev_img = img
            if len(self._map3d) < self.cfg.min_tracked_pts:
                self._extend_map(img)
            self._record(ts)
            return self._T_cur.copy()

        # 3. BA — det(R) check now inside refine_pose_ba
        if self.cfg.use_ba and inlier_mask.sum() >= 6:
            R, t = refine_pose_ba(
                R, t,
                map3d_t[inlier_mask],
                map2d_t[inlier_mask],
                self.K, self.dist,
            )

        # 4. Velocity check
        # ── Rt_to_T + invert_T replace manual construction + np.linalg.inv
        T_cw = Rt_to_T(R, t)
        T_wc = invert_T(T_cw)
        vel  = np.linalg.norm(T_wc[:3, 3] - self._T_cur[:3, 3])

        if vel > self.cfg.max_velocity:
            self._log(f"[{self._frame_id:04d}] vel={vel:.3f} > max – hold")
            self.n_failures += 1
            self._map3d    = map3d_t[inlier_mask]
            self._map2d    = map2d_t[inlier_mask]
            self._prev_img = img
            self._record(ts)
            return self._T_cur.copy()

        # 5. Accept pose
        self._T_prev = self._T_cur.copy()
        self._T_cur  = T_wc
        n_in = inlier_mask.sum()
        self._log(f"[{self._frame_id:04d}] PnP+BA OK  "
                  f"in={n_in}/{n_tracked}  vel={vel:.4f}m  "
                  f"pos={np.round(T_wc[:3,3], 3)}")

        # 6. Reprojection filter
        pts3d_in = map3d_t[inlier_mask]
        pts2d_in = map2d_t[inlier_mask]
        pts3d_f, pts2d_f = self._filter_reproj(
            pts3d_in, pts2d_in, R, t,
            thresh=self.cfg.reproj_thresh,
        )
        self._map3d = pts3d_f if len(pts3d_f) > 10 else pts3d_in
        self._map2d = pts2d_f if len(pts2d_f) > 10 else pts2d_in

        # 7. Extend map if thin
        if len(self._map3d) < self.cfg.min_tracked_pts:
            self._extend_map(img)

        # 8. Cap map
        if len(self._map3d) > self.cfg.max_map_pts:
            idx = np.random.choice(len(self._map3d),
                                   self.cfg.max_map_pts, replace=False)
            self._map3d = self._map3d[idx]
            self._map2d = self._map2d[idx]

        # 9. Local windowed BA
        self._window_poses.append(self._T_cur.copy())
        self._window_pts3d.append(self._map3d.copy())
        self._window_pts2d.append(self._map2d.copy())
        if len(self._window_poses) > self.cfg.local_ba_window:
            self._window_poses.pop(0)
            self._window_pts3d.pop(0)
            self._window_pts2d.pop(0)

        if (self.cfg.use_local_ba
                and self._frame_id % self.cfg.local_ba_every == 0
                and len(self._window_poses) >= 3):
            self._run_local_ba()

        self._prev_img = img
        self._record(ts)
        return self._T_cur.copy()

    # ── local windowed BA ─────────────────────────────────────────────────

    def _run_local_ba(self) -> None:
        all_pts3d = np.vstack(self._window_pts3d)
        all_pts2d = np.vstack(self._window_pts2d)
        if len(all_pts3d) < 10:
            return

        idx      = np.arange(0, len(all_pts3d), 3)
        pts3d_ba = all_pts3d[idx]
        pts2d_ba = all_pts2d[idx]

        # ── cam_from_world replaces manual R.T / -R.T@t ──────────────────
        R_cur, t_cur = cam_from_world(self._T_cur)

        rvec, _ = cv2.Rodrigues(R_cur)
        proj, _ = cv2.projectPoints(
            pts3d_ba.astype(np.float64),
            rvec, t_cur.reshape(3, 1).astype(np.float64),
            self.K, self.dist,
        )
        proj    = proj.reshape(-1, 2)
        h, w    = self.cam.height, self.cam.width
        in_img  = ((proj[:, 0] > 0) & (proj[:, 0] < w) &
                   (proj[:, 1] > 0) & (proj[:, 1] < h))
        err     = np.linalg.norm(proj - pts2d_ba, axis=1)
        visible = in_img & (err < self.cfg.reproj_thresh * 3)

        if visible.sum() < 6:
            return

        R_ref, t_ref = refine_pose_ba(
            R_cur, t_cur,
            pts3d_ba[visible], pts2d_ba[visible],
            self.K, self.dist,
        )

        # ── Rt_to_T + invert_T replace manual construction ────────────────
        T_cw_ref = Rt_to_T(R_ref, t_ref)
        T_wc_ref = invert_T(T_cw_ref)
        delta    = np.linalg.norm(T_wc_ref[:3, 3] - self._T_cur[:3, 3])

        if delta < 0.1:
            self._T_cur = T_wc_ref
            self._window_poses[-1] = T_wc_ref
            self.n_local_ba += 1
            self._log(f"  Local BA correction={delta:.4f}m")

    # ── map extension ─────────────────────────────────────────────────────

    def _extend_map(self, img_cur: np.ndarray) -> None:
        baseline = np.linalg.norm(
            self._T_cur[:3, 3] - self._T_prev[:3, 3])
        if baseline < 1e-5:
            return

        new2d_cur = self.tracker.detect_grid(img_cur)
        if len(new2d_cur) == 0:
            return

        new2d_prev, ok = self._lk_track(img_cur, self._prev_img, new2d_cur)
        if ok.sum() < 5:
            return

        # ── cam_from_world replaces manual _cam_from_world static method ──
        R_prev, t_prev = cam_from_world(self._T_prev)
        R_cur,  t_cur  = cam_from_world(self._T_cur)

        pts3d = triangulate_points(
            R_prev, t_prev, R_cur, t_cur,
            new2d_prev, new2d_cur[ok], self.K,
        )

        d_prev = (R_prev @ pts3d.T).T + t_prev
        d_cur  = (R_cur  @ pts3d.T).T + t_cur
        dist_w = np.linalg.norm(pts3d - self._T_cur[:3, 3], axis=1)

        good = (
            (d_prev[:, 2] > 0.01) & (d_cur[:, 2] > 0.01)
            & np.all(np.isfinite(pts3d), axis=1)
            & (dist_w < 20.0) & (dist_w > 0.02)
        )
        if good.sum() == 0:
            return

        pts3d_g, pts2d_g = self._filter_reproj(
            pts3d[good], new2d_cur[ok][good],
            R_cur, t_cur,
            thresh=self.cfg.reproj_thresh * 2,
        )
        if len(pts3d_g) == 0:
            return

        self._map3d = np.vstack([self._map3d, pts3d_g])
        self._map2d = np.vstack([self._map2d, pts2d_g])
        self.n_map_extends += 1
        self._log(f"  Map +{len(pts3d_g)} → {len(self._map3d)}")

    # ── helpers ───────────────────────────────────────────────────────────

    def _filter_reproj(
        self,
        pts3d: np.ndarray, pts2d: np.ndarray,
        R: np.ndarray, t: np.ndarray,
        thresh: float = 4.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        fb  = np.linalg.norm(p0 - p0b, axis=2).squeeze()
        ok  = (s1.squeeze() == 1) & (s0.squeeze() == 1) & (fb < 2.0)
        return p1.reshape(-1, 2)[ok], ok

    def _record(self, ts: float) -> None:
        self.cur_pts = self._map2d
        self._timestamps.append(ts)
        self._poses.append(self._T_cur.copy())

    def _log(self, msg: str) -> None:
        if self.cfg.verbose:
            print(f"[MonoVO] {msg}")