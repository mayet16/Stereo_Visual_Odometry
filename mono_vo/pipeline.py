"""
Monocular VO — keyframe-based pipeline for TUM VI.

Root cause of previous scale collapse:
  _extend_map was called ~every 2 frames using consecutive-frame pairs
  (baseline ~5 cm, scene depth ~2 m → 1.4° angle).  At that angle,
  triangulation noise inflates depths, making PnP under-estimate
  translation.  After a few hundred frames the entire map was corrupt.

Fix:
  Map extension exclusively from KF→current pairs.
  A new keyframe is created when the baseline/depth ratio exceeds a
  threshold, guaranteeing a useful triangulation angle (≥ 3°).
  Consecutive-frame triangulation is completely removed.

Virtual-depth (VD) reinit uses the actual median scene depth from the
current map, not the fixed expected_depth, so scale is preserved during
pure-rotation phases.
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


@dataclass
class MonoVOConfig:
    feature:              FeatureConfig = field(default_factory=FeatureConfig)
    # ── initialisation ──────────────────────────────────────────────────────
    min_parallax_px:      float = 8.0
    max_parallax_px:      float = 40.0
    expected_depth:       float = 2.0    # scene depth used to set world scale [m]
    # ── PnP ────────────────────────────────────────────────────────────────
    pnp_min_inliers:      int   = 12
    pnp_ransac_th:        float = 6.0
    reproj_thresh:        float = 4.0
    use_ba:               bool  = True
    # ── map ────────────────────────────────────────────────────────────────
    max_map_pts:          int   = 600
    min_tracked_pts:      int   = 50
    # ── keyframe ───────────────────────────────────────────────────────────
    kf_min_parallax_px:   float = 20.0   # proxy parallax to trigger new KF
    kf_max_parallax_px:   float = 60.0   # force KF even without triangulation
    kf_min_baseline_ratio:float = 0.05   # min baseline/depth for triangulation
    # ── motion model ───────────────────────────────────────────────────────
    max_velocity:         float = 0.5    # max per-frame translation [m]
    max_cvm_frames:       int   = 3      # CVM extrapolation limit
    # ── misc ───────────────────────────────────────────────────────────────
    use_clahe:            bool  = False  # CLAHE contrast enhancement before tracking
    # E-matrix reinit has cheirality ambiguity for 180° rotations — disable
    # for corridor-like sequences where the camera turns around.
    use_e_reinit:         bool  = True
    # Minimum camera-frame z accepted by _get_scene_depth().  Raise for
    # rotation-heavy close-range sequences (room2) to exclude very near VD
    # points that are hard to re-track under rotation; keep low (0.1m) for
    # translation-dominant sequences (corridor3) for accurate depth estimation.
    scene_depth_lo_m:     float = 0.1
    verbose:              bool  = False


class MonoVO:

    def __init__(self, cam: CameraIntrinsics,
                 cfg: MonoVOConfig = MonoVOConfig()):
        self.cam  = cam
        self.cfg  = cfg
        self.K    = cam.K
        self.dist = cam.dist_coeffs

        self.tracker = FeatureTracker(cfg.feature)

        # ── public counters ───────────────────────────────────────────────
        self._initialised  = False
        self._frame_id     = 0
        self.n_failures    = 0
        self.n_lk_fails    = 0
        self.n_pnp_fails   = 0
        self.n_vel_fails   = 0
        self.n_kf_updates  = 0
        self.n_vd_reinits  = 0
        self.n_e_reinits   = 0
        self.cur_pts: Optional[np.ndarray] = None

        # ── init state ────────────────────────────────────────────────────
        self._init_img:  Optional[np.ndarray] = None
        self._init_pts:  Optional[np.ndarray] = None

        # ── map ───────────────────────────────────────────────────────────
        self._map3d: Optional[np.ndarray] = None  # (N,3) world frame
        self._map2d: Optional[np.ndarray] = None  # (N,2) current-frame pixels

        # ── keyframe ─────────────────────────────────────────────────────
        self._kf_img: Optional[np.ndarray] = None
        self._kf_T:   Optional[np.ndarray] = None  # T_wc of last keyframe

        # ── pose ─────────────────────────────────────────────────────────
        self._T_cur  = np.eye(4, dtype=np.float64)
        self._T_prev = np.eye(4, dtype=np.float64)
        self._prev_img: Optional[np.ndarray] = None

        # ── constant-velocity model ───────────────────────────────────────
        self._last_delta_T:   np.ndarray = np.eye(4, dtype=np.float64)
        self._has_delta:      bool       = False
        self._n_consec_fails: int        = 0

        # ── E-matrix re-init anchor ───────────────────────────────────────
        self._anchor_img:    Optional[np.ndarray] = None
        self._anchor_pts:    Optional[np.ndarray] = None   # detect_grid features for E-matrix
        self._anchor_map2d:  Optional[np.ndarray] = None   # map 2D positions for PnP primary
        self._anchor_map3d:  Optional[np.ndarray] = None   # map 3D world points for PnP primary
        self._anchor_T:      Optional[np.ndarray] = None
        self._anchor_active: bool                 = False

        # ── trajectory ────────────────────────────────────────────────────
        self._timestamps: List[float]      = []
        self._poses:      List[np.ndarray] = []

        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ── public API ──────────────────────────────────────────────────────────

    def process(self, img: np.ndarray,
                timestamp: float = 0.0) -> Optional[np.ndarray]:
        self._frame_id += 1
        if self.cfg.use_clahe:
            img = self._clahe.apply(img)
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

    # ── initialisation ──────────────────────────────────────────────────────

    def _try_init(self, img: np.ndarray, ts: float) -> Optional[np.ndarray]:
        if self._init_img is None:
            self._init_img = img
            self._init_pts = self.tracker.detect_grid(img)
            self._prev_img = img
            return None

        pts_cur, ok = self._lk_track(self._init_img, img, self._init_pts)
        pts0 = self._init_pts[ok]

        if len(pts_cur) < 20:
            self._init_img = img
            self._init_pts = self.tracker.detect_grid(img)
            self._prev_img = img
            return None

        flow = float(np.median(np.linalg.norm(pts_cur - pts0, axis=1)))

        if flow < self.cfg.min_parallax_px:
            self._prev_img = img
            return None
        if flow > self.cfg.max_parallax_px:
            self._init_img = img
            self._init_pts = self.tracker.detect_grid(img)
            self._prev_img = img
            return None

        # VD init: back-project frame0 features at expected_depth, then PnP
        # to recover the metric init pose.  This anchors scale directly to
        # expected_depth instead of relying on E-matrix triangulation depth,
        # which is unreliable when rotational motion dominates (med_depth ≫
        # expected_depth → scale collapse).
        d0 = self.cfg.expected_depth
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])

        pts3d_world = np.zeros((len(pts0), 3), dtype=np.float64)
        for i, (u, v) in enumerate(pts0):
            ang   = float(np.sqrt(((u - cx) / fx)**2 + ((v - cy) / fy)**2))
            depth = d0 * (1.0 + 0.1 * ang)
            pts3d_world[i] = [(u - cx) / fx * depth,
                              (v - cy) / fy * depth,
                              depth]

        R_init, t_init, inlier_mask = pnp_ransac(
            pts3d_world, pts_cur.astype(np.float32), self.K, self.dist,
            min_inliers=20, ransac_th=self.cfg.pnp_ransac_th,
        )
        if R_init is None:
            self._prev_img = img
            return None

        T_cw     = Rt_to_T(R_init, t_init)
        T_wc_cur = invert_T(T_cw)
        n_in     = int(inlier_mask.sum())
        t_norm   = float(np.linalg.norm(T_wc_cur[:3, 3]))

        print(f"[MonoVO] VD init  d0={d0:.2f}m  n={n_in}  "
              f"flow={flow:.1f}px  t={t_norm:.3f}m")

        self._T_prev = np.eye(4, dtype=np.float64)
        self._T_cur  = T_wc_cur
        self._map3d  = pts3d_world[inlier_mask]
        self._map2d  = pts_cur[inlier_mask].astype(np.float32)

        # Keyframe = init frame (world origin)
        self._kf_img = self._init_img.copy()
        self._kf_T   = np.eye(4, dtype=np.float64)

        self._prev_img    = img
        self._initialised = True
        # CVM not bootstrapped here — set after first tracking frame
        self._record(ts)
        return self._T_cur.copy()

    # ── per-frame tracking ──────────────────────────────────────────────────

    def _track(self, img: np.ndarray, ts: float) -> Optional[np.ndarray]:

        # 1. LK tracking: prev → cur
        map2d_t, ok = self._lk_track(self._prev_img, img, self._map2d)
        map3d_t     = self._map3d[ok]
        n_tracked   = ok.sum()

        if n_tracked < self.cfg.pnp_min_inliers:
            map2d_t, ok = self._lk_track(
                self._prev_img, img, self._map2d, extra_levels=2)
            map3d_t   = self._map3d[ok]
            n_tracked = ok.sum()

        lk_ok = n_tracked >= self.cfg.pnp_min_inliers

        # 2. PnP
        R, t, inlier_mask = None, None, None
        if lk_ok:
            R, t, inlier_mask = pnp_ransac(
                map3d_t, map2d_t, self.K, self.dist,
                min_inliers=self.cfg.pnp_min_inliers,
                ransac_th=self.cfg.pnp_ransac_th,
            )

        # 3. Failure path
        if R is None:
            if not lk_ok:
                self.n_lk_fails += 1
            else:
                self.n_pnp_fails += 1
            self.n_failures      += 1
            self._n_consec_fails += 1

            if self._n_consec_fails == 1:
                self._anchor_img    = self._prev_img.copy()
                self._anchor_pts    = self.tracker.detect_grid(self._prev_img)  # E-matrix
                self._anchor_map2d  = self._map2d.copy()   # PnP primary (aligned w/ map3d)
                self._anchor_map3d  = self._map3d.copy()   # PnP primary (world-frame 3D)
                self._anchor_T      = self._T_cur.copy()
                self._anchor_active = True

            if self._anchor_active and self._n_consec_fails >= 2:
                if self._try_reinit(img, ts):
                    self._prev_img = img
                    return self._T_cur.copy()

            if lk_ok and n_tracked >= self.cfg.pnp_min_inliers:
                self._map3d = map3d_t
                self._map2d = map2d_t

            if self._has_delta and self._n_consec_fails <= self.cfg.max_cvm_frames:
                self._T_prev = self._T_cur.copy()
                self._T_cur  = self._T_cur @ self._last_delta_T

            if len(self._map3d) < self.cfg.pnp_min_inliers * 2:
                self._vd_reinit(img)

            self._prev_img = img
            self._record(ts)
            return self._T_cur.copy()

        # 4. Optional BA
        if self.cfg.use_ba and inlier_mask.sum() >= 6:
            R, t = refine_pose_ba(
                R, t,
                map3d_t[inlier_mask],
                map2d_t[inlier_mask],
                self.K, self.dist,
            )

        T_cw = Rt_to_T(R, t)
        T_wc = invert_T(T_cw)

        # 5. Velocity guard
        vel = float(np.linalg.norm(T_wc[:3, 3] - self._T_cur[:3, 3]))
        if vel > self.cfg.max_velocity:
            self.n_failures      += 1
            self.n_vel_fails     += 1
            self._n_consec_fails += 1
            pts3d_in = map3d_t[inlier_mask]
            pts2d_in = map2d_t[inlier_mask]
            if len(pts3d_in) >= self.cfg.pnp_min_inliers:
                self._map3d = pts3d_in
                self._map2d = pts2d_in
            if self._has_delta and self._n_consec_fails <= self.cfg.max_cvm_frames:
                self._T_prev = self._T_cur.copy()
                self._T_cur  = self._T_cur @ self._last_delta_T
            self._prev_img = img
            self._record(ts)
            return self._T_cur.copy()

        # 6. Accept pose
        self._T_prev = self._T_cur.copy()
        self._T_cur  = T_wc

        self._last_delta_T   = invert_T(self._T_prev) @ self._T_cur
        self._has_delta      = True
        self._n_consec_fails = 0
        self._anchor_active  = False

        # 7. Reprojection filter on PnP inliers
        pts3d_in = map3d_t[inlier_mask]
        pts2d_in = map2d_t[inlier_mask]
        R_cw, t_cw = cam_from_world(self._T_cur)
        pts3d_f, pts2d_f = self._filter_reproj(
            pts3d_in, pts2d_in, R_cw, t_cw, self.cfg.reproj_thresh)
        if len(pts3d_f) >= self.cfg.pnp_min_inliers:
            self._map3d = pts3d_f
            self._map2d = pts2d_f
        else:
            self._map3d = pts3d_in
            self._map2d = pts2d_in

        # 8. Refresh map2d (project from accepted pose → eliminates LK drift)
        self._refresh_map2d(img)

        # 9. Keyframe update + map extension from KF baseline
        self._maybe_update_kf(img)

        # 10. VD augment if map still sparse after KF update
        if len(self._map3d) < self.cfg.min_tracked_pts:
            self._vd_augment(img)

        # 11. Cap map
        if len(self._map3d) > self.cfg.max_map_pts:
            idx = np.random.choice(len(self._map3d),
                                   self.cfg.max_map_pts, replace=False)
            self._map3d = self._map3d[idx]
            self._map2d = self._map2d[idx]

        self._prev_img = img
        self._record(ts)
        return self._T_cur.copy()

    # ── keyframe update ─────────────────────────────────────────────────────

    def _maybe_update_kf(self, img: np.ndarray) -> None:
        """
        Create a new keyframe when the accumulated baseline from the last
        keyframe is large enough for reliable triangulation.

        Triangulated points come exclusively from KF→current pairs to
        guarantee a useful triangulation angle and consistent map scale.
        """
        if self._kf_T is None:
            return

        # Baseline from last KF
        baseline = float(np.linalg.norm(
            self._T_cur[:3, 3] - self._kf_T[:3, 3]))

        # Median scene depth (camera-frame z of current map)
        med_depth = self._get_scene_depth()

        focal     = float(self.K[0, 0])
        approx_px = baseline / max(med_depth, 0.01) * focal

        # Trigger only on sufficient translation-parallax — NOT on map size.
        # Map density is managed separately by _vd_augment so that map_low
        # alone never resets kf_img to a near-identical frame (which would
        # give 1-frame baseline and reject every triangulated point).
        if approx_px < self.cfg.kf_min_parallax_px:
            return

        # Attempt triangulation from KF→current
        new3d, new2d_cur = self._triangulate_from_kf(img, baseline, med_depth)

        # Promote current frame to keyframe
        self._kf_img = img.copy()
        self._kf_T   = self._T_cur.copy()
        self.n_kf_updates += 1

        if len(new3d) == 0:
            return

        # Augment map with correctly-triangulated points
        self._map3d = np.vstack([self._map3d, new3d])
        self._map2d = np.vstack([self._map2d, new2d_cur])

        self._log(f"[{self._frame_id:04d}] KF  +{len(new3d)} pts "
                  f"map={len(self._map3d)}  B={baseline:.3f}m  "
                  f"~{approx_px:.0f}px")

    def _triangulate_from_kf(
        self,
        img: np.ndarray,
        baseline: float,
        med_depth: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect FAST corners in current frame, track back to kf_img,
        triangulate from the KF pose to the current pose.

        Only points whose baseline/depth ratio exceeds kf_min_baseline_ratio
        are returned — this ensures the triangulation angle is large enough
        for the depth to be reliable.
        """
        new2d_cur = self.tracker.detect_grid(img)
        if len(new2d_cur) < 5:
            return np.empty((0, 3), np.float64), np.empty((0, 2), np.float32)

        # Track detected corners back to the KF image
        new2d_kf, ok = self._lk_track(img, self._kf_img, new2d_cur, extra_levels=1)
        if ok.sum() < 5:
            return np.empty((0, 3), np.float64), np.empty((0, 2), np.float32)

        new2d_cur_ok = new2d_cur[ok]

        R_kf, t_kf   = cam_from_world(self._kf_T)
        R_cur, t_cur = cam_from_world(self._T_cur)

        pts3d = triangulate_points(
            R_kf, t_kf, R_cur, t_cur,
            new2d_kf, new2d_cur_ok, self.K,
        )

        pts_cam_kf  = (R_kf  @ pts3d.T).T + t_kf
        pts_cam_cur = (R_cur  @ pts3d.T).T + t_cur
        depths_cur  = pts_cam_cur[:, 2]
        depths_kf   = pts_cam_kf[:, 2]
        dist_world  = np.linalg.norm(pts3d - self._T_cur[:3, 3], axis=1)

        # Baseline/depth ratio filter: ensures triangulation angle is useful
        angle_ratio = baseline / np.maximum(depths_cur, 1e-6)

        good = (
            (depths_cur > 0.1)
            & (depths_kf  > 0.1)
            & np.all(np.isfinite(pts3d), axis=1)
            & (angle_ratio > self.cfg.kf_min_baseline_ratio)
            & (dist_world < 20.0)
            & (dist_world > 0.02)
        )

        if good.sum() == 0:
            return np.empty((0, 3), np.float64), np.empty((0, 2), np.float32)

        # Reprojection filter from current camera
        R_cw, t_cw = cam_from_world(self._T_cur)
        pts3d_g, pts2d_g = self._filter_reproj(
            pts3d[good], new2d_cur_ok[good],
            R_cw, t_cw,
            thresh=self.cfg.reproj_thresh * 2,
        )
        return pts3d_g, pts2d_g

    # ── E-matrix re-initialisation ───────────────────────────────────────────

    def _try_reinit(self, img: np.ndarray, ts: float) -> bool:
        """
        Recover pose after consecutive tracking failures.

        Primary path: direct 3D-PnP using _anchor_map2d/_anchor_map3d (world-
        frame 3D points tracked via LK into the current frame).  No cheirality
        ambiguity — correct for any rotation including 180° corridor turns.

        Fallback: E-matrix + scene-depth scale snap using _anchor_pts (detect_
        grid features, ~300 pts).  Disabled via use_e_reinit=False for sequences
        with large rotations (E-matrix cheirality picks wrong decomposition).
        """
        if (self._anchor_img is None
                or self._anchor_T  is None):
            return False

        # ── Primary: direct 3D-PnP ───────────────────────────────────────────
        if (self._anchor_map2d is not None
                and self._anchor_map3d is not None
                and len(self._anchor_map2d) >= 8):
            pts_cur_pnp, ok_pnp = self._lk_track(
                self._anchor_img, img, self._anchor_map2d, extra_levels=2)
            if len(pts_cur_pnp) >= 8:
                pts3d_ok = self._anchor_map3d[ok_pnp]
                R_d, t_d, inl_d = pnp_ransac(
                    pts3d_ok, pts_cur_pnp, self.K, self.dist,
                    min_inliers=8, ransac_th=self.cfg.pnp_ransac_th * 1.5,
                )
                if R_d is not None:
                    T_cw_d = Rt_to_T(R_d, t_d)
                    T_wc_d = invert_T(T_cw_d)
                    vel    = float(np.linalg.norm(
                        T_wc_d[:3, 3] - self._anchor_T[:3, 3]))
                    if vel < self.cfg.max_velocity * 10:
                        self._T_prev         = self._anchor_T.copy()
                        self._T_cur          = T_wc_d
                        self._map3d          = pts3d_ok[inl_d]
                        self._map2d          = pts_cur_pnp[inl_d].astype(np.float32)
                        self._last_delta_T   = invert_T(self._T_prev) @ self._T_cur
                        self._has_delta      = True
                        self._n_consec_fails = 0
                        self._anchor_active  = False
                        self.n_e_reinits    += 1
                        self._kf_img = self._anchor_img.copy()
                        self._kf_T   = self._anchor_T.copy()
                        self._refresh_map2d(img)
                        self._record(ts)
                        self._log(f"[{self._frame_id:04d}] PnP-reinit OK  "
                                  f"n={inl_d.sum()}")
                        return True

        # ── Fallback: E-matrix + scene-depth scale snap ───────────────────────
        if not self.cfg.use_e_reinit:
            return False

        if (self._anchor_pts is None
                or len(self._anchor_pts) < 20):
            return False

        pts_cur, ok = self._lk_track(
            self._anchor_img, img, self._anchor_pts, extra_levels=2)
        pts0 = self._anchor_pts[ok]

        if len(pts_cur) < 20:
            return False

        flow = float(np.median(np.linalg.norm(pts_cur - pts0, axis=1)))
        if flow < self.cfg.min_parallax_px:
            return False
        if flow > self.cfg.max_parallax_px * 2:
            return False

        E, emask = estimate_essential(pts0, pts_cur, self.K)
        if E is None:
            return False

        R_new, t_new, n_in = recover_pose(E, pts0, pts_cur, self.K, emask)
        if n_in < 20:
            return False

        pts3d_unit = triangulate_points(
            np.eye(3), np.zeros(3), R_new, t_new,
            pts0[emask], pts_cur[emask], self.K,
        )
        valid = (pts3d_unit[:, 2] > 0.01) & np.all(np.isfinite(pts3d_unit), axis=1)
        if valid.sum() < 15:
            return False

        pts3d_v = pts3d_unit[valid]
        pts2d_v = pts_cur[emask][valid].astype(np.float32)

        med_depth_unit = float(np.median(pts3d_v[:, 2]))
        if med_depth_unit < 1e-6:
            return False

        world_depth = self._get_scene_depth()
        scale       = world_depth / med_depth_unit
        pts3d_v     = pts3d_v  * scale
        t_scaled    = t_new    * scale

        T_cw_local = Rt_to_T(R_new, t_scaled)
        T_wc_local = invert_T(T_cw_local)
        T_wc_world = self._anchor_T @ T_wc_local

        R_anc = self._anchor_T[:3, :3]
        t_anc = self._anchor_T[:3, 3]
        pts3d_world = (R_anc @ pts3d_v.T).T + t_anc

        self._T_prev         = self._anchor_T.copy()
        self._T_cur          = T_wc_world
        self._map3d          = pts3d_world
        self._map2d          = pts2d_v
        self._last_delta_T   = invert_T(self._T_prev) @ self._T_cur
        self._has_delta      = True
        self._n_consec_fails = 0
        self._anchor_active  = False
        self.n_e_reinits    += 1

        self._kf_img = self._anchor_img.copy()
        self._kf_T   = self._anchor_T.copy()

        self._refresh_map2d(img)
        self._record(ts)
        self._log(f"[{self._frame_id:04d}] E-reinit OK  "
                  f"flow={flow:.1f}px  n={valid.sum()}")
        return True

    # ── virtual-depth reinit ─────────────────────────────────────────────────

    def _vd_reinit(self, img: np.ndarray) -> bool:
        """
        Back-project fresh FAST corners to the current median scene depth.
        Uses the actual depth from existing map points (not expected_depth)
        to preserve world scale.
        """
        new2d = self.tracker.detect_grid(img)
        if len(new2d) < self.cfg.pnp_min_inliers:
            return False

        d0 = self._get_scene_depth()

        cx, cy = self.K[0, 2], self.K[1, 2]
        fx, fy = self.K[0, 0], self.K[1, 1]

        pts3d_cam = np.zeros((len(new2d), 3), dtype=np.float64)
        for i, (u, v) in enumerate(new2d):
            ang   = np.sqrt(((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2)
            depth = d0 * (1.0 + 0.1 * ang)
            pts3d_cam[i] = [(u - cx) / fx * depth,
                             (v - cy) / fy * depth,
                             depth]

        R_wc = self._T_cur[:3, :3]
        t_wc = self._T_cur[:3, 3]
        pts3d_world = (R_wc @ pts3d_cam.T).T + t_wc

        self._map3d = pts3d_world
        self._map2d = new2d.astype(np.float32)
        self.n_vd_reinits += 1
        self._log(f"[{self._frame_id:04d}] VD-reinit  d0={d0:.2f}m")
        return True

    def _vd_augment(self, img: np.ndarray) -> None:
        """
        Add virtual-depth 3D points to a sparse map without replacing it.

        Called during normal (non-failure) tracking whenever the map falls
        below min_tracked_pts.  Uses the actual median scene depth from the
        existing map so world scale is preserved.  This is NOT consecutive-frame
        triangulation — depths come from back-projection at the current scene
        depth, not from geometry, so there is no small-baseline depth inflation.
        """
        d0 = self._get_scene_depth()
        new2d = self.tracker.detect_grid(img)
        if len(new2d) == 0:
            return

        cx, cy = self.K[0, 2], self.K[1, 2]
        fx, fy = self.K[0, 0], self.K[1, 1]

        pts3d_cam = np.zeros((len(new2d), 3), dtype=np.float64)
        for i, (u, v) in enumerate(new2d):
            ang   = np.sqrt(((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2)
            depth = d0 * (1.0 + 0.1 * ang)
            pts3d_cam[i] = [(u - cx) / fx * depth,
                             (v - cy) / fy * depth,
                             depth]

        R_wc = self._T_cur[:3, :3]
        t_wc = self._T_cur[:3, 3]
        pts3d_world = (R_wc @ pts3d_cam.T).T + t_wc

        if len(self._map3d) > 0:
            self._map3d = np.vstack([self._map3d, pts3d_world])
            self._map2d = np.vstack([self._map2d, new2d.astype(np.float32)])
        else:
            self._map3d = pts3d_world
            self._map2d = new2d.astype(np.float32)

        self._log(f"[{self._frame_id:04d}] VD-augment +{len(new2d)} pts "
                  f"map={len(self._map3d)}  d0={d0:.2f}m")

    # ── map2d refresh ────────────────────────────────────────────────────────

    def _refresh_map2d(self, img: np.ndarray) -> None:
        """Project _map3d to current pose → replace _map2d with exact pixel positions."""
        if self._map3d is None or len(self._map3d) == 0:
            return
        h, w = img.shape[:2]
        R_cw, t_cw = cam_from_world(self._T_cur)
        rvec, _ = cv2.Rodrigues(R_cw)
        proj, _ = cv2.projectPoints(
            self._map3d.astype(np.float64),
            rvec, t_cw.reshape(3, 1).astype(np.float64),
            self.K, self.dist,
        )
        proj    = proj.reshape(-1, 2).astype(np.float32)
        pts_cam = (R_cw @ self._map3d.T).T + t_cw
        in_img  = (
            (proj[:, 0] >= 1) & (proj[:, 0] < w - 1) &
            (proj[:, 1] >= 1) & (proj[:, 1] < h - 1) &
            (pts_cam[:, 2] > 0.01)
        )
        if in_img.sum() < self.cfg.pnp_min_inliers:
            return
        self._map3d = self._map3d[in_img]
        self._map2d = proj[in_img]

    # ── helpers ──────────────────────────────────────────────────────────────

    def _get_scene_depth(self) -> float:
        """Median camera-frame z of current map points."""
        if self._map3d is not None and len(self._map3d) > 0:
            R_cw, t_cw = cam_from_world(self._T_cur)
            pts_cam = (R_cw @ self._map3d.T).T + t_cw
            z = pts_cam[:, 2]
            d_lo = max(0.05, self.cfg.scene_depth_lo_m)
            d_hi = min(20.0, 2.5 * self.cfg.expected_depth)
            vis = z[(z > d_lo) & (z < d_hi)]
            if len(vis) >= 3:
                return float(np.median(vis))
        return self.cfg.expected_depth

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
        self,
        img0: np.ndarray,
        img1: np.ndarray,
        pts0: np.ndarray,
        extra_levels: int = 0,
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
        p0        = pts0.reshape(-1, 1, 2).astype(np.float32)
        p1, s1, _ = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk)
        p0b,s0, _ = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk)
        fb  = np.linalg.norm(p0 - p0b, axis=2).ravel()
        ok  = (s1.ravel() == 1) & (s0.ravel() == 1) & (fb < 2.0)
        return p1.reshape(-1, 2)[ok], ok

    def _record(self, ts: float) -> None:
        self.cur_pts = self._map2d
        self._timestamps.append(ts)
        self._poses.append(self._T_cur.copy())

    def _log(self, msg: str) -> None:
        if self.cfg.verbose:
            print(f"[MonoVO] {msg}")
