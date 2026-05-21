import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection

# Must be set before any cv2.namedWindow call
os.environ.setdefault("QT_QPA_PLATFORM",  "xcb")
os.environ.setdefault("QT_LOGGING_RULES",
                       "*.debug=false;*.warning=false;"
                       "qt.qpa.*=false;"
                       "qt.core.qobject.movethread=false")


_C_TRACKED = (0,   255,   0)
_C_PREV    = (0,   165, 255)
_C_FAIL    = (0,     0, 255)
_C_EST     = (255,  80,  80)
_C_GT      = (80,  200,  80)
_C_CUR     = (0,   255, 255)
_C_START   = (0,   255,   0)
_FONT      = cv2.FONT_HERSHEY_SIMPLEX


def _to_px(x: float, z: float,
           cx: float, cz: float,
           scale: float, H: int, W: int) -> tuple:
    """Map world (x, z) to canvas pixel; Z forward = up on canvas."""
    px = int(W / 2 + (x - cx) * scale)
    py = int(H / 2 - (z - cz) * scale)
    return (int(np.clip(px, 0, W - 1)),
            int(np.clip(py, 0, H - 1)))


def _pts_to_px_array(xz_list: list, cx: float, cz: float,
                     scale: float, H: int, W: int) -> np.ndarray:
    """Convert (x, z) world points to Nx1x2 int32 for cv2.polylines."""
    if len(xz_list) < 2:
        return None
    arr = np.array(xz_list, dtype=np.float32)   # Nx2
    px  = (W / 2 + (arr[:, 0] - cx) * scale).astype(np.int32)
    py  = (H / 2 - (arr[:, 1] - cz) * scale).astype(np.int32)
    px  = np.clip(px, 0, W - 1)
    py  = np.clip(py, 0, H - 1)
    return np.stack([px, py], axis=1).reshape(-1, 1, 2)


class LiveVisualizer:
    """Two OpenCV windows (features + trajectory) updated every N frames.

    Trajectory drawing uses a decimated path (every Nth point) so draw cost
    stays O(N/decimate); running min/max keeps auto-scale computation O(1).
    """

    def __init__(self,
                 title:     str   = "VO",
                 canvas_hw: tuple = (512, 512),
                 show:      bool  = True,
                 decimate:  int   = 3):
        self.title     = title
        self.canvas_hw = canvas_hw
        self.show      = show
        self.decimate  = max(1, decimate)

        self._win_feat = f"{title} — features"
        self._win_traj = f"{title} — trajectory"

        self._xyz_min: "np.ndarray | None" = None
        self._xyz_max: "np.ndarray | None" = None

        self._est_draw3d: list = []
        self._est_cur_3d:   tuple = (0.0, 0.0, 0.0)
        self._est_start_3d: "tuple | None" = None

        self._gt_draw: list = []
        self._gt_cur:  "tuple | None" = None

        self._call_count  = 0
        self._stereo_mode = False

        if self.show:
            H, W = canvas_hw
            cv2.namedWindow(self._win_feat, cv2.WINDOW_NORMAL)
            cv2.namedWindow(self._win_traj, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._win_feat, W, H)
            cv2.resizeWindow(self._win_traj, W, H)
            cv2.moveWindow(self._win_feat, 0,      50)
            cv2.moveWindow(self._win_traj, W + 10, 50)

    def update(self,
               img:        np.ndarray,
               pts_cur:    "np.ndarray | None",
               pose:       np.ndarray,
               gt_pose:    "np.ndarray | None" = None,
               frame_id:   int                 = 0,
               n_failures: int                 = 0,
               extra_info: str                 = "",
               img_right:  "np.ndarray | None" = None,
               pts_right:  "np.ndarray | None" = None) -> None:
        self._call_count += 1

        p = pose[:3, 3]
        x, y, z = float(p[0]), float(p[1]), float(p[2])

        self._est_cur_3d = (x, y, z)

        if self._est_start_3d is None:
            self._est_start_3d = (x, y, z)

        pos = np.array([x, y, z], dtype=np.float64)
        if self._xyz_min is None:
            self._xyz_min = pos.copy()
            self._xyz_max = pos.copy()
        else:
            self._xyz_min = np.minimum(self._xyz_min, pos)
            self._xyz_max = np.maximum(self._xyz_max, pos)

        if self._call_count % self.decimate == 0:
            self._est_draw3d.append((x, y, z))

        if img_right is not None and not self._stereo_mode and self.show:
            self._stereo_mode = True
            H, W = self.canvas_hw
            cv2.resizeWindow(self._win_feat, W * 2 + 2, H)

        feat_img = self._draw_features(
            img, pts_cur, frame_id, n_failures, extra_info,
            img_right=img_right, pts_right=pts_right)
        traj_img = self._draw_trajectory()

        if self.show:
            cv2.imshow(self._win_feat, feat_img)
            cv2.imshow(self._win_traj, traj_img)
            cv2.waitKey(1)

    def close(self) -> None:
        if self.show:
            cv2.destroyWindow(self._win_feat)
            cv2.destroyWindow(self._win_traj)

    def reset(self) -> None:
        self._xyz_min       = None
        self._xyz_max       = None
        self._est_draw3d    = []
        self._gt_draw       = []
        self._est_cur_3d    = (0.0, 0.0, 0.0)
        self._gt_cur        = None
        self._est_start_3d  = None
        self._call_count    = 0
        self._stereo_mode   = False

    def _draw_features(self, img, pts_cur,
                       frame_id, n_failures, extra_info,
                       img_right=None, pts_right=None) -> np.ndarray:
        H, W = self.canvas_hw

        def _render_frame(src_img, pts):
            f = (cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
                 if src_img.ndim == 2 else src_img.copy())
            f = cv2.resize(f, (W, H))
            ih, iw = src_img.shape[:2]
            sx, sy = W / iw, H / ih
            if pts is not None and len(pts) > 0:
                px = (pts * [sx, sy]).astype(np.int32)
                px[:, 0] = np.clip(px[:, 0], 0, W - 1)
                px[:, 1] = np.clip(px[:, 1], 0, H - 1)
                for i in range(len(px)):
                    cv2.circle(f, tuple(px[i]), 3, _C_TRACKED, -1)
            return f

        vis = _render_frame(img, pts_cur)

        n_trk = 0 if pts_cur is None else len(pts_cur)
        lines = [
            (f"Frame    {frame_id:05d}", (200, 200, 200)),
            (f"Tracked  {n_trk}",        _C_TRACKED),
            (f"Failures {n_failures}",   _C_FAIL),
        ]
        if extra_info:
            lines.append((extra_info, (180, 180, 180)))

        box_h = 14 + len(lines) * 18
        cv2.rectangle(vis, (0, 0), (165, box_h), (0, 0, 0),  -1)
        cv2.rectangle(vis, (0, 0), (165, box_h), (60, 60, 60), 1)
        for i, (txt, col) in enumerate(lines):
            cv2.putText(vis, txt, (5, 14 + i * 18),
                        _FONT, 0.48, col, 1, cv2.LINE_AA)

        if img_right is None:
            return vis

        cv2.putText(vis, "LEFT", (W - 42, H - 8),
                    _FONT, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
        vis_r = _render_frame(img_right, pts_right)
        n_r = 0 if pts_right is None else len(pts_right)
        cv2.putText(vis_r, f"RIGHT  n={n_r}", (5, H - 8),
                    _FONT, 0.42, _C_TRACKED, 1, cv2.LINE_AA)
        divider = np.full((H, 2, 3), 60, dtype=np.uint8)
        return np.hstack([vis, divider, vis_r])

    def _draw_trajectory(self) -> np.ndarray:
        H, W = self.canvas_hw
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        if self._xyz_min is None or len(self._est_draw3d) < 2:
            return canvas

        # Auto-select the two highest-range axes (= floor plane) so the view
        # adapts to any camera mounting angle without GT alignment.
        ranges = self._xyz_max - self._xyz_min
        order  = np.argsort(ranges)[::-1]
        a0, a1 = int(order[0]), int(order[1])
        names  = ['X', 'Y', 'Z']

        arr3d   = np.array(self._est_draw3d, dtype=np.float32)
        xz_list = list(zip(arr3d[:, a0].tolist(), arr3d[:, a1].tolist()))

        cur_2d   = (self._est_cur_3d[a0],   self._est_cur_3d[a1])
        start_2d = (self._est_start_3d[a0], self._est_start_3d[a1]) \
                   if self._est_start_3d else None

        pad   = 0.15
        xmin, xmax = float(self._xyz_min[a0]), float(self._xyz_max[a0])
        zmin, zmax = float(self._xyz_min[a1]), float(self._xyz_max[a1])
        rx    = max(xmax - xmin, 0.5) * (1 + pad)
        rz    = max(zmax - zmin, 0.5) * (1 + pad)
        scale = min((W - 20) / rx, (H - 20) / rz)
        cx    = (xmin + xmax) / 2
        cz    = (zmin + zmax) / 2

        grid_px = int(scale)
        if grid_px >= 8:
            x0 = int(np.floor(xmin - 1));  x1 = int(np.ceil(xmax + 1))
            z0 = int(np.floor(zmin - 1));  z1 = int(np.ceil(zmax + 1))
            for xi in range(x0, x1 + 1):
                px = int(W / 2 + (xi - cx) * scale)
                if 0 <= px < W:
                    cv2.line(canvas, (px, 0), (px, H), (30, 30, 30), 1)
            for zi in range(z0, z1 + 1):
                pz = int(H / 2 - (zi - cz) * scale)
                if 0 <= pz < H:
                    cv2.line(canvas, (0, pz), (W, pz), (30, 30, 30), 1)

        pts = _pts_to_px_array(xz_list, cx, cz, scale, H, W)
        if pts is not None:
            cv2.polylines(canvas, [pts], False, _C_EST, 2, cv2.LINE_AA)

        if start_2d is not None:
            sp = _to_px(*start_2d, cx, cz, scale, H, W)
            cv2.circle(canvas, sp, 6, _C_START, -1)

        cur_px = _to_px(*cur_2d, cx, cz, scale, H, W)
        cv2.circle(canvas, cur_px, 6, _C_CUR, -1)

        scale_str  = (f"1m={int(scale)}px" if scale >= 1
                      else f"1px={1/scale:.1f}m")
        view_label = f"Top-down  {names[a0]}–{names[a1]}"
        info_lines = [
            ("EST",         _C_EST),
            (view_label,    (180, 180, 180)),
            (scale_str,     (140, 140, 140)),
            (f"n={len(self._est_draw3d)*self.decimate}", (120, 120, 120)),
        ]
        box_h = 12 + len(info_lines) * 15
        cv2.rectangle(canvas, (0, 0), (130, box_h), (0, 0, 0),  -1)
        cv2.rectangle(canvas, (0, 0), (130, box_h), (50, 50, 50), 1)
        for i, (txt, col) in enumerate(info_lines):
            cv2.putText(canvas, txt, (4, 12 + i * 15),
                        _FONT, 0.40, col, 1, cv2.LINE_AA)

        return canvas


def save_3d_trajectory(
    est_poses: list,
    gt_poses:  list,
    title:     str  = "3D Trajectory",
    out_path:  str  = "traj_3d.png",
    align:     str  = "sim3",
    dpi:       int  = 130,
    show_gt:   bool = True,
) -> None:
    """Save a dark-themed 3-D trajectory figure (PNG)."""
    est     = np.array([T[:3, 3] for T in est_poses], dtype=np.float64)
    gt_list = [T[:3, 3] for T in gt_poses if T is not None]
    gt      = np.array(gt_list, dtype=np.float64) if gt_list else None

    ate_val = float("nan")
    if align and gt is not None and len(gt) > 10:
        try:
            from evaluation.metrics import align_and_evaluate
            # Pair by index — only frames where GT is not None
            paired_est, paired_gt = [], []
            for i, g in enumerate(gt_poses):
                if g is not None and i < len(est_poses):
                    paired_est.append(est_poses[i])
                    paired_gt.append(g)
            if len(paired_est) >= 10:
                result  = align_and_evaluate(paired_est, paired_gt, align=align)
                s       = result.get("s", 1.0)
                R_align = result["R"]
                t_align = result["t"]
                est     = np.array(
                    [s * R_align @ T[:3, 3] + t_align for T in est_poses],
                    dtype=np.float64)
                ate_val = result["ate_rmse"]
        except Exception:
            pass

    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor("#0e0e0e")

    ax3d = fig.add_subplot(121, projection="3d")
    ax3d.set_facecolor("#0e0e0e")
    ax3d.tick_params(colors="white", labelsize=7)
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#333333")

    if show_gt and gt is not None:
        ax3d.plot(gt[:, 0], gt[:, 1], gt[:, 2],
                  color="#A5D6A7", lw=0.7, ls="--", alpha=0.45, label="GT (ref)")
        ax3d.scatter(gt[0, 0], gt[0, 1], gt[0, 2],
                     c="lime",   s=40, zorder=4, marker="*", label="GT start")
        ax3d.scatter(gt[-1, 0], gt[-1, 1], gt[-1, 2],
                     c="gold",   s=40, zorder=4, marker="*", label="GT end")

    ax3d.plot(est[:, 0], est[:, 1], est[:, 2],
              color="#4FC3F7", lw=1.2, label="Estimated")
    ax3d.scatter(est[0, 0],  est[0, 1],  est[0, 2],
                 c="blue", s=50, zorder=6, marker="o", label="Est start")
    ax3d.scatter(est[-1, 0], est[-1, 1], est[-1, 2],
                 c="red",  s=50, zorder=6, marker="D", label="Est end")

    ax3d.set_xlabel("X [m]", color="white", fontsize=8)
    ax3d.set_ylabel("Y [m]", color="white", fontsize=8)
    ax3d.set_zlabel("Z [m] (up)", color="white", fontsize=8)
    ax3d.set_title("3D view", color="white", fontsize=9)
    ax3d.legend(fontsize=7, facecolor="#222222", labelcolor="white",
                loc="upper left")

    ax_xz = fig.add_subplot(322)
    ax_xy = fig.add_subplot(324)
    ax_yz = fig.add_subplot(326)

    proj_defs = [
        (ax_xz,
         est[:, 0], est[:, 1],
         gt[:, 0] if gt is not None else None,
         gt[:, 1] if gt is not None else None,
         "X [m]", "Y [m]", "Top-down  X–Y"),
        (ax_xy,
         est[:, 0], est[:, 2],
         gt[:, 0] if gt is not None else None,
         gt[:, 2] if gt is not None else None,
         "X [m]", "Z [m]", "Front     X–Z"),
        (ax_yz,
         est[:, 1], est[:, 2],
         gt[:, 1] if gt is not None else None,
         gt[:, 2] if gt is not None else None,
         "Y [m]", "Z [m]", "Side      Y–Z"),
    ]

    for ax, xe, ye, xg, yg, xl, yl, ttl in proj_defs:
        ax.set_facecolor("#111111")
        ax.tick_params(colors="white", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.set_xlabel(xl, color="white", fontsize=7)
        ax.set_ylabel(yl, color="white", fontsize=7)
        ax.set_title(ttl, color="white", fontsize=8)

        if show_gt and xg is not None:
            ax.plot(xg, yg, color="#A5D6A7", lw=0.6, ls="--",
                    alpha=0.4, label="GT (ref)")
            ax.scatter(xg[0],  yg[0],  c="lime", s=35, zorder=4, marker="*",
                       label="GT start")
            ax.scatter(xg[-1], yg[-1], c="gold", s=35, zorder=4, marker="*",
                       label="GT end")

        ax.plot(xe, ye, color="#4FC3F7", lw=1.0, label="Est")
        ax.scatter(xe[0],  ye[0],  c="blue", s=35, zorder=7, marker="o", label="Est start")
        ax.scatter(xe[-1], ye[-1], c="red",  s=35, zorder=7, marker="D", label="Est end")

        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=7, facecolor="#222222", labelcolor="white",
                  loc="best")

    ate_str = (f"  |  ATE = {ate_val:.3f} m ({align.upper()})"
               if not np.isnan(ate_val) else "")
    fig.suptitle(f"{title}{ate_str}", color="white",
                 fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"  Saved 3D trajectory → {out_path}")


def save_comparison_3d(
    mono_poses:   list,
    stereo_poses: list,
    gt_poses:     list,
    seq_name:     str   = "",
    out_path:     str   = "comparison_3d.png",
    dpi:          int   = 140,
    full_gt:      bool  = True,
    mono_drift:   float = float("nan"),
    stereo_drift: float = float("nan"),
) -> None:
    """Dark-themed 3-D figure comparing GT, Sim3-aligned mono VO, and SE3-aligned stereo VO."""
    from evaluation.metrics import align_and_evaluate

    gt_T = [T for T in gt_poses if T is not None]
    gt   = np.array([T[:3, 3] for T in gt_T], dtype=np.float64) if gt_T else None

    if full_gt:
        def _align(poses, mode):
            paired_est, paired_gt = [], []
            for i, g in enumerate(gt_poses):
                if g is not None and i < len(poses):
                    paired_est.append(poses[i])
                    paired_gt.append(g)
            if len(paired_gt) < 10:
                return np.array([T[:3, 3] for T in poses], dtype=np.float64), float("nan"), None
            try:
                res = align_and_evaluate(paired_est, paired_gt, align=mode)
                s    = res.get("s", 1.0)
                R, t = res["R"], res["t"]
                arr  = np.array([s * R @ T[:3, 3] + t for T in poses], dtype=np.float64)
                return arr, res["ate_rmse"], res.get("errors")
            except Exception:
                pass
            return np.array([T[:3, 3] for T in poses], dtype=np.float64), float("nan"), None

        mono_arr,   mono_ate,   mono_errors   = _align(mono_poses,   "sim3")
        stereo_arr, stereo_ate, stereo_errors = _align(stereo_poses, "se3")
        mono_lbl   = f"Mono VO Sim3  ATE={mono_ate:.3f}m"
        stereo_lbl = f"Stereo VO SE3  ATE={stereo_ate:.3f}m"
        title_str  = (f"Mono vs Stereo VO — {seq_name}  |  "
                      f"Mono ATE={mono_ate:.3f}m (Sim3)  "
                      f"Stereo ATE={stereo_ate:.3f}m (SE3)")
    else:
        # Trajectories are already in GT world frame (start-aligned by caller)
        mono_arr   = np.array([T[:3, 3] for T in mono_poses],   dtype=np.float64)
        stereo_arr = np.array([T[:3, 3] for T in stereo_poses], dtype=np.float64)
        _dm = f"{mono_drift:.2f}m" if np.isfinite(mono_drift) else "N/A"
        _ds = f"{stereo_drift:.2f}m" if np.isfinite(stereo_drift) else "N/A"
        mono_lbl   = f"Mono VO  drift={_dm}"
        stereo_lbl = f"Stereo VO  drift={_ds}"
        title_str  = (f"Mono vs Stereo VO — {seq_name}  |  "
                      f"mono drift={_dm}  stereo drift={_ds}")

    fig = plt.figure(figsize=(10, 4.8))
    fig.patch.set_facecolor("#0e0e0e")

    ax3 = fig.add_subplot(131, projection="3d")
    ax3.set_facecolor("#0e0e0e")
    ax3.tick_params(colors="white", labelsize=7)
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.fill = False; pane.set_edgecolor("#333333")

    if gt is not None:
        if full_gt:
            ax3.plot(gt[:, 0], gt[:, 1], gt[:, 2],
                     color="#A5D6A7", lw=1.2, ls="--", alpha=0.9, label="GT")
        ax3.scatter(float(gt[0,  0]), float(gt[0,  1]), float(gt[0,  2]),
                    c="lime", s=80, zorder=7, marker="*", label="GT start")
        ax3.scatter(float(gt[-1, 0]), float(gt[-1, 1]), float(gt[-1, 2]),
                    c="gold", s=80, zorder=7, marker="*", label="GT end")
    ax3.plot(mono_arr[:, 0],   mono_arr[:, 1],   mono_arr[:, 2],
             color="#4FC3F7", lw=1.0, label="Mono VO")
    ax3.scatter(mono_arr[0, 0],  mono_arr[0, 1],  mono_arr[0, 2],
                c="cyan",   s=50, zorder=6, marker="o", label="Mono start")
    ax3.scatter(mono_arr[-1, 0], mono_arr[-1, 1], mono_arr[-1, 2],
                c="cyan",   s=50, zorder=6, marker="D", label="Mono end")
    ax3.plot(stereo_arr[:, 0], stereo_arr[:, 1], stereo_arr[:, 2],
             color="#EF9A9A", lw=1.0, label="Stereo VO")
    ax3.scatter(stereo_arr[0, 0],  stereo_arr[0, 1],  stereo_arr[0, 2],
                c="blue", s=50, zorder=6, marker="o", label="Stereo start")
    ax3.scatter(stereo_arr[-1, 0], stereo_arr[-1, 1], stereo_arr[-1, 2],
                c="red",  s=50, zorder=6, marker="D", label="Stereo end")
    ax3.set_xlabel("X [m]", color="white", fontsize=8)
    ax3.set_ylabel("Y [m]", color="white", fontsize=8)
    ax3.set_zlabel("Z [m] (up)", color="white", fontsize=8)
    ax3.set_title("3-D view", color="white", fontsize=9)
    ax3.legend(fontsize=7, facecolor="#222222", labelcolor="white",
               loc="upper left", bbox_to_anchor=(-0.05, 1.12))

    ax_xy = fig.add_subplot(132)
    ax_xy.set_facecolor("#111111")
    ax_xy.tick_params(colors="white", labelsize=7)
    for s in ax_xy.spines.values(): s.set_edgecolor("#333333")
    if gt is not None:
        if full_gt:
            ax_xy.plot(gt[:, 0], gt[:, 1], color="#A5D6A7", lw=1.0,
                       ls="--", alpha=0.9, label="GT")
        ax_xy.scatter(float(gt[0,  0]), float(gt[0,  1]),
                      c="lime", s=80, zorder=7, marker="*", label="GT start")
        ax_xy.scatter(float(gt[-1, 0]), float(gt[-1, 1]),
                      c="gold", s=80, zorder=7, marker="*", label="GT end")
    ax_xy.plot(mono_arr[:, 0],   mono_arr[:, 1],   color="#4FC3F7",
               lw=0.9, label="Mono")
    ax_xy.scatter(mono_arr[0, 0],  mono_arr[0, 1],  c="cyan",   s=50,
                  zorder=6, marker="o", label="Mono start")
    ax_xy.scatter(mono_arr[-1, 0], mono_arr[-1, 1], c="cyan",   s=50,
                  zorder=6, marker="D", label="Mono end")
    ax_xy.plot(stereo_arr[:, 0], stereo_arr[:, 1], color="#EF9A9A",
               lw=0.9, label="Stereo")
    ax_xy.scatter(stereo_arr[0, 0],  stereo_arr[0, 1],  c="blue", s=50,
                  zorder=6, marker="o", label="Stereo start")
    ax_xy.scatter(stereo_arr[-1, 0], stereo_arr[-1, 1], c="red",  s=50,
                  zorder=6, marker="D", label="Stereo end")
    ax_xy.set_xlabel("x [m]", color="white", fontsize=8)
    ax_xy.set_ylabel("y [m]", color="white", fontsize=8)
    ax_xy.set_title("Top-down  x–y", color="white", fontsize=9)
    _xy_arrs = [mono_arr[:, :2], stereo_arr[:, :2]]
    if gt is not None: _xy_arrs.append(gt[:, :2])
    _xy_all = np.concatenate(_xy_arrs)
    if full_gt:
        _cx   = (_xy_all[:, 0].min() + _xy_all[:, 0].max()) / 2
        _cy   = (_xy_all[:, 1].min() + _xy_all[:, 1].max()) / 2
        _half = max(_xy_all[:, 0].max() - _xy_all[:, 0].min(),
                    _xy_all[:, 1].max() - _xy_all[:, 1].min()) / 2 * 1.08
        ax_xy.set_xlim(_cx - _half, _cx + _half)
        ax_xy.set_ylim(_cy - _half, _cy + _half)
        ax_xy.set_aspect("equal", adjustable="box")
    else:
        _pad = 0.05 * max(_xy_all[:, 0].max() - _xy_all[:, 0].min(),
                          _xy_all[:, 1].max() - _xy_all[:, 1].min())
        ax_xy.set_xlim(_xy_all[:, 0].min() - _pad, _xy_all[:, 0].max() + _pad)
        ax_xy.set_ylim(_xy_all[:, 1].min() - _pad, _xy_all[:, 1].max() + _pad)
    ax_z = fig.add_subplot(133)
    ax_z.set_facecolor("#111111")
    ax_z.tick_params(colors="white", labelsize=7)
    for s in ax_z.spines.values(): s.set_edgecolor("#333333")
    if full_gt:
        if mono_errors is not None:
            ax_z.plot(mono_errors,   color="#4FC3F7", lw=0.8, label=f"Mono ATE (RMSE={mono_ate:.3f}m)")
        if stereo_errors is not None:
            ax_z.plot(stereo_errors, color="#EF9A9A", lw=0.8, label=f"Stereo ATE (RMSE={stereo_ate:.3f}m)")
        ax_z.set_xlabel("GT-paired frame", color="white", fontsize=8)
        ax_z.set_ylabel("ATE [m]", color="white", fontsize=8)
        ax_z.set_ylim(bottom=0)
        ax_z.set_title("ATE over time", color="white", fontsize=9)
    else:
        ax_z.plot(stereo_arr[:, 2], color="#EF9A9A", lw=0.9, label="Stereo z")
        ax_z.axhline(stereo_arr[0, 2], color="#A5D6A7", lw=0.7, ls="--",
                     alpha=0.7, label=f"start z={stereo_arr[0,2]:.2f}m")
        ax_z.set_xlabel("frame", color="white", fontsize=8)
        ax_z.set_ylabel("z / height [m]", color="white", fontsize=8)
        ax_z.set_title("Stereo height over time  (start-aligned)", color="white", fontsize=9)
    ax_z.legend(fontsize=7, facecolor="#222222", labelcolor="white")

    fig.suptitle(title_str, color="white", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"  Saved comparison 3D → {out_path}")
