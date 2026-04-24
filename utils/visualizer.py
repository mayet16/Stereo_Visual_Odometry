"""
utils/visualizer.py
===================
Two visual utilities for the Stereo VO project:

1. LiveVisualizer  — TWO separate OpenCV windows shown simultaneously:
       Window A  "title — features" : camera frame with tracked features
                   green dots  = current feature positions
                   orange dots = previous feature positions
                   green lines = optical flow vectors
       Window B  "title — trajectory" : top-down trajectory with AUTO-SCALE
                   blue  = estimated trajectory (always fully visible)
                   green = ground-truth trajectory
                   cyan dot = current camera position

   Performance fixes (problems 1 & 2):
   - Trajectory redraws use a DECIMATED path (every Nth point stored for
     drawing) so cost stays O(N/decimate) not O(N).  Full history is kept
     separately for scale computation — only the last point is added each
     call, so scale computation is O(1) with running min/max.
   - update() is designed to be called only every `update_every` frames
     from main.py — the VO loop is never blocked by drawing.
   - cv2.waitKey(1) is the minimum — no blocking wait.
   - Polylines used instead of per-segment line draws — 10-20× faster for
     long trajectories.

2. save_3d_trajectory — dark-themed 3-D trajectory figure saved as PNG.
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

# Must be set before any cv2.namedWindow call
os.environ.setdefault("QT_QPA_PLATFORM",  "xcb")
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")


# ── colour palette (BGR for OpenCV) ──────────────────────────────────────────
_C_TRACKED = (0,   255,   0)   # green  — current feature
_C_PREV    = (0,   165, 255)   # orange — previous feature
_C_FAIL    = (0,     0, 255)   # red    — failure text
_C_EST     = (255,  80,  80)   # blue   — estimated trajectory
_C_GT      = (80,  200,  80)   # green  — ground-truth trajectory
_C_CUR     = (0,   255, 255)   # cyan   — current position dot
_C_START   = (0,   255,   0)   # green  — start dot
_FONT      = cv2.FONT_HERSHEY_SIMPLEX


# ── helper: world XZ → canvas pixel ──────────────────────────────────────────
def _to_px(x: float, z: float,
           cx: float, cz: float,
           scale: float, H: int, W: int) -> tuple:
    """
    Project world (x, z) → canvas pixel (px, py).
    cx, cz : world coordinates mapped to canvas centre.
    Z forward = up on canvas.
    """
    px = int(W / 2 + (x - cx) * scale)
    py = int(H / 2 - (z - cz) * scale)
    return (int(np.clip(px, 0, W - 1)),
            int(np.clip(py, 0, H - 1)))


def _pts_to_px_array(xz_list: list, cx: float, cz: float,
                     scale: float, H: int, W: int) -> np.ndarray:
    """
    Convert list of (x, z) world points → Nx1x2 int32 array for polylines.
    Uses numpy vectorisation — much faster than per-point _to_px calls.
    """
    if len(xz_list) < 2:
        return None
    arr = np.array(xz_list, dtype=np.float32)   # Nx2
    px  = (W / 2 + (arr[:, 0] - cx) * scale).astype(np.int32)
    py  = (H / 2 - (arr[:, 1] - cz) * scale).astype(np.int32)
    px  = np.clip(px, 0, W - 1)
    py  = np.clip(py, 0, H - 1)
    return np.stack([px, py], axis=1).reshape(-1, 1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LiveVisualizer
# ─────────────────────────────────────────────────────────────────────────────
class LiveVisualizer:
    """
    Two OpenCV windows — features and trajectory — updated every N frames.

    Parameters
    ----------
    title        : base window title
    canvas_hw    : (height, width) in pixels for EACH window
    show         : False = headless, windows not opened
    decimate     : keep every Nth point for trajectory drawing (default 3).
                   Full history kept for auto-scale; only affects draw cost.
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

        # full history for running min/max (scale computation)
        self._xmin = self._xmax =  None
        self._zmin = self._zmax =  None

        # decimated path for drawing — updated every `decimate` calls
        self._est_draw: list = []   # list of (x, z)
        self._gt_draw:  list = []   # list of (x, z)

        # current position (always updated, for the live cyan dot)
        self._est_cur: tuple = (0.0, 0.0)
        self._gt_cur:  tuple = None

        # start position
        self._est_start: tuple = None

        self._call_count = 0   # counts update() calls for decimation
        self._stereo_mode = False

        if self.show:
            H, W = canvas_hw
            cv2.namedWindow(self._win_feat, cv2.WINDOW_NORMAL)
            cv2.namedWindow(self._win_traj, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self._win_feat, W, H)
            cv2.resizeWindow(self._win_traj, W, H)
            cv2.moveWindow(self._win_feat, 0,      50)
            cv2.moveWindow(self._win_traj, W + 10, 50)

    # ── public API ────────────────────────────────────────────────────────────

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
        """
        Update both windows.  Designed to be called every N frames from
        main.py — NOT every frame — to keep the VO loop fast.

        Parameters
        ----------
        img        : left camera frame — grayscale or BGR uint8
        pts_cur    : Nx2 float32 — current feature positions (green dots)
        pose       : 4×4 float64 T_world_cam
        gt_pose    : 4×4 float64 T_world_cam or None
        frame_id   : frame index for overlay text
        n_failures : cumulative failure count
        extra_info : optional one-line string (e.g. "scale=0.020")
        img_right  : right camera frame for stereo side-by-side display (optional)
        pts_right  : Nx2 float32 — right-frame feature positions (optional)
        """
        self._call_count += 1

        # ── update trajectory data ────────────────────────────────────────
        pos = pose[:3, 3]
        x, z = float(pos[0]), float(pos[2])
        self._est_cur = (x, z)

        if self._est_start is None:
            self._est_start = (x, z)

        # update running min/max (O(1) per call)
        if self._xmin is None:
            self._xmin = self._xmax = x
            self._zmin = self._zmax = z
        else:
            self._xmin = min(self._xmin, x)
            self._xmax = max(self._xmax, x)
            self._zmin = min(self._zmin, z)
            self._zmax = max(self._zmax, z)

        if gt_pose is not None:
            g = gt_pose[:3, 3]
            gx, gz = float(g[0]), float(g[2])
            self._gt_cur = (gx, gz)
            self._xmin = min(self._xmin, gx)
            self._xmax = max(self._xmax, gx)
            self._zmin = min(self._zmin, gz)
            self._zmax = max(self._zmax, gz)

        # decimated draw path — append every Nth call
        if self._call_count % self.decimate == 0:
            self._est_draw.append((x, z))
            if gt_pose is not None:
                self._gt_draw.append((gx, gz))

        # ── draw both windows ─────────────────────────────────────────────
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
        """Destroy both OpenCV windows."""
        if self.show:
            cv2.destroyWindow(self._win_feat)
            cv2.destroyWindow(self._win_traj)

    def reset(self) -> None:
        """Clear all history — call between sequences."""
        self._xmin = self._xmax = None
        self._zmin = self._zmax = None
        self._est_draw  = []
        self._gt_draw   = []
        self._est_cur   = (0.0, 0.0)
        self._gt_cur    = None
        self._est_start = None
        self._call_count = 0
        self._stereo_mode = False

    # ── window A: feature frame ───────────────────────────────────────────────

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

        # text overlay with dark background box
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

        # Stereo: side-by-side left | right
        cv2.putText(vis, "LEFT", (W - 42, H - 8),
                    _FONT, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
        vis_r = _render_frame(img_right, pts_right)
        n_r = 0 if pts_right is None else len(pts_right)
        cv2.putText(vis_r, f"RIGHT  n={n_r}", (5, H - 8),
                    _FONT, 0.42, _C_TRACKED, 1, cv2.LINE_AA)
        divider = np.full((H, 2, 3), 60, dtype=np.uint8)
        return np.hstack([vis, divider, vis_r])

    # ── window B: trajectory canvas ───────────────────────────────────────────

    def _draw_trajectory(self) -> np.ndarray:
        H, W = self.canvas_hw
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        # need at least one point
        if self._xmin is None:
            return canvas

        # ── auto-scale: fit all stored points ─────────────────────────────
        pad   = 0.15
        rx    = max(self._xmax - self._xmin, 0.5) * (1 + pad)
        rz    = max(self._zmax - self._zmin, 0.5) * (1 + pad)
        scale = min((W - 20) / rx, (H - 20) / rz)
        cx    = (self._xmin + self._xmax) / 2
        cz    = (self._zmin + self._zmax) / 2

        # ── metric grid ────────────────────────────────────────────────────
        grid_px = int(scale)   # pixels per 1 metre
        if grid_px >= 8:
            x0 = int(np.floor(self._xmin - 1))
            x1 = int(np.ceil(self._xmax  + 1))
            z0 = int(np.floor(self._zmin - 1))
            z1 = int(np.ceil(self._zmax  + 1))
            for xi in range(x0, x1 + 1):
                px = int(W / 2 + (xi - cx) * scale)
                if 0 <= px < W:
                    cv2.line(canvas, (px, 0), (px, H), (30, 30, 30), 1)
            for zi in range(z0, z1 + 1):
                pz = int(H / 2 - (zi - cz) * scale)
                if 0 <= pz < H:
                    cv2.line(canvas, (0, pz), (W, pz), (30, 30, 30), 1)

        # ── GT trajectory using polylines (fast) ──────────────────────────
        if len(self._gt_draw) > 1:
            pts = _pts_to_px_array(self._gt_draw, cx, cz, scale, H, W)
            if pts is not None:
                cv2.polylines(canvas, [pts], False, _C_GT, 1, cv2.LINE_AA)

        # ── estimated trajectory using polylines (fast) ───────────────────
        if len(self._est_draw) > 1:
            pts = _pts_to_px_array(self._est_draw, cx, cz, scale, H, W)
            if pts is not None:
                cv2.polylines(canvas, [pts], False, _C_EST, 2, cv2.LINE_AA)

        # ── start and current position markers ────────────────────────────
        if self._est_start is not None:
            sp = _to_px(*self._est_start, cx, cz, scale, H, W)
            cv2.circle(canvas, sp, 6, _C_START, -1)   # green = start

        cur_px = _to_px(*self._est_cur, cx, cz, scale, H, W)
        cv2.circle(canvas, cur_px, 6, _C_CUR, -1)     # cyan = current pos

        if self._gt_cur is not None:
            gp = _to_px(*self._gt_cur, cx, cz, scale, H, W)
            cv2.circle(canvas, gp, 4, _C_GT, -1)      # green = GT current

        # ── legend ────────────────────────────────────────────────────────
        scale_str = (f"1m={int(scale)}px" if scale >= 1
                     else f"1px={1/scale:.1f}m")
        info_lines = [
            ("EST",          _C_EST),
            ("GT",           _C_GT),
            ("Top-down X-Z", (180, 180, 180)),
            (scale_str,      (140, 140, 140)),
            (f"n={len(self._est_draw)*self.decimate}", (120, 120, 120)),
        ]
        box_h = 12 + len(info_lines) * 15
        cv2.rectangle(canvas, (0, 0), (125, box_h), (0, 0, 0),  -1)
        cv2.rectangle(canvas, (0, 0), (125, box_h), (50, 50, 50), 1)
        for i, (txt, col) in enumerate(info_lines):
            cv2.putText(canvas, txt, (4, 12 + i * 15),
                        _FONT, 0.40, col, 1, cv2.LINE_AA)

        return canvas


# ─────────────────────────────────────────────────────────────────────────────
# 2.  save_3d_trajectory  +  save_comparison_3d
# ─────────────────────────────────────────────────────────────────────────────

def save_3d_trajectory(
    est_poses: list,
    gt_poses:  list,
    title:     str  = "3D Trajectory",
    out_path:  str  = "traj_3d.png",
    align:     str  = "sim3",
    dpi:       int  = 130,
) -> None:
    """
    Save a dark-themed 3-D trajectory figure (PNG).

    Layout: left = full 3-D axes, right = three 2-D projections.
    GT plotted before estimated so axes auto-scale to include both.
    Equal aspect enforced on all 2-D panels.
    """
    est     = np.array([T[:3, 3] for T in est_poses], dtype=np.float64)
    gt_list = [T[:3, 3] for T in gt_poses if T is not None]
    gt      = np.array(gt_list, dtype=np.float64) if gt_list else None

    ate_val = float("nan")
    if align and gt is not None and len(gt) > 10:
        try:
            from evaluation.metrics import align_and_evaluate
            gt_T   = [T for T in gt_poses if T is not None]
            result = align_and_evaluate(list(est_poses), gt_T, align=align)
            est    = np.array(
                [T[:3, 3] for T in result["traj_aligned"]], dtype=np.float64)
            ate_val = result["ate_rmse"]
        except Exception:
            pass

    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor("#0e0e0e")

    # ── left: 3-D view ────────────────────────────────────────────────────
    ax3d = fig.add_subplot(121, projection="3d")
    ax3d.set_facecolor("#0e0e0e")
    ax3d.tick_params(colors="white", labelsize=7)
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#333333")

    if gt is not None:
        ax3d.plot(gt[:, 0], gt[:, 2], gt[:, 1],
                  color="#A5D6A7", lw=1.0, ls="--", alpha=0.85, label="GT")
        ax3d.scatter(*gt[0,  [0, 2, 1]], c="lime", s=20, zorder=4, marker="x")
        ax3d.scatter(*gt[-1, [0, 2, 1]], c="red",  s=20, zorder=4, marker="x")

    ax3d.plot(est[:, 0], est[:, 2], est[:, 1],
              color="#4FC3F7", lw=1.2, label="Estimated")
    ax3d.scatter(*est[0,  [0, 2, 1]], c="lime", s=40, zorder=5, label="Start")
    ax3d.scatter(*est[-1, [0, 2, 1]], c="red",  s=40, zorder=5, label="End")

    ax3d.set_xlabel("X [m]", color="white", fontsize=8)
    ax3d.set_ylabel("Z [m]", color="white", fontsize=8)
    ax3d.set_zlabel("Y [m]", color="white", fontsize=8)
    ax3d.set_title("3D view", color="white", fontsize=9)
    ax3d.legend(fontsize=7, facecolor="#222222", labelcolor="white",
                loc="upper left")

    # ── right: three 2-D projections ─────────────────────────────────────
    ax_xz = fig.add_subplot(322)
    ax_xy = fig.add_subplot(324)
    ax_yz = fig.add_subplot(326)

    proj_defs = [
        (ax_xz,
         est[:, 0], est[:, 2],
         gt[:, 0] if gt is not None else None,
         gt[:, 2] if gt is not None else None,
         "X [m]", "Z [m]", "Top-down  X–Z"),
        (ax_xy,
         est[:, 0], est[:, 1],
         gt[:, 0] if gt is not None else None,
         gt[:, 1] if gt is not None else None,
         "X [m]", "Y [m]", "Front     X–Y"),
        (ax_yz,
         est[:, 2], est[:, 1],
         gt[:, 2] if gt is not None else None,
         gt[:, 1] if gt is not None else None,
         "Z [m]", "Y [m]", "Side      Z–Y"),
    ]

    for ax, xe, ye, xg, yg, xl, yl, ttl in proj_defs:
        ax.set_facecolor("#111111")
        ax.tick_params(colors="white", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.set_xlabel(xl, color="white", fontsize=7)
        ax.set_ylabel(yl, color="white", fontsize=7)
        ax.set_title(ttl, color="white", fontsize=8)

        # GT first — axes scale to include it
        if xg is not None:
            ax.plot(xg, yg, color="#A5D6A7", lw=0.8, ls="--",
                    alpha=0.8, label="GT")
            ax.scatter(xg[0],  yg[0],  c="lime", s=18, zorder=6, marker="x")
            ax.scatter(xg[-1], yg[-1], c="red",  s=18, zorder=6, marker="x")

        ax.plot(xe, ye, color="#4FC3F7", lw=1.0, label="Est")
        ax.scatter(xe[0],  ye[0],  c="lime", s=22, zorder=7)
        ax.scatter(xe[-1], ye[-1], c="red",  s=22, zorder=7)

        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=6, facecolor="#222222", labelcolor="white",
                  loc="best")

    ate_str = (f"  |  ATE = {ate_val:.3f} m ({align.upper()})"
               if not np.isnan(ate_val) else "")
    fig.suptitle(f"{title}{ate_str}", color="white",
                 fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved 3D trajectory → {out_path}")


def save_comparison_3d(
    mono_poses:   list,
    stereo_poses: list,
    gt_poses:     list,
    seq_name:     str = "",
    out_path:     str = "comparison_3d.png",
    dpi:          int = 140,
) -> None:
    """
    Dark-themed 3-D figure comparing GT, Sim3-aligned mono VO, and
    SE3-aligned stereo VO in a single plot.  Required by project spec
    Section VI: "Visualize trajectories in 3D plots comparing VO
    (up-to-scale), stereo VO (metric), and ground truth."
    """
    from evaluation.metrics import align_and_evaluate

    gt_T   = [T for T in gt_poses if T is not None]
    gt     = np.array([T[:3, 3] for T in gt_T], dtype=np.float64) if gt_T else None

    def _align(poses, mode):
        # Pair by index — only frames where GT is not None
        paired_est, paired_gt = [], []
        for i, g in enumerate(gt_poses):
            if g is not None and i < len(poses):
                paired_est.append(poses[i])
                paired_gt.append(g)
        if len(paired_gt) < 10:
            return np.array([T[:3, 3] for T in poses], dtype=np.float64), float("nan")
        try:
            res = align_and_evaluate(paired_est, paired_gt, align=mode)
            s   = res.get("s", 1.0)
            R, t = res["R"], res["t"]
            # Apply alignment to ALL poses for visualization
            arr = np.array([s * R @ T[:3, 3] + t for T in poses], dtype=np.float64)
            return arr, res["ate_rmse"]
        except Exception:
            pass
        return np.array([T[:3, 3] for T in poses], dtype=np.float64), float("nan")

    mono_arr,   mono_ate   = _align(mono_poses,   "sim3")
    stereo_arr, stereo_ate = _align(stereo_poses, "se3")

    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor("#0e0e0e")

    # ── 3-D view ──────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(131, projection="3d")
    ax3.set_facecolor("#0e0e0e")
    ax3.tick_params(colors="white", labelsize=7)
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.fill = False; pane.set_edgecolor("#333333")

    if gt is not None:
        ax3.plot(gt[:, 0], gt[:, 2], gt[:, 1],
                 color="#A5D6A7", lw=1.2, ls="--", alpha=0.9, label="GT")
    ax3.plot(mono_arr[:, 0],   mono_arr[:, 2],   mono_arr[:, 1],
             color="#4FC3F7", lw=1.0,
             label=f"Mono VO Sim3  ATE={mono_ate:.3f}m")
    ax3.plot(stereo_arr[:, 0], stereo_arr[:, 2], stereo_arr[:, 1],
             color="#EF9A9A", lw=1.0,
             label=f"Stereo VO SE3  ATE={stereo_ate:.3f}m")
    ax3.set_xlabel("X [m]", color="white", fontsize=8)
    ax3.set_ylabel("Z [m]", color="white", fontsize=8)
    ax3.set_zlabel("Y [m]", color="white", fontsize=8)
    ax3.set_title("3-D view", color="white", fontsize=9)
    ax3.legend(fontsize=7, facecolor="#222222", labelcolor="white",
               loc="upper left")

    # ── top-down x-y ──────────────────────────────────────────────────────
    ax_xy = fig.add_subplot(132)
    ax_xy.set_facecolor("#111111")
    ax_xy.tick_params(colors="white", labelsize=7)
    for s in ax_xy.spines.values(): s.set_edgecolor("#333333")
    if gt is not None:
        ax_xy.plot(gt[:, 0], gt[:, 1], color="#A5D6A7", lw=1.0,
                   ls="--", alpha=0.9, label="GT")
    ax_xy.plot(mono_arr[:, 0],   mono_arr[:, 1],   color="#4FC3F7",
               lw=0.9, label="Mono (Sim3)")
    ax_xy.plot(stereo_arr[:, 0], stereo_arr[:, 1], color="#EF9A9A",
               lw=0.9, label="Stereo (SE3)")
    ax_xy.set_xlabel("x [m]", color="white", fontsize=8)
    ax_xy.set_ylabel("y [m]", color="white", fontsize=8)
    ax_xy.set_title("Top-down  x–y", color="white", fontsize=9)
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.legend(fontsize=7, facecolor="#222222", labelcolor="white")

    # ── z (height) over frames ────────────────────────────────────────────
    ax_z = fig.add_subplot(133)
    ax_z.set_facecolor("#111111")
    ax_z.tick_params(colors="white", labelsize=7)
    for s in ax_z.spines.values(): s.set_edgecolor("#333333")
    if gt is not None:
        ax_z.plot(gt[:, 2], color="#A5D6A7", lw=0.8, ls="--",
                  alpha=0.9, label="GT z")
    ax_z.plot(mono_arr[:, 2],   color="#4FC3F7", lw=0.8,
              label="Mono z (Sim3)")
    ax_z.plot(stereo_arr[:, 2], color="#EF9A9A", lw=0.8,
              label="Stereo z (SE3)")
    ax_z.set_xlabel("frame", color="white", fontsize=8)
    ax_z.set_ylabel("z / height [m]", color="white", fontsize=8)
    ax_z.set_title("Height over time", color="white", fontsize=9)
    ax_z.legend(fontsize=7, facecolor="#222222", labelcolor="white")

    title_str = (f"Mono vs Stereo VO — {seq_name}  |  "
                 f"Mono ATE={mono_ate:.3f}m (Sim3)  "
                 f"Stereo ATE={stereo_ate:.3f}m (SE3)")
    fig.suptitle(title_str, color="white", fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved comparison 3D → {out_path}")
