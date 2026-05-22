# Usage:
#   python main.py               # runs all sequences with live display
#   SHOW_VIS=0 python main.py    # headless — saves PNGs only

import csv
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.data_loader        import load_run_config, TUMVILoader
from mono_vo.pipeline        import MonoVO, MonoVOConfig
from mono_vo.feature_tracker import FeatureConfig
from stereo_vo.pipeline      import StereoVO, StereoVOConfig
from stereo_vo.disparity     import DisparityConfig
from evaluation.metrics      import align_and_evaluate, start_end_drift, rpe as rpe_frames
from utils.print_utils       import (print_camera_intrinsics, ensure_dir,
                                     print_stereo_params,
                                     print_stereo_extrinsics,
                                     print_sequence_summary)
from utils.visualizer        import LiveVisualizer, save_3d_trajectory, save_comparison_3d


class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
    def flush(self):
        for s in self._streams:
            s.flush()

os.makedirs("outputs", exist_ok=True)
_log_file = open("outputs/run.log", "w", buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)

np.random.seed(42)
print(f"[Reproducibility] np.random.seed(42)  |  Python {__import__('sys').version.split()[0]}  |  NumPy {np.__version__}")

_SHOW = os.environ.get("SHOW_VIS", "0") != "0"

SEQUENCES = [
    "config/tumvi_room2.yaml",
    "config/tumvi_corridor3.yaml",
    "config/tumvi_outdoors5.yaml",
]

FULL_GT = {"room2"}

# Larger LK window + extra pyramid level for TUM-VI's rotation-heavy handheld motion.
_FEAT_MONO = FeatureConfig(
    method         = "orb",
    max_features   = 3000,
    lk_win_size    = 25,
    lk_max_level   = 4,
    grid_rows      = 5,
    grid_cols      = 5,
    fast_threshold = 15,
)


MONO_CONFIGS = {
    "room2": MonoVOConfig(
        feature               = _FEAT_MONO,
        min_tracked_pts       = 50,
        max_map_pts           = 600,
        min_parallax_px       = 20.0,
        max_parallax_px       = 40.0,
        expected_depth        = 2.0,
        scene_depth_lo_m      = 0.6,    # exclude close VD pts hard to track under rotation
        pnp_min_inliers       = 12,
        pnp_ransac_th         = 6.0,
        reproj_thresh         = 4.0,
        use_ba                = True,
        max_velocity          = 0.5,
        kf_min_parallax_px    = 15.0,
        kf_max_parallax_px    = 40.0,
        kf_min_baseline_ratio = 10.0,   # large threshold disables KF triangulation:
                                        # any KF angle causes scale collapse on this
                                        # rotation-dominant sequence
        use_clahe             = True,   # improves PnP conditioning; init t=0.045m vs 0.013m
        max_cvm_frames        = 3,
        verbose               = False,
    ),
    "corridor3": MonoVOConfig(
        feature               = _FEAT_MONO,
        min_tracked_pts       = 70,
        max_map_pts           = 800,
        min_parallax_px       = 12.0,
        max_parallax_px       = 60.0,
        expected_depth        = 5.0,
        pnp_min_inliers       = 15,
        pnp_ransac_th         = 6.0,
        reproj_thresh         = 4.0,
        use_ba                = True,
        max_velocity          = 1.0,
        kf_min_parallax_px    = 10.0,
        kf_max_parallax_px    = 80.0,
        kf_min_baseline_ratio = 0.03,
        max_cvm_frames        = 3,
        use_clahe             = True,   # enhance contrast for homogeneous walls
        use_e_reinit          = False,  # 180° turn: E-matrix cheirality picks wrong rotation
        depth_percentile      = 50,     # median — compromise between drift and failure rate
        verbose               = False,
    ),
    "outdoors5": MonoVOConfig(
        feature               = _FEAT_MONO,
        min_tracked_pts       = 50,
        max_map_pts           = 600,
        min_parallax_px       = 15.0,
        max_parallax_px       = 80.0,
        expected_depth        = 5.0,
        scene_depth_lo_m      = 1.0,    # exclude ground/nearby from VD depth estimate
        pnp_min_inliers       = 12,
        pnp_ransac_th         = 6.0,
        reproj_thresh         = 4.0,
        use_ba                = True,
        max_velocity          = 2.0,
        kf_min_parallax_px    = 12.0,
        kf_max_parallax_px    = 100.0,
        kf_min_baseline_ratio = 0.03,
        max_cvm_frames        = 3,
        use_clahe             = True,   # outdoor lighting varies; CLAHE helps feature visibility
        depth_percentile      = 75,     # deeper reference → smaller scale → less expressed drift
        verbose               = False,
    ),
}

_FEAT_STEREO = FeatureConfig(
    method       = "orb",
    max_features = 3000,
    lk_win_size  = 21,
    lk_max_level = 3,
    grid_rows    = 5,
    grid_cols    = 5,
)


STEREO_CONFIGS = {
    "room2": StereoVOConfig(
        feature         = _FEAT_STEREO,
        disparity       = DisparityConfig(
            num_disparities = 64,
            block_size      = 11,
            min_depth       = 0.5,
            max_depth       = 5.0,
            min_disparity   = 1.5,
            patch_radius    = 3,
        ),
        min_tracked_pts = 150,
        max_map_pts     = 600,
        pnp_min_inliers = 20,
        pnp_ransac_th   = 2.0,
        reproj_thresh   = 2.0,
        use_ba          = True,
        max_velocity    = 0.5,
        use_clahe       = True,   # CLAHE reduces failures 18→3 and ATE 0.785→0.439m on room2
        verbose         = False,
    ),
    "corridor3": StereoVOConfig(
        feature         = _FEAT_STEREO,
        disparity       = DisparityConfig(
            num_disparities = 64,
            block_size      = 21,
            min_depth       = 0.5,
            max_depth       = 10.0,
            min_disparity   = 0.5,
            patch_radius    = 3,
            disp12_diff     = 3,
            uniqueness      = 10,
            min_disp_pixels = 6,
            wls_lambda      = 12000,    # more smoothing for corridor walls
            wls_sigma       = 1.5,
        ),
        min_tracked_pts  = 120,
        max_map_pts      = 1000,
        pnp_min_inliers  = 15,
        pnp_ransac_th    = 3.0,
        reproj_thresh    = 2.5,
        use_ba           = True,
        max_velocity     = 1.0,
        use_depth_update = False,
        use_clahe        = True,
        verbose          = False,
    ),
    "outdoors5": StereoVOConfig(
        feature         = _FEAT_STEREO,
        disparity       = DisparityConfig(
            num_disparities = 128,
            block_size      = 11,
            min_depth       = 0.5,
            max_depth       = 20.0,
            min_disparity   = 1.0,
            patch_radius    = 3,
            min_disp_pixels = 5,
            wls_lambda      = 8000,
            wls_sigma       = 1.2,
        ),
        min_tracked_pts  = 120,
        max_map_pts      = 1200,
        pnp_min_inliers  = 15,
        pnp_ransac_th    = 1.5,
        reproj_thresh    = 1.5,
        use_ba           = True,
        max_velocity     = 2.0,
        use_depth_update = False,
        verbose          = False,
    ),
}



def collect_gt(loader, poses):
    """Return (est_poses, gt_poses) aligned by frame index."""
    gt, est = [], []
    for i, frame in enumerate(loader):
        if i >= len(poses):
            break
        if frame.T_world_cam0 is not None:
            gt.append(frame.T_world_cam0)
            est.append(poses[i])
    return est, gt


def compute_rpe_d1_on_consecutive_gt(loader, poses):
    """RPE d=1 on consecutive-GT segments only; skips mocap gaps in corridor3/outdoors5."""
    all_est, all_gt, all_idx = [], [], []
    for i, frame in enumerate(loader):
        if i >= len(poses):
            break
        if frame.T_world_cam0 is not None:
            all_est.append(poses[i])
            all_gt.append(frame.T_world_cam0)
            all_idx.append(i)

    segments, seg_start = [], 0
    for k in range(1, len(all_idx)):
        if all_idx[k] - all_idx[k - 1] > 1:
            if k - seg_start >= 2:
                segments.append((seg_start, k))
            seg_start = k
    if len(all_idx) - seg_start >= 2:
        segments.append((seg_start, len(all_idx)))

    trans_all, rot_all = [], []
    for s, e in segments:
        r = rpe_frames(all_est[s:e], all_gt[s:e], delta=1)
        trans_all.extend(r["trans_errors"].tolist())
        rot_all.extend(r["rot_errors"].tolist())

    if not trans_all:
        return float("nan"), float("nan")
    te = np.array(trans_all)
    re = np.array(rot_all)
    return float(np.sqrt((te ** 2).mean())), float(np.sqrt((re ** 2).mean()))


def compute_local_ate_blocks(loader, poses, align_mode="se3"):
    """ATE within each mocap block (start + end) independently, for partial-GT sequences."""
    all_est, all_gt, all_idx = [], [], []
    for i, frame in enumerate(loader):
        if i >= len(poses):
            break
        if frame.T_world_cam0 is not None:
            all_est.append(poses[i])
            all_gt.append(frame.T_world_cam0)
            all_idx.append(i)

    if len(all_idx) < 4:
        return {"start_ate": float("nan"), "end_ate": float("nan"),
                "start_n": len(all_idx), "end_n": 0}

    gaps = [(all_idx[k] - all_idx[k - 1], k) for k in range(1, len(all_idx))]
    split = max(gaps, key=lambda x: x[0])[1]

    def _block_ate(est, gt):
        if len(gt) < 10:
            return float("nan")
        try:
            r = align_and_evaluate(est, gt, align=align_mode)
            return r["ate_rmse"]
        except Exception:
            return float("nan")

    return {
        "start_ate": _block_ate(all_est[:split], all_gt[:split]),
        "end_ate":   _block_ate(all_est[split:], all_gt[split:]),
        "start_n":   split,
        "end_n":     len(all_idx) - split,
    }


def collect_gt_timestamps(loader, poses, timestamps):
    ts = []
    for i, frame in enumerate(loader):
        if i >= len(poses):
            break
        if frame.T_world_cam0 is not None:
            ts.append(timestamps[i])
    return np.array(ts)



def run_mono(loader, seq, cfg, show: bool = True):
    """Run monocular VO with live side-by-side display."""
    print(f"\nRunning monocular VO ...")
    vo = MonoVO(loader.calib.cam0_rect, cfg)

    vis = LiveVisualizer(
        title     = f"Mono VO — {seq}",
        canvas_hw = (512, 512),
        show      = show,
    )

    t0 = time.time()
    for frame in loader:
        img_rect = frame.img_left_rect
        vo.process(img_rect, frame.timestamp)

        if vo._initialised:
            vis.update(
                img        = img_rect,
                pts_cur    = vo.cur_pts,
                pose       = vo.trajectory[1][-1],
                gt_pose    = frame.T_world_cam0,
                frame_id   = vo._frame_id,
                n_failures = vo.n_failures,
                extra_info = f"depth={cfg.expected_depth:.1f}m",
            )

        frame.release()

    elapsed = time.time() - t0
    vis.close()

    fps = len(loader) / elapsed
    print(f"  Done: {len(loader)} frames  {elapsed:.1f}s  ({fps:.1f} fps)  "
          f"failures={vo.n_failures}  "
          f"[LK={vo.n_lk_fails}  PnP={vo.n_pnp_fails}  vel={vo.n_vel_fails}]  "
          f"kf_updates={vo.n_kf_updates}  "
          f"vd_reinit={vo.n_vd_reinits}  e_reinit={vo.n_e_reinits}")
    return vo, elapsed


def run_stereo(loader, seq, cfg, show: bool = True):
    """Run stereo VO with live side-by-side display."""
    print(f"\nRunning stereo VO ...")
    vo = StereoVO(loader.calib, cfg)

    vis = LiveVisualizer(
        title     = f"Stereo VO — {seq}",
        canvas_hw = (512, 512),
        show      = show,
    )

    t0 = time.time()
    for frame in loader:
        vo.process(frame.img_left, frame.img_right, frame.timestamp)

        if vo._initialised:
            vis.update(
                img        = frame.img_left,
                pts_cur    = vo.cur_pts,
                pose       = vo.trajectory[1][-1],
                gt_pose    = frame.T_world_cam0,
                frame_id   = vo._frame_id,
                n_failures = vo.n_failures,
                extra_info = "scale=1.000 metric",
                img_right  = vo.cur_right_img,
                pts_right  = vo.cur_pts_right,
            )

        frame.release()

    elapsed = time.time() - t0
    vis.close()

    fps = len(loader) / elapsed
    print(f"  Done: {len(loader)} frames  {elapsed:.1f}s  "
          f"({fps:.1f} fps)  failures={vo.n_failures}  reinits={vo.n_reinits}")
    return vo, elapsed



def save_mono_plot(cfg, loader, mono_vo, mono_time, out_dir, full_gt: bool = True):
    ts  = np.array(mono_vo.trajectory[0]); ts -= ts[0]
    fps = len(loader) / mono_time if mono_time > 0 else 0.0

    all_gt_poses = [f.T_world_cam0 for f in loader if f.T_world_cam0 is not None]
    gt_arr = np.array([T[:3, 3] for T in all_gt_poses]) if all_gt_poses else np.empty((0, 3))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0e0e0e")
    for _a in (ax, ax2):
        _a.set_facecolor("#111111")
        _a.tick_params(colors="white", labelsize=7)
        for _s in _a.spines.values(): _s.set_edgecolor("#333333")

    if full_gt:
        mono_aligned, ate_val, ate_errors = None, float("nan"), None
        est_poses, gt_poses = collect_gt(loader, mono_vo.trajectory[1])
        if len(gt_poses) > 10:
            result       = align_and_evaluate(est_poses, gt_poses, align="sim3")
            s            = result.get("s", 1.0)
            R_a, t_a     = result["R"], result["t"]
            mono_aligned = np.array([s * R_a @ T[:3, 3] + t_a
                                     for T in mono_vo.trajectory[1]])
            ate_val      = result["ate_rmse"]
            ate_errors   = result.get("errors", None)

        if len(gt_arr):
            ax.plot(gt_arr[:, 0], gt_arr[:, 1], color="#A5D6A7", lw=1.0, ls="--",
                    alpha=0.85, label="GT")
            ax.scatter(gt_arr[0,  0], gt_arr[0,  1], c="lime", s=100, zorder=6,
                       marker="*", label="GT start")
            ax.scatter(gt_arr[-1, 0], gt_arr[-1, 1], c="gold", s=100, zorder=6,
                       marker="*", label="GT end")
        if mono_aligned is not None:
            ax.plot(mono_aligned[:, 0], mono_aligned[:, 1], color="#4FC3F7", lw=1.2,
                    label="Mono VO")
            ax.scatter(mono_aligned[0,  0], mono_aligned[0,  1], c="blue", s=60,
                       zorder=5, marker="o", label="est start")
            ax.scatter(mono_aligned[-1, 0], mono_aligned[-1, 1], c="red",  s=60,
                       zorder=5, marker="D", label="est end")
        ax.set_title("Top-down x–y  (Sim3-aligned)", color="white")

        if ate_errors is not None:
            aligned_ts = collect_gt_timestamps(loader, mono_vo.trajectory[1], ts)
            if len(aligned_ts):
                aligned_ts = aligned_ts - aligned_ts[0]
            n = min(len(aligned_ts), len(ate_errors))
            if n:
                ax2.plot(aligned_ts[:n], ate_errors[:n], color="#4FC3F7", lw=0.8, label="Mono ATE")
            ax2.set_xlabel("time [s]", color="white")
            ax2.set_ylabel("ATE [m]", color="white")
            ax2.set_title("ATE over time  (Sim3-aligned)", color="white")
            ax2.set_ylim(bottom=0)
            ax2.legend(fontsize=8, facecolor="#222222", labelcolor="white")

        plt.suptitle(f"Monocular VO — {cfg.sequence_name}  |  "
                     f"failures={mono_vo.n_failures}   ATE={ate_val:.3f}m   fps={fps:.0f}",
                     fontsize=11, color="white")
    else:
        drift_m = float("nan")
        if len(all_gt_poses) >= 2:
            drift_m = start_end_drift(mono_vo.trajectory[1],
                                      [all_gt_poses[0], all_gt_poses[-1]])

        T_a = all_gt_poses[0] @ np.linalg.inv(mono_vo.trajectory[1][0]) \
              if all_gt_poses else np.eye(4)
        mono_sa = np.array([(T_a @ T)[:3, 3] for T in mono_vo.trajectory[1]])

        ax.plot(mono_sa[:, 0], mono_sa[:, 1], color="#4FC3F7", lw=1.2,
                label=f"Mono VO (start-aligned, drift={drift_m:.2f}m)")
        ax.scatter(mono_sa[0,  0], mono_sa[0,  1], c="blue", s=60, zorder=5, marker="o",
                   label="est start")
        ax.scatter(mono_sa[-1, 0], mono_sa[-1, 1], c="red",  s=60, zorder=5, marker="D",
                   label=f"est end  (drift={drift_m:.2f}m)")
        if len(gt_arr) >= 1:
            ax.scatter(gt_arr[0, 0], gt_arr[0, 1], c="lime",  s=100, zorder=6, marker="*",
                       label="GT start")
        if len(gt_arr) >= 2:
            ax.scatter(gt_arr[-1, 0], gt_arr[-1, 1], c="gold", s=100, zorder=6, marker="*",
                       label="GT end")
        ax.set_title("Top-down x–y  (start-aligned to GT)", color="white")

        ax2.plot(ts, mono_sa[:, 0], color="#4FC3F7", lw=0.8, label="x")
        ax2.plot(ts, mono_sa[:, 1], color="#EF9A9A", lw=0.8, label="y")
        ax2.set_xlabel("time [s]", color="white")
        ax2.set_ylabel("position [m]", color="white")
        ax2.set_title("x–y position over time  (start-aligned)", color="white")
        ax2.legend(fontsize=8, facecolor="#222222", labelcolor="white")

        plt.suptitle(f"Monocular VO — {cfg.sequence_name}  |  "
                     f"failures={mono_vo.n_failures}   "
                     f"start-end drift={drift_m:.3f}m   fps={fps:.0f}",
                     fontsize=11, color="white")

    ax.set_xlabel("x [m]", color="white")
    ax.set_ylabel("y [m]", color="white")
    ax.legend(fontsize=8, facecolor="#222222", labelcolor="white", loc="upper left")
    ax.set_aspect("equal")
    plt.tight_layout()
    path = os.path.join(out_dir, "mono_traj.png")
    plt.savefig(path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight"); plt.close()
    print(f"  Saved {path}")


def save_stereo_plot(cfg, loader, stereo_vo, stereo_time, out_dir, full_gt: bool = True):
    ts  = np.array(stereo_vo.trajectory[0]); ts -= ts[0]
    fps = len(loader) / stereo_time if stereo_time > 0 else 0.0

    all_gt_poses = [f.T_world_cam0 for f in loader if f.T_world_cam0 is not None]
    gt_arr = np.array([T[:3, 3] for T in all_gt_poses]) if all_gt_poses else np.empty((0, 3))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0e0e0e")
    for _a in (ax, ax2):
        _a.set_facecolor("#111111")
        _a.tick_params(colors="white", labelsize=7)
        for _s in _a.spines.values(): _s.set_edgecolor("#333333")

    if full_gt:
        est_poses, gt_poses = collect_gt(loader, stereo_vo.trajectory[1])
        stereo_aligned, ate_val, ate_errors = None, float("nan"), None
        if len(gt_poses) > 10:
            result         = align_and_evaluate(est_poses, gt_poses, align="se3")
            s              = result.get("s", 1.0)
            R_a, t_a       = result["R"], result["t"]
            stereo_aligned = np.array([s * R_a @ T[:3, 3] + t_a
                                       for T in stereo_vo.trajectory[1]])
            ate_val        = result["ate_rmse"]
            ate_errors     = result.get("errors", None)

        if stereo_aligned is not None:
            if len(gt_arr):
                ax.plot(gt_arr[:, 0], gt_arr[:, 1], color="#A5D6A7", lw=1.0, ls="--",
                        alpha=0.85, label="GT")
                ax.scatter(gt_arr[0,  0], gt_arr[0,  1], c="lime", s=100, zorder=6,
                           marker="*", label="GT start")
                ax.scatter(gt_arr[-1, 0], gt_arr[-1, 1], c="gold", s=100, zorder=6,
                           marker="*", label="GT end")
            ax.plot(stereo_aligned[:, 0], stereo_aligned[:, 1], color="#EF9A9A", lw=1.0,
                    label="Stereo VO")
            ax.scatter(stereo_aligned[0,  0], stereo_aligned[0,  1], c="blue", s=60,
                       zorder=5, marker="o", label="est start")
            ax.scatter(stereo_aligned[-1, 0], stereo_aligned[-1, 1], c="red",  s=60,
                       zorder=5, marker="D", label="est end")
            ax.set_title("Top-down x–y  (SE3-aligned, metric)", color="white")
        else:
            raw = np.array([T[:3, 3] for T in stereo_vo.trajectory[1]])
            ax.plot(raw[:, 0], raw[:, 1], color="#EF9A9A", lw=1.0, label="Stereo VO (raw)")
            ax.set_title("Top-down x–y  (raw)", color="white")

        if ate_errors is not None:
            aligned_ts = collect_gt_timestamps(loader, stereo_vo.trajectory[1], ts)
            if len(aligned_ts):
                aligned_ts = aligned_ts - aligned_ts[0]
            n = min(len(aligned_ts), len(ate_errors))
            if n:
                ax2.plot(aligned_ts[:n], ate_errors[:n], color="#EF9A9A", lw=0.8, label="Stereo ATE")
            ax2.set_xlabel("time [s]", color="white")
            ax2.set_ylabel("ATE [m]", color="white")
            ax2.set_title("ATE over time  (SE3-aligned)", color="white")
            ax2.set_ylim(bottom=0)
        else:
            raw = np.array([T[:3, 3] for T in stereo_vo.trajectory[1]])
            ax2.plot(ts, raw[:, 0], color="#4FC3F7", lw=0.8, label="x")
            ax2.plot(ts, raw[:, 1], color="#EF9A9A", lw=0.8, label="y")
            ax2.set_xlabel("time [s]", color="white")
            ax2.set_ylabel("position [m]", color="white")
            ax2.set_title("Raw x-y over time", color="white")

        plt.suptitle(f"Stereo VO — {cfg.sequence_name}  |  "
                     f"failures={stereo_vo.n_failures}   ATE={ate_val:.3f}m   "
                     f"scale=1.000 metric   fps={fps:.0f}", fontsize=11, color="white")
    else:
        drift_m = float("nan")
        if len(all_gt_poses) >= 2:
            drift_m = start_end_drift(stereo_vo.trajectory[1],
                                      [all_gt_poses[0], all_gt_poses[-1]])

        T_a = all_gt_poses[0] @ np.linalg.inv(stereo_vo.trajectory[1][0]) \
              if all_gt_poses else np.eye(4)
        stereo_sa = np.array([(T_a @ T)[:3, 3] for T in stereo_vo.trajectory[1]])

        ax.plot(stereo_sa[:, 0], stereo_sa[:, 1], color="#EF9A9A", lw=1.0,
                label=f"Stereo VO (start-aligned, drift={drift_m:.2f}m)")
        ax.scatter(stereo_sa[0,  0], stereo_sa[0,  1], c="blue", s=60, zorder=5, marker="o",
                   label="est start")
        ax.scatter(stereo_sa[-1, 0], stereo_sa[-1, 1], c="red",  s=60, zorder=5, marker="D",
                   label=f"est end  (drift={drift_m:.2f}m)")
        if len(gt_arr) >= 1:
            ax.scatter(gt_arr[0, 0], gt_arr[0, 1], c="lime",  s=100, zorder=6, marker="*",
                       label="GT start")
        if len(gt_arr) >= 2:
            ax.scatter(gt_arr[-1, 0], gt_arr[-1, 1], c="gold", s=100, zorder=6, marker="*",
                       label="GT end")
        ax.set_title("Top-down x–y  (start-aligned to GT)", color="white")

        ax2.plot(ts, stereo_sa[:, 0], color="#EF9A9A", lw=0.8, label="x")
        ax2.plot(ts, stereo_sa[:, 1], color="#4FC3F7", lw=0.8, label="y")
        ax2.set_xlabel("time [s]", color="white")
        ax2.set_ylabel("position [m]", color="white")
        ax2.set_title("x–y position over time  (start-aligned)", color="white")

        plt.suptitle(f"Stereo VO — {cfg.sequence_name}  |  "
                     f"failures={stereo_vo.n_failures}   "
                     f"start-end drift={drift_m:.3f}m   fps={fps:.0f}",
                     fontsize=11, color="white")

    ax.set_xlabel("x [m]", color="white")
    ax.set_ylabel("y [m]", color="white")
    ax.legend(fontsize=8, facecolor="#222222", labelcolor="white", loc="upper left")
    ax.set_aspect("equal")
    ax2.legend(fontsize=8, facecolor="#222222", labelcolor="white")
    plt.tight_layout()
    path = os.path.join(out_dir, "stereo_traj.png")
    plt.savefig(path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight"); plt.close()
    print(f"  Saved {path}")


def save_stereo_sample_visuals(loader, stereo_cfg, out_dir, n_samples: int = 3) -> None:
    """Sample n_samples frames and save rectified pair, disparity map, depth map, and 3D scatter."""
    from stereo_vo.disparity import DisparityComputer
    from mpl_toolkits.mplot3d import Axes3D   # registers 3D projection

    disp_cmp  = DisparityComputer(loader.calib, stereo_cfg.disparity)
    n_frames  = len(loader._frames)
    fracs     = [0.10, 0.50, 0.90][:n_samples]
    indices   = [min(int(n_frames * f), n_frames - 1) for f in fracs]

    print(f"\n  [Stereo visuals] saving disparity/depth/pointcloud for "
          f"{len(indices)} sample frames ...")

    for j, fidx in enumerate(indices):
        frame = loader._frames[fidx]
        img_l = frame.img_left
        img_r = frame.img_right
        if img_r is None:
            continue

        rect_l, rect_r = loader.calib.rectify(img_l, img_r)
        disp  = disp_cmp.compute(rect_l, rect_r, rectified=True)
        valid = disp > disp_cmp.cfg.min_disparity
        depth = disp_cmp.disparity_to_depth(disp)

        h, w = rect_l.shape[:2]
        gx = np.linspace(20, w - 20, 40, dtype=int)
        gy = np.linspace(20, h - 20, 30, dtype=int)
        gxx, gyy = np.meshgrid(gx, gy)
        pts2d_grid = np.stack([gxx.ravel(), gyy.ravel()], axis=1).astype(np.float32)
        pts3d, _, _ = disp_cmp.unproject_points(pts2d_grid, disp)

        n = j + 1

        pair_bgr = cv2.cvtColor(np.hstack([rect_l, rect_r]), cv2.COLOR_GRAY2BGR)
        for row in range(0, h, h // 8):
            cv2.line(pair_bgr, (0, row), (2 * w - 1, row), (0, 200, 0), 1)
        cv2.putText(pair_bgr, "LEFT (rectified)", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(pair_bgr, "RIGHT (rectified)", (w + 10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(out_dir, f"rectified_pair_{n}.png"), pair_bgr)

        disp_u8 = np.zeros(disp.shape, np.uint8)
        if valid.any():
            dlo, dhi = disp[valid].min(), disp[valid].max()
            disp_u8[valid] = np.clip(
                (disp[valid] - dlo) / max(dhi - dlo, 1e-3) * 255, 0, 255
            ).astype(np.uint8)
        disp_col = cv2.applyColorMap(disp_u8, cv2.COLORMAP_JET)
        if valid.any():
            cv2.putText(disp_col, f"disp {dlo:.1f}px", (5, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(disp_col, f"max {dhi:.1f}px", (5, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(out_dir, f"disparity_map_{n}.png"), disp_col)

        valid_d = depth > 0
        depth_u8 = np.zeros(depth.shape, np.uint8)
        if valid_d.any():
            zlo, zhi = depth[valid_d].min(), depth[valid_d].max()
            depth_u8[valid_d] = np.clip(
                (depth[valid_d] - zlo) / max(zhi - zlo, 1e-3) * 255, 0, 255
            ).astype(np.uint8)
        depth_col = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
        if valid_d.any():
            cv2.putText(depth_col, f"depth {zlo:.2f}m", (5, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(depth_col, f"max {zhi:.2f}m", (5, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(out_dir, f"depth_map_{n}.png"), depth_col)

        if len(pts3d) > 0:
            fig = plt.figure(figsize=(8, 6), facecolor="#111111")
            ax  = fig.add_subplot(111, projection="3d")
            ax.set_facecolor("#111111")
            sc  = ax.scatter(
                pts3d[:, 0], pts3d[:, 2], -pts3d[:, 1],
                c=pts3d[:, 2], cmap="jet", s=4, alpha=0.8,
            )
            plt.colorbar(sc, ax=ax, label="Depth Z (m)", shrink=0.6)
            ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_zlabel("-Y (m)")
            ax.set_title(f"3D Point Cloud  frame {fidx}", color="white")
            ax.tick_params(colors="white"); ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white"); ax.zaxis.label.set_color("white")
            ax.view_init(elev=-10, azim=-80)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"pointcloud_{n}.png"),
                        dpi=100, facecolor=fig.get_facecolor())
            plt.close()

        n_pts  = len(pts3d)
        d_info = (f"disp=[{disp[valid].min():.1f},{disp[valid].max():.1f}]px  "
                  f"depth=[{depth[valid_d].min():.2f},{depth[valid_d].max():.2f}]m"
                  if valid.any() and valid_d.any() else "no valid disparity")
        print(f"    frame {fidx}: {n_pts} 3D pts  {d_info}")
        frame.release()


def save_feature_sample_visuals(
    loader,
    feat_cfg,
    out_dir:    str,
    n_samples:  int  = 4,
    use_clahe:  bool = False,
) -> None:
    """Save feature_samples.png: ORB matching (row 0) and LK optical-flow (row 1)."""
    from mono_vo.feature_tracker import FeatureTracker

    tracker = FeatureTracker(feat_cfg)
    clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None

    n_frames = len(loader._frames)
    fracs    = np.linspace(0.10, 0.85, n_samples)
    indices  = [min(int(n_frames * f), n_frames - 2) for f in fracs]

    fig, axes = plt.subplots(2, n_samples, figsize=(6 * n_samples, 9))
    fig.patch.set_facecolor("#0e0e0e")

    orb = cv2.ORB_create(feat_cfg.max_features)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    for col, fidx in enumerate(indices):
        frame0 = loader._frames[fidx]
        frame1 = loader._frames[fidx + 1]
        img0   = frame0.img_left
        img1   = frame1.img_left

        if clahe_op is not None:
            img0 = clahe_op.apply(img0)
            img1 = clahe_op.apply(img1)

        kp0, des0 = orb.detectAndCompute(img0, None)
        kp1, des1 = orb.detectAndCompute(img1, None)

        n_matches  = 0
        match_img  = np.hstack([img0, img1])   # fallback: plain side-by-side
        if des0 is not None and des1 is not None and len(des0) >= 2 and len(des1) >= 2:
            raw  = bf.knnMatch(des0, des1, k=2)
            good = [m for m, n in raw
                    if len((m, n)) == 2 and m.distance < 0.75 * n.distance]
            n_matches = len(good)
            match_img = cv2.drawMatches(
                img0, kp0, img1, kp1, good[:80], None,
                matchColor        = (0, 200, 255),
                singlePointColor  = (80, 80, 80),
                flags             = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)

        ax0 = axes[0, col]
        ax0.imshow(match_img, cmap="gray" if match_img.ndim == 2 else None)
        ax0.set_title(f"Frame {fidx}  |  ORB: {n_matches} matches",
                      color="white", fontsize=8)
        ax0.axis("off")

        pts0 = tracker.detect_grid(img0)
        pts0_good, pts1_good = tracker.track_optical_flow(img0, img1, pts0)

        h, w   = img0.shape[:2]
        lk_img = np.zeros((h, w * 2, 3), np.uint8)
        lk_img[:, :w]  = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        lk_img[:, w:]  = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

        for p0, p1 in zip(pts0_good, pts1_good):
            x0, y0 = int(p0[0]),     int(p0[1])
            x1, y1 = int(p1[0] + w), int(p1[1])
            cv2.circle(lk_img, (x0, y0), 3, (0, 200, 255), -1)
            cv2.circle(lk_img, (x1, y1), 3, (0, 255, 100), -1)
            cv2.line(lk_img, (x0, y0), (x1, y1), (200, 200, 0), 1)

        cv2.line(lk_img, (w, 0), (w, h - 1), (180, 180, 180), 1)

        ax1 = axes[1, col]
        ax1.imshow(cv2.cvtColor(lk_img, cv2.COLOR_BGR2RGB))
        ax1.set_title(f"Frame {fidx}  |  LK: {len(pts0_good)} tracked",
                      color="white", fontsize=8)
        ax1.axis("off")

        frame0.release()

    fig.suptitle(
        "Feature Pipeline  —  ORB Matching (top)  |  LK Optical-Flow Tracking (bottom)",
        color="white", fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "feature_samples.png")
    plt.savefig(out_path, dpi=110, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"  Saved feature samples → {out_path}")


def save_point_cloud_ply(loader, stereo_vo_obj, stereo_cfg,
                         out_dir, max_pts: int = 1_000_000,
                         sor_k: int = 10, sor_sigma: float = 2.0) -> None:
    """Build a dense 3D point cloud from keyframe depth maps; saves binary PLY."""
    from stereo_vo.disparity import DisparityComputer
    from scipy.spatial import cKDTree

    disp_cmp  = DisparityComputer(loader.calib, stereo_cfg.disparity)
    f_px = float(disp_cmp.f)
    cx   = float(disp_cmp.cx)
    cy   = float(disp_cmp.cy)
    fB   = f_px * float(disp_cmp.B)
    min_disp_px = 1.5

    poses    = stereo_vo_obj.trajectory[1]
    n_frames = min(len(loader._frames), len(poses))

    min_trans_m   = 0.05
    min_rot_deg   = 2.0
    max_keyframes = 800    # hard cap to avoid OOM on long sequences

    frame_indices = []
    prev_T = None
    for i in range(n_frames):
        T = poses[i]
        if prev_T is None:
            frame_indices.append(i)
            prev_T = T
            continue
        delta_t = np.linalg.norm(T[:3, 3] - prev_T[:3, 3])
        R_rel   = T[:3, :3] @ prev_T[:3, :3].T
        cos_a   = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        delta_r = np.degrees(np.arccos(cos_a))
        if delta_t >= min_trans_m or delta_r >= min_rot_deg:
            frame_indices.append(i)
            prev_T = T

    if len(frame_indices) > max_keyframes:
        step_kf       = len(frame_indices) // max_keyframes
        frame_indices = frame_indices[::step_kf][:max_keyframes]

    # Auto-scale pixel step: kf * (512/step_px)^2 <= 4M  →  step_px >= 512*sqrt(kf/4M)
    target_raw = 4_000_000
    step_px = int(np.ceil(512.0 * np.sqrt(len(frame_indices) / target_raw)))
    step_px = max(4, step_px + step_px % 2)   # minimum 4, keep even

    all_xyz: list = []
    all_rgb: list = []

    print(f"\n  [PLY] accumulating 3D points: {len(frame_indices)} keyframes "
          f"(motion >= {min_trans_m*100:.0f}cm / {min_rot_deg:.0f}deg), "
          f"pixel step={step_px} (auto-scaled), min_disp={min_disp_px}px ...")

    for fidx in frame_indices:
        frame = loader._frames[fidx]
        img_l = frame.img_left
        img_r = frame.img_right
        if img_r is None:
            frame.release()
            continue

        rect_l, rect_r = loader.calib.rectify(img_l, img_r)
        disp  = disp_cmp.compute(rect_l, rect_r, rectified=True)
        depth = disp_cmp.disparity_to_depth(disp)

        h, w = rect_l.shape[:2]
        v_arr = np.arange(step_px // 2, h, step_px, dtype=np.int32)
        u_arr = np.arange(step_px // 2, w, step_px, dtype=np.int32)
        vv, uu = np.meshgrid(v_arr, u_arr, indexing='ij')
        v_flat = vv.ravel()
        u_flat = uu.ravel()

        Z = depth[v_flat, u_flat]
        D = disp[v_flat, u_flat]
        valid = (Z > 0) & (D >= min_disp_px)
        if valid.sum() < 5:
            frame.release()
            continue

        Z_v = Z[valid].astype(np.float64)
        u_v = u_flat[valid].astype(np.float64)
        v_v = v_flat[valid].astype(np.float64)

        X_v = (u_v - cx) * Z_v / f_px
        Y_v = (v_v - cy) * Z_v / f_px
        pts_cam = np.stack([X_v, Y_v, Z_v], axis=1)

        T_wc  = poses[fidx]
        pts_w = (T_wc[:3, :3] @ pts_cam.T).T + T_wc[:3, 3]

        intensity = rect_l[v_flat[valid], u_flat[valid]]
        rgb = np.stack([intensity, intensity, intensity], axis=1)

        all_xyz.append(pts_w.astype(np.float32))
        all_rgb.append(rgb)
        frame.release()

    if not all_xyz:
        print("  [PLY] no valid points — skipping")
        return

    pts = np.vstack(all_xyz)
    col = np.vstack(all_rgb)

    print(f"  [PLY] {len(pts):,} raw points — running outlier removal ...")
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=sor_k + 1)   # k+1: first hit is self (dist=0)
    mean_nn  = dists[:, 1:].mean(axis=1)
    thr      = mean_nn.mean() + sor_sigma * mean_nn.std()
    keep     = mean_nn < thr
    pts = pts[keep];  col = col[keep]
    print(f"  [PLY] {keep.sum():,} points after outlier removal "
          f"({(~keep).sum():,} removed, thr={thr:.3f}m)")

    if len(pts) > max_pts:
        idx = np.random.choice(len(pts), max_pts, replace=False)
        pts = pts[idx]
        col = col[idx]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Stereo VO 3D reconstruction - TUM VI\n"
        "comment Z = fB/d,  world frame via estimated stereo VO poses\n"
        f"element vertex {len(pts)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    xyz_bytes = pts.astype(np.float32).tobytes()
    n = len(pts)
    vertex_buf = np.empty((n, 15), dtype=np.uint8)
    vertex_buf[:, :12] = np.frombuffer(xyz_bytes, dtype=np.uint8).reshape(n, 12)
    vertex_buf[:, 12:] = col.astype(np.uint8)

    ply_path = os.path.join(out_dir, "pointcloud.ply")
    with open(ply_path, "wb") as fh:
        fh.write(header)
        fh.write(vertex_buf.tobytes())

    size_mb = os.path.getsize(ply_path) / 1e6
    print(f"  [PLY] {len(pts):,} points saved → {ply_path}  ({size_mb:.1f} MB)")

def traj_path_length(poses) -> float:
    """Total arc length in metres from a list of SE3 4×4 matrices."""
    if len(poses) < 2:
        return 0.0
    pts = np.array([T[:3, 3] for T in poses])
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


_CSV_FIELDS = [
    "sequence", "method", "metric_type",
    "ATE_RMSE_m", "ATE_mean_m",
    "RPE_trans_d1_m", "RPE_rot_d1_deg",
    "RPE_trans_100m_m", "RPE_100m_segments",
    "RPE_rot_60deg_deg", "RPE_60deg_segments",
    "est_traj_len_m", "gt_traj_len_m", "scale", "drift_m",
    "local_ate_start_m", "local_ate_end_m",
    "failures", "success_rate_pct", "runtime_fps",
]

def _fmt(v, digits=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    return round(v, digits)

def save_results_csv(results: dict, path: str) -> None:
    """Write one row per (sequence, method) to a flat CSV."""
    rows = []
    for seq, r in results.items():
        if r["type"] == "ate":
            rows.append({
                "sequence":            seq,
                "method":              "mono",
                "metric_type":         "ATE",
                "ATE_RMSE_m":          _fmt(r["mono_ate"]),
                "ATE_mean_m":          _fmt(r["mono_ate_mean"]),
                "RPE_trans_d1_m":      _fmt(r["mono_rpe"]),
                "RPE_rot_d1_deg":      _fmt(r["mono_rpe_rot"]),
                "RPE_trans_100m_m":    _fmt(r["mono_rpe_100m"]),
                "RPE_100m_segments":   r["mono_rpe_segs_100m"],
                "RPE_rot_60deg_deg":   _fmt(r.get("mono_rpe_rot_60deg")),
                "RPE_60deg_segments":  r.get("mono_rpe_segs_60deg", ""),
                "est_traj_len_m":      _fmt(r["mono_traj_len"], 2),
                "gt_traj_len_m":       _fmt(r.get("gt_traj_len"), 2),
                "scale":               _fmt(r["mono_scale"]),
                "drift_m":             "",
                "local_ate_start_m":   "",
                "local_ate_end_m":     "",
                "failures":            r["mono_fail"],
                "success_rate_pct":    _fmt(r.get("mono_success_rate"), 2),
                "runtime_fps":         round(r["mono_fps"], 1),
            })
            rows.append({
                "sequence":            seq,
                "method":              "stereo",
                "metric_type":         "ATE",
                "ATE_RMSE_m":          _fmt(r["stereo_ate"]),
                "ATE_mean_m":          _fmt(r["stereo_ate_mean"]),
                "RPE_trans_d1_m":      _fmt(r["stereo_rpe"]),
                "RPE_rot_d1_deg":      _fmt(r["stereo_rpe_rot"]),
                "RPE_trans_100m_m":    _fmt(r["stereo_rpe_100m"]),
                "RPE_100m_segments":   r["stereo_rpe_segs_100m"],
                "RPE_rot_60deg_deg":   _fmt(r.get("stereo_rpe_rot_60deg")),
                "RPE_60deg_segments":  r.get("stereo_rpe_segs_60deg", ""),
                "est_traj_len_m":      _fmt(r["stereo_traj_len"], 2),
                "gt_traj_len_m":       _fmt(r.get("gt_traj_len"), 2),
                "scale":               _fmt(r["stereo_scale"]),
                "drift_m":             "",
                "local_ate_start_m":   "",
                "local_ate_end_m":     "",
                "failures":            r["stereo_fail"],
                "success_rate_pct":    _fmt(r.get("stereo_success_rate"), 2),
                "runtime_fps":         round(r["stereo_fps"], 1),
            })
        else:
            rows.append({
                "sequence":            seq,
                "method":              "mono",
                "metric_type":         "drift",
                "ATE_RMSE_m":          "",
                "ATE_mean_m":          "",
                "RPE_trans_d1_m":      _fmt(r.get("mono_rpe")),
                "RPE_rot_d1_deg":      _fmt(r.get("mono_rpe_rot")),
                "RPE_trans_100m_m":    "",
                "RPE_100m_segments":   "",
                "RPE_rot_60deg_deg":   "",
                "RPE_60deg_segments":  "",
                "est_traj_len_m":      _fmt(r.get("mono_traj_len"), 2),
                "gt_traj_len_m":       "",
                "scale":               "",
                "drift_m":             _fmt(r["mono_drift"]),
                "local_ate_start_m":   _fmt(r.get("mono_local_ate_start")),
                "local_ate_end_m":     _fmt(r.get("mono_local_ate_end")),
                "failures":            r["mono_fail"],
                "success_rate_pct":    _fmt(r.get("mono_success_rate"), 2),
                "runtime_fps":         round(r["mono_fps"], 1),
            })
            rows.append({
                "sequence":            seq,
                "method":              "stereo",
                "metric_type":         "drift",
                "ATE_RMSE_m":          "",
                "ATE_mean_m":          "",
                "RPE_trans_d1_m":      _fmt(r.get("stereo_rpe")),
                "RPE_rot_d1_deg":      _fmt(r.get("stereo_rpe_rot")),
                "RPE_trans_100m_m":    "",
                "RPE_100m_segments":   "",
                "RPE_rot_60deg_deg":   "",
                "RPE_60deg_segments":  "",
                "est_traj_len_m":      _fmt(r.get("stereo_traj_len"), 2),
                "gt_traj_len_m":       "",
                "scale":               "",
                "drift_m":             _fmt(r["stereo_drift"]),
                "local_ate_start_m":   _fmt(r.get("stereo_local_ate_start")),
                "local_ate_end_m":     _fmt(r.get("stereo_local_ate_end")),
                "failures":            r["stereo_fail"],
                "success_rate_pct":    _fmt(r.get("stereo_success_rate"), 2),
                "runtime_fps":         round(r["stereo_fps"], 1),
            })

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nEvaluation results saved → {path}")


all_results = {}
import cv2
print(cv2.__version__); print(cv2.getBuildInformation()[:800])


_tmp_cfg    = load_run_config(SEQUENCES[0])
_tmp_loader = TUMVILoader.from_config(_tmp_cfg)
print_camera_intrinsics(_tmp_loader.calib.cam0.K, name="cam0 (unrectified)")
print_stereo_params(
    f  = _tmp_loader.calib.P_left[0, 0],
    B  = _tmp_loader.calib.baseline,
    cx = _tmp_loader.calib.P_left[0, 2],
    cy = _tmp_loader.calib.P_left[1, 2],
)
print_stereo_extrinsics(_tmp_loader.calib)

for config_file in SEQUENCES:
    np.random.seed(42)  # reset per-sequence so map capping is reproducible
    cfg     = load_run_config(config_file)
    loader  = TUMVILoader.from_config(cfg)
    seq     = cfg.sequence_name
    out_dir = f"outputs/{seq}"
    ensure_dir(out_dir)

    print_sequence_summary(
        name        = seq,
        n_frames    = len(loader),
        duration    = loader.timestamps[-1] - loader.timestamps[0],
        n_gt        = sum(1 for f in loader if f.T_world_cam0 is not None),
        baseline_cm = loader.calib.baseline * 100,
    )

    mono_cfg   = MONO_CONFIGS[seq]
    stereo_cfg = STEREO_CONFIGS[seq]

    mono_vo, mono_time = run_mono(loader, seq, mono_cfg, show=_SHOW)

    stereo_vo, stereo_time = run_stereo(loader, seq, stereo_cfg, show=_SHOW)

    mono_vo.save_trajectory(os.path.join(out_dir, "mono_traj.txt"))
    stereo_vo.save_trajectory(os.path.join(out_dir, "stereo_traj.txt"))

    _full_gt = seq in FULL_GT
    save_mono_plot(cfg, loader, mono_vo, mono_time, out_dir, full_gt=_full_gt)
    save_stereo_plot(cfg, loader, stereo_vo, stereo_time, out_dir, full_gt=_full_gt)
    save_stereo_sample_visuals(loader, stereo_cfg, out_dir, n_samples=3)
    save_feature_sample_visuals(loader, stereo_cfg.feature, out_dir, n_samples=2,
                                use_clahe=stereo_cfg.use_clahe)
    # room2: tighter SOR (k=20, sigma=1.0) removes CLAHE-induced SGBM speckle on flat walls
    _sor_k, _sor_sigma = (20, 1.0) if seq == "room2" else (10, 2.0)
    save_point_cloud_ply(loader, stereo_vo, stereo_cfg, out_dir,
                         sor_k=_sor_k, sor_sigma=_sor_sigma)

    gt_list = [f.T_world_cam0 for f in loader]

    if _full_gt:
        poses_3d_m = mono_vo.trajectory[1]
        poses_3d_s = stereo_vo.trajectory[1]
        align_m, align_s = "sim3", "se3"
    else:
        _gt_nonnull = [T for T in gt_list if T is not None]
        if _gt_nonnull:
            _Ta_m = _gt_nonnull[0] @ np.linalg.inv(mono_vo.trajectory[1][0])
            _Ta_s = _gt_nonnull[0] @ np.linalg.inv(stereo_vo.trajectory[1][0])
            poses_3d_m = [_Ta_m @ T for T in mono_vo.trajectory[1]]
            poses_3d_s = [_Ta_s @ T for T in stereo_vo.trajectory[1]]
        else:
            poses_3d_m = mono_vo.trajectory[1]
            poses_3d_s = stereo_vo.trajectory[1]
        align_m = align_s = ""
        _pre_drift_m = start_end_drift(mono_vo.trajectory[1],
                                       [_gt_nonnull[0], _gt_nonnull[-1]]) \
                       if len(_gt_nonnull) >= 2 else float("nan")
        _pre_drift_s = start_end_drift(stereo_vo.trajectory[1],
                                       [_gt_nonnull[0], _gt_nonnull[-1]]) \
                       if len(_gt_nonnull) >= 2 else float("nan")

    save_3d_trajectory(
        est_poses = poses_3d_m,
        gt_poses  = gt_list,
        title     = f"Mono VO — {seq}  (Sim3-aligned)",
        out_path  = os.path.join(out_dir, "mono_traj_3d.png"),
        align     = align_m,
        show_gt   = False,
    )
    save_3d_trajectory(
        est_poses = poses_3d_s,
        gt_poses  = gt_list,
        title     = f"Stereo VO — {seq}  (SE3-aligned)",
        out_path  = os.path.join(out_dir, "stereo_traj_3d.png"),
        align     = align_s,
        show_gt   = False,
    )

    save_comparison_3d(
        mono_poses   = poses_3d_m,
        stereo_poses = poses_3d_s,
        gt_poses     = gt_list,
        seq_name     = seq,
        out_path     = os.path.join(out_dir, "comparison_3d.png"),
        full_gt      = _full_gt,
        mono_drift   = _pre_drift_m if not _full_gt else float("nan"),
        stereo_drift = _pre_drift_s if not _full_gt else float("nan"),
    )

    mono_est,   mono_gt   = collect_gt(loader, mono_vo.trajectory[1])
    stereo_est, stereo_gt = collect_gt(loader, stereo_vo.trajectory[1])

    if seq in FULL_GT and len(mono_gt) > 10:
        mono_result   = align_and_evaluate(mono_est,   mono_gt,   align="sim3")
        stereo_result = align_and_evaluate(stereo_est, stereo_gt, align="se3")

        gt_poses_all = [f.T_world_cam0 for f in loader if f.T_world_cam0 is not None]
        gt_traj_len  = traj_path_length(gt_poses_all)

        mono_traj_len_est   = traj_path_length(mono_vo.trajectory[1])
        stereo_traj_len_est = traj_path_length(stereo_vo.trajectory[1])

        n_frames         = len(loader)
        mono_success     = (n_frames - mono_vo.n_failures)   / n_frames * 100
        stereo_success   = (n_frames - stereo_vo.n_failures) / n_frames * 100

        print(f"\n── Evaluation: {seq} {'─' * 30}")
        print(f"{'Metric':<32} {'Mono VO':>12} {'Stereo VO':>12}")
        print("-" * 58)
        for label, mk, sk in [
            ("ATE RMSE [m]",         "ate_rmse",          "ate_rmse"),
            ("ATE mean [m]",         "ate_mean",          "ate_mean"),
            ("RPE trans 100m [m]",   "rpe_trans_100m",    "rpe_trans_100m"),
            ("RPE rot 60deg [deg]",  "rpe_rot_60deg",     "rpe_rot_60deg"),
            ("Sim3/SE3 scale",       "scale",             "scale"),
        ]:
            mv = mono_result[mk]
            sv = stereo_result[sk]
            ms = f"{mv:>12.4f}" if np.isfinite(mv) else f"{'N/A':>12}"
            ss = f"{sv:>12.4f}" if np.isfinite(sv) else f"{'N/A':>12}"
            print(f"{label:<32} {ms}{ss}")
        print(f"{'Est. traj length [m]':<32} {mono_traj_len_est:>12.2f} {stereo_traj_len_est:>12.2f}")
        print(f"{'GT traj length [m]':<32} {gt_traj_len:>12.2f} {gt_traj_len:>12.2f}")
        segs_m = mono_result["rpe_n_segments_100m"]
        segs_s = stereo_result["rpe_n_segments_100m"]
        print(f"{'  (100m segments)':>32} {segs_m:>12d} {segs_s:>12d}")
        segs60_m = mono_result["rpe_n_segments_60deg"]
        segs60_s = stereo_result["rpe_n_segments_60deg"]
        print(f"{'  (60deg segments)':>32} {segs60_m:>12d} {segs60_s:>12d}")
        print(f"{'Tracking failures':<32} "
              f"{mono_vo.n_failures:>12d} "
              f"{stereo_vo.n_failures:>12d}")
        print(f"{'Success rate [%]':<32} "
              f"{mono_success:>12.2f} "
              f"{stereo_success:>12.2f}")
        print(f"{'Runtime [fps]':<32} "
              f"{len(loader) / mono_time:>12.1f} "
              f"{len(loader) / stereo_time:>12.1f}")

        all_results[seq] = {
            "type":                  "ate",
            "mono_ate":              mono_result["ate_rmse"],
            "mono_ate_mean":         mono_result["ate_mean"],
            "mono_rpe":              mono_result["rpe_trans_rmse_d1"],
            "mono_rpe_rot":          mono_result["rpe_rot_rmse_d1"],
            "mono_rpe_100m":         mono_result["rpe_trans_100m"],
            "mono_rpe_segs_100m":    mono_result["rpe_n_segments_100m"],
            "mono_rpe_rot_60deg":    mono_result["rpe_rot_60deg"],
            "mono_rpe_segs_60deg":   mono_result["rpe_n_segments_60deg"],
            "mono_traj_len":         mono_traj_len_est,
            "mono_scale":            mono_result["scale"],
            "stereo_ate":            stereo_result["ate_rmse"],
            "stereo_ate_mean":       stereo_result["ate_mean"],
            "stereo_rpe":            stereo_result["rpe_trans_rmse_d1"],
            "stereo_rpe_rot":        stereo_result["rpe_rot_rmse_d1"],
            "stereo_rpe_100m":       stereo_result["rpe_trans_100m"],
            "stereo_rpe_segs_100m":  stereo_result["rpe_n_segments_100m"],
            "stereo_rpe_rot_60deg":  stereo_result["rpe_rot_60deg"],
            "stereo_rpe_segs_60deg": stereo_result["rpe_n_segments_60deg"],
            "stereo_traj_len":       stereo_traj_len_est,
            "stereo_scale":          stereo_result["scale"],
            "gt_traj_len":           gt_traj_len,
            "mono_fps":              len(loader) / mono_time,
            "stereo_fps":            len(loader) / stereo_time,
            "mono_fail":             mono_vo.n_failures,
            "stereo_fail":           stereo_vo.n_failures,
            "mono_success_rate":     mono_success,
            "stereo_success_rate":   stereo_success,
            "mono_inlier_ratios":    mono_vo.inlier_ratios,
            "stereo_inlier_ratios":  stereo_vo.inlier_ratios,
        }

    else:
        gt_poses = [f.T_world_cam0 for f in loader
                    if f.T_world_cam0 is not None]
        if len(gt_poses) >= 2:
            mono_drift   = start_end_drift(
                mono_vo.trajectory[1],   [gt_poses[0], gt_poses[-1]])
            stereo_drift = start_end_drift(
                stereo_vo.trajectory[1], [gt_poses[0], gt_poses[-1]])
        else:
            mono_drift = stereo_drift = float("nan")

        mono_traj_len_est   = traj_path_length(mono_vo.trajectory[1])
        stereo_traj_len_est = traj_path_length(stereo_vo.trajectory[1])

        n_frames         = len(loader)
        mono_success     = (n_frames - mono_vo.n_failures)   / n_frames * 100
        stereo_success   = (n_frames - stereo_vo.n_failures) / n_frames * 100

        stereo_rpe_d1, stereo_rpe_rot_d1 = compute_rpe_d1_on_consecutive_gt(
            loader, stereo_vo.trajectory[1])

        mono_blocks   = compute_local_ate_blocks(
            loader, mono_vo.trajectory[1],   align_mode="sim3")
        stereo_blocks = compute_local_ate_blocks(
            loader, stereo_vo.trajectory[1], align_mode="se3")

        def _ate_str(v):
            return f"{v:>12.4f}" if np.isfinite(v) else f"{'N/A':>12}"

        print(f"\n── Evaluation: {seq} (start-end drift) {'─' * 15}")
        print(f"{'Metric':<32} {'Mono VO':>12} {'Stereo VO':>12}")
        print("-" * 58)
        print(f"{'Start-end drift [m]':<32} "
              f"{mono_drift:>12.4f} {stereo_drift:>12.4f}")
        rpe_str = f"{stereo_rpe_d1:>12.4f}" if np.isfinite(stereo_rpe_d1) else f"{'N/A':>12}"
        print(f"{'RPE trans d=1 [m] (stereo)':<32} {'N/A':>12} {rpe_str}")
        sn, en = mono_blocks["start_n"], mono_blocks["end_n"]
        print(f"{'Local ATE start block [m]':<32} "
              f"{_ate_str(mono_blocks['start_ate'])} "
              f"{_ate_str(stereo_blocks['start_ate'])}   (n={sn})")
        print(f"{'Local ATE end block [m]':<32} "
              f"{_ate_str(mono_blocks['end_ate'])} "
              f"{_ate_str(stereo_blocks['end_ate'])}   (n={en})")
        print(f"{'Est. traj length [m]':<32} "
              f"{mono_traj_len_est:>12.2f} {stereo_traj_len_est:>12.2f}")
        print(f"{'GT traj length [m]':<32} "
              f"{'N/A (partial GT)':>12} {'N/A (partial GT)':>12}")
        print(f"{'Tracking failures':<32} "
              f"{mono_vo.n_failures:>12d} {stereo_vo.n_failures:>12d}")
        print(f"{'Success rate [%]':<32} "
              f"{mono_success:>12.2f} {stereo_success:>12.2f}")
        print(f"{'Runtime [fps]':<32} "
              f"{len(loader) / mono_time:>12.1f} "
              f"{len(loader) / stereo_time:>12.1f}")

        all_results[seq] = {
            "mono_drift":            mono_drift,
            "stereo_drift":          stereo_drift,
            "mono_fps":              len(loader) / mono_time,
            "stereo_fps":            len(loader) / stereo_time,
            "mono_fail":             mono_vo.n_failures,
            "stereo_fail":           stereo_vo.n_failures,
            "mono_rpe":              float("nan"),
            "mono_rpe_rot":          float("nan"),
            "stereo_rpe":            stereo_rpe_d1,
            "stereo_rpe_rot":        stereo_rpe_rot_d1,
            "mono_local_ate_start":  mono_blocks["start_ate"],
            "mono_local_ate_end":    mono_blocks["end_ate"],
            "stereo_local_ate_start":stereo_blocks["start_ate"],
            "stereo_local_ate_end":  stereo_blocks["end_ate"],
            "mono_traj_len":         mono_traj_len_est,
            "stereo_traj_len":       stereo_traj_len_est,
            "mono_success_rate":     mono_success,
            "stereo_success_rate":   stereo_success,
            "type":                  "drift",
            "mono_inlier_ratios":    mono_vo.inlier_ratios,
            "stereo_inlier_ratios":  stereo_vo.inlier_ratios,
        }

print("\n" + "=" * 70)
print("  FINAL RESULTS — ALL SEQUENCES")
print("=" * 70)
print(f"{'Sequence':<14} {'Metric':<28} {'Mono VO':>10} {'Stereo VO':>10}")
print("-" * 64)
def _pf(v):
    return f"{v:>10.4f}" if np.isfinite(v) else f"{'N/A':>10}"
def _pf1(v):
    return f"{v:>10.1f}" if np.isfinite(v) else f"{'N/A':>10}"
def _pp(v):
    return f"{v:>10.2f}" if np.isfinite(v) else f"{'N/A':>10}"

for seq, r in all_results.items():
    if r["type"] == "ate":
        print(f"{seq:<14} {'ATE RMSE [m]':<28} "
              f"{_pf(r['mono_ate'])} {_pf(r['stereo_ate'])}")
        print(f"{'':<14} {'RPE trans 100m [m]':<28} "
              f"{_pf(r['mono_rpe_100m'])} {_pf(r['stereo_rpe_100m'])}")
        print(f"{'':<14} {'  (100m segments)':<28} "
              f"{r['mono_rpe_segs_100m']:>10d} {r['stereo_rpe_segs_100m']:>10d}")
        print(f"{'':<14} {'RPE rot 60deg [deg]':<28} "
              f"{_pf(r['mono_rpe_rot_60deg'])} {_pf(r['stereo_rpe_rot_60deg'])}")
        print(f"{'':<14} {'Est. traj length [m]':<28} "
              f"{_pf1(r['mono_traj_len'])} {_pf1(r['stereo_traj_len'])}")
        gt_len = r.get('gt_traj_len', float('nan'))
        print(f"{'':<14} {'GT traj length [m]':<28} "
              f"{_pf1(gt_len)} {_pf1(gt_len)}")
        print(f"{'':<14} {'Scale (Sim3/SE3)':<28} "
              f"{_pf(r['mono_scale'])} {'1.0000':>10}")
    else:
        print(f"{seq:<14} {'Start-end drift [m]':<28} "
              f"{_pf(r['mono_drift'])} {_pf(r['stereo_drift'])}")
        print(f"{'':<14} {'Est. traj length [m]':<28} "
              f"{_pf1(r.get('mono_traj_len', float('nan')))} "
              f"{_pf1(r.get('stereo_traj_len', float('nan')))}")
        print(f"{'':<14} {'GT traj length [m]':<28} "
              f"{'N/A (partial)':>10} {'N/A (partial)':>10}")
    print(f"{'':<14} {'Tracking failures':<28} "
          f"{r['mono_fail']:>10d} {r['stereo_fail']:>10d}")
    print(f"{'':<14} {'Success rate [%]':<28} "
          f"{_pp(r.get('mono_success_rate', float('nan')))} "
          f"{_pp(r.get('stereo_success_rate', float('nan')))}")
    print(f"{'':<14} {'Runtime [fps]':<28} "
          f"{r['mono_fps']:>10.1f} {r['stereo_fps']:>10.1f}")
    print("-" * 64)

print("\nAll trajectory files saved under outputs/<sequence>/")

save_results_csv(all_results, "outputs/evaluation_results.csv")


def report_dynamic_sensitivity(all_results: dict,
                               csv_path: str = "outputs/dynamic_results.csv") -> None:
    """Report RANSAC inlier ratio per frame; low ratio indicates dynamic object contamination."""
    LOW_TH  = 0.70
    SEV_TH  = 0.50

    print("\n" + "=" * 72)
    print("  DYNAMIC SCENE SENSITIVITY  —  RANSAC inlier ratio  (Spec §VI)")
    print("  Low inlier ratio => tracked points inconsistent with static-scene")
    print("  ego-motion model => evidence of dynamic object contamination.")
    print("=" * 72)
    hdr = (f"{'Sequence':<12} {'Pipeline':<9} {'Frames':>7} "
           f"{'Mean ratio':>11} {'<0.70':>7} {'<0.50':>7}  Note")
    print(hdr)
    print("-" * 72)

    csv_rows = []
    for seq, res in all_results.items():
        for label, key in [("Mono", "mono_inlier_ratios"),
                           ("Stereo", "stereo_inlier_ratios")]:
            ratios = res.get(key, [])
            if not ratios:
                print(f"{seq:<12} {label:<9} {'N/A':>7}")
                continue
            arr      = np.array(ratios, dtype=float)
            n        = len(arr)
            mean_r   = float(arr.mean())
            pct_low  = float((arr < LOW_TH).mean()) * 100.0
            pct_sev  = float((arr < SEV_TH).mean()) * 100.0

            note = ""
            if seq == "outdoors5":
                note = "pedestrians/cyclists"
            elif seq == "corridor3":
                note = "homogeneous walls"

            print(f"{seq:<12} {label:<9} {n:>7d} "
                  f"{mean_r:>11.3f} {pct_low:>6.1f}% {pct_sev:>6.1f}%  "
                  + (f"<-- {note}" if note else ""))

            csv_rows.append({
                "sequence":   seq,
                "method":     label.lower(),
                "frames":     n,
                "mean_ratio": round(mean_r, 4),
                "pct_low70":  round(pct_low, 2),
                "pct_low50":  round(pct_sev, 2),
                "note":       note,
            })

        print()

    print("-" * 72)
    print("Interpretation:")
    print("  Mean ratio close to 1.0 => most tracked points fit static model (clean).")
    print("  High <0.70 % on outdoors5 => dynamic objects inflate outlier count.")
    print("  <0.50 frames include PnP failures (ratio=0.0) and severe contamination.")
    print()

    _DYN_FIELDS = ["sequence", "method", "frames", "mean_ratio",
                   "pct_low70", "pct_low50", "note"]
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_DYN_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  Dynamic sensitivity results saved → {csv_path}")


report_dynamic_sensitivity(all_results)


import copy

def _run_single_clahe_abl(pipeline_cls, calib_obj, base_cfg, loader,
                           is_full_gt, base_fail, base_metric, align_mode):
    abl_cfg           = copy.deepcopy(base_cfg)
    abl_cfg.use_clahe = not base_cfg.use_clahe
    abl_cfg.verbose   = False

    vo = pipeline_cls(calib_obj, abl_cfg)
    for frame in loader:
        if pipeline_cls is MonoVO:
            vo.process(frame.img_left_rect, frame.timestamp)
        else:
            vo.process(frame.img_left, frame.img_right, frame.timestamp)

    abl_est, abl_gt = collect_gt(loader, vo.trajectory[1])
    if is_full_gt and len(abl_gt) > 10:
        res    = align_and_evaluate(abl_est, abl_gt, align=align_mode)
        metric = res.get("ate_rmse", float("nan"))
    elif len(abl_gt) >= 2:
        metric = start_end_drift(vo.trajectory[1], [abl_gt[0], abl_gt[-1]])
    else:
        metric = float("nan")

    return vo.n_failures, metric


def run_clahe_ablation(all_results: dict,
                       csv_path: str = "outputs/clahe_ablation.csv") -> None:
    """Run both pipelines with CLAHE toggled per sequence; compare failures and ATE/drift."""
    print("\n" + "=" * 74)
    print("  ILLUMINATION SENSITIVITY ABLATION  —  CLAHE on vs off  (Spec §VI)")
    print("=" * 74)
    print(f"{'Sequence':<12} {'Pipeline':<9} {'CLAHE':<7} "
          f"{'Failures':>9} {'ATE / Drift [m]':>16}  Config")
    print("-" * 74)

    def _ms(v):    return f"{v:>16.4f}" if np.isfinite(v) else f"{'N/A':>16}"
    def _clahe(b): return "ON " if b else "OFF"
    def _sign(d):  return "+" if d >= 0 else ""

    csv_rows = []

    for config_file in SEQUENCES:
        seq        = load_run_config(config_file).sequence_name
        is_full_gt = seq in FULL_GT
        metric_lbl = "ATE" if is_full_gt else "drift"
        metric_type = "ATE" if is_full_gt else "drift"

        for pipeline_label, base_cfg, align_mode, fail_key, metric_key in [
            ("Mono",   MONO_CONFIGS[seq],   "sim3",
             "mono_fail",   "mono_ate"   if is_full_gt else "mono_drift"),
            ("Stereo", STEREO_CONFIGS[seq], "se3",
             "stereo_fail", "stereo_ate" if is_full_gt else "stereo_drift"),
        ]:
            np.random.seed(42)
            cfg    = load_run_config(config_file)
            loader = TUMVILoader.from_config(cfg)

            base_fail   = all_results[seq][fail_key]
            base_metric = all_results[seq].get(metric_key, float("nan"))

            if pipeline_label == "Mono":
                abl_fail, abl_metric = _run_single_clahe_abl(
                    MonoVO, loader.calib.cam0, base_cfg,
                    loader, is_full_gt, base_fail, base_metric, "sim3")
            else:
                abl_cfg           = copy.deepcopy(base_cfg)
                abl_cfg.use_clahe = not base_cfg.use_clahe
                abl_cfg.verbose   = False
                vo_s = StereoVO(loader.calib, abl_cfg)
                for frame in loader:
                    vo_s.process(frame.img_left, frame.img_right, frame.timestamp)
                abl_fail = vo_s.n_failures
                abl_est, abl_gt = collect_gt(loader, vo_s.trajectory[1])
                if is_full_gt and len(abl_gt) > 10:
                    res        = align_and_evaluate(abl_est, abl_gt, align="se3")
                    abl_metric = res.get("ate_rmse", float("nan"))
                elif len(abl_gt) >= 2:
                    abl_metric = start_end_drift(
                        vo_s.trajectory[1], [abl_gt[0], abl_gt[-1]])
                else:
                    abl_metric = float("nan")

            fail_d   = abl_fail   - base_fail
            metric_d = abl_metric - base_metric

            print(f"{seq:<12} {pipeline_label:<9} "
                  f"{_clahe(base_cfg.use_clahe):<7} {base_fail:>9}"
                  f" {_ms(base_metric)}  production")
            print(f"{'':12} {'':9} "
                  f"{_clahe(not base_cfg.use_clahe):<7} {abl_fail:>9}"
                  f" {_ms(abl_metric)}  ablation")
            print(f"  effect: failures {_sign(fail_d)}{fail_d}  |  "
                  f"{metric_lbl} {_sign(metric_d)}{metric_d:.4f} m")
            print()

            method = pipeline_label.lower()
            csv_rows.append({
                "sequence": seq, "method": method,
                "clahe": _clahe(base_cfg.use_clahe).strip(),
                "config": "production",
                "failures": base_fail,
                "metric_m": round(float(base_metric), 4) if np.isfinite(base_metric) else "",
                "metric_type": metric_type,
            })
            csv_rows.append({
                "sequence": seq, "method": method,
                "clahe": _clahe(not base_cfg.use_clahe).strip(),
                "config": "ablation",
                "failures": abl_fail,
                "metric_m": round(float(abl_metric), 4) if np.isfinite(abl_metric) else "",
                "metric_type": metric_type,
            })

        print("-" * 74)

    print("CLAHE ablation complete.\n")

    _CLAHE_FIELDS = ["sequence", "method", "clahe", "config",
                     "failures", "metric_m", "metric_type"]
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CLAHE_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  CLAHE ablation results saved → {csv_path}")

_log_file.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
print("Run log saved → outputs/run.log")

run_clahe_ablation(all_results)
