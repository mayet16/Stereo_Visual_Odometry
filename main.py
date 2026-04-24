"""
main.py  —  Stereo_VO project entry point
==========================================
Runs monocular VO + stereo VO on all three TUM VI sequences.

New in this version:
  - LiveVisualizer: real-time side-by-side display (frame + trajectory)
    during both mono and stereo VO runs.
  - save_3d_trajectory: dark-themed 3-D trajectory figure saved after
    each sequence (mono_traj_3d.png / stereo_traj_3d.png).
  - DEBUG prints removed — stereo config is read correctly.
  - show_frames() call removed (caused Qt thread warnings).

Usage:
  python main.py                        # runs all sequences with live display
  SHOW_VIS=0 python main.py             # headless — saves PNGs only
"""

import csv
import os
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


np.random.seed(42)

_SHOW = os.environ.get("SHOW_VIS", "1") != "0"
_SHOW = 0
SEQUENCES = [
    "config/tumvi_room2.yaml",
    # "config/tumvi_corridor3.yaml",
    # "config/tumvi_outdoors5.yaml",
]

# room2 has full GT → ATE.  corridor3/outdoors5 → start-end drift only.
FULL_GT = {"room2"}

_FEAT = FeatureConfig(
    method       = "orb",
    max_features = 3000,
    lk_win_size  = 21,
    lk_max_level = 3,
    grid_rows    = 5,
    grid_cols    = 5,
)

MONO_CONFIGS = {
    "room2": MonoVOConfig(
        feature         = _FEAT,
        min_tracked_pts = 120,
        max_map_pts     = 800,
        min_parallax_px = 8.0,
        max_parallax_px = 40.0,
        init_scale      = 0.02,   # fixed scale — Sim3 alignment corrects it
        pnp_min_inliers = 15,
        pnp_ransac_th   = 6.0,
        reproj_thresh   = 4.0,
        use_ba          = True,
        max_velocity    = 0.5,
        verbose         = False,
    ),
    "corridor3": MonoVOConfig(
        feature         = _FEAT,
        min_tracked_pts = 100,
        max_map_pts     = 1200,
        min_parallax_px = 4.0,
        max_parallax_px = 60.0,
        init_scale      = 0.02,
        pnp_min_inliers = 12,
        pnp_ransac_th   = 6.0,
        reproj_thresh   = 4.0,
        use_ba          = True,
        max_velocity    = 1.0,
        verbose         = False,
    ),
    "outdoors5": MonoVOConfig(
        feature         = _FEAT,
        min_tracked_pts = 120,
        max_map_pts     = 800,
        min_parallax_px = 6.0,
        max_parallax_px = 80.0,
        init_scale      = 0.05,   # outdoor walking — slightly larger scale
        pnp_min_inliers = 15,
        pnp_ransac_th   = 6.0,
        reproj_thresh   = 4.0,
        use_ba          = True,
        max_velocity    = 2.0,
        verbose         = False,
    ),
}

STEREO_CONFIGS = {
    "room2": StereoVOConfig(
        feature         = _FEAT,
        disparity       = DisparityConfig(
            num_disparities = 64,
            block_size      = 11,
            min_depth       = 0.5,
            max_depth       = 5.0,
            min_disparity   = 1.5,
            patch_radius    = 3,
        ),
        min_tracked_pts = 120,
        max_map_pts     = 500,
        pnp_min_inliers = 15,
        pnp_ransac_th   = 4.0,
        reproj_thresh   = 4.0,
        use_ba          = True,
        max_velocity    = 0.5,
        verbose         = False,
    ),
    "corridor3": StereoVOConfig(
        feature         = _FEAT,
        disparity       = DisparityConfig(
            num_disparities = 128,
            block_size      = 11,
            min_depth       = 0.5,
            max_depth       = 8.0,
            min_disparity   = 1.0,
            patch_radius    = 3,
        ),
        min_tracked_pts = 120,
        max_map_pts     = 500,
        pnp_min_inliers = 15,
        pnp_ransac_th   = 4.0,
        reproj_thresh   = 4.0,
        use_ba          = True,
        max_velocity    = 1.0,
        verbose         = False,
    ),
    "outdoors5": StereoVOConfig(
        feature         = _FEAT,
        disparity       = DisparityConfig(
            num_disparities = 64,
            block_size      = 11,
            min_depth       = 0.5,
            max_depth       = 5.0,
            min_disparity   = 1.0,
            patch_radius    = 3,
        ),
        min_tracked_pts = 120,
        max_map_pts     = 600,
        pnp_min_inliers = 15,
        pnp_ransac_th   = 4.0,
        reproj_thresh   = 4.0,
        use_ba          = True,
        max_velocity    = 2.0,
        verbose         = False,
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
    """
    RPE d=1 on segments where original frame indices are consecutive.
    Handles start+end-only GT (corridor3/outdoors5) by skipping the gap
    between the start and end coverage blocks, so only truly consecutive
    frame pairs contribute to the RMSE.
    Returns (rpe_trans_rmse, rpe_rot_rmse) in metres / degrees.
    """
    all_est, all_gt, all_idx = [], [], []
    for i, frame in enumerate(loader):
        if i >= len(poses):
            break
        if frame.T_world_cam0 is not None:
            all_est.append(poses[i])
            all_gt.append(frame.T_world_cam0)
            all_idx.append(i)

    # Split into contiguous segments (no frame-index gaps)
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
    vo = MonoVO(loader.calib.cam0, cfg)

    vis = LiveVisualizer(
        title     = f"Mono VO — {seq}",
        canvas_hw = (512, 512),
        show      = show,
    )

    t0 = time.time()
    for frame in loader:
        vo.process(frame.img_left, frame.timestamp)

        # update live display every frame
        if vo._initialised:
            vis.update(
                img        = frame.img_left,
                pts_cur    = vo.cur_pts,
                pose       = vo.trajectory[1][-1],
                gt_pose    = frame.T_world_cam0,
                frame_id   = vo._frame_id,
                n_failures = vo.n_failures,
                extra_info = (f"scale={cfg.init_scale:.3f}"
                              if cfg.init_scale is not None
                              else f"auto-scale depth={cfg.expected_depth:.1f}m"),
            )

        frame.release()

    elapsed = time.time() - t0
    vis.close()

    fps = len(loader) / elapsed
    print(f"  Done: {len(loader)} frames  {elapsed:.1f}s  "
          f"({fps:.1f} fps)  failures={vo.n_failures}")
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

        # update live display every frame
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
          f"({fps:.1f} fps)  failures={vo.n_failures}")
    return vo, elapsed



def save_plot(cfg, loader, mono_vo, stereo_vo,
              mono_result, stereo_result, out_dir):
    """Save top-down + ATE-over-time comparison plot (room2)."""
    gt_arr    = np.array([f.T_world_cam0[:3, 3]
                          for f in loader if f.T_world_cam0 is not None])
    mono_al   = np.array([T[:3, 3] for T in mono_result["traj_aligned"]])
    stereo_al = np.array([T[:3, 3] for T in stereo_result["traj_aligned"]])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot(gt_arr[:, 0],    gt_arr[:, 1],    "g--", lw=1.5, label="GT")
    ax.plot(mono_al[:, 0],   mono_al[:, 1],   "b-",  lw=1.0, alpha=0.8,
            label=f"Mono VO (Sim3, ATE={mono_result['ate_rmse']:.3f}m)")
    ax.plot(stereo_al[:, 0], stereo_al[:, 1], "r-",  lw=1.2,
            label=f"Stereo VO (SE3, ATE={stereo_result['ate_rmse']:.3f}m)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Top-down (aligned)"); ax.legend(fontsize=8)
    ax.set_aspect("equal")

    ax2 = axes[1]
    mono_ts   = collect_gt_timestamps(
        loader, mono_vo.trajectory[1],
        np.array(mono_vo.trajectory[0]))
    stereo_ts = collect_gt_timestamps(
        loader, stereo_vo.trajectory[1],
        np.array(stereo_vo.trajectory[0]))
    if len(mono_ts):   mono_ts   -= mono_ts[0]
    if len(stereo_ts): stereo_ts -= stereo_ts[0]

    if ("errors" in mono_result
            and len(mono_ts) == len(mono_result["errors"])):
        ax2.plot(mono_ts,   mono_result["errors"],   "b-",
                 lw=0.8, label="Mono ATE")
    if ("errors" in stereo_result
            and len(stereo_ts) == len(stereo_result["errors"])):
        ax2.plot(stereo_ts, stereo_result["errors"], "r-",
                 lw=0.8, label="Stereo ATE")
    ax2.set_xlabel("time [s]"); ax2.set_ylabel("ATE [m]")
    ax2.set_title("ATE over time"); ax2.legend(fontsize=8)
    ax2.set_ylim(bottom=0)

    plt.suptitle(f"Mono vs Stereo VO — {cfg.sequence_name}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "comparison.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved {path}")


def save_mono_plot(cfg, loader, mono_vo, mono_time, out_dir):
    """Save mono VO 2-D trajectory — raw + Sim3-aligned overlay."""
    mono_est = np.array([T[:3, 3] for T in mono_vo.trajectory[1]])
    gt_arr   = np.array([f.T_world_cam0[:3, 3]
                         for f in loader if f.T_world_cam0 is not None])
    ts = np.array(mono_vo.trajectory[0]); ts -= ts[0]

    mono_aligned, ate_val = None, float("nan")
    est_poses, gt_poses = collect_gt(loader, mono_vo.trajectory[1])
    if len(gt_poses) > 10:
        result       = align_and_evaluate(est_poses, gt_poses, align="sim3")
        mono_aligned = np.array([T[:3, 3] for T in result["traj_aligned"]])
        ate_val      = result["ate_rmse"]

    fps = len(loader) / mono_time if mono_time > 0 else 0.0
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    if len(gt_arr):
        ax.plot(gt_arr[:, 0], gt_arr[:, 1], "g--", lw=1.5, label="GT")
    if mono_aligned is not None:
        ax.plot(mono_aligned[:, 0], mono_aligned[:, 1], "b-", lw=1.2,
                label=f"Mono VO (Sim3-aligned, ATE={ate_val:.3f}m)")
        ax.scatter(*mono_aligned[0,  :2], c="blue", s=40, zorder=5,
                   label="start")
        ax.scatter(*mono_aligned[-1, :2], c="red",  s=40, zorder=5,
                   label="end")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Top-down x–y  (Sim3-aligned)  ← main result")
    ax.legend(fontsize=8); ax.set_aspect("equal")

    if mono_aligned is not None:
        ax2.plot(ts[:len(mono_aligned)], mono_aligned[:, 2], "b-", lw=0.9,
                 label="Mono VO z (Sim3-aligned)")
    if len(gt_arr):
        gt_ts = np.linspace(0, ts[-1], len(gt_arr))
        ax2.plot(gt_ts, gt_arr[:, 2], "g--", lw=1.0, label="GT z")
    ax2.set_xlabel("time [s]"); ax2.set_ylabel("z [m]")
    ax2.set_title("Z height over time  (Sim3-aligned)")
    ax2.legend(fontsize=8)

    plt.suptitle(
        f"Monocular VO — {cfg.sequence_name}  |  "
        f"failures={mono_vo.n_failures}   ATE={ate_val:.3f}m   fps={fps:.0f}",
        fontsize=11,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "mono_traj.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved {path}")


def save_stereo_plot(cfg, loader, stereo_vo, stereo_time, out_dir):
    """
    Save stereo VO 2-D trajectory.
    When full GT is available: shows SE3-aligned x-y top-down + ATE over time
    Raw x-z (camera-frame optical axis) is intentionally NOT plotted — it uses
    a different coordinate frame from the gravity-aligned GT and is misleading
    """
    gt_arr = np.array([f.T_world_cam0[:3, 3]
                       for f in loader if f.T_world_cam0 is not None])
    ts  = np.array(stereo_vo.trajectory[0]); ts -= ts[0]
    fps = len(loader) / stereo_time if stereo_time > 0 else 0.0

    # SE3 alignment when full GT is available
    est_poses, gt_poses = collect_gt(loader, stereo_vo.trajectory[1])
    stereo_aligned, ate_val, ate_errors = None, float("nan"), None
    if len(gt_poses) > 10:
        result         = align_and_evaluate(est_poses, gt_poses, align="se3")
        stereo_aligned = np.array([T[:3, 3] for T in result["traj_aligned"]])
        ate_val        = result["ate_rmse"]
        ate_errors     = result.get("errors", None)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    if stereo_aligned is not None:
        if len(gt_arr):
            ax.plot(gt_arr[:, 0], gt_arr[:, 1], "g--", lw=1.2, label="GT")
        ax.plot(stereo_aligned[:, 0], stereo_aligned[:, 1], "r-", lw=1.0,
                label=f"Stereo VO (SE3-aligned, ATE={ate_val:.3f}m)")
        ax.scatter(*stereo_aligned[0,  :2], c="blue", s=40, zorder=5,
                   label="start")
        ax.scatter(*stereo_aligned[-1, :2], c="red",  s=40, zorder=5,
                   label="end")
        ax.set_title("Top-down x–y  (SE3-aligned, metric)")
    else:
        # Drift sequences: raw x-y without alignment
        stereo_raw = np.array([T[:3, 3] for T in stereo_vo.trajectory[1]])
        ax.plot(stereo_raw[:, 0], stereo_raw[:, 1], "r-", lw=1.0,
                label="Stereo VO (raw)")
        ax.set_title("Top-down x–y  (raw, no GT)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(fontsize=8); ax.set_aspect("equal")

    if ate_errors is not None:
        aligned_ts = collect_gt_timestamps(
            loader, stereo_vo.trajectory[1], ts)
        if len(aligned_ts) > 0:
            aligned_ts = aligned_ts - aligned_ts[0]
        n = min(len(aligned_ts), len(ate_errors))
        if n > 0:
            ax2.plot(aligned_ts[:n], ate_errors[:n], "r-", lw=0.8,
                     label="Stereo ATE")
        ax2.set_xlabel("time [s]"); ax2.set_ylabel("ATE [m]")
        ax2.set_title("Position error over time  (SE3-aligned)")
        ax2.set_ylim(bottom=0)
    else:
        stereo_raw = np.array([T[:3, 3] for T in stereo_vo.trajectory[1]])
        ax2.plot(ts, stereo_raw[:, 0], "r-", lw=0.8, label="x")
        ax2.plot(ts, stereo_raw[:, 1], "b-", lw=0.8, label="y")
        ax2.set_xlabel("time [s]"); ax2.set_ylabel("[m]")
        ax2.set_title("Raw x-y over time")
    ax2.legend(fontsize=8)

    plt.suptitle(
        f"Stereo VO — {cfg.sequence_name}  |  "
        f"failures={stereo_vo.n_failures}   ATE={ate_val:.3f}m   "
        f"scale=1.000 metric   fps={fps:.0f}",
        fontsize=11,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "stereo_traj.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved {path}")


def save_drift_comparison_plot(seq, mono_vo, stereo_vo,
                                mono_drift, stereo_drift,
                                gt_poses, out_dir):
    """Save comparison plot for corridor3 / outdoors5 (drift only)."""
    mono_ts   = np.array(mono_vo.trajectory[0]);   mono_ts   -= mono_ts[0]
    stereo_ts = np.array(stereo_vo.trajectory[0]); stereo_ts -= stereo_ts[0]

    # Apply start-pose alignment so trajectories are in GT world frame
    if gt_poses and len(gt_poses) >= 2:
        T_gt0  = gt_poses[0]
        T_a_m  = T_gt0 @ np.linalg.inv(mono_vo.trajectory[1][0])
        T_a_s  = T_gt0 @ np.linalg.inv(stereo_vo.trajectory[1][0])
        mono_arr   = np.array([(T_a_m @ T)[:3, 3] for T in mono_vo.trajectory[1]])
        stereo_arr = np.array([(T_a_s @ T)[:3, 3] for T in stereo_vo.trajectory[1]])
    else:
        mono_arr   = np.array([T[:3, 3] for T in mono_vo.trajectory[1]])
        stereo_arr = np.array([T[:3, 3] for T in stereo_vo.trajectory[1]])

    gt_sparse = (np.array([T[:3, 3] for T in gt_poses]) if gt_poses else None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot(mono_arr[:, 0],   mono_arr[:, 1],   "b-", lw=0.8, alpha=0.8,
            label=f"Mono VO  (drift={mono_drift:.2f}m)")
    ax.plot(stereo_arr[:, 0], stereo_arr[:, 1], "r-", lw=0.8, alpha=0.8,
            label=f"Stereo VO (drift={stereo_drift:.2f}m)")
    if gt_sparse is not None and len(gt_sparse):
        ax.scatter(gt_sparse[[0, -1], 0], gt_sparse[[0, -1], 1],
                   c="green", s=60, zorder=5, marker="*",
                   label="GT start / end")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Top-down  x–y  (start-aligned to GT)")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(mono_ts,   mono_arr[:, 1],   "b-", lw=0.8, label="Mono y")
    ax2.plot(stereo_ts, stereo_arr[:, 1], "r-", lw=0.8, label="Stereo y")
    ax2.set_xlabel("time [s]"); ax2.set_ylabel("y [m]")
    ax2.set_title("Y position over time  (start-aligned)")
    ax2.legend(fontsize=8)

    plt.suptitle(
        f"Mono vs Stereo VO — {seq}  |  "
        f"mono drift={mono_drift:.2f}m  stereo drift={stereo_drift:.2f}m",
        fontsize=11,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "comparison.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"  Saved {path}")



_CSV_FIELDS = [
    "sequence", "method", "metric_type",
    "ATE_RMSE_m", "ATE_mean_m",
    "RPE_trans_d1_m", "RPE_rot_d1_deg",
    "RPE_trans_100m_m", "RPE_rot_100m_deg", "RPE_100m_segments",
    "traj_len_m", "scale", "drift_m",
    "failures", "runtime_fps",
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
                "RPE_rot_100m_deg":    _fmt(r["mono_rpe_rot_100m"]),
                "RPE_100m_segments":   r["mono_rpe_segs_100m"],
                "traj_len_m":          _fmt(r["mono_traj_len"], 2),
                "scale":               _fmt(r["mono_scale"]),
                "drift_m":             "",
                "failures":            r["mono_fail"],
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
                "RPE_rot_100m_deg":    _fmt(r["stereo_rpe_rot_100m"]),
                "RPE_100m_segments":   r["stereo_rpe_segs_100m"],
                "traj_len_m":          _fmt(r["stereo_traj_len"], 2),
                "scale":               _fmt(r["stereo_scale"]),
                "drift_m":             "",
                "failures":            r["stereo_fail"],
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
                "RPE_rot_100m_deg":    "",
                "RPE_100m_segments":   "",
                "traj_len_m":          "",
                "scale":               "",
                "drift_m":             _fmt(r["mono_drift"]),
                "failures":            r["mono_fail"],
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
                "RPE_rot_100m_deg":    "",
                "RPE_100m_segments":   "",
                "traj_len_m":          "",
                "scale":               "",
                "drift_m":             _fmt(r["stereo_drift"]),
                "failures":            r["stereo_fail"],
                "runtime_fps":         round(r["stereo_fps"], 1),
            })

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nEvaluation results saved → {path}")



all_results = {}

# print calibration once using first sequence
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

    save_mono_plot(cfg, loader, mono_vo, mono_time, out_dir)
    save_stereo_plot(cfg, loader, stereo_vo, stereo_time, out_dir)

    gt_list = [f.T_world_cam0 for f in loader]   # None where GT missing

    save_3d_trajectory(
        est_poses = mono_vo.trajectory[1],
        gt_poses  = gt_list,
        title     = f"Mono VO — {seq}",
        out_path  = os.path.join(out_dir, "mono_traj_3d.png"),
        align     = "sim3",
    )
    save_3d_trajectory(
        est_poses = stereo_vo.trajectory[1],
        gt_poses  = gt_list,
        title     = f"Stereo VO — {seq}",
        out_path  = os.path.join(out_dir, "stereo_traj_3d.png"),
        align     = "se3",
    )

    save_comparison_3d(
        mono_poses   = mono_vo.trajectory[1],
        stereo_poses = stereo_vo.trajectory[1],
        gt_poses     = gt_list,
        seq_name     = seq,
        out_path     = os.path.join(out_dir, "comparison_3d.png"),
    )

    mono_est,   mono_gt   = collect_gt(loader, mono_vo.trajectory[1])
    stereo_est, stereo_gt = collect_gt(loader, stereo_vo.trajectory[1])

    if seq in FULL_GT and len(mono_gt) > 10:
        mono_result   = align_and_evaluate(mono_est,   mono_gt,   align="sim3")
        stereo_result = align_and_evaluate(stereo_est, stereo_gt, align="se3")

        print(f"\n── Evaluation: {seq} {'─' * 30}")
        print(f"{'Metric':<32} {'Mono VO':>12} {'Stereo VO':>12}")
        print("-" * 58)
        for label, mk, sk in [
            ("ATE RMSE [m]",         "ate_rmse",          "ate_rmse"),
            ("ATE mean [m]",         "ate_mean",          "ate_mean"),
            ("RPE trans d=1 [m]",    "rpe_trans_rmse_d1", "rpe_trans_rmse_d1"),
            ("RPE rot d=1 [deg]",    "rpe_rot_rmse_d1",   "rpe_rot_rmse_d1"),
            ("RPE trans 100m [m]",   "rpe_trans_100m",    "rpe_trans_100m"),
            ("RPE rot 100m [deg]",   "rpe_rot_100m",      "rpe_rot_100m"),
            ("Traj length [m]",      "total_dist_m",      "total_dist_m"),
            ("Sim3/SE3 scale",       "scale",             "scale"),
        ]:
            mv = mono_result[mk]
            sv = stereo_result[sk]
            ms = f"{mv:>12.4f}" if np.isfinite(mv) else f"{'N/A':>12}"
            ss = f"{sv:>12.4f}" if np.isfinite(sv) else f"{'N/A':>12}"
            print(f"{label:<32} {ms}{ss}")
        segs_m = mono_result["rpe_n_segments_100m"]
        segs_s = stereo_result["rpe_n_segments_100m"]
        print(f"{'  (100m segments)':>32} {segs_m:>12d} {segs_s:>12d}")
        print(f"{'Tracking failures':<32} "
              f"{mono_vo.n_failures:>12d} "
              f"{stereo_vo.n_failures:>12d}")
        print(f"{'Runtime [fps]':<32} "
              f"{len(loader) / mono_time:>12.1f} "
              f"{len(loader) / stereo_time:>12.1f}")

        save_plot(cfg, loader, mono_vo, stereo_vo,
                  mono_result, stereo_result, out_dir)

        all_results[seq] = {
            "type":                  "ate",
            "mono_ate":              mono_result["ate_rmse"],
            "mono_ate_mean":         mono_result["ate_mean"],
            "mono_rpe":              mono_result["rpe_trans_rmse_d1"],
            "mono_rpe_rot":          mono_result["rpe_rot_rmse_d1"],
            "mono_rpe_100m":         mono_result["rpe_trans_100m"],
            "mono_rpe_rot_100m":     mono_result["rpe_rot_100m"],
            "mono_rpe_segs_100m":    mono_result["rpe_n_segments_100m"],
            "mono_traj_len":         mono_result["total_dist_m"],
            "mono_scale":            mono_result["scale"],
            "stereo_ate":            stereo_result["ate_rmse"],
            "stereo_ate_mean":       stereo_result["ate_mean"],
            "stereo_rpe":            stereo_result["rpe_trans_rmse_d1"],
            "stereo_rpe_rot":        stereo_result["rpe_rot_rmse_d1"],
            "stereo_rpe_100m":       stereo_result["rpe_trans_100m"],
            "stereo_rpe_rot_100m":   stereo_result["rpe_rot_100m"],
            "stereo_rpe_segs_100m":  stereo_result["rpe_n_segments_100m"],
            "stereo_traj_len":       stereo_result["total_dist_m"],
            "stereo_scale":          stereo_result["scale"],
            "mono_fps":              len(loader) / mono_time,
            "stereo_fps":            len(loader) / stereo_time,
            "mono_fail":             mono_vo.n_failures,
            "stereo_fail":           stereo_vo.n_failures,
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

        # Per-frame RPE on consecutive GT segments.
        # Stereo is metric (scale=1) → raw RPE is meaningful.
        # Mono has arbitrary scale and often a degenerate frozen trajectory
        # (corridor3: 5013 failures) → skip, report N/A.
        stereo_rpe_d1, stereo_rpe_rot_d1 = compute_rpe_d1_on_consecutive_gt(
            loader, stereo_vo.trajectory[1])

        print(f"\n── Evaluation: {seq} (start-end drift) {'─' * 15}")
        print(f"{'Metric':<32} {'Mono VO':>12} {'Stereo VO':>12}")
        print("-" * 58)
        print(f"{'Start-end drift [m]':<32} "
              f"{mono_drift:>12.4f} {stereo_drift:>12.4f}")
        rpe_str = f"{stereo_rpe_d1:>12.4f}" if np.isfinite(stereo_rpe_d1) else f"{'N/A':>12}"
        print(f"{'RPE trans d=1 [m] (stereo)':<32} {'N/A':>12} {rpe_str}")
        print(f"{'Tracking failures':<32} "
              f"{mono_vo.n_failures:>12d} {stereo_vo.n_failures:>12d}")
        print(f"{'Runtime [fps]':<32} "
              f"{len(loader) / mono_time:>12.1f} "
              f"{len(loader) / stereo_time:>12.1f}")

        save_drift_comparison_plot(
            seq, mono_vo, stereo_vo,
            mono_drift, stereo_drift, gt_poses, out_dir)

        all_results[seq] = {
            "mono_drift":    mono_drift,
            "stereo_drift":  stereo_drift,
            "mono_fps":      len(loader) / mono_time,
            "stereo_fps":    len(loader) / stereo_time,
            "mono_fail":     mono_vo.n_failures,
            "stereo_fail":   stereo_vo.n_failures,
            "mono_rpe":      float("nan"),
            "mono_rpe_rot":  float("nan"),
            "stereo_rpe":    stereo_rpe_d1,
            "stereo_rpe_rot": stereo_rpe_rot_d1,
            "type":          "drift",
        }

print("\n" + "=" * 65)
print("  FINAL RESULTS — ALL SEQUENCES")
print("=" * 65)
print(f"{'Sequence':<14} {'Metric':<22} {'Mono VO':>10} {'Stereo VO':>10}")
print("-" * 58)
def _pf(v):
    return f"{v:>10.4f}" if np.isfinite(v) else f"{'N/A':>10}"

for seq, r in all_results.items():
    if r["type"] == "ate":
        print(f"{seq:<14} {'ATE RMSE [m]':<26} "
              f"{_pf(r['mono_ate'])} {_pf(r['stereo_ate'])}")
        print(f"{'':<14} {'RPE trans d=1 [m]':<26} "
              f"{_pf(r['mono_rpe'])} {_pf(r['stereo_rpe'])}")
        print(f"{'':<14} {'RPE trans 100m [m]':<26} "
              f"{_pf(r['mono_rpe_100m'])} {_pf(r['stereo_rpe_100m'])}")
        print(f"{'':<14} {'  (segments)':<26} "
              f"{r['mono_rpe_segs_100m']:>10d} {r['stereo_rpe_segs_100m']:>10d}")
        print(f"{'':<14} {'Traj length [m]':<26} "
              f"{r['mono_traj_len']:>10.1f} {r['stereo_traj_len']:>10.1f}")
        print(f"{'':<14} {'Scale (Sim3/SE3)':<26} "
              f"{_pf(r['mono_scale'])} {'1.0000':>10}")
    else:
        print(f"{seq:<14} {'Start-end drift [m]':<26} "
              f"{_pf(r['mono_drift'])} {_pf(r['stereo_drift'])}")
    print(f"{'':<14} {'Failures':<26} "
          f"{r['mono_fail']:>10d} {r['stereo_fail']:>10d}")
    print(f"{'':<14} {'Runtime [fps]':<26} "
          f"{r['mono_fps']:>10.1f} {r['stereo_fps']:>10.1f}")
    print("-" * 62)

print("\nAll trajectory files saved under outputs/<sequence>/")

save_results_csv(all_results, "outputs/evaluation_results.csv")
