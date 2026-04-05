"""
main.py  –  Stereo_VO project entry point
Runs monocular VO + stereo VO on all three TUM VI sequences.
Prints matrix outputs, feature stats, and evaluation metrics.
"""


import os
# os.environ["QT_LOGGING_RULES"] = "*.debug=false;*.warning=false"
# os.environ["QT_QPA_PLATFORM"]  = "offscreen"   # no display needed

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — saves PNG, no Qt required
import matplotlib.pyplot as plt

from data.data_loader        import load_run_config, TUMVILoader
from mono_vo.pipeline        import MonoVO, MonoVOConfig
from mono_vo.feature_tracker import FeatureConfig
from stereo_vo.pipeline      import StereoVO, StereoVOConfig
from stereo_vo.disparity     import DisparityConfig
from evaluation.metrics      import (align_and_evaluate,
                                      start_end_drift)
from utils.print_utils       import (print_camera_intrinsics,ensure_dir, show_frames,
                                      print_stereo_params, print_stereo_extrinsics, print_sequence_summary)



# after print_stereo_params(...)


np.random.seed(42)

# ── configs ───────────────────────────────────────────────────────────────────
SEQUENCES = [
    "config/tumvi_room2.yaml",
    "config/tumvi_corridor3.yaml",
    "config/tumvi_outdoors5.yaml",
]

# room2 has full GT  → use ATE
# corridor3/outdoors5 have start+end GT only → use start-end drift
FULL_GT = {"room2"}

# # ── VO configs (identical across sequences) ───────────────────────────────────
# mono_cfg = MonoVOConfig(
#     feature = FeatureConfig(
#         method="orb", max_features=3000, ratio_thresh=0.75,
#         min_matches=50, lk_win_size=21, lk_max_level=3,
#         fast_threshold=15, grid_rows=5, grid_cols=5,
#     ),
#     min_tracked_pts=120, max_map_pts=800,
#     min_parallax_px=8.0, max_parallax_px=40.0,
#     init_scale=0.02,
#     pnp_min_inliers=15, pnp_ransac_th=6.0, reproj_thresh=4.0,
#     use_ba=True, use_local_ba=True,
#     local_ba_window=7, local_ba_every=5,
#     max_velocity=0.5,
#     verbose=False,
# )
#
# stereo_cfg = StereoVOConfig(
#     feature = FeatureConfig(
#         method="orb", max_features=3000, ratio_thresh=0.75,
#         min_matches=50, lk_win_size=21, lk_max_level=3,
#         fast_threshold=15, grid_rows=5, grid_cols=5,
#     ),
#     disparity = DisparityConfig(
#         method="sgbm", num_disparities=64, block_size=11,
#         p1_coeff=8, p2_coeff=32,
#         uniqueness=10, speckle_window=100, speckle_range=2,
#         min_depth=0.5, max_depth=5.0,
#         min_disparity=1.5, patch_radius=3,
#     ),
#     min_tracked_pts=120, max_map_pts=500,
#     pnp_min_inliers=15, pnp_ransac_th=4.0, reproj_thresh=4.0,
#     use_ba=True, max_velocity=0.5,
#     verbose=False,
# )


# Per-sequence VO configs
MONO_CONFIGS = {
    "room2": MonoVOConfig(
        feature=FeatureConfig(method="orb", max_features=3000,
            lk_win_size=21, lk_max_level=3, grid_rows=5, grid_cols=5),
        min_tracked_pts=120, max_map_pts=800,
        min_parallax_px=8.0, max_parallax_px=40.0,
        init_scale=0.02,
        pnp_min_inliers=15, pnp_ransac_th=6.0, reproj_thresh=4.0,
        use_ba=True, max_velocity=0.5, verbose=False,
    ),
    "corridor3": MonoVOConfig(
        feature=FeatureConfig(method="orb", max_features=3000,
            lk_win_size=21, lk_max_level=3, grid_rows=5, grid_cols=5),
        min_tracked_pts=100, max_map_pts=1200,
        min_parallax_px=4.0,    # lower gate — corridor has less lateral motion
        max_parallax_px=60.0,
        init_scale=0.02,
        pnp_min_inliers=12, pnp_ransac_th=6.0, reproj_thresh=4.0,
        use_ba=True, max_velocity=1.0, verbose=False,
    ),
    "outdoors5": MonoVOConfig(
        feature=FeatureConfig(method="orb", max_features=3000,
            lk_win_size=21, lk_max_level=3, grid_rows=5, grid_cols=5),
        min_tracked_pts=120, max_map_pts=800,
        min_parallax_px=6.0,
        max_parallax_px=80.0,   # outdoor motion can be large
        init_scale=0.05,        # faster outdoor walking
        pnp_min_inliers=15, pnp_ransac_th=6.0, reproj_thresh=4.0,
        use_ba=True, max_velocity=2.0,  # allow faster motion
        verbose=False,
    ),
}

STEREO_CONFIGS = {
    "room2": StereoVOConfig(
        feature=FeatureConfig(method="orb", max_features=3000,
            lk_win_size=21, lk_max_level=3, grid_rows=5, grid_cols=5),
        disparity=DisparityConfig(
            num_disparities=64, block_size=11,
            min_depth=0.5, max_depth=5.0, min_disparity=1.5, patch_radius=3),
        min_tracked_pts=120, max_map_pts=500,
        pnp_min_inliers=15, pnp_ransac_th=4.0, reproj_thresh=4.0,
        use_ba=True, max_velocity=0.5, verbose=False,
    ),
    "corridor3": StereoVOConfig(
        feature=FeatureConfig(method="orb", max_features=3000,
            lk_win_size=21, lk_max_level=3, grid_rows=5, grid_cols=5),
        disparity=DisparityConfig(
            num_disparities=128,  # corridor is wider
            block_size=11,
            min_depth=0.5, max_depth=8.0,   # longer range
            min_disparity=1.0, patch_radius=3),
        min_tracked_pts=120, max_map_pts=500,
        pnp_min_inliers=15, pnp_ransac_th=4.0, reproj_thresh=4.0,
        use_ba=True, max_velocity=1.0, verbose=False,
    ),
    "outdoors5": StereoVOConfig(
        feature=FeatureConfig(method="orb", max_features=3000,
            lk_win_size=21, lk_max_level=3, grid_rows=5, grid_cols=5),
        disparity=DisparityConfig(
            num_disparities=64,
            block_size=11,
            min_depth=0.5, max_depth=5.0,  # outdoor scenes are far
            min_disparity=1.0, patch_radius=3),
        min_tracked_pts=120, max_map_pts=600,
        pnp_min_inliers=15, pnp_ransac_th=4.0, reproj_thresh=4.0,
        use_ba=True, max_velocity=2.0, verbose=False,
    ),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def collect_gt(loader, poses):
    """Collect GT poses aligned to estimated trajectory indices."""
    gt, est = [], []
    for i, frame in enumerate(loader):
        if i >= len(poses):
            break
        if frame.T_world_cam0 is not None:
            gt.append(frame.T_world_cam0)
            est.append(poses[i])
    return est, gt


def collect_gt_timestamps(loader, poses, timestamps):
    ts = []
    for i, frame in enumerate(loader):
        if i >= len(poses):
            break
        if frame.T_world_cam0 is not None:
            ts.append(timestamps[i])
    return np.array(ts)


def run_mono(loader, seq, cfg):
    print(f"\nRunning monocular VO ...")
    vo = MonoVO(loader.calib.cam0, cfg)
    t0 = time.time()
    for frame in loader:
        vo.process(frame.img_left, frame.timestamp)
        frame.release()
    elapsed = time.time() - t0
    fps = len(loader) / elapsed
    print(f"  Done: {len(loader)} frames  {elapsed:.1f}s  "
          f"({fps:.1f} fps)  failures={vo.n_failures}")
    return vo, elapsed


def run_stereo(loader, seq, cfg):
    print(f"\nRunning stereo VO ...")
    vo = StereoVO(loader.calib, cfg)
    t0 = time.time()
    for frame in loader:
        vo.process(frame.img_left, frame.img_right, frame.timestamp)
        frame.release()
    elapsed = time.time() - t0
    fps = len(loader) / elapsed
    print(f"  Done: {len(loader)} frames  {elapsed:.1f}s  "
          f"({fps:.1f} fps)  failures={vo.n_failures}")
    return vo, elapsed


def save_plot(cfg, loader, mono_vo, stereo_vo,
              mono_result, stereo_result, out_dir):
    """Save top-down + ATE-over-time comparison plot."""
    gt_arr    = np.array([f.T_world_cam0[:3,3]
                           for f in loader if f.T_world_cam0 is not None])
    mono_al   = np.array([T[:3,3] for T in mono_result['traj_aligned']])
    stereo_al = np.array([T[:3,3] for T in stereo_result['traj_aligned']])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot(gt_arr[:,0],    gt_arr[:,1],    "g--", lw=1.5, label="GT")
    ax.plot(mono_al[:,0],   mono_al[:,1],   "b-",  lw=1.0, alpha=0.8,
            label=f"Mono VO (Sim3, ATE={mono_result['ate_rmse']:.3f}m)")
    ax.plot(stereo_al[:,0], stereo_al[:,1], "r-",  lw=1.2,
            label=f"Stereo VO (SE3, ATE={stereo_result['ate_rmse']:.3f}m)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Top-down (aligned)"); ax.legend(fontsize=8)
    ax.set_aspect("equal")

    ax2 = axes[1]
    mono_ts   = collect_gt_timestamps(
        loader, mono_vo.trajectory[1], np.array(mono_vo.trajectory[0]))
    stereo_ts = collect_gt_timestamps(
        loader, stereo_vo.trajectory[1], np.array(stereo_vo.trajectory[0]))
    if len(mono_ts):
        mono_ts -= mono_ts[0]
    if len(stereo_ts):
        stereo_ts -= stereo_ts[0]

    if 'errors' in mono_result and len(mono_ts) == len(mono_result['errors']):
        ax2.plot(mono_ts,   mono_result['errors'],   "b-", lw=0.8, label="Mono ATE")
    if 'errors' in stereo_result and len(stereo_ts) == len(stereo_result['errors']):
        ax2.plot(stereo_ts, stereo_result['errors'], "r-", lw=0.8, label="Stereo ATE")
    ax2.set_xlabel("time [s]"); ax2.set_ylabel("ATE [m]")
    ax2.set_title("ATE over time"); ax2.legend(fontsize=8)
    ax2.set_ylim(bottom=0)

    plt.suptitle(f"Mono vs Stereo VO — {cfg.sequence_name}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "comparison.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Saved {path}")



# ── main loop over all sequences ──────────────────────────────────────────────

all_results = {}   # sequence_name → dict of metrics

# Print stereo params once (same calibration for all sequences)
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
    cfg    = load_run_config(config_file)
    loader = TUMVILoader.from_config(cfg)
    seq    = cfg.sequence_name
    out_dir = f"outputs/{seq}"
    ensure_dir(out_dir)

    show_frames(loader, title=f"cam0 {seq}", max_frames=100)

    print_sequence_summary(
        name        = seq,
        n_frames    = len(loader),
        duration    = loader.timestamps[-1] - loader.timestamps[0],
        n_gt        = sum(1 for f in loader if f.T_world_cam0 is not None),
        baseline_cm = loader.calib.baseline * 100,
    )

    # ── run both pipelines ────────────────────────────────────────────────
    # mono_vo,   mono_time   = run_mono(loader, seq)
    # stereo_vo, stereo_time = run_stereo(loader, seq)

    mono_cfg = MONO_CONFIGS[seq]
    stereo_cfg = STEREO_CONFIGS[seq]

    mono_vo, mono_time = run_mono(loader, seq, mono_cfg)
    stereo_vo, stereo_time = run_stereo(loader, seq, stereo_cfg)

    # ── save trajectories ─────────────────────────────────────────────────
    mono_vo.save_trajectory(os.path.join(out_dir, "mono_traj.txt"))
    stereo_vo.save_trajectory(os.path.join(out_dir, "stereo_traj.txt"))

    # ── evaluate ──────────────────────────────────────────────────────────
    mono_est,   mono_gt   = collect_gt(loader, mono_vo.trajectory[1])
    stereo_est, stereo_gt = collect_gt(loader, stereo_vo.trajectory[1])

    if seq in FULL_GT and len(mono_gt) > 10:
        # Full ATE evaluation for room2
        mono_result   = align_and_evaluate(mono_est,   mono_gt,   align="sim3")
        stereo_result = align_and_evaluate(stereo_est, stereo_gt, align="se3")

        print(f"\n── Evaluation: {seq} {'─'*30}")
        print(f"{'Metric':<28} {'Mono VO':>12} {'Stereo VO':>12}")
        print("-" * 54)
        for label, mk, sk in [
            ("ATE RMSE [m]",           "ate_rmse",          "ate_rmse"),
            ("ATE mean [m]",           "ate_mean",          "ate_mean"),
            ("RPE trans d=1 [m]",      "rpe_trans_rmse_d1", "rpe_trans_rmse_d1"),
            ("RPE rot d=1 [deg]",      "rpe_rot_rmse_d1",   "rpe_rot_rmse_d1"),
            ("Sim3/SE3 scale",         "scale",             "scale"),
        ]:
            print(f"{label:<28} "
                  f"{mono_result[mk]:>12.4f} "
                  f"{stereo_result[sk]:>12.4f}")
        print(f"{'Tracking failures':<28} "
              f"{mono_vo.n_failures:>12d} "
              f"{stereo_vo.n_failures:>12d}")
        print(f"{'Runtime [fps]':<28} "
              f"{len(loader)/mono_time:>12.1f} "
              f"{len(loader)/stereo_time:>12.1f}")

        save_plot(cfg, loader, mono_vo, stereo_vo,
                  mono_result, stereo_result, out_dir)

        all_results[seq] = {
            "mono_ate":   mono_result["ate_rmse"],
            "stereo_ate": stereo_result["ate_rmse"],
            "mono_rpe":   mono_result["rpe_trans_rmse_d1"],
            "stereo_rpe": stereo_result["rpe_trans_rmse_d1"],
            "mono_fps":   len(loader) / mono_time,
            "stereo_fps": len(loader) / stereo_time,
            "mono_fail":  mono_vo.n_failures,
            "stereo_fail":stereo_vo.n_failures,
            "mono_scale": mono_result["scale"],
            "type":       "ate",
        }

    else:
        # Start-end drift for corridor3 / outdoors5
        # GT only has first and last pose

        # ── 1. Compute drift first ────────────────────────────────────────────
        gt_poses = [f.T_world_cam0 for f in loader if f.T_world_cam0 is not None]
        if len(gt_poses) >= 2:
            mono_drift = start_end_drift(mono_vo.trajectory[1],
                                         [gt_poses[0], gt_poses[-1]])
            stereo_drift = start_end_drift(stereo_vo.trajectory[1],
                                           [gt_poses[0], gt_poses[-1]])
        else:
            mono_drift = stereo_drift = float("nan")

        # ── 2. Print evaluation ───────────────────────────────────────────────
        print(f"\n── Evaluation: {seq} (start-end drift) {'─' * 15}")
        print(f"{'Metric':<28} {'Mono VO':>12} {'Stereo VO':>12}")
        print("-" * 54)
        print(f"{'Start-end drift [m]':<28} "
              f"{mono_drift:>12.4f} {stereo_drift:>12.4f}")
        print(f"{'Tracking failures':<28} "
              f"{mono_vo.n_failures:>12d} {stereo_vo.n_failures:>12d}")
        print(f"{'Runtime [fps]':<28} "
              f"{len(loader) / mono_time:>12.1f} "
              f"{len(loader) / stereo_time:>12.1f}")

        # ── 3. Plot trajectories ──────────────────────────────────────────────
        mono_arr = np.array([T[:3, 3] for T in mono_vo.trajectory[1]])
        stereo_arr = np.array([T[:3, 3] for T in stereo_vo.trajectory[1]])
        gt_sparse = np.array([T[:3, 3] for T in gt_poses]) if gt_poses else None

        mono_ts = np.array(mono_vo.trajectory[0]);
        mono_ts -= mono_ts[0]
        stereo_ts = np.array(stereo_vo.trajectory[0]);
        stereo_ts -= stereo_ts[0]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: top-down x-z plane
        ax = axes[0]
        ax.plot(mono_arr[:, 0], mono_arr[:, 2], "b-", lw=0.8, alpha=0.8,
                label=f"Mono VO  (drift={mono_drift:.2f}m)")
        ax.plot(stereo_arr[:, 0], stereo_arr[:, 2], "r-", lw=0.8, alpha=0.8,
                label=f"Stereo VO (drift={stereo_drift:.2f}m)")
        if gt_sparse is not None and len(gt_sparse):
            ax.scatter(gt_sparse[[0, -1], 0], gt_sparse[[0, -1], 2],
                       c="green", s=60, zorder=5, marker="*",
                       label="GT start / end")
        ax.set_xlabel("x [m]");
        ax.set_ylabel("z [m]")
        ax.set_title("Top-down  x–z  (raw, no alignment)")
        ax.legend(fontsize=8)

        # Right: z over time — shows drift growth rate
        ax2 = axes[1]
        ax2.plot(mono_ts, mono_arr[:, 2], "b-", lw=0.8, label="Mono z")
        ax2.plot(stereo_ts, stereo_arr[:, 2], "r-", lw=0.8, label="Stereo z")
        ax2.set_xlabel("time [s]");
        ax2.set_ylabel("z [m]")
        ax2.set_title("Z position over time  (drift visible)")
        ax2.legend(fontsize=8)

        plt.suptitle(
            f"Mono vs Stereo VO — {seq}  |  "
            f"mono drift={mono_drift:.2f}m  stereo drift={stereo_drift:.2f}m",
            fontsize=11,
        )
        plt.tight_layout()
        plot_path = os.path.join(out_dir, "comparison.png")
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"  Saved {plot_path}")

        # ── 4. Store results ──────────────────────────────────────────────────
        all_results[seq] = {
            "mono_drift": mono_drift,
            "stereo_drift": stereo_drift,
            "mono_fps": len(loader) / mono_time,
            "stereo_fps": len(loader) / stereo_time,
            "mono_fail": mono_vo.n_failures,
            "stereo_fail": stereo_vo.n_failures,
            "type": "drift",
        }

# ── final summary table ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  FINAL RESULTS — ALL SEQUENCES")
print("=" * 65)
print(f"{'Sequence':<14} {'Metric':<22} {'Mono VO':>10} {'Stereo VO':>10}")
print("-" * 58)
for seq, r in all_results.items():
    if r["type"] == "ate":
        print(f"{seq:<14} {'ATE RMSE [m]':<22} "
              f"{r['mono_ate']:>10.4f} {r['stereo_ate']:>10.4f}")
        print(f"{'':<14} {'RPE trans d=1 [m]':<22} "
              f"{r['mono_rpe']:>10.4f} {r['stereo_rpe']:>10.4f}")
        print(f"{'':<14} {'Scale (Sim3/SE3)':<22} "
              f"{r['mono_scale']:>10.4f} {'1.0000':>10}")
    else:
        print(f"{seq:<14} {'Start-end drift [m]':<22} "
              f"{r['mono_drift']:>10.4f} {r['stereo_drift']:>10.4f}")
    print(f"{'':<14} {'Failures':<22} "
          f"{r['mono_fail']:>10d} {r['stereo_fail']:>10d}")
    print(f"{'':<14} {'Runtime [fps]':<22} "
          f"{r['mono_fps']:>10.1f} {r['stereo_fps']:>10.1f}")
    print("-" * 58)

print("\nAll trajectory files saved under outputs/<sequence>/")