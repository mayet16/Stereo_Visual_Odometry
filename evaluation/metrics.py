"""
Trajectory evaluation metrics for TUM VI.

Functions
---------
align_sim3        : Umeyama Sim(3) alignment  (for monocular VO)
align_se3         : Least-squares SE(3) alignment (for stereo VO)
ate_rmse          : Absolute Trajectory Error
rpe               : Relative Pose Error over fixed segments
align_and_evaluate: One-call wrapper used by main.py
"""

import numpy as np
from typing import List, Dict, Optional


# ── Sim(3) alignment (Umeyama 1991) ──────────────────────────────────────────

def align_sim3(
    est:  List[np.ndarray],   # estimated T_world_cam  (4×4 list)
    gt:   List[np.ndarray],   # ground-truth T_world_cam (4×4 list)
) -> Dict:
    """
    Compute optimal Sim(3) alignment: scale s, rotation R, translation t
    such that  s*R*p_est + t  ≈  p_gt  in least-squares sense.

    Returns dict with keys:
      s, R, t          – Sim(3) parameters
      traj_aligned     – list of aligned 4×4 poses
      scale            – recovered scale
    """
    p_est = np.array([T[:3, 3] for T in est])   # (N,3)
    p_gt  = np.array([T[:3, 3] for T in gt])

    n = len(p_est)
    mu_e = p_est.mean(0)
    mu_g = p_gt.mean(0)
    pe   = p_est - mu_e
    pg   = p_gt  - mu_g

    var_e = (pe ** 2).sum() / n
    W     = (pg.T @ pe) / n                      # 3×3 cross-covariance

    U, D, Vt = np.linalg.svd(W)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    s = (D * S.diagonal()).sum() / max(var_e, 1e-10)
    t = mu_g - s * R @ mu_e

    # Apply to all poses
    aligned = []
    for T in est:
        T_new = T.copy()
        T_new[:3, 3] = s * R @ T[:3, 3] + t
        T_new[:3, :3] = R @ T[:3, :3]
        aligned.append(T_new)

    return {"s": s, "R": R, "t": t, "traj_aligned": aligned, "scale": s}


def align_se3(
    est: List[np.ndarray],
    gt:  List[np.ndarray],
) -> Dict:
    """
    SE(3) alignment via horn's method (no scale).
    Used for stereo VO (metric scale already correct).
    """
    p_est = np.array([T[:3, 3] for T in est])
    p_gt  = np.array([T[:3, 3] for T in gt])

    mu_e = p_est.mean(0)
    mu_g = p_gt.mean(0)
    pe   = p_est - mu_e
    pg   = p_gt  - mu_g

    W    = pg.T @ pe
    U, _, Vt = np.linalg.svd(W)
    S    = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R    = U @ S @ Vt
    t    = mu_g - R @ mu_e

    aligned = []
    for T in est:
        T_new = T.copy()
        T_new[:3, 3]  = R @ T[:3, 3] + t
        T_new[:3, :3] = R @ T[:3, :3]
        aligned.append(T_new)

    return {"R": R, "t": t, "traj_aligned": aligned, "scale": 1.0}


# ── ATE ───────────────────────────────────────────────────────────────────────

def ate_rmse(
    aligned: List[np.ndarray],
    gt:      List[np.ndarray],
) -> Dict:
    """
    ATE after alignment.
    Returns dict: ate_rmse, ate_mean, ate_std, ate_max, errors (N,)
    """
    errors = np.array([
        np.linalg.norm(a[:3, 3] - g[:3, 3])
        for a, g in zip(aligned, gt)
    ])
    return {
        "ate_rmse": float(np.sqrt((errors**2).mean())),
        "ate_mean": float(errors.mean()),
        "ate_std":  float(errors.std()),
        "ate_max":  float(errors.max()),
        "errors":   errors,
    }


# ── RPE ───────────────────────────────────────────────────────────────────────

def rpe(
    est: List[np.ndarray],
    gt:  List[np.ndarray],
    delta: int = 1,
) -> Dict:
    """
    Relative Pose Error over segments of length `delta` frames.
    Returns dict: rpe_trans_rmse, rpe_rot_rmse
    """
    trans_errs = []
    rot_errs   = []
    for i in range(len(est) - delta):
        # Relative pose in estimated trajectory
        dT_est = np.linalg.inv(est[i]) @ est[i + delta]
        # Relative pose in GT
        dT_gt  = np.linalg.inv(gt[i])  @ gt[i + delta]
        # Error pose
        E = np.linalg.inv(dT_gt) @ dT_est
        trans_errs.append(np.linalg.norm(E[:3, 3]))
        # Rotation angle from trace
        cos_th = np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1)
        rot_errs.append(float(np.degrees(np.arccos(cos_th))))

    trans_errs = np.array(trans_errs)
    rot_errs   = np.array(rot_errs)
    return {
        "rpe_trans_rmse": float(np.sqrt((trans_errs**2).mean())),
        "rpe_rot_rmse":   float(np.sqrt((rot_errs**2).mean())),
        "trans_errors":   trans_errs,
        "rot_errors":     rot_errs,
    }


# ── start-end drift ───────────────────────────────────────────────────────────

def start_end_drift(
    est: List[np.ndarray],
    gt:  List[np.ndarray],
) -> float:
    """
    ||T_est_end - T_gt_end|| after aligning start poses.
    Used for corridor3 and outdoors5 (start+end GT only).
    """
    # Align start
    T_align = gt[0] @ np.linalg.inv(est[0])
    est_end_aligned = T_align @ est[-1]
    return float(np.linalg.norm(est_end_aligned[:3, 3] - gt[-1][:3, 3]))


# ── distance-based RPE (spec: d=100 m segments) ───────────────────────────────

def rpe_distance_based(
    est:         List[np.ndarray],
    gt:          List[np.ndarray],
    target_dist: float = 100.0,
) -> Dict:
    """
    RPE over segments whose path length ≈ target_dist metres.
    For each frame i, finds j so that the estimated trajectory
    travels target_dist metres from i to j, then computes the
    relative-pose error against the matching GT segment.

    Returns nan metrics (with n_segments=0) when the total trajectory
    is shorter than target_dist (e.g. room2 ~15 m < 100 m).
    """
    n = len(est)
    step_lengths = np.zeros(n)
    for i in range(1, n):
        step_lengths[i] = np.linalg.norm(est[i][:3, 3] - est[i-1][:3, 3])
    cumulative = np.cumsum(step_lengths)
    total_dist = float(cumulative[-1])

    trans_errs, rot_errs = [], []
    for i in range(n - 1):
        target = cumulative[i] + target_dist
        if target > cumulative[-1]:
            break
        j = int(np.searchsorted(cumulative, target))
        if j >= n:
            break
        dT_est = np.linalg.inv(est[i]) @ est[j]
        dT_gt  = np.linalg.inv(gt[i])  @ gt[j]
        E      = np.linalg.inv(dT_gt)  @ dT_est
        trans_errs.append(float(np.linalg.norm(E[:3, 3])))
        cos_th = float(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1))
        rot_errs.append(float(np.degrees(np.arccos(cos_th))))

    if len(trans_errs) == 0:
        return {
            "rpe_trans_rmse": float("nan"),
            "rpe_rot_rmse":   float("nan"),
            "n_segments":     0,
            "total_dist_m":   total_dist,
            "target_dist_m":  target_dist,
        }

    te = np.array(trans_errs)
    re = np.array(rot_errs)
    return {
        "rpe_trans_rmse": float(np.sqrt((te**2).mean())),
        "rpe_rot_rmse":   float(np.sqrt((re**2).mean())),
        "n_segments":     len(te),
        "total_dist_m":   total_dist,
        "target_dist_m":  target_dist,
        "trans_errors":   te,
        "rot_errors":     re,
    }


# ── rotation-based RPE (spec: θ=60° segments) ────────────────────────────────

def rpe_rotation_based(
    est:            List[np.ndarray],
    gt:             List[np.ndarray],
    target_rot_deg: float = 60.0,
) -> Dict:
    """
    RPE_rot over segments whose cumulative ESTIMATED rotation ≈ target_rot_deg.
    Spec: θ=60° for rotation — measures how much rotation drifts per 60° of travel.

    Segment endpoints defined from estimated trajectory (consistent with
    rpe_distance_based which uses estimated path length for 100m segments).
    """
    n = len(est)
    target_rot_rad = np.radians(target_rot_deg)

    # Cumulative rotation along estimated trajectory
    step_rots = np.zeros(n)
    for i in range(1, n):
        dR = est[i - 1][:3, :3].T @ est[i][:3, :3]
        cos_th = float(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        step_rots[i] = np.arccos(cos_th)
    cumulative_rot = np.cumsum(step_rots)
    total_rot_deg  = float(np.degrees(cumulative_rot[-1]))

    rot_errs = []
    for i in range(n - 1):
        target = cumulative_rot[i] + target_rot_rad
        if target > cumulative_rot[-1]:
            break
        j = int(np.searchsorted(cumulative_rot, target))
        if j >= n:
            break
        dT_est = np.linalg.inv(est[i]) @ est[j]
        dT_gt  = np.linalg.inv(gt[i])  @ gt[j]
        E      = np.linalg.inv(dT_gt)  @ dT_est
        cos_th = float(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1))
        rot_errs.append(float(np.degrees(np.arccos(cos_th))))

    if len(rot_errs) == 0:
        return {
            "rpe_rot_rmse":   float("nan"),
            "n_segments":     0,
            "total_rot_deg":  total_rot_deg,
            "target_rot_deg": target_rot_deg,
        }

    re = np.array(rot_errs)
    return {
        "rpe_rot_rmse":   float(np.sqrt((re ** 2).mean())),
        "n_segments":     len(re),
        "total_rot_deg":  total_rot_deg,
        "target_rot_deg": target_rot_deg,
        "rot_errors":     re,
    }


# ── one-call wrapper ──────────────────────────────────────────────────────────

def align_and_evaluate(
    est:   List[np.ndarray],
    gt:    List[np.ndarray],
    align: str = "sim3",       # "sim3" for mono, "se3" for stereo
) -> Dict:
    """
    Align, compute ATE and frame-based + distance-based RPE.
    """
    if align == "sim3":
        result = align_sim3(est, gt)
    else:
        result = align_se3(est, gt)

    ate    = ate_rmse(result["traj_aligned"], gt)
    rpe_1  = rpe(result["traj_aligned"], gt, delta=1)
    rpe_10 = rpe(result["traj_aligned"], gt, delta=10)
    rpe_d   = rpe_distance_based(result["traj_aligned"], gt, target_dist=100.0)
    rpe_r   = rpe_rotation_based(result["traj_aligned"], gt, target_rot_deg=60.0)

    result.update(ate)
    result["rpe_trans_rmse_d1"]    = rpe_1["rpe_trans_rmse"]
    result["rpe_rot_rmse_d1"]      = rpe_1["rpe_rot_rmse"]
    result["rpe_trans_rmse_d10"]   = rpe_10["rpe_trans_rmse"]
    result["rpe_rot_rmse_d10"]     = rpe_10["rpe_rot_rmse"]
    result["rpe_trans_100m"]       = rpe_d["rpe_trans_rmse"]
    result["rpe_n_segments_100m"]  = rpe_d["n_segments"]
    result["total_dist_m"]         = rpe_d["total_dist_m"]
    result["rpe_rot_60deg"]        = rpe_r["rpe_rot_rmse"]
    result["rpe_n_segments_60deg"] = rpe_r["n_segments"]
    result["total_rot_deg"]        = rpe_r["total_rot_deg"]
    return result