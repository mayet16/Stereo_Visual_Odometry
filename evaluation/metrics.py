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


# ── one-call wrapper ──────────────────────────────────────────────────────────

def align_and_evaluate(
    est:   List[np.ndarray],
    gt:    List[np.ndarray],
    align: str = "sim3",       # "sim3" for mono, "se3" for stereo
) -> Dict:
    """
    Align, compute ATE and RPE, return combined result dict.
    """
    if align == "sim3":
        result = align_sim3(est, gt)
    else:
        result = align_se3(est, gt)

    ate = ate_rmse(result["traj_aligned"], gt)
    rpe_1  = rpe(result["traj_aligned"], gt, delta=1)
    rpe_10 = rpe(result["traj_aligned"], gt, delta=10)

    result.update(ate)
    result["rpe_trans_rmse_d1"]  = rpe_1["rpe_trans_rmse"]
    result["rpe_rot_rmse_d1"]    = rpe_1["rpe_rot_rmse"]
    result["rpe_trans_rmse_d10"] = rpe_10["rpe_trans_rmse"]
    result["rpe_rot_rmse_d10"]   = rpe_10["rpe_rot_rmse"]
    return result