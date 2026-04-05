"""
SE(3) math utilities.
stable than np.linalg.inv for rotation matrices.
"""

import numpy as np


def Rt_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4×4 SE(3) matrix from R (3×3) and t (3,) or (3,1)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = np.asarray(t, dtype=np.float64).ravel()
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    """
    Exact SE(3) inverse using R.T — faster and more stable than np.linalg.inv.
    For rotation matrices R is orthogonal so R⁻¹ = Rᵀ exactly.

    T_wc = invert_T(T_cw)
    """
    R  = T[:3, :3]
    t  = T[:3,  3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3,  3] = -R.T @ t
    return Ti


def compose_T(T_a_b: np.ndarray, T_b_c: np.ndarray) -> np.ndarray:
    """Compose two SE(3) transforms: T_a_c = T_a_b @ T_b_c."""
    return T_a_b @ T_b_c


def cam_from_world(T_wc: np.ndarray):
    """
    Extract R_cw, t_cw from T_world_cam.
    Returns (R_cw, t_cw) suitable for cv2.projectPoints / solvePnP.
    """
    R_wc = T_wc[:3, :3]
    t_wc = T_wc[:3,  3]
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    return R_cw, t_cw