import numpy as np


def Rt_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4×4 SE(3) matrix from R (3×3) and t (3,)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = np.asarray(t, dtype=np.float64).ravel()
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    """SE(3) inverse via R.T — avoids np.linalg.inv numerical error on rotation blocks."""
    R  = T[:3, :3]
    t  = T[:3,  3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3,  3] = -R.T @ t
    return Ti


def compose_T(T_a_b: np.ndarray, T_b_c: np.ndarray) -> np.ndarray:
    return T_a_b @ T_b_c


def cam_from_world(T_wc: np.ndarray):
    """Extract (R_cw, t_cw) from T_world_cam for use with cv2.projectPoints/solvePnP."""
    R_wc = T_wc[:3, :3]
    t_wc = T_wc[:3,  3]
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    return R_cw, t_cw