"""
Essential matrix estimation, pose recovery, triangulation, PnP, and
motion-only bundle adjustment.
  - Full input validation (None check, shape check, min points)
  - det(R) sanity check after recoverPose
  - Descriptive failure reasons
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from utils.math_utils import Rt_to_T, invert_T


def estimate_essential(
    pts0: np.ndarray,
    pts1: np.ndarray,
    K:    np.ndarray,
    ransac_th:  float = 1.0,
    confidence: float = 0.999,
) -> Tuple[Optional[np.ndarray], np.ndarray]:

    if pts0 is None or pts1 is None:
        return None, np.zeros(0, bool)

    pts0 = np.asarray(pts0, dtype=np.float64)
    pts1 = np.asarray(pts1, dtype=np.float64)

    if pts0.ndim != 2 or pts0.shape[1] != 2:
        return None, np.zeros(len(pts0), bool)
    if len(pts0) < 8:
        return None, np.zeros(len(pts0), bool)

    # ── from old feature_frontend.py: finite point guard ─────────────────
    finite = np.isfinite(pts0).all(axis=1) & np.isfinite(pts1).all(axis=1)
    pts0 = pts0[finite]
    pts1 = pts1[finite]
    if len(pts0) < 8:
        return None, np.zeros(len(pts0), bool)

    # ── from old feature_frontend.py: cv2.error guard ────────────────────
    try:
        E, mask = cv2.findEssentialMat(
            pts0, pts1, K,
            method=cv2.RANSAC,
            prob=confidence,
            threshold=ransac_th,
        )
    except cv2.error:
        return None, np.zeros(len(pts0), bool)

    if E is None or mask is None:
        return None, np.zeros(len(pts0), bool)
    if E.shape != (3, 3):
        return None, np.zeros(len(pts0), bool)

    return E, mask.ravel().astype(bool)


def recover_pose(
    E:    np.ndarray,
    pts0: np.ndarray,
    pts1: np.ndarray,
    K:    np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Recover R, t from Essential matrix with cheirality check.
    Includes det(R) sanity check (from old geometry.py).
    Returns (R, t_flat, n_inliers).
    """
    n_in, R, t, _ = cv2.recoverPose(
        E, pts0, pts1, K,
        mask=mask.astype(np.uint8),
    )

    # ── det(R) sanity check (from old geometry.py) ────────────────────────
    if R is not None:
        det = float(np.linalg.det(R))
        if not (0.99 < det < 1.01):
            # Return but signal failure via n_inliers=0
            return R, t.ravel(), 0

    return R, t.ravel(), int(n_in)


def triangulate_points(
    R0: np.ndarray, t0: np.ndarray,
    R1: np.ndarray, t1: np.ndarray,
    pts0: np.ndarray,
    pts1: np.ndarray,
    K:    np.ndarray,
) -> np.ndarray:
    """
    Triangulate 3D points from two camera poses and matching 2D points.
    Returns (N, 3) array of 3D points in the frame of camera 0.
    """
    P0 = K @ np.hstack([R0, t0.reshape(3, 1)])
    P1 = K @ np.hstack([R1, t1.reshape(3, 1)])
    pts4 = cv2.triangulatePoints(
        P0, P1,
        pts0.T.astype(np.float64),
        pts1.T.astype(np.float64),
    )
    return (pts4[:3] / pts4[3]).T.astype(np.float64)


def pnp_ransac(
    pts3d:       np.ndarray,
    pts2d:       np.ndarray,
    K:           np.ndarray,
    dist:        np.ndarray,
    ransac_th:   float = 4.0,
    confidence:  float = 0.999,
    min_inliers: int   = 15,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """
    PnP + RANSAC pose estimation.
    Returns (R, t, inlier_mask_bool) or (None, None, zeros) on failure.
    """
    # ── input validation ──────────────────────────────────────────────────
    if pts3d is None or pts2d is None:
        return None, None, np.zeros(0, bool)
    if len(pts3d) < min_inliers:
        return None, None, np.zeros(len(pts3d), bool)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d.astype(np.float64),
        pts2d.astype(np.float64),
        K, dist,
        iterationsCount   = 200,
        reprojectionError = ransac_th,
        confidence        = confidence,
        flags             = cv2.SOLVEPNP_ITERATIVE,
    )

    if not ok or inliers is None or len(inliers) < min_inliers:
        return None, None, np.zeros(len(pts3d), bool)

    R, _ = cv2.Rodrigues(rvec)

    # ── det(R) sanity check ───────────────────────────────────────────────
    det = float(np.linalg.det(R))
    if not (0.99 < det < 1.01):
        return None, None, np.zeros(len(pts3d), bool)

    mask = np.zeros(len(pts3d), bool)
    mask[inliers.ravel()] = True
    return R, tvec.ravel(), mask


def refine_pose_ba(
    R:     np.ndarray,
    t:     np.ndarray,
    pts3d: np.ndarray,
    pts2d: np.ndarray,
    K:     np.ndarray,
    dist:  np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Motion-only bundle adjustment via Levenberg-Marquardt.
    Refines (R, t) by minimising reprojection error over inlier points.
    Returns refined (R, t).
    """
    if len(pts3d) < 6:
        return R, t

    rvec, _ = cv2.Rodrigues(R)
    try:
        rvec, tvec, *_ = cv2.solvePnPRefineLM(
            pts3d.astype(np.float64),
            pts2d.astype(np.float64),
            K, dist,
            rvec.astype(np.float64),
            t.reshape(3, 1).astype(np.float64),
        )
        R_ref, _ = cv2.Rodrigues(rvec)

        # ── det(R) check on refined result ────────────────────────────────
        det = float(np.linalg.det(R_ref))
        if not (0.99 < det < 1.01):
            return R, t   # reject refinement, keep original

        return R_ref, tvec.ravel()
    except cv2.error:
        return R, t   # cv2 LM failed — keep original