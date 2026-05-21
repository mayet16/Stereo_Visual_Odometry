import numpy as np
import cv2
from typing import Optional

from pathlib import Path

def ensure_dir(path: str) -> Path:
    """Create directory and parents if needed; return resolved Path."""
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p

def show_frames(loader, title: str = "cam0", max_frames: int = 300) -> None:
    """Display frames via cv2.imshow; press ESC to stop early."""
    import cv2
    print(f"\nShowing frames (press ESC to stop) ...")
    for i, frame in enumerate(loader):
        if i == 0:
            print(f"  First frame shape: {frame.img_left.shape}")
        cv2.imshow(title, frame.img_left)
        if cv2.waitKey(30) == 27:
            break
        if i >= max_frames:
            break
        frame.release()
    cv2.destroyWindow(title)

def print_camera_intrinsics(K: np.ndarray, name: str = "cam0") -> None:
    print(f"\n── Camera intrinsics K  ({name}) {'─'*20}")
    print(f"  fx = {K[0,0]:.4f}   fy = {K[1,1]:.4f}")
    print(f"  cx = {K[0,2]:.4f}   cy = {K[1,2]:.4f}")
    print(f"  K =\n{_fmt_matrix(K)}")


def print_relative_pose(R: np.ndarray, t: np.ndarray,
                         label: str = "init pair") -> None:
    rvec, _ = cv2.Rodrigues(R)
    angle   = float(np.linalg.norm(rvec) * 180.0 / np.pi)
    ax      = rvec.ravel() / (np.linalg.norm(rvec) + 1e-12)
    t_flat  = t.ravel()
    dom     = ["x (lateral)", "y (vertical)", "z (forward)"][int(np.argmax(np.abs(t_flat)))]
    print(f"\n── Relative pose  ({label}) {'─'*20}")
    print(f"  R =\n{_fmt_matrix(R)}")
    print(f"  Rotation angle  : {angle:.3f}°  "
          f"axis = [{ax[0]:.3f}, {ax[1]:.3f}, {ax[2]:.3f}]")
    print(f"  t (unit dir)    : [{t_flat[0]:.4f}, {t_flat[1]:.4f}, {t_flat[2]:.4f}]")
    print(f"  Dominant motion : {dom}")


def print_essential_matrix(E: np.ndarray) -> None:
    sv = np.linalg.svd(E, compute_uv=False)
    print(f"\n── Essential matrix E {'─'*25}")
    print(f"  E =\n{_fmt_matrix(E)}")
    print(f"  Singular values : [{sv[0]:.4f}, {sv[1]:.4f}, {sv[2]:.4f}]")
    print(f"  Quality check   : σ1≈σ2, σ3≈0 → "
          f"{'PASS' if abs(sv[0]-sv[1])/sv[0] < 0.1 and sv[2] < 0.01 else 'WARN'}")


def print_fundamental_matrix(F: np.ndarray) -> None:
    sv = np.linalg.svd(F, compute_uv=False)
    print(f"\n── Fundamental matrix F {'─'*23}")
    print(f"  F =\n{_fmt_matrix(F)}")
    print(f"  Singular values : [{sv[0]:.6f}, {sv[1]:.6f}, {sv[2]:.6f}]")
    print(f"  Rank check      : rank should be 2 → σ3≈0: "
          f"{'PASS' if sv[2] < 1e-3 else 'WARN'}")
    print(f"  det(F)          : {np.linalg.det(F):.2e}  (should be ~0)")


def print_feature_stats(n_kp0: int, n_kp1: int,
                         n_matches: int, n_ransac: int,
                         n_pose: int) -> None:
    print(f"\n── Feature matching  {'─'*27}")
    print(f"  Keypoints frame 0 / 1 : {n_kp0} / {n_kp1}")
    print(f"  After ratio test       : {n_matches}  "
          f"({100*n_matches/max(n_kp0,1):.1f}% of kp0)")
    print(f"  After F-RANSAC         : {n_ransac}  "
          f"(ratio {n_ransac/max(n_matches,1):.2f})")
    print(f"  After cheirality       : {n_pose}  "
          f"(recoverPose inliers)")


def print_map_point(X: np.ndarray, uv: np.ndarray,
                     desc_shape: tuple, lid: int = 0) -> None:
    print(f"\n── Example landmark  id={lid} {'─'*22}")
    print(f"  3D position X : [{X[0]:.4f}, {X[1]:.4f}, {X[2]:.4f}]")
    print(f"  Pixel obs uv  : [{uv[0]:.1f}, {uv[1]:.1f}]")
    print(f"  Descriptor    : shape {desc_shape}  (ORB binary)")


def print_stereo_params(f: float, B: float, cx: float, cy: float) -> None:
    print(f"\n── Stereo calibration  {'─'*25}")
    print(f"  Focal length  f  : {f:.4f} px")
    print(f"  Baseline      B  : {B*100:.4f} cm  =  {B:.6f} m")
    print(f"  Principal pt     : cx={cx:.4f}  cy={cy:.4f}")
    print(f"  fB product       : {f*B:.4f} px·m")
    print(f"  Depth at d=10px  : {f*B/10:.4f} m")
    print(f"  Depth at d=5px   : {f*B/5:.4f} m")


def print_sequence_summary(name: str, n_frames: int, duration: float,
                            n_gt: int, baseline_cm: float) -> None:
    print(f"\n{'='*50}")
    print(f"  Sequence  : {name}")
    print(f"  Frames    : {n_frames}   Duration: {duration:.1f}s")
    print(f"  GT poses  : {n_gt}/{n_frames}")
    print(f"  Baseline  : {baseline_cm:.2f} cm")
    print(f"{'='*50}")


def _fmt_matrix(M: np.ndarray, indent: int = 4) -> str:
    sp = " " * indent
    rows = []
    for row in M:
        r = "  ".join(f"{v:10.6f}" for v in row)
        rows.append(f"{sp}[{r}]")
    return "\n".join(rows)



def print_map_init(pts3d: np.ndarray, pts2d: np.ndarray,
                   frame_i: int, frame_j: int,
                   reproj_thresh: float = 3.0) -> None:
    print(f"\n── Map initialisation {'─'*30}")
    print(f"  Init pair        : i={frame_i} → j={frame_j}  "
          f"(gap={frame_j - frame_i} frames)")
    print(f"  Triangulated pts : {len(pts3d)}")
    print(f"  Valid 3D pts     : {len(pts3d)}  "
          f"(cheirality + reproj < {reproj_thresh}px)")
    print(f"  Map size         : {len(pts3d)}")
    if len(pts3d) > 0:
        print(f"\n  Example landmark id=0")
        print(f"    X   : [{pts3d[0,0]:.8f}  "
              f"{pts3d[0,1]:.8f}  {pts3d[0,2]:.8f}]")
        print(f"    uv  : [{pts2d[0,0]:.1f},  {pts2d[0,1]:.1f}] px")
        print(f"    desc: shape (32,)  ORB binary 256-bit")
        zvals = pts3d[:, 2]
        print(f"\n  Depth stats (Z)  : "
              f"min={zvals.min():.3f}  "
              f"max={zvals.max():.3f}  "
              f"median={np.median(zvals):.3f}  [scene units]")



def print_stereo_extrinsics(calib) -> None:
    print(f"\n── Stereo extrinsics  (cam1 relative to cam0) {'─'*10}")

    R  = calib.R
    t  = calib.t.ravel()

    rvec, _ = cv2.Rodrigues(R)
    angle   = float(np.linalg.norm(rvec) * 180.0 / np.pi)

    print(f"  Rotation R  (cam1 from cam0):")
    print(f"    [{R[0,0]:10.6f}  {R[0,1]:10.6f}  {R[0,2]:10.6f}]")
    print(f"    [{R[1,0]:10.6f}  {R[1,1]:10.6f}  {R[1,2]:10.6f}]")
    print(f"    [{R[2,0]:10.6f}  {R[2,1]:10.6f}  {R[2,2]:10.6f}]")
    print(f"  Rotation angle     : {angle:.4f}°  (near-zero = good stereo rig)")

    print(f"\n  Translation t  (cam1 from cam0):")
    print(f"    tx = {t[0]:+.8f} m   ← baseline")
    print(f"    ty = {t[1]:+.8f} m")
    print(f"    tz = {t[2]:+.8f} m")
    print(f"  Baseline |tx|      : {abs(t[0])*100:.4f} cm")
    print(f"  Lateral offset ty  : {abs(t[1])*1000:.4f} mm  (ideal = 0)")
    print(f"  Axial  offset tz   : {abs(t[2])*1000:.4f} mm  (ideal = 0)")


def print_mono_reproj_error(pts3d: np.ndarray, pts2d: np.ndarray,
                             R: np.ndarray, t: np.ndarray,
                             K: np.ndarray, dist: np.ndarray,
                             label: str = "mono init") -> None:

    if len(pts3d) == 0:
        return
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(
        pts3d.astype(np.float64),
        rvec, t.reshape(3,1).astype(np.float64),
        K, dist,
    )
    err = np.linalg.norm(
        proj.reshape(-1, 2) - pts2d.astype(np.float64), axis=1)

    print(f"\n── Reprojection error  ({label}) {'─'*20}")
    print(f"  Points evaluated : {len(err)}")
    print(f"  Mean  error      : {err.mean():.4f} px")
    print(f"  Median error     : {float(np.median(err)):.4f} px")
    print(f"  Std   error      : {err.std():.4f} px")
    print(f"  Max   error      : {err.max():.4f} px")
    print(f"  < 1px            : {(err < 1.0).sum():>4d} / {len(err)}  "
          f"({100*(err<1.0).mean():.1f}%)")
    print(f"  < 2px            : {(err < 2.0).sum():>4d} / {len(err)}  "
          f"({100*(err<2.0).mean():.1f}%)")
    print(f"  < 3px            : {(err < 3.0).sum():>4d} / {len(err)}  "
          f"({100*(err<3.0).mean():.1f}%)")


def print_stereo_reproj_error(pts3d_cam: np.ndarray,
                               pts2d_left: np.ndarray,
                               K_rect: np.ndarray,
                               label: str = "stereo init") -> None:
    if len(pts3d_cam) == 0:
        return

    rvec = np.zeros(3, dtype=np.float64)
    tvec = np.zeros(3, dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

    proj, _ = cv2.projectPoints(
        pts3d_cam.astype(np.float64),
        rvec, tvec, K_rect, dist,
    )
    err = np.linalg.norm(
        proj.reshape(-1, 2) - pts2d_left.astype(np.float64), axis=1)

    print(f"\n── Reprojection error  ({label}) {'─'*20}")
    print(f"  Points evaluated : {len(err)}")
    print(f"  Mean  error      : {err.mean():.4f} px")
    print(f"  Median error     : {float(np.median(err)):.4f} px")
    print(f"  Std   error      : {err.std():.4f} px")
    print(f"  Max   error      : {err.max():.4f} px")
    print(f"  < 1px            : {(err < 1.0).sum():>4d} / {len(err)}  "
          f"({100*(err<1.0).mean():.1f}%)")
    print(f"  < 2px            : {(err < 2.0).sum():>4d} / {len(err)}  "
          f"({100*(err<2.0).mean():.1f}%)")
    print(f"  < 3px            : {(err < 3.0).sum():>4d} / {len(err)}  "
          f"({100*(err<3.0).mean():.1f}%)")