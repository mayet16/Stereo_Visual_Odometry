"""
TUM VI Data Loader
==================
Usage from any other module:

    from data.data_loader import load_run_config, TUMVILoader

    cfg    = load_run_config("config/tumvi_room2.yaml")
    loader = TUMVILoader.from_config(cfg)

    for frame in loader:
        img_left  = frame.img_left
        img_right = frame.img_right
        T_gt      = frame.T_world_cam0
        frame.release()
"""

import os
import csv
import numpy as np
import cv2
import yaml
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Iterator


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    sequence_root: str
    camchain_path: str
    sequence_name: str
    stereo:        bool         = True
    start_frame:   int          = 0
    max_frames:    Optional[int] = None
    preload:       bool         = False

    def __str__(self):
        return (
            f"RunConfig({self.sequence_name})\n"
            f"  sequence_root : {self.sequence_root}\n"
            f"  camchain      : {self.camchain_path}\n"
            f"  stereo        : {self.stereo}\n"
            f"  start_frame   : {self.start_frame}\n"
            f"  max_frames    : {self.max_frames}\n"
            f"  preload       : {self.preload}"
        )


def load_run_config(config_yaml: str) -> RunConfig:
    config_yaml = os.path.abspath(config_yaml)
    if not os.path.isfile(config_yaml):
        raise FileNotFoundError(f"Config not found: {config_yaml}")

    with open(config_yaml, "r") as f:
        data = yaml.safe_load(f)

    seq_root = data["sequence_root"]
    if not os.path.isabs(seq_root):
        seq_root = os.path.abspath(
            os.path.join(os.path.dirname(config_yaml), seq_root)
        )

    if not os.path.isdir(seq_root):
        raise NotADirectoryError(f"sequence_root not found: {seq_root}")

    # ── fix: use os.path.join instead of Path / operator ─────────────────
    mav0_dir = os.path.join(seq_root, "mav0")
    if not os.path.isdir(mav0_dir):
        raise FileNotFoundError(f"mav0 folder not found: {mav0_dir}")

    camchain_path = os.path.abspath(
        os.path.join(seq_root, data["camchain_file"])
    )
    if not os.path.isfile(camchain_path):
        raise FileNotFoundError(f"camchain not found: {camchain_path}")

    base     = os.path.splitext(os.path.basename(config_yaml))[0]
    seq_name = base.replace("tumvi_", "")

    return RunConfig(
        sequence_root = seq_root,
        camchain_path = camchain_path,
        sequence_name = seq_name,
        stereo        = bool(data.get("stereo",      True)),
        start_frame   = int (data.get("start_frame", 0)),
        max_frames    =      data.get("max_frames",  None),
        preload       = bool(data.get("preload",     False)),
    )


# ---------------------------------------------------------------------------
# Camera calibration
# ---------------------------------------------------------------------------

@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width:  int
    height: int
    dist_coeffs: np.ndarray        # [k1, k2, p1, p2] radtan

    @property
    def K(self) -> np.ndarray:
        return np.array([
            [self.fx,    0,    self.cx],
            [   0,    self.fy, self.cy],
            [   0,       0,      1   ],
        ], dtype=np.float64)


@dataclass
class StereoPair:
    cam0:        CameraIntrinsics
    cam1:        CameraIntrinsics
    T_cam1_cam0: np.ndarray             # 4×4  cam1 pose in cam0 frame
    R:           np.ndarray             # rotation    cam1←cam0 (3×3)
    t:           np.ndarray             # translation cam1←cam0 (3,)
    map1_left:   Optional[np.ndarray] = field(default=None, repr=False)
    map2_left:   Optional[np.ndarray] = field(default=None, repr=False)
    map1_right:  Optional[np.ndarray] = field(default=None, repr=False)
    map2_right:  Optional[np.ndarray] = field(default=None, repr=False)
    P_left:      Optional[np.ndarray] = field(default=None, repr=False)
    P_right:     Optional[np.ndarray] = field(default=None, repr=False)
    Q:           Optional[np.ndarray] = field(default=None, repr=False)
    baseline:    float = 0.0

    @property
    def T_cam0_cam1(self) -> np.ndarray:
        return np.linalg.inv(self.T_cam1_cam0)

    @property
    def cam0_rect(self) -> "CameraIntrinsics":
        """Rectified left-camera intrinsics: K from P_left, zero distortion.
        Use this (not cam0) whenever operating on rectified images."""
        assert self.P_left is not None, "Call compute_rectification() first."
        return CameraIntrinsics(
            fx           = float(self.P_left[0, 0]),
            fy           = float(self.P_left[1, 1]),
            cx           = float(self.P_left[0, 2]),
            cy           = float(self.P_left[1, 2]),
            width        = self.cam0.width,
            height       = self.cam0.height,
            dist_coeffs  = np.zeros(4, dtype=np.float64),
        )

    def compute_rectification(self) -> None:
        """
        TUM VI uses the equidistant (Kannala-Brandt KB4) fisheye model.
        cv2.fisheye must be used — cv2.stereoRectify assumes radtan and
        gives wrong maps for this dataset.
        """
        h, w  = self.cam0.height, self.cam0.width
        R_rel = self.T_cam1_cam0[:3, :3].copy()          # must be contiguous
        t_rel = self.T_cam1_cam0[:3,  3].copy().reshape(3, 1)

        # D must be (1,4) for cv2.fisheye
        D0 = self.cam0.dist_coeffs.reshape(1, 4)
        D1 = self.cam1.dist_coeffs.reshape(1, 4)

        # fov_scale=0.33 recovers f≈190px (same as original intrinsics) so that
        # the fB product stays ≈19 m·px and SGBM disparities land in 2-64 px.
        R_left, R_right, P_left, P_right, Q = cv2.fisheye.stereoRectify(
            self.cam0.K, D0,
            self.cam1.K, D1,
            (w, h), R_rel, t_rel,
            flags=cv2.CALIB_ZERO_DISPARITY,
            fov_scale=0.33,
        )
        self.map1_left,  self.map2_left  = cv2.fisheye.initUndistortRectifyMap(
            self.cam0.K, D0, R_left,  P_left,  (w, h), cv2.CV_32FC1)
        self.map1_right, self.map2_right = cv2.fisheye.initUndistortRectifyMap(
            self.cam1.K, D1, R_right, P_right, (w, h), cv2.CV_32FC1)

        self.P_left   = P_left
        self.P_right  = P_right
        self.Q        = Q
        self.baseline = abs(P_right[0, 3] / P_right[0, 0])

    def rectify(self, img_left: np.ndarray, img_right: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
        if self.map1_left is None:
            raise RuntimeError("Call compute_rectification() first.")
        rl = cv2.remap(img_left,  self.map1_left,  self.map2_left,  cv2.INTER_LINEAR)
        rr = cv2.remap(img_right, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
        return rl, rr

    def rectify_left(self, img_left: np.ndarray) -> np.ndarray:
        if self.map1_left is None:
            raise RuntimeError("Call compute_rectification() first.")
        return cv2.remap(img_left, self.map1_left, self.map2_left, cv2.INTER_LINEAR)


def load_calibration(yaml_path: str) -> StereoPair:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    def _parse(entry: dict) -> CameraIntrinsics:
        intr = entry["intrinsics"]
        res  = entry["resolution"]
        dist = entry.get("distortion_coeffs", [0, 0, 0, 0])
        return CameraIntrinsics(
            fx=float(intr[0]), fy=float(intr[1]),
            cx=float(intr[2]), cy=float(intr[3]),
            width=int(res[0]),  height=int(res[1]),
            dist_coeffs=np.array(dist, dtype=np.float64),
        )

    cam0        = _parse(data["cam0"])
    cam1        = _parse(data["cam1"])
    T_cam1_cam0 = np.array(data["cam1"]["T_cn_cnm1"],
                            dtype=np.float64).reshape(4, 4)

    # ── extract R and t from T_cam1_cam0 ─────────────────────────────────
    R_extr = T_cam1_cam0[:3, :3].copy()
    t_extr = T_cam1_cam0[:3,  3].copy()

    pair = StereoPair(
        cam0        = cam0,
        cam1        = cam1,
        T_cam1_cam0 = T_cam1_cam0,
        R           = R_extr,
        t           = t_extr,     #
    )
    pair.compute_rectification()
    return pair


# ---------------------------------------------------------------------------
# Ground-truth poses
# ---------------------------------------------------------------------------

def _quat_trans_to_T(qw, qx, qy, qz, tx, ty, tz) -> np.ndarray:
    n = qw*qw + qx*qx + qy*qy + qz*qz
    if abs(n - 1.0) > 1e-6:
        s = 1.0 / np.sqrt(n)
        qw, qx, qy, qz = qw*s, qx*s, qy*s, qz*s
    R = np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = [tx, ty, tz]
    return T


def _load_gt_poses(mocap_csv: str) -> dict:
    poses = {}
    if not os.path.isfile(mocap_csv):
        return poses
    with open(mocap_csv, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            ts_ns = int(row[0].strip())
            tx, ty, tz = float(row[1]), float(row[2]), float(row[3])
            qw, qx, qy, qz = float(row[4]), float(row[5]), float(row[6]), float(row[7])
            poses[ts_ns] = _quat_trans_to_T(qw, qx, qy, qz, tx, ty, tz)
    return poses


def _nearest_pose(poses: dict, ts_ns: int,
                  max_dt_ns: int = 5_000_000) -> Optional[np.ndarray]:
    if not poses:
        return None
    ts_arr = np.array(list(poses.keys()), dtype=np.int64)
    idx    = int(np.argmin(np.abs(ts_arr - ts_ns)))
    if abs(int(ts_arr[idx]) - ts_ns) > max_dt_ns:
        return None
    return poses[ts_arr[idx]]


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    index:          int
    timestamp:      float
    timestamp_ns:   int
    img_path_left:  str
    img_path_right: Optional[str]
    calib:          Optional[StereoPair]
    T_world_cam0:   Optional[np.ndarray]
    _img_left:      Optional[np.ndarray] = field(default=None, repr=False)
    _img_right:     Optional[np.ndarray] = field(default=None, repr=False)


    def _load(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        return img

    @property
    def img_left(self) -> np.ndarray:
        if self._img_left is None:
            self._img_left = self._load(self.img_path_left)
        return self._img_left

    @property
    def img_right(self) -> Optional[np.ndarray]:
        if self.img_path_right is None:
            return None
        if self._img_right is None:
            self._img_right = self._load(self.img_path_right)
        return self._img_right

    @property
    def img_left_rect(self) -> np.ndarray:
        if self.calib is None:
            raise RuntimeError("No calibration attached.")
        return self.calib.rectify_left(self.img_left)

    @property
    def stereo_rectified(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.calib is None:
            raise RuntimeError("No calibration attached.")
        if self.img_right is None:
            raise RuntimeError("No right image.")
        return self.calib.rectify(self.img_left, self.img_right)

    def release(self):
        self._img_left  = None
        self._img_right = None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class TUMVILoader:

    @classmethod
    def from_config(cls, cfg: RunConfig) -> "TUMVILoader":
        print(f"[TUMVILoader] sequence : {cfg.sequence_name}")
        calib = load_calibration(cfg.camchain_path)
        return cls(
            sequence_root = cfg.sequence_root,
            calib         = calib,
            stereo        = cfg.stereo,
            start_frame   = cfg.start_frame,
            max_frames    = cfg.max_frames,
            preload       = cfg.preload,
        )

    def __init__(self, sequence_root: str,
                 calib:       Optional[StereoPair] = None,
                 stereo:      bool = True,
                 start_frame: int  = 0,
                 max_frames:  Optional[int] = None,
                 preload:     bool = False):

        self.root        = os.path.abspath(sequence_root)
        self.stereo      = stereo
        self.start_frame = start_frame
        self.max_frames  = max_frames
        self.calib       = calib

        self._frames: List[Frame] = self._build_frame_list()

        print(f"[TUMVILoader] {len(self._frames)} frames  |  "
              f"baseline = {self.calib.baseline*100:.2f} cm" if self.calib
              else f"[TUMVILoader] {len(self._frames)} frames  |  no calibration")

        if preload:
            self._preload_images()

    # ── internal ─────────────────────────────────────────────────────────────

    def _read_image_csv(self, cam_dir: str) -> List[Tuple[int, str]]:
        csv_path = os.path.join(cam_dir, "data.csv")
        img_dir  = os.path.join(cam_dir, "data")
        entries  = []
        if not os.path.isfile(csv_path):
            for fname in sorted(f for f in os.listdir(img_dir)
                                if f.endswith(".png")):
                try:
                    entries.append((int(os.path.splitext(fname)[0]),
                                    os.path.join(img_dir, fname)))
                except ValueError:
                    pass
            return entries
        with open(csv_path, newline="") as f:
            for row in csv.reader(f):
                if not row or row[0].startswith("#"):
                    continue
                ts_ns = int(row[0].strip())
                fname = row[1].strip() if len(row) > 1 else f"{ts_ns}.png"
                entries.append((ts_ns, os.path.join(img_dir, fname)))
        return entries

    def _build_frame_list(self) -> List[Frame]:
        mav0        = os.path.join(self.root, "mav0")
        left_ents   = self._read_image_csv(os.path.join(mav0, "cam0"))
        right_ents  = self._read_image_csv(os.path.join(mav0, "cam1")) \
                      if self.stereo else []
        gt_poses    = _load_gt_poses(os.path.join(mav0, "mocap0", "data.csv"))

        right_ts    = np.array([e[0] for e in right_ents], dtype=np.int64) \
                      if right_ents else np.array([], dtype=np.int64)

        end = len(left_ents)
        if self.max_frames is not None:
            end = min(end, self.start_frame + self.max_frames)

        frames = []
        for idx, (ts_ns, left_path) in enumerate(
                left_ents[self.start_frame:end]):
            right_path = None
            if right_ents:
                i  = int(np.argmin(np.abs(right_ts - ts_ns)))
                if abs(int(right_ts[i]) - ts_ns) < 2_000_000:
                    right_path = right_ents[i][1]

            frames.append(Frame(
                index          = idx + self.start_frame,
                timestamp      = ts_ns * 1e-9,
                timestamp_ns   = ts_ns,
                img_path_left  = left_path,
                img_path_right = right_path,
                calib          = self.calib,
                T_world_cam0   = _nearest_pose(gt_poses, ts_ns),
            ))
        return frames

    def _preload_images(self):
        print("[TUMVILoader] preloading ...")
        for f in self._frames:
            _ = f.img_left
            if f.img_path_right:
                _ = f.img_right

    # ── public API ───────────────────────────────────────────────────────────

    def __len__(self):              return len(self._frames)
    def __getitem__(self, i):       return self._frames[i]
    def __iter__(self):             return iter(self._frames)

    @property
    def timestamps(self) -> np.ndarray:
        return np.array([f.timestamp for f in self._frames])

    @property
    def gt_poses(self) -> List[Optional[np.ndarray]]:
        return [f.T_world_cam0 for f in self._frames]

    def has_stereo(self) -> bool:
        return bool(self._frames and self._frames[0].img_path_right)

    def has_gt(self) -> bool:
        return any(f.T_world_cam0 is not None for f in self._frames)

    def summary(self) -> str:
        n_gt = sum(1 for f in self._frames if f.T_world_cam0 is not None)
        dur  = (self._frames[-1].timestamp - self._frames[0].timestamp
                if self._frames else 0)
        lines = [
            f"Sequence  : {self.root}",
            f"Frames    : {len(self._frames)}",
            f"Duration  : {dur:.1f} s",
            f"Stereo    : {self.has_stereo()}",
            f"GT poses  : {n_gt}/{len(self._frames)}",
        ]
        if self.calib:
            lines.append(f"Baseline  : {self.calib.baseline*100:.2f} cm")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trajectory I/O
# ---------------------------------------------------------------------------

def _R_to_quaternion_xyzw(R: np.ndarray) -> np.ndarray:
    """
    Robust Shepperd method for R → quaternion [qx, qy, qz, qw].
    Integrated from old project utils_io.py.
    Handles all 4 numerical cases correctly.
    """
    R  = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)

    if tr > 0.0:
        S  = np.sqrt(tr + 1.0) * 2.0      # S = 4*qw
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S  = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0   # S = 4*qx
        qw = (R[2,1] - R[1,2]) / S
        qx = 0.25 * S
        qy = (R[0,1] + R[1,0]) / S
        qz = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S  = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0   # S = 4*qy
        qw = (R[0,2] - R[2,0]) / S
        qx = (R[0,1] + R[1,0]) / S
        qy = 0.25 * S
        qz = (R[1,2] + R[2,1]) / S
    else:
        S  = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0   # S = 4*qz
        qw = (R[1,0] - R[0,1]) / S
        qx = (R[0,2] + R[2,0]) / S
        qy = (R[1,2] + R[2,1]) / S
        qz = 0.25 * S

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= (np.linalg.norm(q) + 1e-12)
    return q


def save_trajectory_tum(
    path:       str,
    timestamps: list,
    poses:      list,
) -> None:
    """
    Save trajectory in TUM RGB-D format:
    timestamp tx ty tz qx qy qz qw
    Uses robust Shepperd quaternion conversion.
    """
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(path, "w") as f:
        for ts, T in zip(timestamps, poses):
            t = T[:3, 3]
            R = T[:3, :3]
            qx, qy, qz, qw = _R_to_quaternion_xyzw(R)
            f.write(f"{ts:.9f} "
                    f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

    print(f"[save_trajectory_tum] {len(poses)} poses → {path}")


def load_trajectory_tum(path: str) -> Tuple[List[float], List[np.ndarray]]:
    timestamps, poses = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            v  = line.split()
            ts = float(v[0])
            tx, ty, tz = float(v[1]), float(v[2]), float(v[3])
            qx, qy, qz, qw = float(v[4]), float(v[5]), float(v[6]), float(v[7])
            poses.append(_quat_trans_to_T(qw, qx, qy, qz, tx, ty, tz))
            timestamps.append(ts)
    return timestamps, poses