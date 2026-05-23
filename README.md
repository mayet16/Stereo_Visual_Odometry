# From Monocular to Stereo Visual Odometry (Metric Scale) on TUM VI

> Master's Course Project — Computer Vision and Robotics  
> Classical geometry-based monocular VO and metric stereo VO on the TUM VI benchmark dataset.

---

## Overview

This project implements a complete classical visual odometry pipeline in Python, progressing from
feature-based **monocular VO** (up-to-scale) to **metric stereo VO** using a calibrated stereo rig.
No deep learning is used — the entire system is built on epipolar geometry, PnP+RANSAC, disparity
estimation, and bundle adjustment.

The system is evaluated on three sequences from the [TUM VI benchmark](https://vision.in.tum.de/data/datasets/visual-inertial-dataset):
**room2**, **corridor3**, and **outdoors5**.

---

## Pipeline Overview

### Figures

| Monocular VO | Stereo VO |
|:---:|:---:|
| ![mono pipeline](outputs/pipeline_mono.png) | ![stereo pipeline](outputs/pipeline_stereo.png) |

---

### Mermaid (interactive / GitHub)

### Monocular VO

```mermaid
flowchart LR
    A(["Left Frame\n512×512"]) --> B["ORB Detect\ngrid 5×5 · max 3000"]
    B --> C{Initialized?}

    C -->|No| D["LK Track\nprev → cur"]
    D --> E{Parallax\n≥ min_px?}
    E -->|No| D
    E -->|Yes| F["Essential Matrix\nRANSAC · 5-pt"]
    F --> G["recoverPose\n+ Triangulate"]
    G --> H["Map Init\nscale = init_scale"]
    H --> Z

    C -->|Yes| I["LK Track\nprev → cur"]
    I --> J["PnP + RANSAC\nsolvePnP"]
    J --> K{"Velocity &\nRot Gates"}
    K -->|Pass| L["Local BA\nwindow = 7"]
    L --> M["Update Pose"]
    K -->|Fail| N["Hold Pose\nfailures ++"]
    M --> O["Extend Map\nif baseline > 0"]
    O --> Z
    N --> Z

    Z[("Pose Store\nTUM format")]
```

### Stereo VO

```mermaid
flowchart LR
    A(["Left + Right\nFrames 512×512"]) --> B["Rectify\ncalibrated stereo"]
    B --> C["SGBM Disparity\nStereoSGBM"]
    C --> D["Unproject 3D\nZ = fB / d"]
    D --> E{Initialized?}

    E -->|No| F["Grid Detect\nBuild 3D Map"]
    F --> G["World = cam₀\nmetric origin"]
    G --> Z

    E -->|Yes| H["LK Track\nleft  t → t+1"]
    H --> I["PnP + RANSAC\n3D → 2D"]
    I --> J{"Velocity &\nRot Gates"}
    J -->|Pass| K["Local BA\nwindow = 7"]
    K --> L["Update Pose"]
    J -->|Fail| M["Hold / Reinit\nfailures ++"]
    L --> N["Add new SGBM\npoints to map"]
    N --> Z
    M --> Z

    Z[("Pose Store\nTUM format · scale = 1")]
```

---

## Key Results

### room2

| Metric | Mono VO | Stereo VO | Winner |
|---|---|---|---|
| ATE RMSE (Sim3 / SE3) | 1.206 m | **0.436 m** | ✓ Stereo |
| Scale | 0.114 (×9 underscaled) | 1.000 (metric) | |
| RPE trans 100m (d=100m) | N/A (90.0 m < 100 m) | 1.005 m / 885 seg | |
| RPE rot 60° (θ=60°) | 32.35° / 2496 seg | 2.03° / 2559 seg | |
| Est. traj length | 90.0 m | 143.3 m (GT: 140.9 m) | |
| Tracking failures | 5 / 2882 frames | 3 / 2882 frames | |
| Success rate | 99.83% | 99.90% | |
| Runtime | 107.5 fps | 37.9 fps | |

### corridor3

| Metric | Mono VO | Stereo VO | Winner |
|---|---|---|---|
| Start-end drift | 11.46 m | **5.75 m** | ✓ Stereo |
| RPE trans d=1 ¹ | N/A | 0.010 m/frame | |
| Local ATE start block | 0.748 m (n=512) | 0.093 m (n=512) | |
| Local ATE end block | 0.744 m (n=676) | 0.216 m (n=676) | |
| Est. traj length | 179.1 m | 301.3 m | GT: N/A (partial) |
| Tracking failures | 12 / 5802 frames | 14 / 5802 frames | |
| Success rate | 99.79% | 99.76% | |
| Runtime | 105.4 fps | 48.9 fps | |

### outdoors5

| Metric | Mono VO | Stereo VO | Winner |
|---|---|---|---|
| Start-end drift | **15.90 m** | 32.51 m | ✓ Mono ² |
| RPE trans d=1 ¹ | N/A | 0.026 m/frame | |
| Local ATE start block | 0.677 m (n=1182) | 0.337 m (n=1182) | |
| Local ATE end block | 0.891 m (n=1555) | 1.679 m (n=1555) | |
| Est. traj length | 455.8 m | 926.7 m | GT: N/A (partial) |
| Tracking failures | 2 / 17747 frames | 12 / 17747 frames | |
| Success rate | 99.99% | 99.93% | |
| Runtime | 91.9 fps | 39.5 fps | |

¹ Computed only on consecutive GT-covered frames (start and end blocks separately).
Mono RPE omitted for drift sequences — arbitrary scale makes it uninformative.

² Outdoors5 mono wins the drift metric but underscales the trajectory (estimated path ~456 m for
a ~887 s outdoor walk; ORB+E init scale ≈ 0.103). Stereo has correct metric scale but accumulates
heading drift from noisy disparity at outdoor distances (fB/d noise at 15 m ≈ 6 m). See Analysis section.

Mono VO runs at **92–108 fps** (~4.6–5.4× real-time at 20 Hz).
Stereo VO runs at **38–49 fps** (~1.9–2.4× real-time).

---

### Trajectory Plots

#### room2

| Mono VO | Stereo VO |
|---|---|
| ![mono](outputs/room2/mono_traj.png) | ![stereo](outputs/room2/stereo_traj.png) |

| Mono VO 3D | Stereo VO 3D |
|---|---|
| ![mono3d](outputs/room2/mono_traj_3d.png) | ![stereo3d](outputs/room2/stereo_traj_3d.png) |

| 3D Comparison (Mono vs Stereo vs GT) |
|---|
| ![3d](outputs/room2/comparison_3d.png) |

| Feature Tracking Samples — room2 |
|---|
| ![feat](outputs/room2/feature_samples.png) |

> **Row 1**: ORB descriptor matching (ratio test 0.75, up to 80 matches shown) between consecutive frames.
> **Row 2**: Lucas-Kanade optical flow tracking — feature positions in the previous frame (circles) and current frame (lines), shown side-by-side. Samples taken at 10% and 85% of the sequence.

#### corridor3

| Mono VO | Stereo VO |
|---|---|
| ![mono](outputs/corridor3/mono_traj.png) | ![stereo](outputs/corridor3/stereo_traj.png) |

| Mono VO 3D | Stereo VO 3D |
|---|---|
| ![mono3d](outputs/corridor3/mono_traj_3d.png) | ![stereo3d](outputs/corridor3/stereo_traj_3d.png) |

| 3D Comparison (Mono vs Stereo, start-aligned) |
|---|
| ![3d](outputs/corridor3/comparison_3d.png) |

| Feature Tracking Samples — corridor3 |
|---|
| ![feat](outputs/corridor3/feature_samples.png) |

> **Row 1**: ORB descriptor matching between consecutive frames.
> **Row 2**: LK optical flow tracking. Samples at 10% and 85% of the sequence.

#### outdoors5

| Mono VO | Stereo VO |
|---|---|
| ![mono](outputs/outdoors5/mono_traj.png) | ![stereo](outputs/outdoors5/stereo_traj.png) |

| Mono VO 3D | Stereo VO 3D |
|---|---|
| ![mono3d](outputs/outdoors5/mono_traj_3d.png) | ![stereo3d](outputs/outdoors5/stereo_traj_3d.png) |

| 3D Comparison (Mono vs Stereo, start-aligned) |
|---|
| ![3d](outputs/outdoors5/comparison_3d.png) |

| Feature Tracking Samples — outdoors5 |
|---|
| ![feat](outputs/outdoors5/feature_samples.png) |

> **Row 1**: ORB descriptor matching between consecutive frames.
> **Row 2**: LK optical flow tracking. Samples at 10% and 85% of the sequence.

---

## 3D Reconstruction Outputs

For each sequence the stereo pipeline saves:

**Rectified stereo pairs** (`rectified_pair_N.png`) — left/right frames after fisheye undistortion and stereo rectification, for 3 sample timestamps. Epipolar lines are horizontal after rectification.

**Disparity maps** (`disparity_map_N.png`) — SGBM+WLS disparity heat maps (jet colormap) for 3 frames (10%, 50%, 90% of sequence). Measured disparity and depth ranges are annotated on each image.

**Depth maps** (`depth_map_N.png`) — metric depth heat maps (Z = fB/d) for the same 3 frames.

**Metric 3D point cloud** (`pointcloud.ply`, binary little-endian PLY) — accumulated from up to 800 keyframes selected by a motion threshold (Δt ≥ 5 cm or Δθ ≥ 2°). Points are sampled at an auto-scaled pixel step so raw counts stay ≤ 4M, filtered by minimum disparity, then cleaned by Statistical Outlier Removal. room2 uses k=20, σ=1.0 (tighter, to remove CLAHE-induced SGBM wall speckle); corridor3 and outdoors5 use k=10, σ=2.0. Final clouds: ~1M points, 15 MB.

| Sequence | Raw pts | After SOR | Removed | SOR threshold |
|---|---|---|---|---|
| room2    | 2,562,160 | 2,346,604 | 8.4% | 0.067 m |
| corridor3 | 2,219,598 | 2,182,964 | 1.7% | 0.242 m |
| outdoors5 | 1,768,976 | 1,703,928 | 3.8% | 0.708 m |

Open PLY in MeshLab: **File → Import Mesh → pointcloud.ply**

### room2

| Rectified Pair (frame 288) | Disparity Map (frame 288) | Depth Map (frame 288) | 3D Point Cloud (frame 288) |
|---|---|---|---|
| ![rect](outputs/room2/rectified_pair_1.png) | ![disp](outputs/room2/disparity_map_1.png) | ![depth](outputs/room2/depth_map_1.png) | ![pc](outputs/room2/pointcloud_1.png) |

| Full Point Cloud (depth-coloured) | Full Point Cloud (RGB) |
|---|---|
| ![pcf](outputs/room2/pointcloud_full.png) | ![pcfrgb](outputs/room2/pointcloud_full_rgb.png) |

---

### corridor3

| Rectified Pair (frame 580) | Disparity Map (frame 580) | Depth Map (frame 580) | 3D Point Cloud (frame 580) |
|---|---|---|---|
| ![rect](outputs/corridor3/rectified_pair_1.png) | ![disp](outputs/corridor3/disparity_map_1.png) | ![depth](outputs/corridor3/depth_map_1.png) | ![pc](outputs/corridor3/pointcloud_1.png) |

| Full Point Cloud (depth-coloured) | Full Point Cloud (RGB) |
|---|---|
| ![pcf](outputs/corridor3/pointcloud_full.png) | ![pcfrgb](outputs/corridor3/pointcloud_full_rgb.png) |

---

### outdoors5

| Rectified Pair (frame 1774) | Disparity Map (frame 1774) | Depth Map (frame 1774) | 3D Point Cloud (frame 1774) |
|---|---|---|---|
| ![rect](outputs/outdoors5/rectified_pair_1.png) | ![disp](outputs/outdoors5/disparity_map_1.png) | ![depth](outputs/outdoors5/depth_map_1.png) | ![pc](outputs/outdoors5/pointcloud_1.png) |

| Full Point Cloud (depth-coloured) | Full Point Cloud (RGB) |
|---|---|
| ![pcf](outputs/outdoors5/pointcloud_full.png) | ![pcfrgb](outputs/outdoors5/pointcloud_full_rgb.png) |

---

## Project Structure

```
Stereo_VO/
├── config/
│   ├── tumvi_room2.yaml
│   ├── tumvi_corridor3.yaml
│   └── tumvi_outdoors5.yaml
├── data/
│   └── data_loader.py          # TUMVILoader, StereoPair, Frame, save_trajectory_tum
├── mono_vo/
│   ├── __init__.py
│   ├── feature_tracker.py      # FeatureTracker (ORB + LK optical flow)
│   ├── epipolar.py             # Essential matrix, recoverPose, triangulate, PnP+RANSAC
│   └── pipeline.py             # MonoVO, MonoVOConfig
├── stereo_vo/
│   ├── __init__.py
│   ├── disparity.py            # DisparityComputer, DisparityConfig (SGBM + WLS)
│   └── pipeline.py             # StereoVO, StereoVOConfig  (n_reinits counter)
├── evaluation/
│   ├── __init__.py
│   └── metrics.py              # ATE (Sim3/SE3), RPE, start-end drift, align_and_evaluate
├── utils/
│   ├── __init__.py
│   ├── math_utils.py           # Rt_to_T, invert_T, compose_T, cam_from_world
│   └── print_utils.py          # Calibration display, reprojection error, trajectory plots
├── outputs/
│   ├── room2/
│   │   ├── mono_traj.txt           # TUM format trajectory
│   │   ├── stereo_traj.txt
│   │   ├── mono_traj.png / stereo_traj.png
│   │   ├── mono_traj_3d.png / stereo_traj_3d.png / comparison_3d.png
│   │   ├── rectified_pair_*.png    # rectified stereo pairs (3 sample frames)
│   │   ├── disparity_map_*.png     # SGBM+WLS disparity heat maps (3 sample frames)
│   │   ├── depth_map_*.png         # metric depth heat maps (3 sample frames)
│   │   ├── pointcloud_*.png        # point cloud renders (3 views + full + RGB)
│   │   └── pointcloud.ply          # metric 3D point cloud ~1M pts (15 MB, binary PLY)
│   ├── corridor3/  (same structure)
│   ├── outdoors5/  (same structure)
│   └── evaluation_results.csv
└── main.py   # Entry point: all 3 sequences, evaluation, plots, PLY, CLAHE ablation
```

---

## Dataset

[TUM VI Benchmark](https://vision.in.tum.de/data/datasets/visual-inertial-dataset) —
hand-held global-shutter stereo camera, 512×512, 20 Hz.

| Sequence | Frames | Duration | GT poses | Used metric |
|---|---|---|---|---|
| room2 | 2882 | 144.1s | 2587/2882 (full) | ATE RMSE |
| corridor3 | 5802 | 290.1s | 1196/5802 (start+end) | Start-end drift |
| outdoors5 | 17747 | 887.3s | 2747/17747 (start+end) | Start-end drift |

Download the sequences:
```
dataset-room2_512_16
dataset-corridor3_512_16
dataset-outdoors5_512_16
```
from https://vision.in.tum.de/data/datasets/visual-inertial-dataset

---

## Camera Calibration

Calibration is loaded directly from the Kalibr `camchain.yaml` provided with the dataset.
**No re-calibration is performed.**

| Parameter | cam0 (left) | cam1 (right) |
|---|---|---|
| fx | 190.9785 px | 190.4451 px |
| fy | 190.9733 px | 190.4451 px |
| cx | 254.9317 px | 252.5998 px |
| cy | 256.8974 px | 254.9967 px |
| Distortion model | equidistant (k1–k4) | equidistant (k1–k4) |

**Stereo extrinsics (T_cam1_cam0):**

| Parameter | Value |
|---|---|
| Baseline B | 10.11 cm |
| Rotation angle | 2.69° |
| ty offset | 1.98 mm |
| tz offset | 1.18 mm |
| Rectified focal length | 187.13 px |
| fB product | 18.92 px·m |

---

## Coordinate Systems

The project uses two distinct frames throughout. Understanding them is essential for reading the trajectory plots and evaluation metrics.

### TUM-VI Ground-Truth World Frame (GT frame)

Ground-truth poses are stored in this frame (loaded from `mav0/mocap0/data.csv`):

```
        Z  ↑  (vertical / up)
           │
           │
           └──────────── X  (horizontal, floor plane)
          /
         /
        Y  (horizontal, floor plane)
```

| Axis | Direction | Role |
|---|---|---|
| X | right (horizontal) | floor traversal |
| Y | forward (horizontal) | floor traversal |
| Z | up (vertical) | height |

Floor motion appears in the **X–Y plane**; camera height appears along **Z**.
Verification from room2 GT: X span = 3.27 m, Y span = 2.59 m, Z span = 0.66 m — X–Y is the floor loop, Z is near-constant height.

### Camera / VO World Frame (VO frame)

Estimated poses accumulate in this frame during visual odometry:

```
        Z  (forward, optical axis)
       /
      /
     └──────── X  (right, image column direction)
     │
     ↓
     Y  (down, image row direction)
```

| Axis | Direction | Role |
|---|---|---|
| X | right (image x) | floor traversal |
| Y | down (image y) | height (inverted) |
| Z | forward (optical) | floor traversal |

For a roughly horizontal camera, floor motion appears in the **X–Z plane**; height appears along **Y**.
The origin is the camera position at frame 0 (T = I₄).

### How They Are Used

| Output | Frame used | Alignment |
|---|---|---|
| Saved trajectory plots (`mono_traj.png`, `stereo_traj.png`, `*_3d.png`) | **GT world frame** | Sim3 (mono) or SE3 (stereo) global best-fit |
| `comparison_3d.png` | **GT world frame** | Same as above |
| Live trajectory window | **VO frame** (raw) | None — auto-selects the two highest-variance axes (X–Z for horizontal cameras) |

The 3D trajectory plots label axes as X [m], Y [m], Z [m] (up) in the GT world frame. The top-down panel shows the X–Y floor plan, the front panel shows X–Z (horizontal vs height), and the side panel shows Y–Z.

---

## Dependencies

```
Python       >= 3.10
OpenCV       >= 4.5     (cv2)
NumPy        >= 1.23
SciPy        >= 1.9
Matplotlib   >= 3.5
PyYAML       >= 6.0
```

Install all dependencies:

```bash
pip install numpy opencv-python scipy matplotlib pyyaml
```

---

## Quick Start

**1. Clone the repository**
```bash
git clone https://github.com/mayet16/Stereo_Visual_Odometry.git
cd Stereo_VO
```

**2. Download TUM VI sequences** and place them under a common root, e.g.:
```
~/datasets/
    dataset-room2_512_16/
    dataset-corridor3_512_16/
    dataset-outdoors5_512_16/
```

**3. Edit config files** to point to your dataset root:
```yaml
# config/tumvi_room2.yaml
sequence_root: /home/user/datasets/dataset-room2_512_16
camchain_file: dso/camchain.yaml
```

**4. Run the full pipeline:**
```bash
python main.py
```

This will run monocular VO and stereo VO on all three sequences, print evaluation metrics,
and save trajectory files and plots to `outputs/<sequence>/`.

**5. Run a single sequence** (edit `SEQUENCES` list in `main.py`):
```python
SEQUENCES = ["config/tumvi_room2.yaml"]   # room2 only
```

**6. Toggle the live visualizer on / off**

The live display opens two OpenCV windows per sequence — *features* (camera frame with tracked points) and *trajectory* (top-down path). It is **off by default** — the pipeline runs headless and all results are saved to `outputs/<sequence>/`.

| Method | Command |
|---|---|
| Run headless (default) — results saved to `outputs/` | `python main.py` |
| Run with live display | `SHOW_VIS=1 python main.py` |

Alternatively, hard-code the flag at the top of `main.py` (line 43):
```python
_SHOW = False   # headless — no OpenCV windows (default)
_SHOW = True    # enable live feature and trajectory windows
```

> All trajectory plots, 3D figures, point clouds, and evaluation CSVs are always written to `outputs/<sequence>/` regardless of the `_SHOW` setting.

---

## Reproducibility

```
OS          : Ubuntu 22.04
CPU         : Intel Core i7 (8 cores)
RAM         : 32 GB
Python      : 3.13.9
OpenCV      : 4.13.0
NumPy       : 2.3.5
NumPy seed  : np.random.seed(42)
```

Runtime per sequence (measured, Python 3.13.9, OpenCV 4.13.0, no GPU):
- room2:     mono ~27s   (107.5 fps), stereo ~76s   (37.9 fps)
- corridor3: mono ~55s   (105.4 fps), stereo ~119s  (48.9 fps)
- outdoors5: mono ~193s  (91.9 fps),  stereo ~449s  (39.5 fps)

---

## Evaluation Protocol

Trajectory alignment follows the TUM VI benchmark standard:

- **Monocular VO**: Sim3 alignment (7 DoF — rotation, translation, scale correction)
- **Stereo VO**: SE3 alignment (6 DoF — rotation and translation only, no scale correction)

This means the mono ATE benefits from scale correction while stereo ATE is evaluated honestly
at metric scale. The ORB-SLAM2 paper [2] explicitly notes: *"The better accuracy of pure monocular
compared with stereo is only apparent: the monocular solution is up-to-scale and aligned with
ground-truth with 7 DoFs, while stereo provides the true scale and is aligned with 6 DoFs."*

**Full-trajectory plotting with GT-derived alignment**: The trajectory plots show the complete estimated VO path (all frames), not only the GT-overlapping subset. In all three sequences the alignment transform (R, t, s) is still computed from the GT-paired frames only (2587 for room2, start+end blocks for corridor3/outdoors5) — that is the correct and sufficient set for a robust Sim3/SE3 fit. Only the *application* of that transform is extended to all estimated frames so the full path is visible. The ATE error values and the error-over-time plots are unaffected: they are computed exclusively against the GT-paired subset.

---

## Implementation Notes

### Monocular VO
- **Initialization**: ORB detect → BFMatcher kNN + Lowe ratio test → Essential matrix RANSAC (5-pt) → recoverPose → triangulate; scale recovered from the Nth-percentile of depth-filtered inliers (room2: 25th pct, corridor3: 50th pct, outdoors5: 75th pct relative to expected_depth d₀). A FAST+LK parallax gate ensures sufficient baseline before ORB+E is attempted.
- **Tracking**: LK optical flow → PnP+RANSAC → solvePnPRefineLM (local BA) → velocity/rotation check → map extension
- **Relocalization** (spec §VI): when tracking is lost, `_try_reinit()` performs recovery in two stages: (1) **anchor 3D-PnP** (`e_reinit`) — solve pose from stored anchor map points using `cv2.solvePnPRansac`; (2) **VD fallback** (`vd_reinit`) — FAST+LK re-initialization at estimated scene median depth. Both counters are reported in the per-sequence tracking summary.
- **Local BA**: sliding window of 7 poses, run every 5 frames

### Stereo VO
- **Motion estimation method**: **3D–2D PnP** — 3D map points from the previous frame are matched to 2D feature positions in the current frame, then `cv2.solvePnPRansac` recovers the relative pose. The spec also allows 3D–3D ICP alignment; 3D–2D PnP was chosen because it works directly with the sparse LK-tracked keypoints without requiring dense overlapping point sets.
- **RANSAC**: applied inside `pnp_ransac()` (`mono_vo/epipolar.py:108`) — confidence 0.999, per-sequence threshold 2–4 px.
- **Features**: ORB keypoints detected on a 5×5 grid (`FeatureTracker.detect_grid`), tracked frame-to-frame with Lucas-Kanade optical flow.
- **Init**: detect_grid → SGBM disparity → unproject via Z = fB/d → world = camera at t=0
- **Tracking**: LK track prev→cur → PnP+RANSAC → local BA → velocity/rotation check → add new stereo points
- **Depth**: 3D points lifted once from disparity, tracked in 2D via LK thereafter
- **SGBM params per sequence**:

| Sequence | num_disparities | block_size | max_depth | min_disparity | Notes |
|---|---|---|---|---|---|
| room2 | 64 | 11 | 5.0 m | 1.5 px | room diameter ~5 m |
| corridor3 | 64 | 21 | 10.0 m | 0.5 px | large block for wall texture; `use_depth_update=False` |
| outdoors5 | 128 | 11 | 20.0 m | 1.0 px | wider search for far outdoor features; `use_depth_update=False` |

### Why LK alongside SGBM?

SGBM is spatial (computes depth from left vs right at the **same** timestamp).
LK is temporal (tracks 2D feature positions across **consecutive** left frames).
They solve orthogonal problems: SGBM provides the metric 3D map; LK provides
2D correspondences to feed into PnP for pose estimation. Neither can replace the other.

---

## Reference Paper Mapping

### Monocular VO → Scaramuzza & Fraundorfer (2011) [1]

| Our module | Paper section | Description |
|---|---|---|
| `FeatureTracker.detect_grid()` | §II-A Feature detection | ORB keypoints on a uniform grid |
| `FeatureTracker.track_lk()` | §II-B Feature matching | Lucas-Kanade optical flow |
| `epipolar.estimate_essential()` | §III-A Essential matrix | 5-point RANSAC via `cv2.findEssentialMat` |
| `epipolar.recover_pose()` | §III-A Decomposition | `cv2.recoverPose` + cheirality check |
| `epipolar.triangulate_points()` | §III-B Triangulation | `cv2.triangulatePoints` |
| `epipolar.pnp_ransac()` | §III-C Pose from 3D–2D | `cv2.solvePnPRansac` |
| `MonoVO._try_init()` | §III-D Scale initialisation | ORB+E init: triangulate → depth-percentile scale (d₀ = 2–5 m per sequence); `_vd_reinit()` used as fallback only |
| `epipolar.refine_pose_ba()` | §IV Local optimisation | Sliding-window bundle adjustment (7 poses) |

Reference: D. Scaramuzza and F. Fraundorfer, "Visual Odometry [Tutorial]," IEEE R&A Magazine, 2011.

### Stereo VO → Mur-Artal & Tardós, ORB-SLAM2 (2017) [2]

| Our module | ORB-SLAM2 section | Description |
|---|---|---|
| `calib.rectify()` | §III-A Stereo initialisation | `cv2.fisheye.initUndistortRectifyMap` every frame |
| `DisparityComputer.compute()` | §III-A Depth computation | StereoSGBM + WLS filter → disparity map |
| `DisparityComputer.disparity_to_depth()` | §III-A Eq. (1) | Z = fB/d; X = (u−cx)Z/f; Y = (v−cy)Z/f |
| `FeatureTracker.track_lk()` | §III-B Tracking | LK optical flow for 2D correspondences |
| `epipolar.pnp_ransac()` | §III-C Motion estimation | 3D–2D PnP + RANSAC (3D–3D ICP not implemented) |
| `epipolar.refine_pose_ba()` | §IV Local BA | Sliding-window bundle adjustment |
| `StereoVO._add_points()` | §III-D Map maintenance | SGBM reprojection when map drops below threshold |

Reference: R. Mur-Artal and J. D. Tardós, "ORB-SLAM2," IEEE T-RO, 2017.

---

## Analysis and Findings

### room2 — Stereo advantage: metric scale in small environments
Stereo achieves **64% lower ATE** (0.436 m vs 1.206 m). The mono scale is 0.114 (~9× underscaled)
because ORB features in the small room bias toward far-wall textures, over-estimating scene depth
and under-estimating scale at ORB+E init time. Stereo's metric depth from disparity anchors the
map correctly and maintains scale=1.000 throughout. CLAHE is enabled for both pipelines on room2:
the structured indoor scene benefits from adaptive contrast, reducing stereo failures from 18→3
and ATE from 0.788→0.436 m.

### corridor3 — Stereo advantage: heading accuracy over long hallways
Stereo drift is **2× lower** (5.75 m vs 11.46 m). The corridor's long straight path amplifies any
heading error: mono accumulates scale-and-heading drift over 290 s, while stereo's metric 3D map
keeps the heading constrained through PnP. The mono drift is higher than VD-init results because
the ORB+E init produces a more geometrically honest (less compressed) trajectory — the expressed
drift reflects true heading error rather than scale-compression artefact. CLAHE and a large block
size (21 px) stabilise the SGBM disparity on homogeneous wall textures. Mono config improvements
(`pnp_min_inliers=15`, `min_parallax_px=12`, `min_tracked_pts=70`, `max_map_pts=800`) reduced
mono drift by 21.5% relative to the initial baseline.

### outdoors5 — Scale ambiguity and heading drift over a long outdoor sequence
Mono wins the drift metric (15.9 m vs 32.5 m), but this apparent advantage is an artefact of
scale compression. The ORB+E scale initialises to ≈0.103 (75th-percentile at d₀=5.0 m),
compressing the estimated path to ≈456 m against the stereo-estimated metric path of ≈927 m.
A 15.9 m error on a non-metric, 2×-compressed path cannot be directly compared to stereo's
32.5 m error on a metric 926.7 m path. Mono tracking failures dropped to 2 with the ORB+E init,
reflecting stable feature matching on rich outdoor texture.

Stereo maintains correct metric scale throughout but accumulates **heading drift** over the
17,747-frame sequence — the same mechanism as corridor3, amplified by 3× greater feature depth
(5–20 m) and 3× longer sequence. With fB = 18.92 px·m, a feature at 15 m yields only ≈1.3 px
disparity; 0.5 px noise gives ≈6 m depth error per map point, biasing PnP rotation at each frame.
Over nearly 900 s of outdoor travel, this per-frame bias accumulates as heading drift. Setting
Z_max = 20 m is necessary to keep sufficient 3D map points for PnP; restricting to 5 m starves
the tracker below the minimum inlier count, causing hundreds of failures.

For metric localisation, stereo remains the preferred choice — it provides accurate metric scale
(≈927 m estimated path) throughout the sequence. Reducing drift on long outdoor sequences requires
loop closure or IMU fusion to correct accumulated heading error.

### Tracking robustness

| Sequence | Mono failures | Mono relocalization attempts | Stereo failures | Stereo reinits |
|---|---|---|---|---|
| room2 | 5 | e_reinit=0, vd_reinit=0 | 3 | 3 |
| corridor3 | 12 | e_reinit=0, vd_reinit=0 | 14 | 14 |
| outdoors5 | **2** | e_reinit=0, vd_reinit=0 | 12 | 12 |

**Mono relocalization** (per spec §VI): when tracking fails, `_try_reinit()` attempts pose recovery via two strategies — `e_reinit` (anchor 3D-PnP: solve pose from stored anchor map points) and `vd_reinit` (VD-fallback: FAST+LK re-initialization at median scene depth). These are the system's relocalization attempts as defined in §VI of the project specification.

Stereo `n_reinits` counts failure-recovery events only (PnP failure, insufficient tracked points, velocity spike) — equals `n_failures` because every failure triggers immediate SGBM-based map replenishment.

### Dynamic scene sensitivity (RANSAC inlier ratio)

Per spec §VI: the RANSAC inlier ratio measures how many tracked points are consistent with the static-scene ego-motion model. A low ratio indicates dynamic object contamination or degenerate geometry.

| Sequence | Pipeline | Frames | Mean ratio | % frames < 0.70 | % frames < 0.50 | Note |
|---|---|---|---|---|---|---|
| room2 | Mono | 2815 | 0.982 | 0.2% | 0.1% | |
| room2 | Stereo | 2881 | 0.978 | 0.2% | 0.1% | |
| corridor3 | Mono | 5793 | 0.984 | 0.2% | 0.2% | homogeneous walls |
| corridor3 | Stereo | 5801 | 0.977 | 0.3% | 0.3% | homogeneous walls |
| outdoors5 | Mono | 17736 | 0.996 | 0.0% | 0.0% | pedestrians/cyclists |
| outdoors5 | Stereo | 17746 | 0.987 | 0.2% | 0.1% | pedestrians/cyclists |

Mean ratios close to 1.0 across all sequences confirm the static-scene ego-motion model holds well. The near-zero < 0.50 percentages on outdoors5 show that pedestrians and cyclists do not significantly contaminate the RANSAC inlier set — the robust estimator rejects dynamic points as outliers rather than including them in the pose estimate. Full per-frame ratios are saved to `outputs/dynamic_results.csv`.

### Illumination sensitivity (CLAHE ablation)

CLAHE (clip limit 2.0, tile 8×8) is applied selectively per sequence. Production rule: ON for both pipelines on room2 and corridor3 (indoor structured scenes), ON for mono only on outdoors5, OFF for stereo on outdoors5 (sky/foliage corrupt SGBM). Ablation: toggling CLAHE opposite to production setting:

| Sequence | Pipeline | CLAHE | Failures | ATE / Drift |
|---|---|---|---|---|
| room2 | Mono | ON (prod) | 5 | 1.206 m ATE |
| room2 | Mono | OFF (abl) | 42 | 1.200 m ATE |
| room2 | Stereo | ON (prod) | 3 | **0.436 m ATE** |
| room2 | Stereo | OFF (abl) | 18 | 0.788 m ATE |
| corridor3 | Mono | ON (prod) | 12 | 11.46 m drift |
| corridor3 | Mono | OFF (abl) | 221 | 13.71 m drift |
| corridor3 | Stereo | ON (prod) | 14 | 5.75 m drift |
| corridor3 | Stereo | OFF (abl) | 83 | 11.28 m drift |
| outdoors5 | Mono | ON (prod) | 2 | 15.90 m drift |
| outdoors5 | Mono | OFF (abl) | 74 | 24.26 m drift |
| outdoors5 | Stereo | OFF (prod) | 12 | 32.51 m drift |
| outdoors5 | Stereo | ON  (abl) | 13 | **201.08 m drift** |

Key findings:
- **Room2**: CLAHE ON for both pipelines — structured indoor scene benefits from adaptive contrast. Disabling mono CLAHE raises failures 5→42 (+37) with negligible ATE change (1.206→1.200 m). Disabling stereo CLAHE raises failures 3→18 (+15) and ATE 0.436→0.788 m (+0.352 m).
- **Corridor3**: CLAHE is critical for tracking stability — disabling it raises mono failures 12→221 (+209) and stereo failures 14→83 (+69, +5.5 m drift). The drift increase on mono is modest (11.46→13.71 m) because the held-pose fallback masks drift accumulation, but the failure spike confirms that homogeneous walls need adaptive contrast enhancement.
- **Outdoors5 mono**: CLAHE is critical for ORB+E init — disabling it raises failures 2→74 (+72) and drift 15.90→24.26 m (+8.4 m). Without CLAHE the ORB detector finds fewer high-quality matches in variable outdoor lighting, degrading the depth-percentile scale estimate at init.
- **Outdoors5 stereo**: CLAHE is harmful — enabling it increases drift 32.51→201.08 m (+168.6 m). CLAHE over-enhances sky/foliage gradients, corrupting SGBM disparity on outdoor scenes.

---

## References

[1] D. Scaramuzza and F. Fraundorfer, "Visual Odometry [Tutorial]: Part I — The First 30 Years
    and Fundamentals," IEEE Robotics & Automation Magazine, vol. 18, no. 4, pp. 80–92, 2011.

[2] R. Hartley and A. Zisserman, Multiple View Geometry in Computer Vision, 2nd ed.
    Cambridge University Press, 2004.

[3] R. Mur-Artal, J. M. M. Montiel, and J. D. Tardós, "ORB-SLAM: A Versatile and Accurate
    Monocular SLAM System," IEEE Transactions on Robotics, vol. 31, no. 5, pp. 1147–1163, 2015.

[4] R. Mur-Artal and J. D. Tardós, "ORB-SLAM2: An Open-Source SLAM System for Monocular,
    Stereo, and RGB-D Cameras," IEEE Transactions on Robotics, vol. 33, no. 5, pp. 1255–1262, 2017.

[5] C. Campos, R. Elvira, J. J. G. Rodríguez, J. M. M. Montiel, and J. D. Tardós,
    "ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial, and Multimap SLAM,"
    IEEE Transactions on Robotics, vol. 37, no. 6, pp. 1874–1890, Dec. 2021.

[6] D. Schubert et al., "The TUM VI Benchmark for Evaluating Visual-Inertial Odometry,"
    IEEE/RSJ IROS, pp. 1680–1687, 2018.

[7] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, "Vision Meets Robotics: The KITTI Dataset,"
    International Journal of Robotics Research, vol. 32, no. 11, pp. 1231–1237, 2013.

[8] NVIDIA Corporation, "Isaac ROS Visual SLAM (cuVSLAM),"
    GitHub repository, NVIDIA-ISAAC-ROS, 2023.

[9] Z. Teed and J. Deng, "DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras,"
    Proc. NeurIPS, 2021.

[10] F. Rameau, D. Sidibé, C. Demonceaux, and D. Fofi,
     "Structure from Motion Using a Hybrid Stereo-Vision System,"
     Proc. 12th Int. Conf. Ubiquitous Robots and Ambient Intelligence (URAI 2015),
     Goyang City, South Korea, Oct. 2015.

[11] D. Gálvez-López and J. D. Tardós,
     "Bags of Binary Words for Fast Place Recognition in Image Sequences,"
     IEEE Transactions on Robotics, vol. 28, no. 5, pp. 1188–1197, 2012.

[12] J. Laconte, "How to Write a Robotics Paper,"
     Online resource, INRAE MathNum, 2023.

[13] J. Laconte, "Figures for Robotics Papers,"
     Online resource, INRAE MathNum, 2023.

[14] E. Rublee, V. Rabaud, K. Konolige, and G. Bradski,
     "ORB: An Efficient Alternative to SIFT or SURF,"
     Proc. IEEE ICCV, pp. 2564–2571, 2011.

[15] D. Nistér, "An Efficient Solution to the Five-Point Relative Pose Problem,"
     IEEE Trans. Pattern Anal. Mach. Intell., vol. 26, no. 6, pp. 756–777, Jun. 2004.

[16] B. D. Lucas and T. Kanade,
     "An Iterative Image Registration Technique with an Application to Stereo Vision,"
     Proc. DARPA Image Understanding Workshop, pp. 121–130, 1981.

[17] H. Hirschmüller, "Stereo Processing by Semiglobal Matching and Mutual Information,"
     IEEE Trans. Pattern Anal. Mach. Intell., vol. 30, no. 2, pp. 328–341, Feb. 2008.

[18] N. Muhammad, D. Fofi, and S. Ainouz-Zemouche,
     "Current State-of-the-Art of Vision-Based SLAM,"
     Proc. IS&T/SPIE Electronic Imaging: Machine Vision Applications II,
     San Jose, CA, USA, Jan. 2009.

[19] C. Jiang, D. P. Paudel, Y. D. Fougerolle, D. Fofi, and C. Demonceaux,
     "Static-Map and Dynamic Object Reconstruction in Outdoor Scenes Using 3-D Motion Segmentation,"
     IEEE Robotics and Automation Letters, vol. 1, no. 1, pp. 324–331, 2016.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
