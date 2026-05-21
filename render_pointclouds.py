# Usage: python render_pointclouds.py

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection
import os


SEQUENCES = {
    "room2": {
        "ply":   "outputs/room2/pointcloud.ply",
        "out":   "outputs/room2/pointcloud_full.png",
        "title": "room2 — 3D Reconstruction",
        "elev":  25,
        "azim":  -60,
    },
    "corridor3": {
        "ply":   "outputs/corridor3/pointcloud.ply",
        "out":   "outputs/corridor3/pointcloud_full.png",
        "title": "corridor3 — 3D Reconstruction",
        "elev":  20,
        "azim":  -70,
    },
    "outdoors5": {
        "ply":   "outputs/outdoors5/pointcloud.ply",
        "out":   "outputs/outdoors5/pointcloud_full.png",
        "title": "outdoors5 — 3D Reconstruction",
        "elev":  30,
        "azim":  -45,
    },
}


def read_ply(path):
    """Read binary-little-endian PLY with float xyz + uchar rgb."""
    with open(path, "rb") as f:
        n_verts = 0
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            if line == "end_header":
                break
        raw = np.frombuffer(f.read(n_verts * 15), dtype=np.uint8).reshape(n_verts, 15)
    xyz = raw[:, :12].copy().view(np.float32).reshape(n_verts, 3)
    rgb = raw[:, 12:].astype(np.float32) / 255.0
    return xyz, rgb


def render(seq, cfg):
    ply_path = cfg["ply"]
    if not os.path.exists(ply_path):
        print(f"  [{seq}] PLY not found: {ply_path}")
        return

    print(f"  [{seq}] loading {ply_path} ...", end=" ", flush=True)
    xyz, rgb = read_ply(ply_path)
    print(f"{len(xyz):,} points")

    fig = plt.figure(figsize=(5, 4), facecolor="#0d0d0d")
    ax  = fig.add_subplot(111, projection="3d", facecolor="#0d0d0d")

    # Camera frame: X=right, Y=down, Z=forward.
    # Display as: X (m) = right, Z (m) = depth, −Y (m) = up.
    depth_vals = xyz[:, 2]
    sc = ax.scatter(
        xyz[:, 0], xyz[:, 2], -xyz[:, 1],
        c=depth_vals, cmap="plasma", s=0.3, alpha=0.8, linewidths=0,
    )
    cbar = plt.colorbar(sc, ax=ax, label="Depth Z (m)", shrink=0.55, pad=0.12)
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white", labelsize=7)

    ax.set_xlabel("X (m)", color="white", labelpad=4)
    ax.set_ylabel("Z (m)", color="white", labelpad=4)
    ax.set_zlabel("−Y (m)", color="white", labelpad=4)
    ax.set_title(cfg["title"], color="white", pad=8, fontsize=11)

    ax.tick_params(colors="white", labelsize=7)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#555555")
    ax.grid(True, color="#333333", linewidth=0.4)

    ax.view_init(elev=cfg["elev"], azim=cfg["azim"])

    plt.tight_layout(pad=0.5)
    plt.savefig(cfg["out"], dpi=120, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close()
    print(f"  [{seq}] saved → {cfg['out']}")


if __name__ == "__main__":
    np.random.seed(42)
    for seq, cfg in SEQUENCES.items():
        render(seq, cfg)
    print("\nDone.")
