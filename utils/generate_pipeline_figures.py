#!/usr/bin/env python3
"""
Generate pipeline diagram PNG figures for the Stereo VO IEEE paper.

Outputs:
    outputs/pipeline_mono.png
    outputs/pipeline_stereo.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

os.makedirs('outputs', exist_ok=True)

P = dict(
    frame   = '#1565C0',   # blue        – input frame
    feat    = '#6A1B9A',   # purple      – feature ops
    geom    = '#BF360C',   # deep orange – geometry / math
    gate    = '#B71C1C',   # deep red    – decision
    opt     = '#1B5E20',   # dark green  – optimisation / BA
    hold    = '#546E7A',   # blue-grey   – hold / fail path
    out     = '#004D40',   # dark teal   – pose output
    arrow   = '#37474F',
    init_bg = '#E3F2FD',
    trk_bg  = '#E8F5E9',
    dep_bg  = '#FFFDE7',
    init_e  = '#1565C0',
    trk_e   = '#2E7D32',
    dep_e   = '#F9A825',
)

BW, BH = 1.68, 0.66   # default box  width / height
DW, DH = 1.55, 0.76   # diamond      width / height



def bg(ax, x, y, w, h, fc, ec, label):
    r = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.12',
                       facecolor=fc, edgecolor=ec,
                       linewidth=1.3, linestyle='--', zorder=1)
    ax.add_patch(r)
    ax.text(x + 0.18, y + h - 0.16, label,
            ha='left', va='top', fontsize=8.5,
            color=ec, fontweight='bold', style='italic', zorder=2)


def bx(ax, cx, cy, text, color, w=BW, h=BH, fs=8.8):
    r = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                       boxstyle='round,pad=0.07',
                       facecolor=color, edgecolor='white',
                       linewidth=2.1, zorder=3)
    ax.add_patch(r)
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fs, color='white', fontweight='bold',
            zorder=4, multialignment='center', linespacing=1.3)


def dm(ax, cx, cy, text, color, w=DW, h=DH, fs=8.3):
    pts = np.array([[cx, cy+h/2], [cx+w/2, cy],
                    [cx, cy-h/2], [cx-w/2, cy]])
    ax.add_patch(plt.Polygon(pts, facecolor=color, edgecolor='white',
                             linewidth=2.1, zorder=3))
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fs, color='white', fontweight='bold',
            zorder=4, multialignment='center', linespacing=1.3)


def arr(ax, x1, y1, x2, y2, label='', lside='top',
        color=None, lw=1.6, rad=0.0, lfs=7.6):
    c = color or P['arrow']
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=lw,
                                mutation_scale=14,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        dy = 0.14 if lside == 'top' else -0.16
        ax.text(mx, my + dy, label, ha='center', va='center',
                fontsize=lfs, color=c, style='italic')


def h_chain(ax, xs, y, labels, colors, widths=None):
    """Draw a horizontal chain of boxes with connecting arrows."""
    ws = widths or [BW] * len(xs)
    for i, (x, lab, col, w) in enumerate(zip(xs, labels, colors, ws)):
        bx(ax, x, y, lab, col, w=w)
    for i in range(1, len(xs)):
        arr(ax, xs[i-1] + ws[i-1]/2, y, xs[i] - ws[i]/2, y)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 – Monocular VO
# ══════════════════════════════════════════════════════════════════════════════

fig1, ax1 = plt.subplots(figsize=(15, 5.8))
ax1.set_xlim(0, 15)
ax1.set_ylim(0, 5.8)
ax1.axis('off')
fig1.patch.set_facecolor('white')
ax1.set_facecolor('white')

# title
ax1.text(7.5, 5.55, 'Monocular VO Pipeline',
         ha='center', fontsize=14, fontweight='bold', color='#1A237E')

# section backgrounds
bg(ax1,  0.18, 3.05, 14.64, 2.10,
   P['init_bg'], P['init_e'], 'Initialization  (first time only)')
bg(ax1,  0.18, 0.35, 14.64, 2.45,
   P['trk_bg'],  P['trk_e'], 'Per-frame tracking loop')

yi = 3.88
xi = [1.1,  3.1,  5.35,  7.75,  10.25, 12.65]
li = ['Left\nFrame', 'ORB\nDetect', 'LK Track\n(parallax gate)',
      'E-Matrix\n+ RANSAC', 'recoverPose\n+ Triangulate', 'Map Init\nscale = s₀']
ci = [P['frame'], P['feat'], P['feat'], P['geom'], P['geom'], P['opt']]
wi = [BW, BW, 1.80, BW, 1.90, 1.70]
h_chain(ax1, xi, yi, li, ci, wi)

yt = 1.48
xt = [1.1,  3.0,  5.15,  7.50,  9.85, 12.20, 14.0]
lt = ['Left\nFrame', 'LK Track\nprev → cur', 'PnP + RANSAC\nsolvePnP',
      'Vel / Rot\nGate', 'Local BA\nwindow = 7', 'Pose Update\n(TUM format)', '']
ct = [P['frame'], P['feat'], P['geom'], P['gate'], P['opt'], P['out'], 'white']
wt = [BW, 1.72, 1.82, DW, 1.72, 1.82, 0.01]

for i, (x, lab, col, w) in enumerate(zip(xt[:-1], lt[:-1], ct[:-1], wt[:-1])):
    if lab == 'Vel / Rot\nGate':
        dm(ax1, x, yt, lab, col)
    else:
        bx(ax1, x, yt, lab, col, w=w)

# arrows in tracking row
re = [x + (DW/2 if 'Gate' in lab else w/2) for x, lab, w in zip(xt, lt, wt)]
le = [x - (DW/2 if 'Gate' in lab else w/2) for x, lab, w in zip(xt, lt, wt)]
for i in range(1, 6):
    arr(ax1, re[i-1], yt, le[i], yt)

# "pass" label
ax1.text((re[3] + le[4])/2, yt + 0.14, 'pass',
         ha='center', fontsize=7.8, color=P['opt'],
         style='italic', fontweight='bold')

# "fail" branch: gate → hold pose (below)
yh = 0.72
bx(ax1, xt[3], yh, 'Hold Pose\nfailures ++', P['hold'])
arr(ax1, xt[3], yt - DH/2, xt[3], yh + BH/2,
    label='fail', lside='right', color=P['gate'])

# hold pose rejoins at pose update (dashed)
ax1.annotate('', xy=(xt[5] - wt[5]/2, yh),
             xytext=(xt[3] + DW/2, yh),
             arrowprops=dict(arrowstyle='-|>', color=P['hold'], lw=1.3,
                             linestyle='dashed', mutation_scale=11,
                             connectionstyle='arc3,rad=-0.3'), zorder=2)
ax1.annotate('', xy=(xt[5], yt - BH/2),
             xytext=(xt[5], yh + BH/2),
             arrowprops=dict(arrowstyle='-|>', color=P['hold'], lw=1.3,
                             mutation_scale=11), zorder=2)

# Init → tracking connection (map ready)
arr(ax1, xi[-1] + wi[-1]/2 - 0.02, yi,
    xt[1] + DW/2 + 0.05, yt + BH/2 + 0.03,
    color=P['opt'], rad=-0.45, lw=1.5)
ax1.text(13.6, (yi + yt)/2 + 0.1, 'map\nready',
         ha='center', va='center', fontsize=7, color=P['opt'], style='italic')

# "extend map" annotation (bottom dashed)
ax1.annotate('', xy=(xt[1] + wt[1]/2, yt - BH/2 - 0.08),
             xytext=(xt[4] + wt[4]/2, yt - BH/2 - 0.08),
             arrowprops=dict(arrowstyle='<-', color='#90A4AE', lw=1.1,
                             linestyle='dashed', mutation_scale=10), zorder=2)
ax1.text((xt[1] + xt[4])/2, yt - BH/2 - 0.21,
         'extend map  (new triangulated pts)',
         ha='center', fontsize=7.1, color='#78909C', style='italic')

fig1.tight_layout(pad=0.3)
p1 = 'outputs/pipeline_mono.png'
fig1.savefig(p1, dpi=180, bbox_inches='tight', facecolor='white')
plt.close(fig1)
print(f'Saved  {p1}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 – Stereo VO
# ══════════════════════════════════════════════════════════════════════════════

fig2, ax2 = plt.subplots(figsize=(15, 5.8))
ax2.set_xlim(0, 15)
ax2.set_ylim(0, 5.8)
ax2.axis('off')
fig2.patch.set_facecolor('white')
ax2.set_facecolor('white')

ax2.text(7.5, 5.55, 'Stereo VO Pipeline  (Metric Scale)',
         ha='center', fontsize=14, fontweight='bold', color='#1A237E')

# section backgrounds
bg(ax2,  0.18, 3.05, 14.64, 2.10,
   P['dep_bg'], P['dep_e'], 'Stereo depth estimation  (per frame)')
bg(ax2,  0.18, 0.35, 14.64, 2.45,
   P['trk_bg'],  P['trk_e'], 'Per-frame pose estimation loop')

yd = 3.88
xd = [1.20,  3.30,  5.35,  7.60,  10.05]
ld = ['Left Frame\n+ Right Frame', 'Rectify\n(calibrated)',
      'SGBM\nDisparity', 'Unproject 3D\nZ = fB / d', '3D Point Map\n(metric)']
cd = [P['frame'], P['feat'], P['geom'], P['geom'], P['opt']]
wd = [1.85, BW, BW, 1.85, 1.72]

h_chain(ax2, xd, yd, ld, cd, wd)

yp = 1.48
xp = [1.20,  3.20,  5.30,  7.65,  9.95, 12.25, 14.10]
lp = ['Left Frame\nt → t+1', 'LK Track\nprev → cur',
      'PnP + RANSAC\n3D → 2D', 'Vel / Rot\nGate',
      'Local BA\nwindow = 7', 'Metric\nPose', '']
cp = [P['frame'], P['feat'], P['geom'], P['gate'], P['opt'], P['out'], 'white']
wp = [1.72, 1.72, 1.85, DW, 1.72, 1.55, 0.01]

for i, (x, lab, col, w) in enumerate(zip(xp[:-1], lp[:-1], cp[:-1], wp[:-1])):
    if 'Gate' in lab:
        dm(ax2, x, yp, lab, col)
    else:
        bx(ax2, x, yp, lab, col, w=w)

# arrows
rp = [x + (DW/2 if 'Gate' in l else w/2) for x, l, w in zip(xp, lp, wp)]
lpe = [x - (DW/2 if 'Gate' in l else w/2) for x, l, w in zip(xp, lp, wp)]
for i in range(1, 6):
    arr(ax2, rp[i-1], yp, lpe[i], yp)

# "pass" label
ax2.text((rp[3] + lpe[4])/2, yp + 0.14, 'pass',
         ha='center', fontsize=7.8, color=P['opt'],
         style='italic', fontweight='bold')

# "fail" branch → hold/reinit
yh2 = 0.72
bx(ax2, xp[3], yh2, 'Hold / Reinit\nfailures ++', P['hold'])
arr(ax2, xp[3], yp - DH/2, xp[3], yh2 + BH/2,
    label='fail', lside='right', color=P['gate'])

# hold rejoins pose (dashed)
ax2.annotate('', xy=(xp[5] - wp[5]/2, yh2),
             xytext=(xp[3] + DW/2, yh2),
             arrowprops=dict(arrowstyle='-|>', color=P['hold'], lw=1.3,
                             linestyle='dashed', mutation_scale=11,
                             connectionstyle='arc3,rad=-0.3'), zorder=2)
ax2.annotate('', xy=(xp[5], yp - BH/2),
             xytext=(xp[5], yh2 + BH/2),
             arrowprops=dict(arrowstyle='-|>', color=P['hold'], lw=1.3,
                             mutation_scale=11), zorder=2)

# Depth map feeds PnP (vertical arrow)
arr(ax2, xd[-1], yd - BH/2,
    xp[2] + wp[2]/2 - 0.05, yp + BH/2 + 0.04,
    label='3D points', lside='right', color=P['opt'], rad=-0.2, lw=1.5)

# "Add new SGBM points" annotation
ax2.annotate('', xy=(xd[-1], yd - BH/2 - 0.05),
             xytext=(xp[1] + wp[1]/2, yd - BH/2 - 0.05),
             arrowprops=dict(arrowstyle='<-', color='#90A4AE', lw=1.1,
                             linestyle='dashed', mutation_scale=10), zorder=2)
ax2.text((xd[-1] + xp[1])/2 + 0.5, yd - BH/2 - 0.19,
         'refresh map  (new SGBM points each cycle)',
         ha='center', fontsize=7.1, color='#78909C', style='italic')

# Note about first frame
ax2.text(0.28, 0.19,
         '* First frame: grid detect on left image → SGBM depth → '
         '3D map built at camera origin  (metric, scale = 1).',
         ha='left', va='center', fontsize=7.2, color='#555555', style='italic')

fig2.tight_layout(pad=0.3)
p2 = 'outputs/pipeline_stereo.png'
fig2.savefig(p2, dpi=180, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print(f'Saved  {p2}')
