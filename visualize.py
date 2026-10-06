#!/usr/bin/env python3
"""Open3D viewer for the clustered frame: every cluster gets its own color.

usage: python3 visualize.py data/points.bin out/labels.bin [--min-size N] [--ply out.ply] [--point-size 2]
"""
import argparse, colorsys
import numpy as np
import open3d as o3d


def read_points(path):
    with open(path, "rb") as f:
        n = int(np.frombuffer(f.read(4), "<i4")[0])
        return np.frombuffer(f.read(), "<f4").reshape(n, 4)


def read_labels(path):
    with open(path, "rb") as f:
        n = int(np.frombuffer(f.read(4), "<i4")[0])
        return np.frombuffer(f.read(), "<i4")[:n]


def cluster_colors(labels, min_size):
    """One distinct RGB per cluster id (golden-ratio hue stepping). Clusters smaller than min_size -> grey."""
    ids, inv, cnt = np.unique(labels, return_inverse=True, return_counts=True)
    palette = np.empty((len(ids), 3))
    for k in range(len(ids)):
        h = (k * 0.61803398875) % 1.0
        s = (0.95, 0.75, 0.55)[k % 3]          # vary sat/value so neighbouring hues still differ
        v = (1.00, 0.85)[k % 2]
        palette[k] = colorsys.hsv_to_rgb(h, s, v)
    palette[cnt < min_size] = 0.35
    return palette[inv], len(ids), int((cnt >= min_size).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("points"); ap.add_argument("labels")
    ap.add_argument("--min-size", type=int, default=1,
                    help="clusters with fewer points are drawn dark grey (default 1 = color everything)")
    ap.add_argument("--point-size", type=float, default=2.0)
    ap.add_argument("--ply", default=None, help="also save the colored cloud to this .ply file")
    a = ap.parse_args()

    P = read_points(a.points)
    L = read_labels(a.labels)
    assert len(P) == len(L), "points/labels length mismatch"

    colors, n_clusters, n_shown = cluster_colors(L, a.min_size)
    print(f"{len(P)} points, {n_clusters} clusters ({n_shown} colored, min_size={a.min_size})")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P[:, :3].astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if a.ply:
        o3d.io.write_point_cloud(a.ply, pcd)

    vis = o3d.visualization.Visualizer()
    vis.create_window("Clustered LiDAR frame", width=1600, height=900)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.0, 0.0, 0.0])
    opt.point_size = a.point_size
    # x forward, y left, z up: view from behind the sensor, looking along +x
    ctl = vis.get_view_control()
    ctl.set_front([-1.0, 0.0, 0.3]); ctl.set_lookat(P[:, :3].mean(axis=0).tolist()); ctl.set_up([0.0, 0.0, 1.0])
    ctl.set_zoom(0.25)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
