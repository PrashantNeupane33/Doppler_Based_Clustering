#!/usr/bin/env python3
"""
    convert data (shperical to rectagular): {x,y,z,v}
"""
import argparse, struct
import numpy as np

def load_xyzv(path):
    a = np.load(path)
    need = {"azimuth", "elevation", "range", "velocity", "valid"}
    if a.dtype.names is None or not need <= set(a.dtype.names):
        raise SystemExit(f"unexpected format, need fields {need}, got {a.dtype}")
    keep = np.flatnonzero(a["valid"] & (a["range"] > 0) & np.isfinite(a["range"]))
    s = a[keep]
    az, el, r = np.deg2rad(s["azimuth"]), np.deg2rad(s["elevation"]), s["range"]
    xyzv = np.column_stack([r*np.cos(el)*np.cos(az),
                            r*np.cos(el)*np.sin(az),
                            r*np.sin(el),
                            s["velocity"]]).astype(np.float32)
    return a, keep, xyzv

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npy"); ap.add_argument("out_bin"); ap.add_argument("out_idx")
    args = ap.parse_args()
    a, keep, xyzv = load_xyzv(args.npy)
    with open(args.out_bin, "wb") as f:
        f.write(struct.pack("<i", len(xyzv)))
        f.write(np.ascontiguousarray(xyzv).tobytes())
    np.save(args.out_idx, keep.astype(np.int64))   # row in original frame for each cluster-input point
