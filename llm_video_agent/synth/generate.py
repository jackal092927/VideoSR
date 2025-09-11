#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
import numpy as np


COLORS_BGR = {
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "white": (255, 255, 255),
}


def gen_projectile(t, W, H, params):
    x0 = params.get("x0", 50)
    y0 = params.get("y0", 50)
    v0x = params.get("v0x", 120)
    v0y = params.get("v0y", 0)
    g = params.get("g", 200)  # px/s^2, y down positive
    x = x0 + v0x * t
    y = y0 + v0y * t + 0.5 * g * t * t
    return x, y


def gen_constant_velocity(t, W, H, params):
    x0 = params.get("x0", 50)
    y0 = params.get("y0", H//2)
    vx = params.get("vx", 150)
    vy = params.get("vy", 0)
    x = x0 + vx * t
    y = y0 + vy * t
    return x, y


def gen_free_fall(t, W, H, params):
    x0 = params.get("x0", W//2)
    y0 = params.get("y0", 30)
    v0y = params.get("v0y", 0)
    g = params.get("g", 250)
    x = x0 + 0 * t
    y = y0 + v0y * t + 0.5 * g * t * t
    return x, y


def gen_uniform_accel_1d(t, W, H, params):
    # Motion along a line at angle theta
    x0 = params.get("x0", 40)
    y0 = params.get("y0", 40)
    v0 = params.get("v0", 50)
    a = params.get("a", 80)
    theta = np.deg2rad(params.get("theta_deg", 30))
    s = v0 * t + 0.5 * a * t * t
    x = x0 + s * np.cos(theta)
    y = y0 + s * np.sin(theta)
    return x, y


GEN_MAP = {
    "projectile": gen_projectile,
    "constant_velocity": gen_constant_velocity,
    "free_fall": gen_free_fall,
    "uniform_acceleration_1d": gen_uniform_accel_1d,
}


def main():
    ap = argparse.ArgumentParser(description="Generate a simple synthetic 2D video")
    ap.add_argument("--out", required=True, help="Output mp4 path")
    ap.add_argument("--motion", choices=list(GEN_MAP.keys()), default="projectile")
    ap.add_argument("--color", choices=list(COLORS_BGR.keys()), default="blue")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--radius", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.0, help="pixel noise std dev")
    args = ap.parse_args()

    out_path = Path(args.out)
    W, H = args.width, args.height
    N = int(args.seconds * args.fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (W, H))

    gen_fn = GEN_MAP[args.motion]
    for i in range(N):
        t = i / args.fps
        x, y = gen_fn(t, W, H, {})
        if args.noise > 0:
            x += np.random.normal(0, args.noise)
            y += np.random.normal(0, args.noise)
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        cx = int(np.clip(x, args.radius, W-args.radius-1))
        cy = int(np.clip(y, args.radius, H-args.radius-1))
        cv2.circle(frame, (cx, cy), args.radius, COLORS_BGR[args.color], -1)
        writer.write(frame)
    writer.release()
    print(f"[synth] Wrote {out_path}")


if __name__ == "__main__":
    main()

