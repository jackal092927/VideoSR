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
    ap.add_argument("--motion", choices=list(GEN_MAP.keys()) + ["sho"], default="projectile")
    ap.add_argument("--color", choices=list(COLORS_BGR.keys()), default="blue")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--radius", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.0, help="(deprecated) trajectory jitter std dev [px]")
    ap.add_argument("--traj-jitter", type=float, default=0.0, help="Gaussian jitter on trajectory positions [px]")
    ap.add_argument("--sensor-noise", type=float, default=0.0, help="Gaussian noise sigma added to image intensities")
    ap.add_argument("--color-drift", type=float, default=0.0, help="Per-frame hue drift in degrees (approx)")
    ap.add_argument("--jpeg-quality", type=int, default=0, help="If >0, re-encode frames as JPEG with this quality (e.g., 28-32)")
    ap.add_argument("--motion-blur", type=int, default=0, help="Motion blur kernel length (e.g., 5-11); 0 to disable")
    ap.add_argument("--motion-blur-angle", type=float, default=0.0, help="Motion blur angle in degrees (e.g., 10-25)")
    ap.add_argument("--shake-px", type=float, default=0.0, help="Amplitude of random camera shake in pixels (e.g., 2-4)")
    ap.add_argument("--occlude", action="store_true", help="Enable an occlusion window near the object")
    ap.add_argument("--occ-frames", type=int, nargs=2, default=(0, 0), help="Occlusion duration range: min max frames (e.g., 8 20)")
    ap.add_argument("--occ-alpha", type=float, default=0.5, help="Occlusion opacity (0..1)")
    # Level C extras
    ap.add_argument("--distractors", type=int, default=0, help="Number of distractor objects")
    ap.add_argument("--distractor-radius-delta", type=int, default=0, help="Radius +/- delta for distractors")
    ap.add_argument("--distractor-colors", type=str, default="", help="Comma-separated distractor colors (blue,red,green,yellow,white)")
    ap.add_argument("--distractor-shapes", type=str, default="", help="Comma-separated shapes for distractors: ball|square")
    ap.add_argument("--flicker", type=float, default=0.0, help="Illumination flicker amplitude (fraction, e.g., 0.15 for ±15%)")
    ap.add_argument("--rolling-shutter-px", type=int, default=0, help="Row-wise horizontal shift magnitude in pixels (0 disables)")
    ap.add_argument("--bg-stripes", action="store_true", help="Draw moving background stripes (parallax)")
    ap.add_argument("--bg-speed", type=float, default=1.0, help="Background stripes speed (px/frame)")
    # (duplicate Level C extras removed)
    args = ap.parse_args()

    out_path = Path(args.out)
    W, H = args.width, args.height
    N = int(args.seconds * args.fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (W, H))

    gen_fn = GEN_MAP.get(args.motion, gen_constant_velocity)

    # Prepare motion blur kernel if requested
    klen = int(args.motion_blur)
    kernel = None
    if klen and klen > 1:
        kernel = np.zeros((klen, klen), dtype=np.float32)
        angle = np.deg2rad(args.motion_blur_angle)
        cxk = klen//2; cyk = klen//2
        for i in range(klen):
            xk = int(cxk + (i - klen//2) * np.cos(angle))
            yk = int(cyk + (i - klen//2) * np.sin(angle))
            if 0 <= xk < klen and 0 <= yk < klen:
                kernel[yk, xk] = 1.0
        if kernel.sum() == 0:
            kernel[:, klen//2] = 1.0
        kernel /= max(1.0, kernel.sum())

    # Occlusion schedule
    occ_start = occ_end = -1
    if args.occlude and args.occ_frames[1] > 0:
        import random
        minf, maxf = args.occ_frames
        dur = random.randint(minf, maxf)
        occ_start = max(0, N//3 - dur//2)
        occ_end = min(N-1, occ_start + dur)

    # Prepare distractors (constant velocity)
    rng = np.random.default_rng(1234)
    distractors = []
    # Optional explicit color/shape lists
    color_list = [c.strip() for c in str(args.distractor_colors).split(',') if c.strip()]
    shape_list = [s.strip().lower() for s in str(args.distractor_shapes).split(',') if s.strip()]
    for k in range(int(args.distractors)):
        dx0 = float(rng.uniform(args.radius, W-args.radius))
        dy0 = float(rng.uniform(args.radius, H-args.radius))
        dvx = float(rng.uniform(-60, 60))
        dvy = float(rng.uniform(-60, 60))
        dr = max(2, args.radius + int(rng.integers(-abs(args.distractor_radius_delta), abs(args.distractor_radius_delta)+1)))
        # Pick distractor color: from provided list, else choose different than main
        if color_list:
            dc = color_list[min(k, len(color_list)-1)]
            dc = dc if dc in COLORS_BGR else args.color
        else:
            alt = [c for c in COLORS_BGR.keys() if c != args.color]
            dc = alt[k % len(alt)] if alt else args.color
        color = list(COLORS_BGR[dc])
        # Optionally apply small hue drift to distractor color too
        if args.color_drift != 0:
            sw = np.zeros((1,1,3), dtype=np.uint8)
            sw[0,0] = color
            hsv = cv2.cvtColor(sw, cv2.COLOR_BGR2HSV)
            hsv[...,0] = (hsv[...,0].astype(np.int16) + int(rng.uniform(-args.color_drift, args.color_drift))) % 180
            sw = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            color = sw[0,0].tolist()
        # Pick shape: from list or alternate
        if shape_list:
            shape = shape_list[min(k, len(shape_list)-1)]
            shape = shape if shape in ("ball","square") else "ball"
        else:
            shape = "square" if (k % 2 == 0) else "ball"
        distractors.append({"x0":dx0,"y0":dy0,"vx":dvx,"vy":dvy,"r":int(dr),"color":color, "shape": shape})

    # GT arrays
    gt_base = []
    gt_pert = []
    for i in range(N):
        t = i / args.fps
        # Generate motion
        if args.motion == 'sho':
            # Horizontal SHO
            A = 80.0; omega = 2*np.pi/2.0; x0=W//2; y0=H//2
            x = x0 + A*np.cos(omega*t)
            y = y0
        else:
            x, y = gen_fn(t, W, H, {})

        jitter = args.traj_jitter if args.traj_jitter > 0 else args.noise
        xj, yj = x, y
        if jitter > 0:
            xj += np.random.normal(0, jitter)
            yj += np.random.normal(0, jitter)
        gt_base.append((i, t, float(x), float(y)))
        gt_pert.append((i, t, float(xj), float(yj)))
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        cx = int(np.clip(xj, args.radius, W-args.radius-1))
        cy = int(np.clip(yj, args.radius, H-args.radius-1))

        # Background stripes (parallax)
        if args.bg_stripes:
            for yrow in range(0, H, 20):
                offset = int((i*args.bg_speed) % 40)
                xstart = (yrow + offset) % 40
                cv2.rectangle(frame, (xstart, yrow), (min(W-1, xstart+10), min(H-1, yrow+20)), (10,10,10), -1)
        # Draw ball with optional color drift
        color = list(COLORS_BGR[args.color])
        if args.color_drift != 0:
            ball = np.zeros((1,1,3), dtype=np.uint8)
            ball[0,0] = color
            hsv = cv2.cvtColor(ball, cv2.COLOR_BGR2HSV)
            hsv[...,0] = (hsv[...,0].astype(np.int16) + int(np.random.uniform(-args.color_drift, args.color_drift))) % 180
            ball = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            color = ball[0,0].tolist()
        cv2.circle(frame, (cx, cy), args.radius, color, -1)

        # Draw distractors
        for d in distractors:
            xd = d["x0"] + d["vx"]*t
            yd = d["y0"] + d["vy"]*t
            cxd = int(np.clip(xd, d["r"], W-d["r"]-1))
            cyd = int(np.clip(yd, d["r"], H-d["r"]-1))
            if d.get("shape") == "square":
                cv2.rectangle(frame, (cxd-d["r"], cyd-d["r"]), (cxd+d["r"], cyd+d["r"]), d["color"], -1)
            else:
                cv2.circle(frame, (cxd, cyd), d["r"], d["color"], -1)

        # Occlusion near the object
        if args.occlude and occ_start <= i <= occ_end:
            occ = frame.copy()
            x0 = max(0, cx - 2*args.radius); y0 = max(0, cy - 2*args.radius)
            x1 = min(W-1, cx + 2*args.radius); y1 = min(H-1, cy + 2*args.radius)
            cv2.rectangle(occ, (x0, y0), (x1, y1), (0,0,0), -1)
            frame = cv2.addWeighted(occ, args.occ_alpha, frame, 1-args.occ_alpha, 0)

        # Motion blur
        if kernel is not None:
            frame = cv2.filter2D(frame, -1, kernel)

        # Camera shake
        if args.shake_px > 0:
            sx = int(np.random.uniform(-args.shake_px, args.shake_px))
            sy = int(np.random.uniform(-args.shake_px, args.shake_px))
            M = np.float32([[1,0,sx],[0,1,sy]])
            frame = cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

        # Sensor noise
        if args.sensor_noise > 0:
            noise = np.random.normal(0, args.sensor_noise, frame.shape).astype(np.float32)
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Illumination flicker (gamma/brightness)
        if args.flicker > 0:
            gain = 1.0 + np.random.uniform(-args.flicker, args.flicker)
            frame = np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)

        # JPEG round-trip to simulate compression artifacts
        if args.jpeg_quality and args.jpeg_quality > 0:
            enc_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]
            ok, buf = cv2.imencode('.jpg', frame, enc_param)
            if ok:
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        # Rolling shutter warp (horizontal per-row shift proportional to row index)
        if args.rolling_shutter_px and args.rolling_shutter_px != 0:
            rs = np.zeros_like(frame)
            for yrow in range(H):
                shift = int((args.rolling_shutter_px * yrow) / H)
                rs[yrow] = np.roll(frame[yrow], shift, axis=0)  # shift columns
            frame = rs
        writer.write(frame)
    writer.release()
    print(f"[synth] Wrote {out_path}")

    # Save GT CSVs next to the video
    import csv
    gt_path = out_path.with_name(out_path.stem + "_gt.csv")
    pert_path = out_path.with_name(out_path.stem + "_perturbed_gt.csv")
    with open(gt_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(["frame","t","x","y"]) ; w.writerows(gt_base)
    with open(pert_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(["frame","t","x","y"]) ; w.writerows(gt_pert)
    print(f"[synth] Wrote {gt_path.name} and {pert_path.name}")


if __name__ == "__main__":
    main()
