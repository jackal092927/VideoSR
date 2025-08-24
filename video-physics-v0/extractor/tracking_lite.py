import cv2
import numpy as np


def _range_mask(hsv, rng):
    lower = np.array(rng["lower"], dtype=np.uint8)
    upper = np.array(rng["upper"], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def track_centroid_hsv(cap, cfg):
    color = cfg.get("color_hsv")
    color_alt = cfg.get("color_hsv_alt")
    min_area = int(cfg.get("min_area", 50))
    mk_open = int(cfg.get("morph", {}).get("open_kernel", 3))
    mk_close = int(cfg.get("morph", {}).get("close_kernel", 5))

    tracks = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = _range_mask(hsv, color)
        
        if color_alt is not None:
            mask = cv2.bitwise_or(mask, _range_mask(hsv, color_alt))

        if mk_open > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk_open, mk_open))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        if mk_close > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk_close, mk_close))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cx = cy = area = np.nan
        ok = False
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(c))
            if area >= min_area:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = float(M["m10"]/M["m00"])
                    cy = float(M["m01"]/M["m00"])
                    ok = True
        
        tracks.append({
            "frame": frame_idx,
            "t": None,  # fill later
            "x": cx,
            "y": cy,
            "area": area,
            "ok": ok
        })
        frame_idx += 1

    # Fill time column after reading fps in caller
    return tracks