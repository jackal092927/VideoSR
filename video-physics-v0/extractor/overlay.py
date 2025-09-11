import cv2
import numpy as np


def write_overlay(cap, df, out_path, width, height, fps, equations=None):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    pts = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(df):
            break
        
        row = df.iloc[frame_idx]
        if np.isfinite(row["x"]) and np.isfinite(row["y"]):
            pts.append((int(row["x"]), int(row["y"])))
            cv2.circle(frame, pts[-1], 5, (0, 255, 0), -1)
        
        if len(pts) > 1:
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], (255, 0, 0), 2)

        # HUD - Tracking info
        txt = f"t={row['t']:.3f}s x={row['x']:.1f} y={row['y']:.1f}"
        cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2)
        
        # Display discovered physics equations if available
        if equations:
            y_offset = 70
            cv2.putText(frame, "Discovered Equations:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
            
            for i, eq in enumerate(equations[:3]):  # Show first 3 equations
                if y_offset < height - 50:  # Don't go off screen
                    cv2.putText(frame, f"{i+1}. {eq}", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    y_offset += 25
        
        writer.write(frame)
        frame_idx += 1

    writer.release()