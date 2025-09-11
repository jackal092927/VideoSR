#!/usr/bin/env python3
"""
debug_incline.py - Debug script to help identify incline line points
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from analyze_incline import load_config, load_video, sample_background, detect_incline_line

def visualize_background_and_edges(video_path, config_path=None):
    """Visualize background image, edges, and detected lines to help debug incline detection"""
    
    # Load configuration
    cfg = load_config(Path(config_path) if config_path else None)
    
    # Load video and sample background
    cap, fps, (W, H), N = load_video(Path(video_path))
    bg = sample_background(Path(video_path), fps, N, cfg.fg)
    cap.release()
    
    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, cfg.incline.hough_canny1, cfg.incline.hough_canny2)
    
    # Try to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                           threshold=cfg.incline.hough_threshold,
                           minLineLength=cfg.incline.hough_min_line_len, 
                           maxLineGap=cfg.incline.hough_max_line_gap)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original background
    axes[0, 0].imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Background Image')
    axes[0, 0].axis('off')
    
    # Grayscale
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale Background')
    axes[0, 1].axis('off')
    
    # Edges
    axes[1, 0].imshow(edges, cmap='gray')
    axes[1, 0].set_title(f'Canny Edges (thresholds: {cfg.incline.hough_canny1}, {cfg.incline.hough_canny2})')
    axes[1, 0].axis('off')
    
    # Lines detected
    line_img = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB).copy()
    valid_lines = []
    
    if lines is not None:
        print(f"Found {len(lines)} lines total")
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            angle = 180 - angle if angle > 90 else angle
            length = (dx*dx + dy*dy)**0.5
            
            # Check if line meets angle criteria
            if cfg.incline.min_angle_deg <= angle <= cfg.incline.max_angle_deg:
                valid_lines.append((length, angle, (x1, y1, x2, y2)))
                # Draw valid lines in green
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Add text with angle and length
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.putText(line_img, f'{angle:.1f}° L:{length:.0f}', 
                           (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                # Draw invalid lines in red
                cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                
        print(f"Found {len(valid_lines)} valid lines (angle between {cfg.incline.min_angle_deg}° and {cfg.incline.max_angle_deg}°)")
        
        if valid_lines:
            # Sort by length and show the best candidate
            valid_lines.sort(key=lambda x: x[0], reverse=True)
            best_length, best_angle, (x1, y1, x2, y2) = valid_lines[0]
            print(f"Best line: angle={best_angle:.1f}°, length={best_length:.1f}, points=({x1},{y1}) to ({x2},{y2})")
            
            # Draw best line in blue with thicker line
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 255, 0), 4)
            cv2.circle(line_img, (x1, y1), 5, (0, 0, 255), -1)  # Start point in red
            cv2.circle(line_img, (x2, y2), 5, (255, 0, 0), -1)  # End point in blue
            
    else:
        print("No lines detected!")
    
    axes[1, 1].imshow(line_img)
    axes[1, 1].set_title('Detected Lines (Green=Valid, Red=Invalid, Yellow=Best)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save the debug image
    debug_path = Path('debug_incline_detection.png')
    plt.savefig(debug_path, dpi=150, bbox_inches='tight')
    print(f"Debug visualization saved to: {debug_path}")
    
    # Also save the background image separately for manual inspection
    bg_path = Path('background_image.png')
    cv2.imwrite(str(bg_path), bg)
    print(f"Background image saved to: {bg_path}")
    
    plt.show()
    
    # Print current Hough parameters
    print("\nCurrent Hough parameters:")
    print(f"  Canny thresholds: {cfg.incline.hough_canny1}, {cfg.incline.hough_canny2}")
    print(f"  Hough threshold: {cfg.incline.hough_threshold}")
    print(f"  Min line length: {cfg.incline.hough_min_line_len}")
    print(f"  Max line gap: {cfg.incline.hough_max_line_gap}")
    print(f"  Angle range: {cfg.incline.min_angle_deg}° to {cfg.incline.max_angle_deg}°")
    
    # Suggest manual points if no valid lines found
    if lines is None or not valid_lines:
        print("\n" + "="*60)
        print("SUGGESTION: Use manual points instead of auto-detection")
        print("="*60)
        print("1. Look at the background image and identify the incline line")
        print("2. Choose two points (x1,y1) and (x2,y2) along the incline")
        print("3. Update your config file with:")
        print('   "incline": {')
        print('     "auto_detect": false,')
        print('     "manual_points": [x1, y1, x2, y2]')
        print('   }')
        print("\nTip: Click on the saved background_image.png to get pixel coordinates")

def mouse_callback(event, x, y, flags, param):
    """Mouse callback to help select manual points"""
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked point: ({x}, {y})")
        param['points'].append((x, y))
        
        # Draw point on image
        cv2.circle(param['img'], (x, y), 5, (0, 0, 255), -1)
        cv2.putText(param['img'], f'({x},{y})', (x+10, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow('Select Points', param['img'])
        
        if len(param['points']) == 2:
            x1, y1 = param['points'][0]
            x2, y2 = param['points'][1]
            cv2.line(param['img'], (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow('Select Points', param['img'])
            print(f"Manual points: [{x1}, {y1}, {x2}, {y2}]")
            print("Press any key to finish...")

def select_manual_points(video_path, config_path=None):
    """Interactive tool to select manual incline points"""
    
    cfg = load_config(Path(config_path) if config_path else None)
    cap, fps, (W, H), N = load_video(Path(video_path))
    bg = sample_background(Path(video_path), fps, N, cfg.fg)
    cap.release()
    
    # Create window and set mouse callback
    img_display = bg.copy()
    param = {'img': img_display, 'points': []}
    
    cv2.namedWindow('Select Points', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Select Points', mouse_callback, param)
    
    print("Click two points along the incline line to set manual points")
    print("Press any key when done, or ESC to cancel")
    
    cv2.imshow('Select Points', img_display)
    
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if len(param['points']) == 2:
        x1, y1 = param['points'][0]
        x2, y2 = param['points'][1]
        return [x1, y1, x2, y2]
    else:
        return None

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug incline detection')
    parser.add_argument('--video', type=str, default='./2419_1744339511.mp4', 
                       help='Path to video file')
    parser.add_argument('--config', type=str, default='./incline_config.json',
                       help='Path to config file')
    parser.add_argument('--interactive', action='store_true',
                       help='Launch interactive point selection tool')
    
    args = parser.parse_args()
    
    if args.interactive:
        points = select_manual_points(args.video, args.config)
        if points:
            print(f"\nSelected manual points: {points}")
            print("Add these to your config file:")
            print(f'  "manual_points": {points}')
    else:
        visualize_background_and_edges(args.video, args.config)
