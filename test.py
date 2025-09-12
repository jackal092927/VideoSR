# save as check_hsv.py
import cv2, sys, numpy as np
vid = sys.argv[1]
lower = np.array([100, 80, 40], np.uint8)
upper = np.array([130,255,255], np.uint8)

cap = cv2.VideoCapture(vid)
ok, f0 = cap.read(); assert ok, "Cannot read video"
hsv = cv2.cvtColor(f0, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower, upper)
print("mask nonzero ratio:", (mask>0).mean())
cv2.imwrite("mask_check.png", mask)