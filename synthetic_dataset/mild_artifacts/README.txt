mild_artifacts
Timestamp: 1756184360

This is a near-ideal synthetic projectile for end-to-end validation.
- mild Gaussian noise only
- larger target
- ground truth clamped to the visible frame to avoid off-frame bias
- interior-frame fitting (exclude border-clamped frames)

True params:
- g_true = 140.00 px/s^2
- vy0_true = -131.06 px/s

Recovered (interior fit):
- g_est = 140.0542
- vy0_est = -131.2046
- fit_MSE_y = 0.015849
- track_RMSE_px = 0.243931

Files:
- Projectile_mild_1756184360.mp4
- Projectile_mild_1756184360_groundtruth.csv
- Projectile_mild_1756184360_track.csv
- Projectile_mild_1756184360_overlay.mp4
- Projectile_mild_1756184360_fit_interior.png

Suggested use:
1) Run your extractor on Projectile_mild_1756184360.mp4 (or use the provided track CSV).
2) Mask frames near borders or use the provided interior-fit approach.
3) Fit y(t)=p0+p1*t+0.5*p2*t^2 and compare g_est, vy0_est to true.
