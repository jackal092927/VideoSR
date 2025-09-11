Noisy-trajectory dataset (perturbations on the path itself)
Timestamp: 1756271692

This dataset renders a projectile whose *true smooth* path is known, but the *visualized path* includes
stochastic perturbations (low-freq jitter + high-freq noise + a few impulses). The tracker therefore
recovers a noisy trajectory, and we use a Kalman smoother + quadratic fit to recover the ideal trend.

Files
-----
- Projectile_noisyTrajectory_1756271692.mp4                  (video rendered from perturbed path)
- Projectile_noisyTrajectory_1756271692_clean_gt.csv         (clean ideal trajectory: t,x,y,vx,vy,ax,ay)
- Projectile_noisyTrajectory_1756271692_perturbed_gt.csv     (the exact noisy path used for rendering: t,x,y)
- Projectile_noisyTrajectory_1756271692_track.csv            (HSV tracker measurements from the video)
- Projectile_noisyTrajectory_1756271692_smoothed.csv         (Kalman-RTS smoothed path from measurements)
- Projectile_noisyTrajectory_1756271692_fitted.csv           (x from smoothed, y from ideal quadratic fit)
- Projectile_noisyTrajectory_1756271692_overlay.mp4          (overlay: red=measured, green=smoothed, blue=fitted)
- Projectile_noisyTrajectory_1756271692_fit.png              (plot of y: measured vs smoothed vs fitted)

Quick metrics
-------------
- g_est ≈ 4.023  (true g = 180.0)
- vy0_est ≈ 39.297  (true vy0 = -173.362)
- RMSE(measured vs clean) ≈ 229.27 px
- RMSE(smoothed  vs clean) ≈ 227.87 px

Suggested usage
---------------
1) Read Projectile_noisyTrajectory_1756271692_track.csv as your raw experimental data (noisy).
2) Apply your preferred smoother/denoiser or parameteric fit; compare against Projectile_noisyTrajectory_1756271692_clean_gt.csv.
3) For presentations, use Projectile_noisyTrajectory_1756271692_overlay.mp4 to show recovery: raw jitter → smooth → ideal fit.
