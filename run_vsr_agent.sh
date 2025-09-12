#!/usr/bin/env bash
set -euo pipefail

# Simple runner to generate Level A/B/C synthetic videos and run the hybrid agent.
# Configurable via env vars or simple flags.

LLM_HOST=${LLM_HOST:-http://127.0.0.1}
LLM_PORT=${LLM_PORT:-8000}
LEVEL=${LEVEL:-A}               # A, B, C, or ALL
DATA_DIR=${DATA_DIR:-datasets}  # where to write generated videos
RUNS_DIR=${RUNS_DIR:-runs}      # where to write agent outputs
DENOISER=${DENOISER:-none}      # none|savgol|kalman
AUTO_HSV_FLAG=${AUTO_HSV_FLAG:---auto-hsv}  # set to empty to disable
OVERLAY_NO_FIT=${OVERLAY_NO_FIT:---overlay-no-fit}

mkdir -p "${DATA_DIR}" "${RUNS_DIR}"

gen_level_A() {
  mkdir -p "${DATA_DIR}/A"
  echo "[Gen] Level A (mild) videos → ${DATA_DIR}/A"
  python -m llm_video_agent.synth.generate --out "${DATA_DIR}/A/A_projectile.mp4"         --motion projectile          --color blue --seconds 3 --fps 30 --sensor-noise 3 --jpeg-quality 30 --color-drift 3 --traj-jitter 0.5
  python -m llm_video_agent.synth.generate --out "${DATA_DIR}/A/A_cv.mp4"                 --motion constant_velocity   --color blue --seconds 3 --fps 30 --sensor-noise 3 --jpeg-quality 30 --color-drift 3 --traj-jitter 0.5
  python -m llm_video_agent.synth.generate --out "${DATA_DIR}/A/A_freefall.mp4"           --motion free_fall           --color blue --seconds 3 --fps 30 --sensor-noise 3 --jpeg-quality 30 --color-drift 3 --traj-jitter 0.5
  python -m llm_video_agent.synth.generate --out "${DATA_DIR}/A/A_sho.mp4"                --motion sho                 --color blue --seconds 3 --fps 30 --sensor-noise 3 --jpeg-quality 30 --color-drift 3 --traj-jitter 0.5
}

gen_level_B() {
  mkdir -p "${DATA_DIR}/B"
  echo "[Gen] Level B (moderate) videos → ${DATA_DIR}/B"
  python -m llm_video_agent.synth.generate --out "${DATA_DIR}/B/B_projectile.mp4"         --motion projectile          --color blue --seconds 3 --fps 30 --sensor-noise 4 --jpeg-quality 30 --color-drift 3 --motion-blur 9 --motion-blur-angle 15 --shake-px 3 --occlude --occ-frames 8 20 --occ-alpha 0.6 --traj-jitter 1.0
  python -m llm_video_agent.synth.generate --out "${DATA_DIR}/B/B_sho.mp4"                --motion sho                 --color blue --seconds 3 --fps 30 --sensor-noise 4 --jpeg-quality 30 --color-drift 3 --motion-blur 9 --motion-blur-angle 20 --shake-px 3 --occlude --occ-frames 8 16 --occ-alpha 0.5 --traj-jitter 1.0
}
gen_level_C() {
  mkdir -p "${DATA_DIR}/C"
  echo "[Gen] Level C (hard) videos → ${DATA_DIR}/C"
  python -m llm_video_agent.synth.generate --out "${DATA_DIR}/C/C_projectile.mp4" \
    --motion projectile --color blue --seconds 3 --fps 30 \
    --sensor-noise 4 --jpeg-quality 30 --color-drift 3 \
    --motion-blur 9 --motion-blur-angle 15 --shake-px 3 \
    --occlude --occ-frames 8 20 --occ-alpha 0.6 --traj-jitter 1.0 \
    --distractors 3 --distractor-radius-delta 2 --flicker 0.15 \
    --rolling-shutter-px 8 --bg-stripes --bg-speed 1.5
  python -m llm_video_agent.synth.generate --out "${DATA_DIR}/C/C_sho.mp4" \
    --motion sho --color blue --seconds 3 --fps 30 \
    --sensor-noise 4 --jpeg-quality 30 --color-drift 3 \
    --motion-blur 9 --motion-blur-angle 20 --shake-px 3 \
    --occlude --occ-frames 8 16 --occ-alpha 0.5 --traj-jitter 1.0 \
    --distractors 2 --distractor-radius-delta 2 --flicker 0.15 \
    --rolling-shutter-px 6 --bg-stripes --bg-speed 1.0
}

prompt_for() {
  local name="$1"
  if [[ "$name" == *projectile* ]]; then
    if [[ "$name" == C_* ]]; then
      echo "Track the blue ball doing projectile motion. Distractors: one red square and one yellow ball. Use HSV for blue with morphology + nearest-neighbor gating; avoid Kalman unless necessary; ignore red wrap."
    else
      echo "blue ball doing projectile motion"
    fi
  elif [[ "$name" == *freefall* ]] || [[ "$name" == *free_fall* ]]; then
    echo "blue ball free fall"
  elif [[ "$name" == *cv* ]]; then
    echo "blue ball constant velocity motion"
  elif [[ "$name" == *sho* ]]; then
    if [[ "$name" == C_* ]]; then
      echo "Track the blue ball in simple harmonic motion. Distractors: one red square and one yellow ball. Use HSV for blue with morphology + nearest-neighbor gating; avoid Kalman; ignore red wrap."
    else
      echo "blue ball simple harmonic motion"
    fi
  else
    echo "blue ball moving"
  fi
}

run_folder() {
  local folder="$1"
  local out_sub="$2"
  mkdir -p "${RUNS_DIR}/${out_sub}"
  shopt -s nullglob
  for vid in "${folder}"/*.mp4; do
    local base="$(basename "$vid" .mp4)"
    local outdir="${RUNS_DIR}/${out_sub}/${base}"
    local prompt
    prompt="$(prompt_for "$base")"
    echo "[Run] $base → $outdir"
    python -m llm_video_agent.main \
      --video "$vid" \
      --prompt "$prompt" \
      --outdir "$outdir" \
      --backend llmsr_hybrid \
      --llm-host "$LLM_HOST" \
      --llm-port "$LLM_PORT" \
      $AUTO_HSV_FLAG $OVERLAY_NO_FIT \
      --denoiser "$DENOISER"
    gt_csv="${vid%.mp4}_gt.csv"
    if [[ -f "$gt_csv" ]]; then
      cp "$gt_csv" "$outdir/gt.csv" || true
    fi
  done
}

case "$LEVEL" in
  A|a)
    gen_level_A
    run_folder "${DATA_DIR}/A" "A"
    ;;
  B|b)
    gen_level_B
    run_folder "${DATA_DIR}/B" "B"
    ;;
  C|c)
    gen_level_C
    run_folder "${DATA_DIR}/C" "C"
    ;;
  ALL|all)
    gen_level_A; gen_level_B; gen_level_C
    run_folder "${DATA_DIR}/A" "A"
    run_folder "${DATA_DIR}/B" "B"
    run_folder "${DATA_DIR}/C" "C"
    ;;
  *)
    echo "Unknown LEVEL='$LEVEL' (use A, B, C, or ALL)" >&2; exit 1;;
esac

echo "[Done] Outputs in $RUNS_DIR; videos in $DATA_DIR"
