#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="validation_2025"
WORKSPACE_DIR="/home/ikr/gitrepos/sunflow-scores"
OUTPUT_DIR="/dmidata/projects/weather2x/Energivejr_historical_data/sunflow_validation_scores/v1.0.0/2025"
NWC_BASE="/dmidata/projects/weather2x/Energivejr_historical_data/sunflow_validation_output/v1.0.0"
OBS_BASE="/dmidata/projects/weather2x/Energivejr_historical_data/KNMI_MSGCPP_reproj"
LOG_FILE="${OUTPUT_DIR}/run_validation_2025_tmux.log"
INNER_SCRIPT="$(mktemp /tmp/run_validation_2025_inner.XXXXXX.sh)"

export WORKSPACE_DIR OUTPUT_DIR NWC_BASE OBS_BASE LOG_FILE

mkdir -p "$OUTPUT_DIR"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists"
  exit 1
fi

cat >"$INNER_SCRIPT" <<'EOF'
#!/usr/bin/env bash
set +e
set +u
set +o pipefail

exec > >(tee -a "${LOG_FILE}") 2>&1

cd "${WORKSPACE_DIR}"

# Make sure the uv environment is ready.
uv sync
SYNC_STATUS=$?
if [[ $SYNC_STATUS -ne 0 ]]; then
  echo "uv sync failed with status $SYNC_STATUS"
fi

cleanup() {
  if [[ -n "${KINIT_PID:-}" && "${KINIT_PID}" =~ ^[0-9]+$ ]]; then
    kill "${KINIT_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "tmux runner started in ${WORKSPACE_DIR}"

# Keep the Kerberos ticket alive if it is renewable.
(
  while true; do
    kinit -R >/dev/null 2>&1 || true
    sleep 1800
  done
) &
KINIT_PID=$!

for m in $(seq -w 1 12); do
  month=2025${m}
  nwc_dir="${NWC_BASE}/${month}"
  obs_dir="${OBS_BASE}/${month}"

  if [[ ! -d "${nwc_dir}" || ! -d "${obs_dir}" ]]; then
    echo "SKIP month ${month}: missing input folder"
    continue
  fi

  last_day=$(date -d "2025-${m}-01 +1 month -1 day" +%d)
  for d in $(seq -w 1 "${last_day}"); do
    day=2025-${m}-${d}
    echo "Running ${day}"
    uv run python run_validation.py \
      --start "${day}" \
      --end "${day}" \
      --nwc-dir "${nwc_dir}" \
      --obs-dir "${obs_dir}" \
      --output-dir "${OUTPUT_DIR}" \
      --nowcast_ghi_var probabilistic_advection \
      --obs_ghi_var sds \
      --obs_cs_ghi_var sds_cs
    RUN_STATUS=$?
    if [[ $RUN_STATUS -ne 0 ]]; then
      echo "Day ${day} failed with status ${RUN_STATUS}, continuing"
    fi
  done
done
echo "tmux runner finished all months"
echo "Keeping tmux session alive. Log file: ${LOG_FILE}"
while true; do
  sleep 3600
done
EOF

chmod +x "$INNER_SCRIPT"
tmux new-session -d -s "$SESSION_NAME" "env WORKSPACE_DIR='$WORKSPACE_DIR' OUTPUT_DIR='$OUTPUT_DIR' NWC_BASE='$NWC_BASE' OBS_BASE='$OBS_BASE' LOG_FILE='$LOG_FILE' bash '$INNER_SCRIPT'"

echo "Started tmux session '$SESSION_NAME'. Attach with: tmux attach -t $SESSION_NAME"
