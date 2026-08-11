
common="--log-level INFO --skip-existing --simulator fftvis32-coupled --force-remake-obsparams --not-redundant --layout HEX --no-compress"

# 1 hour chunks, do three chunks
dimensions=" --n-time-chunks 24 --do-time-chunks 0~3 --channels 746~843 --n-unique-beams 42"

uv run vsim --uv runsim ${common} --cpu ${dimensions}  --prefix coupled --slurm-override time '03:00:00' --slurm-override ntasks-per-node 128 --slurm-override partition RM --sky-model gsm_nside1024 --spline-interp-order 3 --coupled




