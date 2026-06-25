
common="--log-level DEBUG --skip-existing --profile --profile-timer-unit 1e-2 --simulator fftvis32-coupled --force-remake-obsparams --not-redundant --layout HEX --no-compress"

# We do the two different spline interp orders with different channels or time chunks so they don't overwrite each other.

# Diffuse long-time-axis tests. 3 hours for two channels to check FRF.
dimensions=" --n-time-chunks 8640 --do-time-chunks 0 --channels 1 --n-unique-beams 42"

uv run vsim --uv runsim ${common} --cpu ${dimensions}  --prefix coupled --slurm-override time '03:00:00' --slurm-override ntasks-per-node 128 --slurm-override partition RM --sky-model ptsrc1024 --spline-interp-order 3 --coupled




