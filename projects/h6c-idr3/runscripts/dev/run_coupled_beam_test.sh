
common="--log-level DEBUG --skip-existing --profile --profile-timer-unit 1e-2 --simulator fftvis --force-remake-obsparams --not-redundant --layout HEX"

# We do the two different spline interp orders with different channels or time chunks so they don't overwrite each other.

# Diffuse long-time-axis tests. 3 hours for two channels to check FRF.
dimensions=" --n-time-chunks 8640 --do-time-chunks 0 --channels 1 --n-unique-beams 42"

./vsim.py runsim ${common} ${dimensions}  --prefix coupled --slurm-override time '03:00:00' --slurm-override ntasks 3 --sky-model gsm_nside1024 --spline-interp-order 3 --coupled




