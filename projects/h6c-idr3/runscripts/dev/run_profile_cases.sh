
common="--log-level INFO --skip-existing --gpu --do-time-chunks 0 --channels 100 --profile --slurm-override time '01:00:00' --not-redundant --prefix h6c-idr3-profiling  --sky-model ptsrc1024 --layout FULL --uv"

# These two simply check that 120 time chunks is sufficient to avoid excessive overhead.
uv run vsim runsim ${common} --n-time-chunks 120 --simulator matvis
uv run vsim runsim ${common} --n-time-chunks 60 --simulator matvis

# Then, we'll want some that test fftvis vs matvis,

# And some that tests multi-beam vs single-beam
uv run vsim runsim ${common} --n-time-chunks 120 --simulator matvis --n-unique-beams 350
uv run vsim runsim ${common} --n-time-chunks 60 --simulator matvis --n-unique-beams 350

# And some that test polarized sky vs not
