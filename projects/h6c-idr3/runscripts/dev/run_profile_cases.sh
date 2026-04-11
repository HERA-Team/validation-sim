
common="--log-level INFO --skip-existing --gpu --do-time-chunks 0 --channels 100 --profile --slurm-override time '01:00:00' --redundant"

# These two simply check that 120 time chunks is sufficient to avoid excessive overhead.
uv run vsim runsim ${common} --n-time-chunks 120 --sky-model ptsrc1024 --layout FULL --simulator matvis --prefix h6c-idr3-profiling-1beam
uv run vsim runsim ${common} --n-time-chunks 60 --sky-model ptsrc1024 --layout FULL --simulator matvis --prefix h6c-idr3-profiling-1beam

# Then, we'll want some that test fftvis vs matvis,

# And some that tests multi-beam vs single-beam
uv run vsim runsim ${common} --n-time-chunks 120 --sky-model ptsrc1024 --layout FULL --simulator matvis --n-unique-beams 350 --prefix h6c-idr3-profiling-350beam --not-redundant
uv run vsim runsim ${common} --n-time-chunks 60 --sky-model ptsrc1024 --layout FULL --simulator matvis --n-unique-beams 350 --prefix h6c-idr3-profiling-350beam --not-redundant

# And some that test polarized sky vs not
