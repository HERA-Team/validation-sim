
common="--log-level INFO --skip-existing --do-time-chunks 0 --channels 100 --profile --slurm-override time '01:00:00' --not-redundant  --sky-model ptsrc1024 --layout FULL"

# These two simply check that 135 time chunks is sufficient to avoid excessive overhead (corresponds to 128 times in each chunk)
# uv run vsim --uv runsim ${common} --gpu --n-time-chunks 135 --simulator matvis  --prefix h6c-idr3-profiling-matvis
# uv run vsim --uv runsim ${common} --gpu --n-time-chunks 270 --simulator matvis --prefix h6c-idr3-profiling-matvis

# And some that tests multi-beam vs single-beam
# uv run vsim --uv runsim ${common} --gpu --n-time-chunks 135 --simulator matvis --n-unique-beams 350 --prefix h6c-idr3-profiling-matvis
# uv run vsim --uv runsim ${common} --gpu --n-time-chunks 270 --simulator matvis --n-unique-beams 350 --prefix h6c-idr3-profiling-matvis

# Then, we'll want some that test fftvis vs matvis,
uv run vsim --uv runsim ${common} --cpu --n-time-chunks 135 --simulator fftvis32 --n-unique-beams 1 --prefix h6c-idr3-profiling-fftvis-cpu --slurm-override ntasks-per-node 64
uv run vsim --uv runsim ${common} --cpu --n-time-chunks 135 --simulator fftvis32 --n-unique-beams 350 --prefix h6c-idr3-profiling-fftvis-cpu --slurm-override ntasks-per-node 64

# And some that test polarized sky vs not
