
common="--log-level INFO --skip-existing --do-time-chunks 0 --channels 100 --profile --slurm-override time '01:00:00' --not-redundant  --sky-model ptsrc1024 --layout FULL --no-compress"

# These two simply check that 135 time chunks is sufficient to avoid excessive overhead (corresponds to 128 times in each chunk)
# uv run vsim --uv runsim ${common} --gpu --n-time-chunks 2880 --simulator matvis  --prefix h6c-idr3-profiling-matvis
# uv run vsim --uv runsim ${common} --gpu --n-time-chunks 270 --simulator matvis --prefix h6c-idr3-profiling-matvis

# CPU runs for MatVis, just to compare with GPU
#uv run vsim --uv runsim ${common} --cpu --n-time-chunks 2880 --simulator matvis  --prefix h6c-idr3-profiling-matvis-cpu

# And some that tests multi-beam vs single-beam
# uv run vsim --uv runsim ${common} --gpu --n-time-chunks 135 --simulator matvis --n-unique-beams 350 --prefix h6c-idr3-profiling-matvis
# uv run vsim --uv runsim ${common} --gpu --n-time-chunks 270 --simulator matvis --n-unique-beams 350 --prefix h6c-idr3-profiling-matvis

# Then, we'll want some that test fftvis.
#uv run vsim --uv runsim ${common} --cpu --n-time-chunks 135 --simulator fftvis32 --n-unique-beams 1 --prefix h6c-idr3-profiling-fftvis-cpu --slurm-override ntasks-per-node 64
#uv run vsim --uv runsim ${common} --cpu --n-time-chunks 27  --simulator fftvis32 --n-unique-beams 1 --prefix h6c-idr3-profiling-fftvis-cpu --slurm-override ntasks-per-node 64
# uv run vsim --uv runsim ${common} --cpu --n-time-chunks 135 --simulator fftvis32 --n-unique-beams 10 --prefix h6c-idr3-profiling-fftvis-cpu --slurm-override ntasks-per-node 128 --slurm-override partition RM
#uv run vsim --uv runsim ${common} --cpu --n-time-chunks 135 --simulator fftvis32 --n-unique-beams 30 --prefix h6c-idr3-profiling-fftvis-cpu --slurm-override ntasks-per-node 128 --slurm-override partition RM

# See how many cores we can get going simultaneously on fftvis. Try using all cores as separate processes,
# and increase nchunks to buy back memory. Need to set number of integrations higher or else fftvis
# short-circuits out and uses one process.
#uv run vsim --uv runsim ${common} --cpu --n-time-chunks 27 --simulator fftvis128 --n-unique-beams 30 --prefix h6c-idr3-profiling-fftvis128-cpu --slurm-override ntasks-per-node 128 --slurm-override partition RM
#uv run vsim --uv runsim ${common} --cpu --n-time-chunks 135 --simulator fftvis64 --n-unique-beams 30 --prefix h6c-idr3-profiling-fftvis64-cpu --slurm-override ntasks-per-node 128 --slurm-override partition RM


uv run vsim --uv runsim ${common} --cpu --n-time-chunks 135 --simulator fftvis64 --n-unique-beams 60 --prefix h6c-idr3-profiling-fftvis64-cpu --slurm-override ntasks-per-node 128 --slurm-override partition RM


# And some that test polarized sky vs not
