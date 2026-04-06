"""Rescale EOR sims to have different power spectrum level"""

import numpy as np
import h5py
from pathlib import Path

scale = np.load("eor_rescaling.npy")

# These are the files that go into the mock_data script, but they are not the ones that go into the mock_lstbin_data script
#files = sorted(Path("/lustre/aoc/projects/hera/Validation/H6C_IDR2/sim_data/eor/").glob("*.uvh5"))

# These files go into the mock_lstbin_data script
#files = sorted(Path("/lustre/aoc/projects/hera/Validation/H6C_IDR2/chunked-ideal-data/eor-grf-1024").glob("*.uvh5"))

# These files are the outputs of the mock_lstbin_data script. We only run this because we used the wrong files
# above the first time when running the mock_lstbin_data script.
#files = sorted(Path("/lustre/aoc/projects/hera/Validation/H6C_IDR2/lstbin-outputs/eor-only").glob("*.sum.uvh5"))

# But actually I needed to do this on the corner-turned single-baseline files, so run on these files once:
files = sorted(Path("/lustre/aoc/projects/hera/Validation/H6C_IDR2/lstbin-outputs/eor-only/single_baseline_files").glob("*.sum.uvh5"))

for i, fl in enumerate(files):
    print(f"Doing file {i+1} of {len(files)}")
    with h5py.File(fl, 'a') as _fl:
        history = _fl['Header']['history'][()].decode()
        if "Rescaled to match noise power" in history:
            warnings.warn(f"File {fl} already has scaled EOR power")
            continue
             
        data = _fl['Data']['visdata'][()]
 
        _fl['Data']['visdata'][...] = data * scale[None, :, None]
        
        history += "\nRescaled to match noise power at k=1"
        del _fl['Header']['history']
        _fl['Header']['history'] = history