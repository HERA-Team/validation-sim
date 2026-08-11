#!/usr/bin/env python3
"""Utilities for running a full simulation."""

import logging
import subprocess
from importlib.metadata import version
from pathlib import Path

import hera_sim
import yaml

from . import paths
from .cli._utils import _get_sbatch_program
from .obsparams import make_hera_obsparam

logger = logging.getLogger(__name__)

_HS_VERSION = hera_sim.__version__


def calculate_expected_time(chunks: int) -> int:
    """
    Calculate the expected time for a given number of chunks.

    Calibrated to Bridges GPU.
    """
    nt = 17280 // chunks

    # 5 min buffer + (nt / 5760) * 1 hr
    return int(max(5, (nt / 5760) * 60))


def run_validation_sim(
    layout: str,
    channels: list[int],
    sky_model: str,
    n_time_chunks: int,
    gpu: bool,
    slurm_override,
    skip_existing: bool,
    force_remake_obsparams: bool,
    log_level: str,
    dry_run: bool,
    ideal_layout: bool = True,
    simulator: str = "matvis",
    freq_interp_kind: str = "cubic",
    spline_interp_order: int = 1,
    do_time_chunks: list[int] | None = None,
    profile: bool = False,
    profile_timer_unit=1e-2,
    redundant: bool = False,
    compress: bool = True,
    prefix: str = "default",
    phase_center_name: str = "zenith",
    n_unique_beams: int = 1,
    conda: bool = True,
    coupled: bool = False,
):
    """Run a full validation sim on SLURM compute."""
    sgpu = "gpu" if gpu else "cpu"
    simulator_config = paths.SIMULATOR_SPECS / f"{simulator}-{sgpu}.yaml"

    assert simulator_config.exists(), (
        f"Simulator config file {simulator_config.name} does not exist."
    )

    with open(simulator_config) as fl:
        sc = yaml.load(fl, Loader=yaml.FullLoader)

    logger.info(f"Frequency channels to run: {channels}")

    layout_file = make_hera_obsparam(
        layout=layout,
        ideal_layout=ideal_layout,
        channels=channels,
        sky_model=sky_model,
        chunks=n_time_chunks,
        freq_interp_kind=freq_interp_kind,
        spline_interp_order=spline_interp_order,
        force=force_remake_obsparams,
        do_chunks=do_time_chunks,
        beam_interpolator=sc.get("interpolation_function", "az_za_map_coordinates"),
        redundant=redundant,
        prefix=prefix,
        n_unique_beams=n_unique_beams,
        coupled_beams=coupled,
    )
    if not layout_file.exists():
        raise ValueError(f"Error in creating layout file: {layout_file}")
    else:
        pass

    logger.info(f"Created obsparam layout file: {layout_file}")

    if not do_time_chunks:
        do_time_chunks = list(range(n_time_chunks))

    time_est = calculate_expected_time(n_time_chunks)

    modeldir = paths.get_direc(
        sky_model=sky_model,
        chunks=n_time_chunks,
        layout=layout_file.stem,
        redundant=redundant,
        prefix=prefix,
        coupled=coupled,
    )

    # We want to override the job-name to be <sky_model>-<fch>-<ch>, but the last two
    # variables have to be accessed in the loop, so we will be instead override it to
    # a Python string formatting pattern and format it in the loop.
    logdir = paths.LOGDIR / "vis" / modeldir
    logdir.mkdir(parents=True, exist_ok=True)

    # Note that click default `slurm_overrride` to (), and we want it to be "2D" tuple
    slurm_override = (
        *slurm_override,
        ("job-name", "{jobname}"),
        ("output", "{logdir}/{jobname}-%J.out"),
    )

    if "time" not in [x[0] for x in slurm_override]:
        slurm_override = (*slurm_override, ("time", f"0-00:{time_est}:00"))

    # Make the SBATCH script minus hera-sim-vis.py command
    program = _get_sbatch_program(gpu=gpu, conda=conda, slurm_override=slurm_override)

    out_dir = paths.OUTDIR / modeldir
    obsp_dir = paths.OBSPDIR / modeldir
    out_dir.mkdir(parents=True, exist_ok=True)
    obsp_dir.mkdir(parents=True, exist_ok=True)

    compress_cache = paths.COMPRESSDIR / paths.COMPRESS_FMT.format(
        chunks=n_time_chunks, layout_file=layout_file.stem
    )
    if not paths.COMPRESSDIR.exists():
        paths.COMPRESSDIR.mkdir(parents=True)

    # Option for hera-sim-vis.py. Let's just keep this fixed.
    sim_options = (
        f"--normalize_beams --fix_autos --log-level {log_level} "
        f"--phase-center-name {phase_center_name}"
    )

    if not redundant and compress:
        # Compress the results by redundancy unless unless the user explicitly said not
        # to, or if we can send in purely the redundant info to the simulator to begin
        # with. This is because the compression step is not free, and if we can avoid it
        # by just simulating the redundant info, then that is preferable.
        sim_options += f" --compress {compress_cache}"

    for fch in channels:
        for ch in do_time_chunks:
            logger.info(f"Working on frequency channel {fch} chunk {ch}")
            jobname = modeldir / paths.get_file(chunk=ch, channel=fch, with_dir=False)
            outfile = (paths.OUTDIR / jobname).with_suffix(".uvh5")
            obsp = paths.OBSPDIR / jobname
            if not obsp.exists():
                raise ValueError(f"No obsparam file exists: {obsp}!")

            logger.info(f"{outfile.as_posix()}")
            # Check if output file already existed, if clobber is False
            if skip_existing and outfile.exists():
                logger.warning(f"File {outfile} exists, skipping")
                continue

            trace = ""
            profilestr = ""
            if profile:
                proflabel = jobname / f"mv{version('matvis')}-hs{version('hera_sim')}"
                profout = Path("profiling") / proflabel
                profout.parent.mkdir(parents=True, exist_ok=True)

                if gpu:
                    trace = (
                        f"nsys profile -w true -t cuda,cublas -s cpu -f true -x true -o {profout} "
                    )
                else:
                    trace = ""
                profilestr = (
                    f"--profile --profile-timer-unit {profile_timer_unit} "
                    f"--profile-output {profout}.profile.txt"
                )
                prof_funcs = [
                    "hera_sim.visibilities.simulators:ModelData.from_config",
                    "pyuvsim.simsetup:initialize_catalog_from_params",
                    "pyradiosky:SkyModel.from_file",
                ]
                if simulator == "matvis":
                    prof_funcs.extend(
                        [
                            f"matvis.{'gpu' if gpu else 'cpu'}:simulate",
                            "hera_sim.visibilities.matvis:MatVis",
                        ]
                    )
                elif simulator == "fftvis":
                    dstr = "gpu" if gpu else "cpu"
                    prof_funcs.extend(
                        [
                            f"fftvis.{dstr}.{dstr}_simulate:{dstr.upper()}SimulationEngine",
                            "hera_sim.visibilities.fftvis:FFTVis",
                        ]
                    )

                prof_funcs = ",".join(prof_funcs)
                profilestr += f' --profile-funcs "{prof_funcs}"'

            cmd = f"{trace}hera-sim-vis.py {sim_options} {profilestr} {obsp} {simulator_config}"

            if paths.HPC_CONFIG["slurm"]:
                # Write job script and submit
                sbatch_dir = paths.REPODIR / "batch_scripts/vis"
                sbatch_dir.mkdir(parents=True, exist_ok=True)

                sbatch_file = sbatch_dir / jobname
                sbatch_file.parent.mkdir(parents=True, exist_ok=True)

                logger.info(f"Creating sbatch file: {sbatch_file}")
                # Now, join the job script with the hera-sim-vis.py command
                # and format the job-name
                job_script = "\n".join([program, "", cmd, ""]).format(
                    jobname=jobname, logdir=paths.LOGDIR / "vis"
                )
                with open(sbatch_file, "w") as fl:
                    fl.write(job_script)

                if not dry_run:
                    subprocess.call(f"sbatch {sbatch_file}".split())

                logger.debug(f"\n===Job Script===\n{job_script}\n===END===\n")
            else:
                logger.info(f"Running the simulation locally\nCommand: {cmd}")
                if not dry_run:
                    subprocess.call(cmd.split())
