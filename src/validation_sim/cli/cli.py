#!/usr/bin/env python3
"""Entry point for top-level CLI."""

import logging
import subprocess
from pathlib import Path

import click
from rich.logging import RichHandler

from .. import paths, set_project_path
from . import _utils
from .monitor import type_click_app as monitor_app

# TODO: this should be better refactored into a "profiling" sub-group
from .process_fftvis_profile import typer_click_app as process_fftvis_profile_app
from .rechunk_fast import click_app as rechunk_fast_app

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": 100}

logger = logging.getLogger(__name__)


@click.group(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging level to use.",
)
@click.option(
    "-p",
    "--project-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=Path(),
    help="Path to the root of the validation sim repository. If not given, defaults to the current working directory.",
)
@click.option(
    "--conda/--uv",
    default=True,
    help="Whether to use conda or uv for environment management.",
)
@click.pass_context
def cli(ctx, log_level, project_dir, conda):
    """Make job scripts and run visibility simulations via hera-sim-vis.py."""
    logger.setLevel(log_level)
    if project_dir is not None:
        set_project_path(project_dir)

    # Put all options that should be passed through to subcommands in here.
    ctx.obj = {"top-level-args": f"--log-level {log_level}", "conda": conda}


@cli.command
@_utils.opts.add_opts
@click.option(
    "--simulator",
    type=click.Choice(["fftvis", "matvis", "fftvis64", "fftvis32", "fftvis128", "matvis-cpu"]),
    default="matvis",
)
@click.option(
    "--n-unique-beams",
    default=1,
    help="Number of unique beams to use in the simulation (for testing performance of interpolation).",
)
@click.option(
    "--coupled/--isolated", default=False, help="Whether to use coupled beams or isolated beams"
)
@click.option(
    "--compress/--no-compress",
    default=True,
)
@click.pass_context
def runsim(ctx, channels, freq_range, n_unique_beams, compress, coupled, **kwargs):
    """Run HERA validation simulations.

    Use the default parameters, configuration files, and directories for HERA sims
    (see make_obsparams.py).
    """
    from ..run_sim import run_validation_sim

    channels = _utils.parse_channels(channels, freq_range)
    kwargs.pop("beam_interpolator", None)
    run_validation_sim(
        channels=channels,
        n_unique_beams=n_unique_beams,
        conda=ctx.obj["conda"],
        compress=compress,
        coupled=coupled,
        **kwargs,
    )


@cli.command("make-obsparams")
@_utils.opts.layout
@_utils.opts.ants
@_utils.opts.ideal_layout
@_utils.opts.channels
@_utils.opts.freq_range
@_utils.opts.sky_model
@_utils.opts.n_time_chunks
@_utils.opts.spline_interp_order
@_utils.opts.redundant
@_utils.opts.do_time_chunks
@click.option(
    "--n-unique-beams",
    default=1,
    help="Number of unique beams to use in the simulation (for testing performance of interpolation).",
)
@click.option(
    "--coupled/--isolated", default=False, help="Whether to use coupled beams or isolated beams"
)
@click.option("--beam-interpolator", default="az_za_map_coordinates")
def make_obsparams(
    layout,
    ideal_layout,
    freq_range,
    channels,
    sky_model,
    n_time_chunks,
    spline_interp_order,
    beam_interpolator,
    redundant,
    do_time_chunks,
    n_unique_beams,
    coupled,
):
    """Make obsparams for H4C simulations given a sky model and frequencies."""
    from ..obsparams import make_hera_obsparam

    channels = _utils.parse_channels(channels, freq_range)

    make_hera_obsparam(
        layout=layout,
        ideal_layout=ideal_layout,
        channels=channels,
        sky_model=sky_model,
        chunks=n_time_chunks,
        spline_interp_order=spline_interp_order,
        beam_interpolator=beam_interpolator,
        redundant=redundant,
        do_chunks=do_time_chunks,
        n_unique_beams=n_unique_beams,
        coupled=coupled,
    )


option_nside = click.option("--nside", default=256, show_default=True)


@cli.command("sky-model")
@click.argument("sky_model", type=click.Choice(["gsm", "diffuse", "ptsrc", "grf-eor"]))
@_utils.opts.channels
@_utils.opts.freq_range
@_utils.opts.slurm_override
@_utils.opts.skip_existing
@_utils.opts.dry_run
@option_nside
@click.option("--local/--slurm", default=False)
@click.option("--split-freqs/--no-split-freqs", default=False)
@click.option("--label", default="")
@click.option("--with-confusion/--no-confusion", default=True)
@click.option(
    "--per-channel-files/--single-file",
    default=False,
    help="Whether to output one skyh5 file per channel, or a single file with all channels (only for ptsrc model).",
)
@click.pass_context
def sky_model(
    ctx,
    sky_model,
    freq_range,
    channels,
    nside,
    local,
    slurm_override,
    split_freqs,
    skip_existing,
    dry_run,
    label,
    with_confusion,
    per_channel_files,
):
    """Make SkyModel at given frequencies.

    Frequencies are based on H4C data.
    Outputs are written to the default directories, i.e. "./sky_models/<type>".
    """
    if per_channel_files and sky_model != "ptsrc":
        raise ValueError("Per-channel files are only supported for the ptsrc sky model.")

    channels = _utils.parse_channels(channels, freq_range)
    if local:
        from .. import sky_model as sm

        if sky_model == "gsm":
            sm.make_gsm_model(channels, nside, label=label)
        elif sky_model == "diffuse":
            sm.make_diffuse_model(channels, nside, with_confusion=with_confusion, label=label)
        elif sky_model == "ptsrc":
            sm.make_ptsrc_model(channels, nside, label=label, per_channel_files=per_channel_files)
        elif sky_model == "grf-eor":
            sm.make_grf_eor_model(
                f"healpix-maps{nside}{label}.h5",
                channels=channels,
                label=label,
            )
        else:
            raise ValueError(f"Unknown sky model: {sky_model}")
    else:
        from ..run_sky_model import run_make_sky_model

        run_make_sky_model(
            sky_model,
            channels,
            nside,
            slurm_override=slurm_override,
            skip_existing=skip_existing,
            dry_run=dry_run,
            split_freqs=split_freqs,
            label=label,
            with_confusion=with_confusion,
            per_channel_files=per_channel_files,
            top_level_args=ctx.obj["top-level-args"],
            conda=ctx.obj["conda"],
        )


@cli.command
@click.option("--nside", type=int, required=True)
@click.option("--seed", type=int, default=2038)
@click.option("--low-memory/--fast-cpu", default=True)
@click.option("--local/--slurm", default=False)
@click.pass_context
def grf_realization(ctx, nside, seed, low_memory, local):
    from ..grf_realization import run_compute_grf_realization

    run_compute_grf_realization(
        nside=nside, seed=seed, low_memory=low_memory, conda=ctx.obj["conda"]
    )


@cli.command
@click.option("--test-mode/--production", default=False)
@click.option("--ell-max", default=1250)
@click.option("--local/--slurm", default=False)
@click.pass_context
def grf_covariance(ctx, test_mode, ell_max, local):
    from ..grf_covariance import compute_grf_covariance, run_compute_grf_covariance

    if local:
        compute_grf_covariance(test_mode, ell_max=ell_max)
    else:
        run_compute_grf_covariance(test_mode, ell_max=ell_max, conda=ctx.obj["conda"])


@cli.command("cornerturn")
@_utils.opts.sky_model
@click.option("-c", "--time-chunk", default=0)
@click.option("-n", "--new-chunk-size", default=2)
@click.option("--nchunks-sim", default=3, type=int)
@click.option("--conjugate/--no-conjugate", default=False)
@click.option("--remove-cross-pols/--keep-cross-pols", default=False)
@click.option(
    "--direc",
    default=None,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
)
@click.option(
    "--channels",
    default=None,
    type=str,
    help="Channels to use, e.g. '0~1536'. If not given, all channels are used.",
)
@_utils.opts.layout
@_utils.opts.log_level
@_utils.opts.dry_run
@_utils.opts.slurm_override
@_utils.opts.redundant
@_utils.opts.prefix
@click.pass_context
def cornerturn(
    ctx,
    sky_model,
    time_chunk,
    slurm_override,
    new_chunk_size,
    dry_run,
    nchunks_sim,
    conjugate: bool,
    remove_cross_pols: bool,
    direc: Path | None,
    channels: str | None,
    log_level: str,
    layout: str,
    redundant: bool,
    prefix: str,
):
    """Perform a cornerturn on simulation files.

    This takes multiple files, each with a single frequency and many times (snapshots),
    and reforms them into files with all frequencies and a set number of times (generally
    smaller). Note that the input files may be partial in frequency *and* time.

    Output files have the following prototype:
        zen.LST.{lst:.7f}[.{sky_cmp}].uvh5
    """
    logger.setLevel(log_level)

    # Make sure that the slurm log directory exists.
    # Otherwise, the job will terminate
    log_dir = Path(f"logs/chunk/{sky_model}")
    log_dir.mkdir(parents=True, exist_ok=True)

    if direc is None:
        simdir = paths.OUTDIR / paths.get_direc(
            sky_model=sky_model,
            chunks=nchunks_sim,
            layout=layout,
            redundant=redundant,
            prefix=prefix,
        )
    else:
        simdir = Path(direc)

    outdir = simdir / "rechunk"
    outdir.mkdir(parents=True, exist_ok=True)

    conjugate = "--conjugate" if conjugate else ""
    remove_cross_pols = "--remove-cross-pols" if remove_cross_pols else ""

    if channels is None:
        allfiles = sorted(simdir.glob(f"fch????_chunk{time_chunk:05d}.uvh5"))
        maxchan = int(allfiles[-1].name.split("fch")[1][:4])
        if len(allfiles) != maxchan + 1:
            raise ValueError(f"Missing files in {simdir}")
        channels = f"0~{maxchan + 1}"

    nchannels = int(channels.split("~")[1]) - int(channels.split("~")[0])
    estimated_time = 36 * nchannels / 1536  # hours

    estimated_minutes = max(int(estimated_time - int(estimated_time)) * 60, 10)

    if estimated_time > 24:
        estimated_time = f"1-{int(estimated_time) - 24:02d}:{estimated_minutes:02d}:00"
    else:
        estimated_time = f"{int(estimated_time):02d}:{estimated_minutes:02d}:00"

    slurm_override = (
        *slurm_override,
        ("job-name", f"{sky_model}-ct"),
        ("output", f"{log_dir}/%J.out"),
        ("nodes", "1"),
        ("ntasks", "1"),
        ("cpus-per-task", "16"),
        ("mem", "31GB"),
        ("time", estimated_time),
    )

    sbatch = _utils._get_sbatch_program(
        gpu=False, conda=ctx.obj["conda"], slurm_override=slurm_override
    )

    cmd = f"""
    time {ctx.obj["top-level-args"]} vsim rechunk-fast \
    --r-prototype "fch{{channel:04d}}_chunk{time_chunk:05d}.uvh5" \
    --chunk-size {new_chunk_size} \
    --channels {channels} \
    --sky-cmp {sky_model}\
    --assume-same-blt-layout \
    --is-rectangular \
    --nthreads 16 \
    {conjugate} \
    {remove_cross_pols} \
    {simdir} \
    {outdir} \
    """
    sbatch_dir = paths.REPODIR / "batch_scripts/rechunk"
    sbatch_dir.mkdir(parents=True, exist_ok=True)

    sbatch_file = sbatch_dir / f"{sky_model}_ch{time_chunk:03d}_{layout}.sbatch"

    sbatch = "\n".join([sbatch, "", cmd, ""])
    with open(sbatch_file, "w") as fl:
        fl.write(sbatch)

    if not dry_run:
        subprocess.call(f"sbatch {sbatch_file}".split())

    logger.debug(f"\n===Job Script===\n{sbatch}\n===END===\n")


cli.add_command(monitor_app, name="monitor")
cli.add_command(process_fftvis_profile_app, name="process-fftvis-profile")
cli.add_command(rechunk_fast_app, name="rechunk-fast")
