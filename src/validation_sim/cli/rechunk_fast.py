"""Perform a cornerturn on simulation files.

This takes multiple files, each with a single frequency and many times (snapshots),
and reforms them into files with all frequencies and a set number of times (generally
smaller). Note that the input files may be partial in frequency *and* time.

Output files have the following prototype:
    zen.LST.{lst:.7f}[.{sky_cmp}].uvh5
"""

import logging
import operator
import os
from collections import Counter
from functools import partial, reduce
from multiprocessing import Pool, cpu_count, shared_memory
from pathlib import Path

import h5py
import numpy as np
import psutil
import typer
from pyuvdata.uvdata import FastUVH5Meta

logger = logging.getLogger("rechunk")
ps = psutil.Process()


def find_all_files(
    base_dir: Path,
    channels: list[int],
    prototype: str,
    ignore_missing_channels: bool = False,
    assume_blt_layout: bool = False,
    is_rectangular: bool | None = None,
):
    """Find all the files that need to be read to be chunked."""
    all_files = {}
    for ch in channels:
        all_files[ch] = {}
        p = prototype.format(channel=ch)
        if files := sorted(base_dir.glob(p)):
            all_files[ch] = files

        else:
            msg = f"No files with prototype {p} for channel {ch}"
            if not ignore_missing_channels:
                raise FileNotFoundError(msg)
            else:
                logger.warning(msg)
    if not all_files:
        raise FileNotFoundError(f"No files found with prototype {prototype} in {base_dir}")

    nchunks_counter = Counter([len(flist) for flist in all_files.values()])
    if len(nchunks_counter) > 1:
        most_common_nchunks = nchunks_counter.most_common(1)[0][0]
        for fllist in all_files.values():
            if len(fllist) != most_common_nchunks:
                logger.warning(
                    "Number of files for different channels is not the same. "
                    f"Got {len(fllist)} for channel {ch} and "
                    f"{len(all_files[channels[0]])} for channel {channels[0]}"
                )

    for nc, count in nchunks_counter.items():
        logger.info(f"Found {count} channels with {nc} chunks")

    # Turn them all into metadata objects.
    fl0 = FastUVH5Meta(all_files[channels[0]][0], blts_are_rectangular=is_rectangular)

    if not fl0.blts_are_rectangular:
        raise ValueError(
            "Your first file is not rectangular. This script only works for rectangular files..."
        )

    for ch in channels:
        all_files[ch] = [
            FastUVH5Meta(
                fl,
                blts_are_rectangular=fl0.blts_are_rectangular if assume_blt_layout else None,
                time_axis_faster_than_bls=fl0.time_axis_faster_than_bls
                if assume_blt_layout
                else None,
            )
            for fl in all_files[ch]
        ]

    return all_files


def get_file_time_slices(
    meta_list: list[FastUVH5Meta],
    lsts_per_chunk: int,
    lst_wrap: float,
):
    """Get the time slices in each file that need to be taken."""
    dlst = -1
    i = 0
    while dlst < 0:
        dlst = meta_list[0].lsts[i + 1] - meta_list[0].lsts[i]
        i += 1

    for i, meta in enumerate(meta_list):
        if meta.lsts[0] > (lst_wrap + dlst) or meta.lsts[-1] < lst_wrap:
            continue
        else:
            time_index = np.argwhere(meta.lsts >= lst_wrap).flatten()[0]
            fl_index = i
            break

    Ntimes = meta_list[0].Ntimes

    total_times = Ntimes * len(meta_list)
    logger.info(f"Total times: {total_times}")
    nchunks = int(np.ceil(total_times / lsts_per_chunk))
    logger.info(f"Total chunks: {nchunks}")
    chunks = [[] for _ in range(nchunks)]

    for chunk in chunks:
        times_remaining = lsts_per_chunk

        while True:
            chunk.append((fl_index, slice(time_index, min(Ntimes, time_index + times_remaining))))

            if time_index + lsts_per_chunk >= Ntimes:
                # We have to use the start of the next file
                times_remaining -= Ntimes - time_index
                time_index = 0
                fl_index += 1
                if fl_index >= len(meta_list):
                    fl_index = 0
            else:
                times_remaining = 0
                time_index += lsts_per_chunk

            if times_remaining == 0:
                break

    logger.debug(f"CHUNKS: {chunks}")
    return chunks


def reset_time_arrays(uvd, meta, times, lsts, ras, pas, slc, time_first):
    """Update UVData metadata to new times."""
    nt = slc.stop - (slc.start or 0)

    if nt != uvd.Ntimes:
        uvd.Nblts = uvd.Nbls * nt
        uvd.integration_time = uvd.integration_time[0] * np.ones(uvd.Nblts)
        uvd.phase_center_app_dec = uvd.phase_center_app_dec[0] * np.ones(uvd.Nblts)
        uvd.phase_center_id_array = np.zeros(uvd.Nblts, dtype=int)

    if time_first:
        uvd.time_array = np.tile(times[slc], uvd.Nbls)
        uvd.phase_center_app_ra = np.tile(ras[slc], uvd.Nbls)
        uvd.phase_center_frame_pa = np.tile(pas[slc], uvd.Nbls)
        uvd.lst_array = np.tile(lsts[slc], uvd.Nbls)
        if nt != uvd.Ntimes:
            uvd.uvw_array = np.repeat(uvd.uvw_array[:: meta.Ntimes], nt, axis=0)
            uvd.ant_1_array = np.repeat(meta.unique_antpair_1_array, nt)
            uvd.ant_2_array = np.repeat(meta.unique_antpair_2_array, nt)
            uvd.baseline_array = np.repeat(meta.unique_baseline_array, nt)
    else:
        uvd.time_array = np.repeat(times[slc], uvd.Nbls)
        uvd.phase_center_app_ra = np.repeat(ras[slc], uvd.Nbls)
        uvd.phase_center_frame_pa = np.repeat(pas[slc], uvd.Nbls)
        uvd.lst_array = np.repeat(lsts[slc], uvd.Nbls)
        if nt != uvd.Ntimes:
            uvd.uvw_array = np.tile(uvd.uvw_array[: meta.Nbls], (nt, 1))
            uvd.ant_1_array = np.tile(meta.unique_antpair_1_array, nt)
            uvd.ant_2_array = np.tile(meta.unique_antpair_2_array, nt)
            uvd.baseline_array = np.tile(meta.unique_baseline_array, nt)

    uvd.Ntimes = nt


def _check_rectangularity_consistency(
    raw_files: dict[int, list[Path]], time_first: bool, channels: list[int]
):
    # Ensure all the files have rectangular blts with the same ordering.
    # This is a requirement for the chunking to work.
    for ch in channels:
        for meta in raw_files[ch]:
            if not meta.blts_are_rectangular:
                raise ValueError("Not all files have rectangular blts.")
            if meta.time_axis_faster_than_bls != time_first:
                raise ValueError("Not all files have the same blt ordering.")


def chunk_files(
    channels,
    r_prototype: str,
    save_dir: Path,
    base_dir: Path,
    prototype: str,
    lst_wrap: float = np.pi,
    n_times_per_file: int = 180,
    ignore_missing_channels: bool = False,
    assume_blt_layout: bool = False,
    blt_order="determine",
    max_mem_mb: int = 100000,
    nthreads: int | None = None,
    is_rectangular: bool | None = None,
    max_freq_chunk_size: int = 100000000,
    remove_cross_pols: bool = False,
    conjugate: bool = False,
    clobber: bool = False,
):
    """Chunk given files."""
    # Load the read files, and check that the read prototype is valid if provided.
    raw_files = find_all_files(
        base_dir,
        channels,
        r_prototype,
        ignore_missing_channels,
        assume_blt_layout,
        is_rectangular=is_rectangular,
    )

    logger.info(f"Number of raw data files: {len(raw_files)}")

    time_first = raw_files[channels[0]][0].time_axis_faster_than_bls

    if not assume_blt_layout:
        _check_rectangularity_consistency(raw_files, time_first)

    # Read the metadata from a file to get the times.
    logging.info("Reading reference metadata...")

    # Get all the frequencies we're gonna use.
    freqs = []
    for ch in channels:
        meta = raw_files[ch][0]
        try:
            freqs.append(meta.freq_array)
        except OSError as e:
            raise OSError(f"{e!s} in file {meta.path}") from e

    freqs = np.concatenate(freqs)

    # Get the times we're gonna use.
    times = []
    lsts = []
    for meta in raw_files[channels[0]]:
        times.append(meta.times)
        lsts.append(meta.lsts)
    times = np.concatenate(times)
    lsts = np.concatenate(lsts)

    if lst_wrap is None:
        lst_wrap = np.min(lsts)

    # Roll the times and lsts around so it's easier to chunk them.
    lsts[lsts < lst_wrap] += 2 * np.pi
    time_index = np.argwhere(lsts >= lst_wrap)[0][0]
    times = np.roll(times, -time_index)
    lsts = np.roll(lsts, -time_index) % (2 * np.pi)

    # Make a prototype UVData object for the chunked data.
    # this has too many times, and only one frequency. We update that manually.
    uvd = meta.to_uvdata()
    uvd.use_future_array_shapes()
    uvd.freq_array = freqs
    uvd.Nfreqs = len(freqs)
    uvd.channel_width = np.ones_like(freqs) * (np.diff(freqs)[0])
    # Enforce fixed spectral window (required to pass check)
    # TODO: Support flexible spectral window in the future
    uvd.flex_spw = False
    uvd.flex_spw_id_array = np.zeros_like(freqs, dtype=np.integer)

    if remove_cross_pols:
        pol_indices = [i for i, pol in enumerate(meta.pols) if pol[0] == pol[1]]
    else:
        pol_indices = [i for i, pol in enumerate(meta.pols)]
    uvd.Npols = len(pol_indices)
    uvd.polarization_array = meta.polarization_array[pol_indices]

    DTYPE = np.dtype(complex) if uvd.data_array is None else uvd.data_array.dtype

    if time_first:
        phase_center_ra = uvd.phase_center_app_ra[: meta.Ntimes]
        phase_center_pa = uvd.phase_center_frame_pa[: meta.Ntimes]
    else:
        phase_center_ra = uvd.phase_center_app_ra[:: meta.Nbls]
        phase_center_pa = uvd.phase_center_frame_pa[:: meta.Nbls]

    phase_center_ra = np.roll(phase_center_ra, -time_index)
    phase_center_pa = np.roll(phase_center_pa, -time_index)

    logger.info(f"Data has {uvd.Nbls} baselines and {uvd.Ntimes} times per file")

    # Get the slices we'll need for each chunk.
    logger.info("Getting time slices for each output file...")
    chunk_slices = get_file_time_slices(raw_files[channels[0]], n_times_per_file, lst_wrap)
    logger.info("Got all time slices")

    # Figure out how many frequencies we can fit in the data at once.
    current_mem = ps.memory_info().rss
    mem_left = max_mem_mb * (1024**2) - current_mem - 100 * (1024**2)  # leave 100MB for overhead
    mem_per_freq = n_times_per_file * uvd.Nbls * uvd.Npols * 8
    nfreqs = min(mem_left // mem_per_freq, uvd.Nfreqs, max_freq_chunk_size)

    nfreq_chunks = int(np.ceil(uvd.Nfreqs / nfreqs))
    logger.info(f"Going to use {nfreq_chunks} frequency chunks of {nfreqs} frequencies each.")
    logger.info(
        f"This is estimated to use {mem_per_freq * nfreqs / 1024**2:.2f} MB "
        f"of memory (of the {mem_left / 1024**2} MB left)."
    )
    logger.info("")

    MAXSHAPE = (uvd.Nblts, nfreqs, uvd.Npols)

    try:
        shm = shared_memory.SharedMemory(
            create=True,
            size=DTYPE.itemsize * np.prod(MAXSHAPE),
            name="FULLDSET",
        )
    except FileExistsError:
        # Shared memory "FULLDSET" was not unlink properly. We must close it.
        shm = shared_memory.SharedMemory(name="FULLDSET", create=False)
        shm.unlink()
        # Now re-initialize
        shm = shared_memory.SharedMemory(
            create=True,
            size=DTYPE.itemsize * np.prod(MAXSHAPE),
            name="FULLDSET",
        )

    for outfile_index, this_chunk_slices in enumerate(chunk_slices):
        logger.info(f"Creating data for file {outfile_index + 1}")
        # Update the metadata for this chunk.
        time_slice = slice(
            outfile_index * n_times_per_file,
            min(len(times), (outfile_index + 1) * n_times_per_file),
        )
        reset_time_arrays(
            uvd,
            meta,
            times=times,
            lsts=lsts,
            ras=phase_center_ra,
            pas=phase_center_pa,
            time_first=time_first,
            slc=time_slice,
        )

        # Check on first, second and last.
        if outfile_index in [0, 1, len(chunk_slices) - 1]:
            uvd.check()  # quick check to make sure we set everything up correctly.

        fname = prototype.format(lst=uvd.lst_array[0])
        pth = save_dir / fname

        # Check if file already exists and only proceed if clobber is True
        if not pth.exists() or clobber:
            # This just writes the header.
            logger.info("Initializing UVH5 file...")
            uvd.initialize_uvh5_file(pth, clobber=True)

            pool = Pool(nthreads or cpu_count())
            raw_file_paths = {ch: [f.path for f in raw_files[ch]] for ch in channels}

            for freq_chunk in range(nfreq_chunks):
                logger.info(f"Obtaining frequency chunk {freq_chunk + 1}/{nfreq_chunks}...")
                freq_slice = slice(freq_chunk * nfreqs, min((freq_chunk + 1) * nfreqs, uvd.Nfreqs))
                this_nfreq = freq_slice.stop - freq_slice.start
                SHAPE = (uvd.Nblts, this_nfreq, uvd.Npols)
                full_dset = np.ndarray(SHAPE, dtype=DTYPE, buffer=shm.buf)

                # Now we need to actually write the data.
                pool.map(
                    partial(
                        write_freq_chunk,
                        ntimes=uvd.Ntimes,
                        chunk_slices=this_chunk_slices,
                        nbls=uvd.Nbls,
                        raw_files=raw_file_paths,
                        time_first=time_first,
                        shape=SHAPE,
                        channels=channels,
                        start_index=freq_slice.start,
                        pol_indices=pol_indices,
                        DTYPE=DTYPE,
                    ),
                    range(freq_slice.start, freq_slice.stop),
                )

                # Now write the data.
                with h5py.File(pth, "a") as fl:
                    if conjugate:
                        full_dset = np.conjugate(full_dset)

                    fl["/Data/visdata"][:, freq_slice] = full_dset
                del full_dset
        else:
            logger.info(
                f"File {outfile_index + 1} already exists, skipping write out. "
                "Set --clobber to overwrite."
            )

    shm.close()
    shm.unlink()


def write_freq_chunk(
    ich: int,
    ntimes,
    chunk_slices,
    nbls,
    raw_files,
    time_first,
    shape,
    channels,
    start_index,
    DTYPE,
    pol_indices: list[int],
):
    """Write out a particular frequency chunk to file."""
    ntimes_left = ntimes
    nblts_so_far = 0
    ch = channels[ich]

    shm = shared_memory.SharedMemory(name="FULLDSET")
    full_dset = np.ndarray(shape, dtype=DTYPE, buffer=shm.buf)

    for flidx, slc in chunk_slices:
        if ntimes_left == 0:
            break

        this_ntimes = min(slc.stop - slc.start, ntimes_left)
        this_nblts = this_ntimes * nbls

        # Get the data for this chunk.
        pth = raw_files[ch][flidx]
        with h5py.File(pth, "r") as fl:
            if time_first:
                slices = [slice(slc.start + n, slc.start + this_ntimes + n) for n in range(nbls)]
            else:
                slices = slice(nbls * slc.start, nbls * (slc.start + this_ntimes))

            data = fl["/Data/visdata"]

            logger.debug(
                f"Reading file {pth} with slices {slices} for channel {ch}."
                f"Data has shape {data.shape}. This chunk has {this_ntimes} times and "
                f"{this_nblts} blts."
            )

            data = data[slices]

            logger.debug(f"After slicing out times, data has shape {data.shape}.")

            if data.ndim > 3:
                raise ValueError("Data has old array shapes. Please make it future array shapes.")

            if data.shape[0] < this_nblts:
                raise ValueError(
                    f"Data in {pth} has fewer blts than expected ({data.shape[0]} < {this_nblts})."
                )

        full_dset[nblts_so_far : nblts_so_far + this_nblts, ich - start_index] = data[
            :this_nblts, 0, pol_indices
        ]

        ntimes_left -= this_ntimes
        nblts_so_far += this_nblts


app = typer.Typer()


@app.command()
def main(
    base_dir: Path,
    save_dir: Path,
    r_prototype: str = "",
    channels: list[str] = (),
    sky_cmp: str | None = None,
    chunk_size: int = 180,
    lst_wrap: float | None = None,
    clobber: bool = False,
    ignore_missing_channels: bool = False,
    assume_same_blt_layout: bool = False,
    is_rectangular: bool | None = None,
    max_mem: int = 1e9,
    nthreads: int | None = None,
    max_freq_chunk_size: int = 1e6,
    remove_cross_pols: bool = False,
):
    """
    Re-chunk simulation files in time.

    Parameters
    ----------
    base_dir
        The path to direcotry containing simulation data
    save_dir
        Path to write new files to. Note that this directory must already exist and be
        writable. The chunked files will be written to the same subdirectories as the
        input files, but with "rechunk" appended to the name.
    r_prototype
        glob-parsable prototype of files to read. Can include a {channel} format string
        which will be replaced by the channel internally. For example, if your files are
        named like "zen.LST.12345.678.fch0000.uvh5", "zen.LST.12345.678.fch0001.uvh5",
        etc, then you could use "zen.LST.*.fch{channel:04d}.uvh5" as the prototype, and
        the script will replace {channel:04d} with the channel number (e.g. 0, 1, etc)
        to find the files to read in.
    channels
        The channels to read, in the form "low~high", eg "0~1536". If not given, all
        channels matching the prototype will be used.
    sky_cmp
        Optional sky component to include in the output file name, e.g. "diffuse".
    chunk_size
        The number of integrations per chunk.
    lst_wrap
        Where to perform the wrap in LST. Default is lowest LST in data.
    clobber
        If set, rechunk and overwrite existing files. Otherwise, skip the chunk if it
        already exists.
    ignore_missing_channels
        If set, merely warn if there are no files for a particular channel. Otherwise, raise
        an error.
    assume_same_blt_layout
        Whether to assume each file has the same layout of baselines/times. If False,
        the script will check that all files have the same blt layout and ordering. If True,
        the script will skip this check and just assume they are the same.
    is_rectangular
        Whether the blts in the files are rectangular. If None, will try to infer from
        the first file. If True, will assume they are rectangular. If False, will assume
        they are not rectangular. Note that this script only works for rectangular files, so
        if this is False, the script will raise an error.
    max_mem
        Maximum memory to use in MB. The script will try to chunk the frequencies such that
        it does not exceed this memory usage. Note that this is just an estimate, and actual
        memory usage may vary. Setting this too low may result in very long run times, while
        setting it too high may result in out of memory errors. Default is 1e9 MB (1 TB).
    nthreads
        Number of threads to use for reading and writing data. If None, will use all available
        threads. Note that using more threads may speed up the process, but also may increase
        memory usage and contention, so use with caution.
    max_freq_chunk_size
        Maximum number of frequencies to read at once. Setting --max-mem will try autodetect
        optimal setting.
    remove_cross_pols
        Whether to remove cross-pols from the data. If True, only XX and YY will be kept. If False,
        all polarizations will be kept.
    """
    # Check that the read/write directories actually exist with proper permissions.
    if not base_dir.exists():
        raise FileNotFoundError(f"The provided base directory '{base_dir}' does not exist.")

    if not save_dir.exists():
        raise FileNotFoundError("The provided save directory does not exist.")

    if not os.access(save_dir, os.W_OK):
        # Test that we have write privileges here to save time.
        raise PermissionError(
            "You do not have permission to write to the provided save "
            "directory. Please update the directory choice or permissions."
        )

    # Build the save file prototype.
    prototype = "zen.LST.{lst:.7f}." + (f"{sky_cmp}.uvh5" if sky_cmp else "uvh5")

    channels = reduce(
        operator.iadd, (list(range(*tuple(map(int, ch.split("~"))))) for ch in channels), []
    )

    chunk_files(
        prototype=prototype,
        channels=channels,
        lst_wrap=lst_wrap,
        n_times_per_file=chunk_size,
        save_dir=save_dir,
        base_dir=base_dir,
        r_prototype=r_prototype,
        ignore_missing_channels=ignore_missing_channels,
        assume_blt_layout=assume_same_blt_layout,
        is_rectangular=is_rectangular,
        max_mem_mb=max_mem,
        nthreads=nthreads,
        max_freq_chunk_size=max_freq_chunk_size,
        remove_cross_pols=remove_cross_pols,
        clobber=clobber,
    )


click_app = typer.main.get_command(app)
