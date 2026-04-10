"""Specification of project paths."""

import logging
from os import environ
from pathlib import Path

import numpy as np
import yaml
from parse import parse

logger = logging.getLogger(__name__)


class Paths:
    """Class to hold all paths for the validation sims."""

    def __init__(self, repodir: Path | None = None):
        # These paths and variables define some of the default directories and shared
        # config files for H4C validation simulations
        self.REPODIR = repodir if repodir is not None else Path()
        self.DIRFMT = "{sky_model}/{prefix}/nt17280-{chunks:05d}chunks-{layout}-{redundant}"
        self.FLFMT = "fch{fch:04d}_chunk{ch:05d}"
        self.COMPRESS_FMT = "ch{chunks}_{layout_file}.npy"

    @property
    def SKYDIR(self):
        """Path to sky model files."""
        return self.REPODIR / "sky_models"

    @property
    def RAWSKYDIR(self):
        """Path to raw sky model files."""
        return self.SKYDIR / "raw"

    @property
    def CFGDIR(self):
        """Path to config files."""
        return self.REPODIR / "config_files"

    @property
    def OUTDIR(self):
        """Path to output files."""
        return self.REPODIR / "outputs"

    @property
    def HPCDIR(self):
        """Path to HPC configuration files."""
        return self.REPODIR / "hpc-configs"

    @property
    def OBSPDIR(self):
        """Path to obsparam files."""
        return self.CFGDIR / "obsparams"

    @property
    def COMPRESSDIR(self):
        """Path to compression cache files."""
        return self.REPODIR / "compression-cache"

    @property
    def LOGDIR(self):
        """Path to log files."""
        return self.REPODIR / "logs"

    @property
    def BEAMDIR(self):
        """Path to beam files."""
        return self.REPODIR / "beams"

    @property
    def LAYOUTDIR(self):
        """Path to array layout files."""
        return self.CFGDIR / "array_layouts"

    @property
    def FULL_HERA_LAYOUT(self):
        """Path to the full HERA array layout file."""
        return self.LAYOUTDIR / "array_layout_hera_350.txt"

    @property
    def IDEAL_HERA_LAYOUT(self):
        """Path to the ideal HERA array layout file."""
        return self.LAYOUTDIR / "array_layout_hera_350_ideal.txt"

    @property
    def SIMULATOR_SPECS(self):
        """Path to the simulator specifications folder."""
        return self.REPODIR / "simulator-specs"

    def get_direc(
        self,
        sky_model: str,
        chunks: int,
        layout: str,
        redundant: bool,
        prefix: str = "default",
    ) -> Path:
        """Get a directory path for a given set of parameters.

        The directory structure is defined by the DIRFMT variable, and is of the form
        {sky_model}/{prefix}/nt17280-{chunks:05d}chunks-{layout}-{redundant}.
        The prefix is optional and may be omitted.
        """
        return Path(
            self.DIRFMT.format(
                sky_model=sky_model,
                prefix=prefix,
                chunks=chunks,
                layout=layout,
                redundant="red" if redundant else "nonred",
            )
        )

    def get_file(
        self, chunk: int, channel: int, with_dir: bool = True, ext: str | None = None, **kw
    ):
        """Get a file path for a given chunk and channel."""
        stem = self.FLFMT.format(fch=channel, ch=chunk)

        fl = self.get_direc(**kw) / stem if with_dir else Path(stem)
        if ext:
            fl = fl.with_suffix(ext)
        return fl

    def parse_fname(self, fname: str):
        """Parse a filename to extract the parameters in the filename structure."""
        return parse(self.FLFMT, fname).named

    def parse_direc(self, direc: Path):
        """Parse a directory name to extract the parameters in the directory structure.

        The expected format is
        {sky_model}/{prefix}/nt17280-{chunks:05d}chunks-{layout}-{redundant}, but the
        prefix is optional and may be omitted. If the directory name does not match this
        format, an error is raised.
        """
        parents = direc.parents
        if len(parents) > 2:
            name = str(direc.relative_to(parents[2]))
        else:
            name = str(direc.relative_to(parents[1]))
        return parse(self.DIRFMT, name).named

    def parse_path(self, path: Path, only_model: bool = False):
        """Parse a path to extract the parameters encoded in the path structure.

        If only_model is True, only parse the sky_model from the directory name and
        ignore the rest of the parameters.
        """
        path = Path(path)

        if not path.exists():
            raise ValueError(f"Path {path} does not exist.")

        if path.is_file():
            if only_model:
                out = self.parse_direc(path.parent)
            else:
                out = self.parse_fname(path.stem) | self.parse_direc(path.parent)
        else:
            out = self.parse_direc(path)

        if not out:
            raise ValueError(f"path {path} did not adhere to any specifications")

    @property
    def HPC_CONFIG(self) -> None | dict:
        """Load the HPC config file specified by the VALIDATION_SYSTEM_NAME environment variable."""
        HPC = environ.get("VALIDATION_SYSTEM_NAME")

        if HPC is None:
            logger.warning(
                "You must set the VALIDATION_SYSTEM_NAME environment variable to your system "
                "name, corresponding to a file in hpc-configs. Assuming local system.",
            )
            return None
        else:
            with open(self.HPCDIR / f"{HPC}.yaml") as fl:
                return yaml.load(fl, Loader=yaml.FullLoader)

    def possible_layouts(self):
        """Return a list of possible layout names."""
        return ["H4C", "HEX", "FULL", "HERA19", "MINIMAL"]

    def get_layout_indices(self, name: str) -> np.ndarray:
        """Get the antenna indices for a given layout name. If the name is not recognized, raise an error."""
        if name.upper() not in self.possible_layouts():
            raise ValueError(
                f"Layout name {name} not recognized. Must be one of {self.possible_layouts()}"
            )

        ANTS_DICT = {
            "H4C": np.genfromtxt(self.LAYOUTDIR / "h4c_ants.txt").astype(int),
            "HEX": np.arange(320),
            "FULL": np.arange(350),
            "HERA19": [0, 1, 2, 11, 12, 13, 14, 23, 24, 25, 26, 27, 37, 38, 39, 40, 52, 53, 54],
            "MINIMAL": [0, 1, 2, 4, 11, 23, 52],  # 3 N-S bls, 3 E-W bls (1,2,4 units)
        }
        return ANTS_DICT[name.upper()]

    def phase_two_freqs(self):
        """Get the HERA Phase II frequencies in Hz."""
        with open(self.REPODIR / "h4c_freqs.yaml") as fl:
            freq_info = yaml.load(fl, Loader=yaml.FullLoader)

        return np.arange(
            freq_info["start"], freq_info["end"] + freq_info["delta"] / 2, freq_info["delta"]
        )

    def make_hera_layout(
        self, name: str, ants: np.ndarray | None = None, ideal: bool = True, n_unique_beams: int = 1
    ) -> Path:
        """Create a HERA layout."""
        if ants is None:
            ants = self.get_layout_indices(name)

        direc = self.LAYOUTDIR / "tmp"
        if not direc.exists():
            direc.mkdir()

        full_layout = np.genfromtxt(
            self.IDEAL_HERA_LAYOUT if ideal else self.FULL_HERA_LAYOUT, skip_header=1
        )

        # Get list of beam indices.
        beam_idx = (
            np.arange(len(ants)) % n_unique_beams if n_unique_beams > 0 else np.arange(len(ants))
        )

        pth = direc / f"{name}_nbeams{n_unique_beams}.txt"
        with pth.open("w") as fl:
            fl.write("Name    Number  BeamID  E       N       U\n")
            for i, ant in enumerate(ants):
                pos = full_layout[ant][1:]
                fl.write(f"HH{ant}\t{ant}\t{beam_idx[i]}\t{pos[0]}\t{pos[1]}\t{pos[2]}\n")

        return pth


paths = Paths()


def set_project_path(repodir: Path):
    """Set the project paths to a new repository directory."""
    paths.REPODIR = repodir
