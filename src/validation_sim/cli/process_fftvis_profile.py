from pathlib import Path

import typer

app = typer.Typer()


def print_important_lines(fname):
    with open(fname) as fl:
        important = [
            "Total time:",
            "beams._evaluate_beam(",
            "nufft3d3(",
            "nufft2d3(",
            "ModelData.from_config(",
            "write_uvh5(",
        ]

        for line in fl:
            if any(imp in line for imp in important):
                pass


def get_peak_mem(logfile):
    mem = 0
    with open(logfile) as fl:
        for line in fl:
            if "Memory usage" in line:
                _m = float(line.split(": ")[1].split(" G")[0].strip())
                if _m > mem:
                    mem = _m
    return mem


@app.command()
def main():
    d = Path("profiling")
    allfiles = sorted(d.glob("*-fftvis-*"))

    for fl in allfiles:
        skymod, _gpu, nt, layout, _code, _version, _hsim = fl.name.split("-")

        logdir = Path(f"logs/vis/{skymod}/nt17280-{nt[2:]}chunks-{layout}")
        if not logdir.exists():
            pass
        else:
            logfile = sorted(logdir.glob("fch0001-ch000_*.out"))[-1]
            get_peak_mem(logfile)

        f"SKY={skymod}, NTIMES={nt[2:]}, LAYOUT={layout}"

        print_important_lines(fl)


typer_click_app = typer.main.get_command(app)
