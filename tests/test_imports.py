"""Very simple smoke test to check that imports work."""

from click.testing import CliRunner

from validation_sim import cli

runner = CliRunner()


def test_imports():
    pass


def test_cli_sky_model_obsparams(tmp_path):
    result = runner.invoke(cli.cli, ["sky-model", "ptsrc", "--nside", "32", "--local"])
    assert result.exit_code == 0

    result = runner.invoke(
        cli.cli, ["make-obsparams", "--layout", "FULL", "--sky-model", "ptsrc32"]
    )
    assert result.exit_code == 0

    result = runner.invoke(
        cli.cli,
        ["make-obsparams", "--layout", "FULL", "--sky-model", "ptsrc32", "--n-unique-beams", "2"],
    )
    assert result.exit_code == 0
