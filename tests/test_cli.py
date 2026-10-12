import numpy as np
import pytest

import lunar_ir_tools
from lunar_ir_tools import cli
from conftest import IIRS_WL, synthetic_solar_flux, write_iirs_product


def test_version_and_api():
    assert lunar_ir_tools.__version__ == "0.1.0"
    for name in lunar_ir_tools.__all__:
        assert hasattr(lunar_ir_tools, name), name
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0


def test_sp_read(sp_label, tmp_path, capsys):
    out = tmp_path / "spec.csv"
    assert cli.main(["sp-read", sp_label, "--csv", str(out)]) == 0
    text = capsys.readouterr().out
    assert "footprints (N) : 3" in text
    assert "CHOSEN footprint row 0" in text
    assert "850 nm band depth" in text
    assert out.exists() and open(out).readline().startswith("wavelength_nm,ref1,ref2,qa")
    assert cli.main(["sp-read", sp_label, "--near", "10.0,20.0", "--window", "500", "700"]) == 0


def test_iirs_read(iirs_l0_bundle, tmp_path, capsys):
    root, pid, cube = iirs_l0_bundle
    assert cli.main(["iirs-read", root]) == 0
    text = capsys.readouterr().out
    assert pid in text and "level          : Raw" in text
    out = tmp_path / "s.csv"
    assert cli.main(["iirs-read", root, "--spectrum", "1", "2", "--csv", str(out)]) == 0
    rows = np.loadtxt(out, delimiter=",", skiprows=1, usecols=(0, 1))
    assert rows.shape == (256, 2)
    np.testing.assert_allclose(rows[:, 1], cube[:, 1, 2])


def test_iirs_reflectance_and_thermal_commands(tmp_path, capsys):
    root = str(tmp_path / "l1")
    nl, ns = 10, 3
    write_iirs_product(root, np.full((256, nl, ns), 50.0, "<f4"), typ="nci")
    flux_path = tmp_path / "flux.txt"
    np.savetxt(flux_path, np.column_stack([IIRS_WL, synthetic_solar_flux(IIRS_WL)]))
    out = tmp_path / "refl.npz"
    assert cli.main(["iirs-reflectance", root, "--lines", "0", "4", "--solar-flux", str(flux_path),
                     "--out", str(out), "--no-thermal"]) == 0
    z = np.load(out)
    assert z["refl"].shape == (256, 4, ns) and z["line1"] == 4
    out2 = tmp_path / "aniso.npz"
    assert cli.main(["iirs-thermal", root, "--lines", "0", "4", "--solar-flux", str(flux_path),
                     "--out", str(out2), "--n-sample", "30"]) == 0
    z2 = np.load(out2)
    assert z2["refl"].shape == (256, 4, ns)
    assert "f=" in capsys.readouterr().out


def test_mi_sample_requires_input(capsys):
    assert cli.main(["mi-sample", "x.tif"]) == 2
