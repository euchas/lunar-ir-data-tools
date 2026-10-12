import os

import numpy as np
import pytest

from lunar_ir_tools import kaguya_sp as sp
from conftest import default_sp_arrays, write_sp_product, SP_WAV_NM


def test_parse_label_pointers_objects_columns(sp_label):
    pointers, objects, cols = sp.parse_label(sp_label)
    assert pointers["ANCILLARY_AND_SUPPLEMENT_DATA"] == 0          # 1-based -> 0-based
    assert pointers["SP_SPECTRUM_WAV"] == 3 * 64                    # right after the table
    assert set(sp.SPECTRUM_OBJECTS) <= set(objects)
    assert objects["SP_SPECTRUM_REF1"]["SCALING_FACTOR"] == "0.0001"
    assert objects["SP_SPECTRUM_RAD"]["UNIT"] == "W/m**2/sr/um"
    names = [c["name"] for c in cols]
    assert names[:2] == ["SP_INDEX", "INCIDENCE_ANGLE"]
    assert cols[1] == {"name": "INCIDENCE_ANGLE", "dtype": "IEEE_REAL", "start0": 4, "bytes": 8}


def test_load_sp_values_and_geometry(sp_label):
    a = default_sp_arrays()
    p = sp.load_sp(sp_label)
    assert p.nrows == 3 and p.nbands == SP_WAV_NM.size
    np.testing.assert_allclose(p.wav, SP_WAV_NM)
    np.testing.assert_allclose(p.ref1, a["ref1"], atol=1e-4 / 2)
    np.testing.assert_allclose(p.ref2, a["ref2"], atol=1e-4 / 2)
    np.testing.assert_allclose(p.rad, a["rad"], atol=1e-2 / 2)
    np.testing.assert_array_equal(p.raw, a["raw"])
    np.testing.assert_array_equal(p.qa, a["qa"])
    for k, v in a["geo"].items():
        np.testing.assert_allclose(p.geo[k], v, rtol=1e-6)
    assert p.spc_path.endswith(".spc") and os.path.exists(p.spc_path)


def test_optional_grids_and_missing_columns(tmp_path):
    lbl = write_sp_product(str(tmp_path), stem="SP_2C_02_00002_N001_E0010",
                           include_rad=False, include_raw=False)
    p = sp.load_sp(lbl, ancillary_columns=["INCIDENCE_ANGLE", "NOT_A_COLUMN", "SP_TIME"])
    assert p.rad is None and p.raw is None
    assert np.isnan(p.geo["NOT_A_COLUMN"]).all()
    assert np.isnan(p.geo["SP_TIME"]).all()            # CHARACTER column -> unsupported -> NaN
    np.testing.assert_allclose(p.geo["INCIDENCE_ANGLE"], [40.0, 20.0, 95.0])


def test_explicit_spc_path(sp_label, tmp_path):
    spc = os.path.splitext(sp_label)[0] + ".spc"
    moved = tmp_path / "elsewhere.bin"
    os.replace(spc, moved)
    with pytest.raises(FileNotFoundError):
        sp.load_sp(sp_label)
    p = sp.load_sp(sp_label, spc_path=str(moved))
    assert p.nrows == 3


def test_nearest_band_and_band_depth():
    wav = SP_WAV_NM
    assert sp.nearest_band(wav, 845.0) == (4, 850.0)
    spec = np.full(wav.size, 0.10)
    spec[4] = 0.08
    assert sp.simple_band_depth(wav, spec, 850, 750, 950) == pytest.approx(0.2)
    assert np.isnan(sp.simple_band_depth(wav, np.zeros(wav.size)))


def test_select_footprint(sp_label):
    p = sp.load_sp(sp_label)
    # row 1 is too dark, row 2 is night-side -> only row 0 is valid
    assert sp.select_footprint(p) == 0
    assert sp.select_footprint(p, near=(10.2, 20.0)) == 0
    # with no valid footprint the smallest incidence wins
    p.ref1[:] = 0.0
    assert sp.select_footprint(p) == 1


def test_read_spectrum_grid_short_buffer():
    with pytest.raises(ValueError):
        sp.read_spectrum_grid(b"\x00" * 10, 0, 2, 4)
