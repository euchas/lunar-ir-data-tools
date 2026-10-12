import os

import numpy as np
import pytest

from lunar_ir_tools import iirs
from conftest import (IIRS_WL, IIRS_T0, write_iirs_product, default_geometry_grid)


def test_read_label_and_product_basics(iirs_l0_bundle):
    root, pid, cube = iirs_l0_bundle
    p = iirs.IirsProduct(root)
    assert p.product_id == pid
    assert p.label["level"] == "Raw" and p.label["dtype"] == "<u2"
    assert p.label["axes"] == ["Band", "Line", "Sample"]
    assert (p.nbands, p.nlines, p.nsamples) == (256, 5, 3)
    np.testing.assert_allclose(p.wl, IIRS_WL)
    assert p.label["exposure"] == "e1" and p.label["gain"] == "g2"
    assert not p.is_descending()
    assert p.band(1555.0) == int(np.argmin(np.abs(IIRS_WL - 1555.0)))
    np.testing.assert_array_equal(p.cube[:, 2, 1], cube[:, 2, 1])
    np.testing.assert_array_equal(p.spectrum(4, 2), cube[:, 4, 2].astype(float))


def test_product_from_qub_and_xml_paths(iirs_l0_bundle):
    root, pid, cube = iirs_l0_bundle
    qub = os.path.join(root, "data", "raw", "20210123", pid + ".qub")
    assert iirs.IirsProduct(qub).product_id == pid
    assert iirs.IirsProduct(qub[:-4] + ".xml").product_id == pid
    assert iirs.IirsProduct(qub).geometry_grid() is None


def test_cube_size_mismatch_raises(iirs_l0_bundle):
    root, pid, cube = iirs_l0_bundle
    qub = os.path.join(root, "data", "raw", "20210123", pid + ".qub")
    with open(qub, "ab") as f:
        f.write(b"\x00\x00")
    with pytest.raises(ValueError):
        iirs.IirsProduct(root).cube


def test_good_bands_masks(iirs_l0_bundle):
    root, pid, _ = iirs_l0_bundle
    p = iirs.IirsProduct(root)
    g = p.good_bands()
    assert not g[:6].any() and not g[251:].any()           # detector ends
    assert not g[28:34].any() and not g[161:171].any()     # OSF junctions
    assert g[100]
    assert p.good_bands(include_osf=True)[165]
    p.label["exposure"] = "e2"
    assert not p.good_bands()[161:].any() and p.good_bands()[150]


def test_line_times_are_linear_between_label_start_and_stop(iirs_l0_bundle):
    root, _, _ = iirs_l0_bundle
    p = iirs.IirsProduct(root)
    t = p.line_times()
    assert t[0] == IIRS_T0
    steps = np.diff(t) / np.timedelta64(1, "ms")
    assert steps.min() >= 52 and steps.max() <= 54      # 53.06 ms nominal, ms resolution


def test_spm_and_line_incidence(iirs_l0_bundle):
    root, _, _ = iirs_l0_bundle
    p = iirs.IirsProduct(root)
    sp = p.spm()
    assert sp["t"].dtype == np.dtype("datetime64[ms]")
    np.testing.assert_allclose(sp["sun_el"], 50.0)
    np.testing.assert_allclose(sp["phase"], 40.0)
    np.testing.assert_allclose(sp["sun_az"], 90.0)
    np.testing.assert_allclose(p.line_incidence(), 40.0)


def test_oat_and_line_latlon_unwrap(iirs_l0_bundle):
    root, _, _ = iirs_l0_bundle
    p = iirs.IirsProduct(root)
    o = p.oat()
    assert o["orbit"][0] == 1234 and o["raw"].shape[1] == 49
    np.testing.assert_allclose(o["alt_km"], 100.0)
    ll = p.line_latlon()
    assert ll.shape == (p.nlines, 2)
    assert np.all(np.diff(ll[:, 0]) > 0)                     # latitude increasing with time
    assert np.all(np.abs(ll[:, 1]) <= 180)
    # longitudes were written increasing across 180 deg: no 360 deg jumps after unwrap
    lon_unwrapped = np.unwrap(np.deg2rad(ll[:, 1]))
    assert np.all(np.diff(lon_unwrapped) > 0)


def test_oat_must_cover_line_times(tmp_path):
    cube = np.zeros((256, 4, 2), "<u2")
    root = str(tmp_path / "b")
    write_iirs_product(root, cube, typ="nri", level="Raw", dtype_name="UnsignedLSB2",
                       with_oat=True)
    p = iirs.IirsProduct(root)
    p.label["t0"] = "2021-01-23T10:00:00.000Z"
    p.label["t1"] = "2021-01-23T10:00:01.000Z"
    with pytest.raises(ValueError):
        p.line_latlon()


def test_missing_misc_files_raise(tmp_path):
    cube = np.zeros((256, 2, 2), "<u2")
    root = str(tmp_path / "nomisc")
    write_iirs_product(root, cube, typ="nri", level="Raw", dtype_name="UnsignedLSB2",
                       with_spm=False, with_oat=False)
    p = iirs.IirsProduct(root)
    with pytest.raises(FileNotFoundError):
        p.spm()
    with pytest.raises(FileNotFoundError):
        p.oat()
    with pytest.raises(FileNotFoundError):
        iirs.IirsProduct(str(tmp_path / "does_not_exist"))


def test_geometry_grid_and_pixel_latlon(tmp_path):
    cube = np.zeros((256, 5, 5), "<f4")
    root = str(tmp_path / "l1")
    write_iirs_product(root, cube, typ="nci", with_geometry=True)
    p = iirs.IirsProduct(root)
    g = p.geometry_grid()
    assert g is not None and set(g.dtype.names) == {"Longitude", "Latitude", "Pixel", "Scan"}
    # grid nodes are reproduced exactly
    ll = p.pixel_latlon([0, 4, 2], [0, 4, 2])
    np.testing.assert_allclose(ll[0], [10.0, 355.0], atol=1e-9)
    np.testing.assert_allclose(ll[1], [12.0, 11.0], atol=1e-9)
    np.testing.assert_allclose(ll[2], [11.0, 3.0], atol=1e-9)
    # midway between lon 355 and 3 (=363): wrap-free interpolation gives ~359, not ~179
    mid = p.pixel_latlon([0], [1])[0]
    assert abs(((mid[1] - 359.0) + 180) % 360 - 180) < 0.1


def test_grid_latlon_direct():
    rows = default_geometry_grid()
    g = np.array(rows, dtype=[("Longitude", "f8"), ("Latitude", "f8"), ("Pixel", "f8"), ("Scan", "f8")])
    out = iirs.grid_latlon(g, [1.0], [3.0])
    assert out.shape == (1, 2)
    assert 10.0 < out[0, 0] < 11.0
