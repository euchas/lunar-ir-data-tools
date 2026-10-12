import numpy as np
import pytest

from lunar_ir_tools import iirs_reflectance as rf
from lunar_ir_tools.iirs import IirsProduct, RAD_SCALE_MW
from conftest import IIRS_WL, IIRS_BW, synthetic_solar_flux, write_iirs_product


def test_planck_and_brightness_temperature_roundtrip():
    wl_um = np.array([2.0, 3.5, 4.7])
    for T in (200.0, 300.0, 400.0):
        B = rf.planck_mwcm(wl_um, T)
        np.testing.assert_allclose(rf.brightness_temperature(rf.EMISSIVITY * B, wl_um), T, rtol=1e-10)
    # value check against the SI Planck function (W m-2 sr-1 um-1 -> mW cm-2 sr-1 um-1 is x0.1)
    lm = 4.7e-6
    B_si = 2 * rf.H * rf.C ** 2 / lm ** 5 / np.expm1(rf.H * rf.C / (lm * rf.K * 300.0)) * 1e-6
    assert rf.planck_mwcm(4.7, 300.0) == pytest.approx(0.1 * B_si)


def test_sun_moon_distance_formula():
    t = np.array(["2021-01-04T00:00", "2021-07-04T00:00"], dtype="datetime64[ms]")
    d = rf.sun_moon_distance_au(t)
    assert d[0] == pytest.approx(1.00014 - 0.01671, abs=2e-5)     # perihelion
    assert d[1] == pytest.approx(1.00014 + 0.01671, abs=2e-4)     # aphelion
    assert rf.sun_moon_distance_au(np.datetime64("2021-01-04T00:00")) == pytest.approx(d[0])


def test_load_and_resample_solar_flux(tmp_path):
    flux = synthetic_solar_flux(IIRS_WL)
    path = tmp_path / "flux.txt"
    with open(path, "w") as f:
        f.write("# wavelength_nm irradiance_mW_cm-2_um-1\n")
        for w, v in zip(IIRS_WL, flux):
            f.write("%.6f\t%.8f\n" % (w, v))
    np.testing.assert_allclose(rf.load_solar_flux(str(path), IIRS_WL), flux, rtol=1e-6)
    with pytest.raises(ValueError):
        rf.load_solar_flux(str(path), IIRS_WL + 5.0)
    # resampling a fine reference spectrum onto the bands reproduces a smooth input
    ref_wl = np.arange(600.0, 5200.0, 0.5)
    ref = synthetic_solar_flux(ref_wl)
    res = rf.resample_solar_flux(ref_wl, ref, IIRS_WL, IIRS_BW)
    np.testing.assert_allclose(res, flux, rtol=2e-3)
    res2 = rf.resample_solar_flux(ref_wl, ref, IIRS_WL)
    np.testing.assert_allclose(res2, flux, rtol=2e-3)
    # unit sanity: ~136 mW cm-2 um-1 near 712 nm for a 1 AU solar spectrum
    assert 120 < flux[0] < 150
    # as_solar_flux accepts tables and arrays
    np.testing.assert_allclose(rf.as_solar_flux(np.column_stack([IIRS_WL, flux]), IIRS_WL), flux)
    with pytest.raises(ValueError):
        rf.as_solar_flux(flux[:10], IIRS_WL)


def _forward_l1_cube(R, T, cosi, d_au, flux, emissivity=rf.EMISSIVITY):
    """Isothermal forward model: L = R S + eps B(T) (mW cm-2 sr-1 um-1) -> stored x1000."""
    wl_um = IIRS_WL / 1000.0
    S = flux[:, None, None] * cosi / (np.pi * d_au ** 2)
    L = R[:, None, None] * S + emissivity * rf.planck_mwcm(wl_um[:, None, None], T[None])
    return (L / RAD_SCALE_MW).astype("<f4")


def _make_l1(tmp_path, R, T, sun_el=50.0, exposure="e1"):
    root = str(tmp_path / ("l1_" + exposure))
    nl, ns = T.shape
    flux = synthetic_solar_flux(IIRS_WL)
    # placeholder cube to get the product's line times (depend only on nl)
    pid = write_iirs_product(root, np.zeros((256, nl, ns), "<f4"), typ="nci", sun_el=sun_el,
                             exposure=exposure)
    p = IirsProduct(root)
    d_au = float(np.mean(rf.sun_moon_distance_au(p.line_times())))
    cosi = np.cos(np.deg2rad(90.0 - sun_el))
    cube = _forward_l1_cube(R, T, cosi, d_au, flux)
    cube.tofile(p.qub_path)
    return IirsProduct(root), flux, d_au, cosi


def test_reflectance_window_recovers_truth_exactly_when_far_bands_are_thermal_only(tmp_path):
    x = (IIRS_WL - 700) / (5000 - 700)
    R = 0.08 + 0.04 * x
    R[IIRS_WL > 4400] = 0.0                      # no reflected light in the T-estimation bands
    T = 290.0 + 5.0 * np.arange(6)[:, None] + 2.0 * np.arange(4)[None, :]
    p, flux, d_au, cosi = _make_l1(tmp_path, R, T)
    refl, aux = rf.reflectance_window(p, 0, 6, flux)
    assert refl.shape == (256, 6, 4) and refl.dtype == np.float32
    np.testing.assert_allclose(aux["T"], T, atol=0.05)
    np.testing.assert_allclose(aux["cosi"], cosi, rtol=1e-6)
    assert aux["d_au"] == pytest.approx(d_au)
    good = p.good_bands()
    np.testing.assert_allclose(refl[good], np.broadcast_to(R[good, None, None], (good.sum(), 6, 4)),
                               atol=2e-4)
    # a line window and a solar-flux table on disk give the same result
    path = tmp_path / "flux.txt"
    np.savetxt(path, np.column_stack([IIRS_WL, flux]))
    refl2, _ = rf.reflectance_window(p, 2, 5, str(path))
    np.testing.assert_allclose(refl2, refl[:, 2:5, :], atol=1e-6)


def test_reflectance_window_realistic_truth_small_bias(tmp_path):
    x = (IIRS_WL - 700) / (5000 - 700)
    R = 0.08 + 0.04 * x
    T = np.full((3, 2), 300.0)
    p, flux, d_au, cosi = _make_l1(tmp_path, R, T)
    refl, aux = rf.reflectance_window(p, 0, 3, flux)
    # reflected light at 4.5-4.9 um biases the brightness temperature by a few K at most
    assert np.all(np.abs(aux["T"] - 300.0) < 5.0)
    short = p.good_bands() & (IIRS_WL < 2600)
    truth_short = np.broadcast_to(R[short, None, None], (short.sum(), 3, 2))
    np.testing.assert_allclose(refl[short], truth_short, atol=1e-3)
    # no thermal removal: plain I/F of the full radiance, larger beyond 3 um
    refl_nt, aux_nt = rf.reflectance_window(p, 0, 3, flux, thermal=False)
    assert "T" not in aux_nt
    assert np.all(refl_nt[IIRS_WL > 4000] > refl[IIRS_WL > 4000])
    np.testing.assert_allclose(refl_nt[short], truth_short, atol=2e-3)


def test_incidence_and_distance_overrides(tmp_path):
    R = np.full(256, 0.1)
    T = np.full((2, 2), 280.0)
    p, flux, d_au, cosi = _make_l1(tmp_path, R, T)
    a, _ = rf.reflectance_window(p, 0, 2, flux, thermal=False)
    b, _ = rf.reflectance_window(p, 0, 2, flux, thermal=False, incidence_deg=40.0, d_au=d_au)
    np.testing.assert_allclose(a, b, rtol=1e-6)
    c, aux = rf.reflectance_window(p, 0, 2, flux, thermal=False, incidence_deg=[60.0, 89.9])
    assert aux["cosi"][1] == pytest.approx(rf.MIN_COS_INCIDENCE)   # grazing guard
    assert np.all(c[:, 0, :] > a[:, 0, :])


def test_non_e1_exposure_refused_for_thermal(tmp_path):
    R = np.full(256, 0.1)
    T = np.full((2, 2), 280.0)
    p, flux, _, _ = _make_l1(tmp_path, R, T, exposure="e2")
    with pytest.raises(ValueError):
        rf.reflectance_window(p, 0, 2, flux)
    refl, _ = rf.reflectance_window(p, 0, 2, flux, thermal=False)
    assert refl.shape == (256, 2, 2)
