"""Injection-recovery tests of the two-temperature correction.

A synthetic radiance field is built from a known reflectance (with planted
absorption features inside the protected windows) and a known two-component
thermal field at the warm temperatures the correction is meant for
(300-322 K). The fit must detect the anisothermality, beat the isothermal
correction on the 3.5-4.2 um continuum and preserve the planted band depths.
Note that (f, dT, T0) are partly degenerate (a hotter minority fraction can
trade against a lower mean temperature), so the tests check the recovered
reflectance rather than the parameter values themselves.
"""
import numpy as np
import pytest

from lunar_ir_tools import iirs_thermal as th
from lunar_ir_tools.iirs import IirsProduct, RAD_SCALE_MW
from lunar_ir_tools.iirs_reflectance import planck_mwcm, sun_moon_distance_au
from conftest import IIRS_WL, synthetic_solar_flux, write_iirs_product

WL = IIRS_WL
PLANTED = (3400.0, 3950.0)            # feature centres (nm), one in each protected window
PLANTED_DEPTH = 0.04

# the isothermal solution is NaN in whole bands near the reflected/emitted null
pytestmark = pytest.mark.filterwarnings("ignore:All-NaN slice encountered")


def gauss(wl, c, depth, fwhm):
    return depth * np.exp(-0.5 * ((wl - c) / (fwhm / 2.355)) ** 2)


def truth_reflectance(depth=PLANTED_DEPTH):
    """Smooth reddish continuum, a 3 um feature and two planted features."""
    x = (WL - 700) / (5000 - 700)
    cont = 0.05 + 0.10 * x
    r = cont - gauss(WL, 2950, 0.03, 250)
    for c in PLANTED:
        r = r - gauss(WL, c, depth, 90)
    return r, cont


def good_mask():
    good = np.ones(256, bool)
    good[:6] = good[251:] = False
    good[161:171] = False                                      # OSF junction
    return good


def make_scene(npx=1500, f_true=0.15, dT_true=80.0, T0_lo=300, T0_hi=322,
               cosi=0.35, d_au=1.0, noise=0.004, R_true=None):
    flux = synthetic_solar_flux(WL)
    wl_um = WL / 1000.0
    S = flux * cosi / (np.pi * d_au ** 2)                      # (nb,)
    if R_true is None:
        R_true, _ = truth_reflectance()
    T0 = T0_lo + (T0_hi - T0_lo) * (np.arange(npx) % npx) / npx   # deterministic spread
    E = (1 - f_true) * planck_mwcm(wl_um[:, None], T0[None]) \
        + f_true * planck_mwcm(wl_um[:, None], T0[None] + dT_true)
    L = R_true[:, None] * S[:, None] + (1 - R_true[:, None]) * E
    ph = np.outer(np.arange(256), np.arange(npx))
    L = L * (1 + noise * np.sin(12.9898 * ph))                 # deterministic pseudo-noise
    return L, S, R_true, T0, wl_um


def band_depth(R, wl, c, halfwin=60):
    b = (wl >= c - halfwin) & (wl <= c + halfwin)
    sh = ((wl >= c - 200) & (wl <= c - 120)) | ((wl >= c + 120) & (wl <= c + 200))
    return float(np.nanmean(R[sh]) - np.nanmin(R[b]))


def closer(a, b, truth):
    """True when a is closer to truth than b (a NaN b counts as worse)."""
    if not np.isfinite(a):
        return False
    return (not np.isfinite(b)) or abs(a - truth) < abs(b - truth)


def test_window_constants_do_not_overlap_protected_windows():
    excl = th.band_mask(WL, th.EXCLUDED_WINDOWS)
    for anchor in (th.FAR_ANCHOR, th.MID_ANCHOR, th.REFL_NOM):
        assert not (th.band_mask(WL, anchor) & excl).any()
    assert excl.sum() >= 20                                    # the windows do contain bands
    for c in PLANTED:
        assert excl[np.argmin(np.abs(WL - c))]


def test_emission_reduces_to_planck_for_f0_or_dT0():
    wl_um = WL / 1000.0
    B = planck_mwcm(wl_um, 250.0)
    np.testing.assert_allclose(th.emission(wl_um, 250.0, 80.0, 0.0), B)
    np.testing.assert_allclose(th.emission(wl_um, 250.0, 0.0, 0.3), B)
    hot = th.emission(wl_um, 250.0, 80.0, 0.2)
    assert np.all(hot >= B)


def test_flat_isothermal_scene_is_solved_exactly_and_fit_falls_back():
    """With a flat reflectance the nominal-R assumption is exact: T0 and R are
    recovered to numerical precision and the (f, dT) fit keeps the isothermal
    solution because nothing is left to improve."""
    R_flat = np.full(256, 0.10)
    L, S, R_true, T0, wl_um = make_scene(npx=300, f_true=0.0, dT_true=0.0, T0_lo=192,
                                         T0_hi=208, noise=0.0, R_true=R_flat)
    R, T0_fit = th.solve_reflectance(L, S, wl_um, good_mask(), 0.0, 0.0)
    np.testing.assert_allclose(T0_fit, T0, atol=1e-3)
    good = good_mask()
    assert not np.isnan(R[good]).any()
    np.testing.assert_allclose(R[good], 0.10, atol=1e-6)
    f, dT, iso_err, ani_err = th.fit_scene_anisothermality(L, wl_um, S, good, verbose=False)
    assert (f, dT) == (0.0, 0.0) and ani_err == iso_err


def test_recovers_anisothermality_and_preserves_planted_features():
    """Validator scene: T0 300-322 K, f = 0.15, dT = 80 K, cos(i) = 0.35."""
    L, S, R_true, T0_true, wl_um = make_scene()
    good = good_mask()
    f, dT, iso_err, ani_err = th.fit_scene_anisothermality(L, wl_um, S, good, verbose=False)
    assert f > 0 and dT > 0                                    # anisothermality detected
    assert ani_err < iso_err
    R_iso, _ = th.solve_reflectance(L, S, wl_um, good, 0.0, 0.0)
    R_ani, T0_fit = th.solve_reflectance(L, S, wl_um, good, f, dT)
    m_iso, m_ani = np.nanmedian(R_iso, axis=1), np.nanmedian(R_ani, axis=1)
    cont = good & th.band_mask(WL, th.MID_ANCHOR)
    err_iso = np.nanmean(np.abs(m_iso[cont] - R_true[cont]))
    err_ani = np.nanmean(np.abs(m_ani[cont] - R_true[cont]))
    assert err_ani < 0.05 and err_ani < 0.25 * err_iso        # continuum upturn removed
    # the planted features inside the protected windows keep their depth
    for c in PLANTED:
        d_true = band_depth(R_true, WL, c)
        d_iso, d_ani = band_depth(m_iso, WL, c), band_depth(m_ani, WL, c)
        assert abs(d_ani - d_true) < 0.5 * d_true
        assert closer(d_ani, d_iso, d_true)
    reliable = np.isfinite(R_ani[(WL >= 3900) & (WL <= 4000)]).all(axis=0).mean()
    assert reliable > 0.9
    # short wavelengths are thermal-free and unaffected by the model
    short = good & (WL < 2600)
    np.testing.assert_allclose(m_ani[short], R_true[short], atol=3e-3)


def test_too_few_anchor_bands_raise():
    L, S, R_true, T0, wl_um = make_scene(npx=50, noise=0.0)
    good = good_mask()
    good[WL > 4300] = False
    with pytest.raises(ValueError):
        th.fit_scene_anisothermality(L, wl_um, S, good, verbose=False)


def test_product_level_pipeline(tmp_path):
    """refl_window_aniso on a small fake L1 bundle with a two-temperature truth."""
    nl, ns = 12, 4
    f_true, dT_true = 0.15, 80.0
    root = str(tmp_path / "l1")
    write_iirs_product(root, np.zeros((256, nl, ns), "<f4"), typ="nci", sun_el=50.0)
    p = IirsProduct(root)
    flux = synthetic_solar_flux(WL)
    wl_um = WL / 1000.0
    d_au = float(np.mean(sun_moon_distance_au(p.line_times())))
    cosi = np.cos(np.deg2rad(40.0))
    S = flux * cosi / (np.pi * d_au ** 2)
    R_true, _ = truth_reflectance()
    T0 = 300.0 + 16.0 * np.arange(nl * ns).reshape(nl, ns) / (nl * ns)
    E = ((1 - f_true) * planck_mwcm(wl_um[:, None, None], T0[None])
         + f_true * planck_mwcm(wl_um[:, None, None], T0[None] + dT_true))
    L = R_true[:, None, None] * S[:, None, None] + (1 - R_true[:, None, None]) * E
    (L / RAD_SCALE_MW).astype("<f4").tofile(p.qub_path)
    p = IirsProduct(root)

    R, aux = th.refl_window_aniso(p, 2, 10, flux, verbose=False, n_sample=nl * ns)
    assert R.shape == (256, 8, ns) and R.dtype == np.float32
    for k in ("T0", "reliable", "cosi", "d_au", "f", "dT", "iso_resid", "aniso_resid", "reliable_frac"):
        assert k in aux
    assert aux["f"] > 0 and aux["dT"] > 0
    assert aux["reliable"].all() and aux["reliable_frac"] == 1.0
    np.testing.assert_allclose(aux["cosi"], cosi, rtol=1e-6)
    assert aux["d_au"] == pytest.approx(d_au)
    assert np.all(np.abs(aux["T0"] - T0[2:10]) < 40.0)        # degenerate with (f, dT)
    good = p.good_bands(include_osf=False)
    short = good & (WL < 2600)
    np.testing.assert_allclose(R[short], np.broadcast_to(R_true[short, None, None], (short.sum(), 8, ns)),
                               atol=5e-3)
    # the isothermal solution of the same window is worse on the mid continuum and at 3.95 um
    R_iso, _ = th.refl_chunk_aniso(p, 2, 10, 0.0, 0.0, flux)
    m_ani = np.nanmedian(R.reshape(256, -1), axis=1)
    m_iso = np.nanmedian(R_iso.reshape(256, -1), axis=1)
    cont = good & th.band_mask(WL, th.MID_ANCHOR)
    assert np.nanmean(np.abs(m_ani[cont] - R_true[cont])) < np.nanmean(np.abs(m_iso[cont] - R_true[cont]))
    d_true = band_depth(R_true, WL, 3950.0)
    assert closer(band_depth(m_ani, WL, 3950.0), band_depth(m_iso, WL, 3950.0), d_true)
    # the same (f, dT) applied through refl_chunk_aniso gives the same window
    R2, aux2 = th.refl_chunk_aniso(p, 2, 10, aux["f"], aux["dT"], flux)
    np.testing.assert_allclose(R2, R, equal_nan=True)
    # overrides for incidence and distance reproduce the SPM-based result
    R3, _ = th.refl_chunk_aniso(p, 2, 10, aux["f"], aux["dT"], flux, incidence_deg=40.0, d_au=d_au)
    np.testing.assert_allclose(R3, R, equal_nan=True, rtol=1e-5)
    # e2 exposure is refused for the fit
    p.label["exposure"] = "e2"
    with pytest.raises(ValueError):
        th.fit_strip_anisothermality(p, flux, verbose=False)
