"""Chandrayaan-2 IIRS Level 1 radiance to reflectance (single-temperature thermal removal).

Implements the radiance-to-reflectance step of Verma, Chauhan & Chauhan
(2022, Icarus 383, 115075) for IIRS L1 (``nci``) PDS4 products from ISSDC
PRADAN:

1. brightness temperature from the 4.5-4.9 um bands (reflected sunlight is
   assumed negligible there) with emissivity :data:`EMISSIVITY` = 0.95,
   averaged over those bands to one temperature T per pixel;
2. subtract ``EMISSIVITY * B(lambda, T)`` from the whole spectrum (isothermal
   pixel; see :mod:`lunar_ir_tools.iirs_thermal` for a two-temperature model
   that matters beyond ~3.5 um);
3. I/F: ``refl = pi * L * d**2 / (F_sun(lambda) * cos(i))`` with ``F_sun`` the
   solar spectral irradiance at 1 AU on the 256 IIRS bands, ``d`` the
   Sun-Moon distance in AU and ``i`` the per-line solar incidence angle from
   the bundle's SPM file (flat surface).

Units: L1 stored float x :data:`~lunar_ir_tools.iirs.RAD_SCALE_MW` gives
radiance in mW cm-2 sr-1 um-1 (SIS App. J); solar flux in mW cm-2 um-1;
Planck radiance in mW cm-2 sr-1 um-1; temperature in K; wavelength in um
inside this module unless the name ends in ``_nm``.

Only exposure-``e1`` scenes are accepted when thermal removal is on: ``e2``-``e4``
saturate the 4.5-4.9 um bands (SIS App. H).

The solar-flux table is **not shipped** with this package because its
provenance could not be established (see the README section "Solar-flux
table"). Pass a two-column text file ``wavelength_nm  irradiance_mW_cm-2_um-1``
on the product's band grid, or build one from a public reference spectrum with
:func:`resample_solar_flux`.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from .iirs import IirsProduct, RAD_SCALE_MW

__all__ = [
    "EMISSIVITY",
    "T_EST_RANGE_NM",
    "T_VALID_RANGE_K",
    "MIN_COS_INCIDENCE",
    "planck_mwcm",
    "brightness_temperature",
    "sun_moon_distance_au",
    "load_solar_flux",
    "resample_solar_flux",
    "as_solar_flux",
    "reflectance_window",
]

#: Surface emissivity assumed for the thermal term (Verma et al. 2022).
EMISSIVITY = 0.95
#: Bands used for the brightness-temperature estimate, nm.
T_EST_RANGE_NM = (4500.0, 4900.0)
#: Brightness temperatures outside this range (K) are ignored in the average.
T_VALID_RANGE_K = (100.0, 450.0)
#: Lower clip on cos(incidence) to guard grazing illumination.
MIN_COS_INCIDENCE = 0.05
# Planck constants (SI).
H, C, K = 6.62607015e-34, 2.99792458e8, 1.380649e-23


def planck_mwcm(wl_um, T):
    """Planck spectral radiance ``B(lambda, T)`` in mW cm-2 sr-1 um-1.

    Parameters
    ----------
    wl_um
        Wavelength(s) in um.
    T
        Temperature(s) in K, broadcastable against ``wl_um``.
    """
    lm = np.asarray(wl_um, dtype=np.float64) * 1e-6
    c1 = 2 * H * C * C / lm ** 5            # W m-3 sr-1
    x = H * C / (lm * K * np.asarray(T, dtype=np.float64))
    return 0.1 * (c1 / np.expm1(x)) * 1e-6  # -> W m-2 sr-1 um-1 -> mW cm-2 sr-1 um-1


def brightness_temperature(L_mw, wl_um, emissivity: float = EMISSIVITY):
    """Invert Planck's law per band assuming ``L = emissivity * B(lambda, T)``.

    Parameters
    ----------
    L_mw
        Radiance in mW cm-2 sr-1 um-1.
    wl_um
        Wavelength in um, broadcastable against ``L_mw``.

    Returns
    -------
    numpy.ndarray
        Temperature in K (NaN or inf where the radiance is not positive).
    """
    lm = np.asarray(wl_um, dtype=np.float64) * 1e-6
    c1 = 2 * H * C * C / lm ** 5
    with np.errstate(divide="ignore", invalid="ignore"):
        T = (H * C / (lm * K)) / np.log1p(emissivity * 0.1 * c1 * 1e-6 / np.asarray(L_mw, dtype=np.float64))
    return T


def sun_moon_distance_au(t):
    """Sun-Moon distance in AU from the observation date.

    Uses the Earth-orbit approximation
    ``1.00014 - 0.01671 * cos(2 pi (doy - 3.5) / 365.25)`` with ``doy`` the
    fractional day of year; the Moon's +/-0.0026 AU excursion around the
    Earth-Moon barycentre is ignored (it is within the formula's own accuracy).

    Parameters
    ----------
    t
        ``numpy.datetime64`` scalar or array.
    """
    t = np.asarray(t)
    doy = (t - t.astype("datetime64[Y]").astype("datetime64[D]")) / np.timedelta64(1, "D")
    return 1.00014 - 0.01671 * np.cos(2 * np.pi * (doy - 3.5) / 365.25)


def load_solar_flux(path: str, wl_nm=None, atol_nm: float = 1.0) -> np.ndarray:
    """Read a two-column solar-flux table ``wavelength_nm  irradiance``.

    Parameters
    ----------
    path
        Text file, one band per row, whitespace separated, ``#`` comments
        allowed. Irradiance must be in mW cm-2 um-1 at 1 AU.
    wl_nm
        Optional product band centres (nm); when given the table's wavelength
        column must match them within ``atol_nm``.

    Returns
    -------
    numpy.ndarray
        ``(nbands,)`` irradiance in mW cm-2 um-1.
    """
    flux = np.loadtxt(path, comments="#", ndmin=2)
    if flux.ndim != 2 or flux.shape[1] < 2:
        raise ValueError("solar-flux table must have two columns: wavelength_nm irradiance")
    if wl_nm is not None:
        wl_nm = np.asarray(wl_nm, dtype=np.float64)
        if flux.shape[0] != wl_nm.size or not np.allclose(flux[:, 0], wl_nm, atol=atol_nm):
            raise ValueError("solar-flux wavelength grid does not match the product bands")
    return flux[:, 1].astype(np.float64)


def resample_solar_flux(wl_nm, irradiance, band_centres_nm, band_widths_nm=None) -> np.ndarray:
    """Resample a reference solar spectrum onto instrument bands by boxcar averaging.

    Parameters
    ----------
    wl_nm, irradiance
        Reference spectrum (e.g. ASTM E-490, Gueymard 2004 or Thuillier et
        al. 2003), already converted to mW cm-2 um-1 at 1 AU
        (``W m-2 nm-1`` x 100 = mW cm-2 um-1).
    band_centres_nm
        Band centres (nm), e.g. ``IirsProduct.wl``.
    band_widths_nm
        Band widths (nm), e.g. ``IirsProduct.label["bw"]``; when None, half the
        spacing to the neighbouring centres is used on each side.

    Returns
    -------
    numpy.ndarray
        ``(nbands,)`` mean irradiance inside each band; bands without any
        reference sample fall back to linear interpolation at the centre.
    """
    wl_nm = np.asarray(wl_nm, dtype=np.float64)
    irradiance = np.asarray(irradiance, dtype=np.float64)
    centres = np.asarray(band_centres_nm, dtype=np.float64)
    order = np.argsort(wl_nm)
    wl_nm, irradiance = wl_nm[order], irradiance[order]
    if band_widths_nm is None:
        edges = np.concatenate([[centres[0] - (centres[1] - centres[0]) / 2],
                                0.5 * (centres[1:] + centres[:-1]),
                                [centres[-1] + (centres[-1] - centres[-2]) / 2]])
        lo, hi = edges[:-1], edges[1:]
    else:
        widths = np.asarray(band_widths_nm, dtype=np.float64)
        lo, hi = centres - widths / 2, centres + widths / 2
    out = np.empty(centres.size)
    for i in range(centres.size):
        m = (wl_nm >= lo[i]) & (wl_nm <= hi[i])
        out[i] = irradiance[m].mean() if m.any() else np.interp(centres[i], wl_nm, irradiance)
    return out


def as_solar_flux(solar_flux: Any, wl_nm, atol_nm: float = 1.0) -> np.ndarray:
    """Normalise a solar-flux argument to a ``(nbands,)`` array in mW cm-2 um-1.

    ``solar_flux`` may be a path to a two-column table, a ``(nbands, 2)``
    array (wavelength_nm, irradiance) or a ``(nbands,)`` irradiance array
    already on the product grid ``wl_nm``.
    """
    if isinstance(solar_flux, (str, os.PathLike)):
        return load_solar_flux(os.fspath(solar_flux), wl_nm, atol_nm)
    arr = np.asarray(solar_flux, dtype=np.float64)
    wl_nm = np.asarray(wl_nm, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        if arr.shape[0] != wl_nm.size or not np.allclose(arr[:, 0], wl_nm, atol=atol_nm):
            raise ValueError("solar-flux wavelength grid does not match the product bands")
        return arr[:, 1]
    if arr.ndim == 1 and arr.size == wl_nm.size:
        return arr
    raise ValueError("solar_flux must be a path, an (nbands, 2) table or an (nbands,) array")


def _illumination(product: IirsProduct, line0: int, line1: int,
                  incidence_deg=None, d_au=None) -> tuple[np.ndarray, float]:
    """Per-line cos(incidence) (clipped) and mean Sun-Moon distance for a line window."""
    if d_au is None:
        d_au = float(np.mean(sun_moon_distance_au(product.line_times()[line0:line1])))
    if incidence_deg is None:
        incidence_deg = product.line_incidence()[line0:line1]
    inc = np.broadcast_to(np.asarray(incidence_deg, dtype=np.float64), (line1 - line0,))
    cosi = np.clip(np.cos(np.deg2rad(inc)), MIN_COS_INCIDENCE, None)
    return cosi, float(d_au)


def reflectance_window(product: IirsProduct, line0: int, line1: int, solar_flux,
                       thermal: bool = True, emissivity: float = EMISSIVITY,
                       incidence_deg=None, d_au=None) -> tuple[np.ndarray, dict]:
    """Reflectance for ``cube[:, line0:line1, :]`` of an L1 (``nci``) product.

    Parameters
    ----------
    product
        An L1 :class:`~lunar_ir_tools.iirs.IirsProduct`.
    line0, line1
        Line window (0-based, end exclusive).
    solar_flux
        Solar irradiance at 1 AU on the product bands, mW cm-2 um-1: a file
        path, an ``(nbands, 2)`` table or an ``(nbands,)`` array
        (see :func:`as_solar_flux`).
    thermal
        Remove the single-temperature thermal term first (requires exposure
        ``e1``).
    emissivity
        Emissivity used for the brightness temperature and the subtracted
        thermal term (default :data:`EMISSIVITY`).
    incidence_deg
        Optional per-line solar incidence angles in degrees (scalar or
        ``(line1 - line0,)``); default: from the bundle's SPM file.
    d_au
        Optional Sun-Moon distance in AU; default: from the line times.

    Returns
    -------
    refl
        ``(nbands, nlines, nsamples)`` float32 reflectance (I/F, dimensionless).
    aux
        ``{"T": (nlines, nsamples) float32 K (only with thermal),
        "cosi": (nlines,) float32, "d_au": float}``.
    """
    exp = str(product.label.get("exposure", "")).lower()
    if thermal and exp != "e1":
        raise ValueError("exposure %r saturates the 4.5-4.9 um temperature-estimation bands; "
                         "thermal removal is only valid for e1 scenes" % exp)
    wl_um = product.wl / 1000.0
    L = np.asarray(product.cube[:, line0:line1, :], dtype=np.float64) * RAD_SCALE_MW
    aux: dict = {}
    if thermal:
        tb = (wl_um >= T_EST_RANGE_NM[0] / 1000) & (wl_um <= T_EST_RANGE_NM[1] / 1000)
        tb &= product.good_bands(include_osf=True)      # OSF bands are fine for the T average
        Ts = brightness_temperature(L[tb], wl_um[tb, None, None], emissivity)
        ok = np.isfinite(Ts) & (Ts > T_VALID_RANGE_K[0]) & (Ts < T_VALID_RANGE_K[1])
        T = np.nanmean(np.where(ok, Ts, np.nan), axis=0)
        aux["T"] = T.astype(np.float32)
        L = L - emissivity * planck_mwcm(wl_um[:, None, None], T[None, :, :])
    flux = as_solar_flux(solar_flux, product.wl)
    cosi, d_au = _illumination(product, line0, line1, incidence_deg, d_au)
    refl = (np.pi * L * d_au ** 2
            / (flux[:, None, None] * cosi[None, :, None]))
    aux["cosi"] = cosi.astype(np.float32)
    aux["d_au"] = d_au
    return refl.astype(np.float32), aux
