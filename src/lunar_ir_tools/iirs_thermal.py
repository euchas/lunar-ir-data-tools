"""Two-temperature (anisothermal) thermal-emission removal for Chandrayaan-2 IIRS L1 radiance.

Extends the single-temperature correction in :mod:`lunar_ir_tools.iirs_reflectance`
(Verma et al. 2022) for IIRS L1 (``nci``) PDS4 products from ISSDC PRADAN.
On a rough ~80 m pixel the sub-pixel kinetic-temperature distribution is
wide: Sun-facing facets run far hotter than the mean, and because the Wien
tail is steep that hot fraction dominates the thermal radiance at 3.5-4.2 um
much more than at 4.5-4.9 um. A single temperature fitted to 4.5-4.9 um
therefore under-predicts the thermal term near 3.9 um and leaves a spurious
continuum upturn between 3.5 and 4.2 um.

Forward model, per pixel and band::

    L_obs = R * S + (1 - R) * E(T0)
      S(lambda)  = F_sun(lambda) cos(i) / (pi d^2)          reflected-solar transfer
      E(T0)      = (1 - f) B(lambda, T0) + f B(lambda, T0 + dT)   two-component emission
      emissivity = 1 - R                                     (Kirchhoff)
    => R(lambda) = (L_obs - E) / (S - E)

``(f, dT)`` describe the anisothermality *shape* and are held fixed for a
scene (regolith roughness assumed uniform); the mean temperature ``T0`` is
solved per pixel (albedo, slope and illumination vary). ``f = 0`` or
``dT = 0`` reduces exactly to the isothermal case.

Wavelength windows (nm; also listed in the README "Known limitations and
assumptions"):

* :data:`FAR_ANCHOR` 4400-4900: absolute-radiance anchor for ``T0`` per pixel
  (bulk thermal emission, beyond the features of interest);
* :data:`MID_ANCHOR` 3600-3850 and 4050-4350: together with the far anchor,
  the continuum whose residual curvature the global ``(f, dT)`` fit minimises;
* :data:`REFL_NOM` 1700-2100: thermal-free region giving a nominal
  reflectance per pixel that weakly enters the ``T0`` anchor;
* :data:`EXCLUDED_WINDOWS` 3300-3520 and 3850-4050: never used as anchors, so
  that absorption features in those windows are neither fitted away nor
  biased by the correction. They lie between the mid anchors by construction.

Guards:

* crossover null: where ``|S - E|`` is small (reflected ~ emitted) ``R`` is
  ill-posed and set to NaN (:data:`CROSSOVER_GUARD`);
* physical bounds: ``R`` outside (-0.05, 1.5) is set to NaN;
* isothermal fallback when the two-temperature fit does not reduce the
  continuum curvature by at least 10 %.

Units: radiance in mW cm-2 sr-1 um-1, solar flux in mW cm-2 um-1,
wavelength in um unless a name ends in ``_nm``, temperatures in K.
"""
from __future__ import annotations

import logging

import numpy as np

from .iirs import IirsProduct, RAD_SCALE_MW
from .iirs_reflectance import (H, C, K, planck_mwcm, sun_moon_distance_au,
                               as_solar_flux, MIN_COS_INCIDENCE)

__all__ = [
    "EXCLUDED_WINDOWS",
    "FAR_ANCHOR",
    "MID_ANCHOR",
    "REFL_NOM",
    "CROSSOVER_GUARD",
    "band_mask",
    "emission",
    "solve_reflectance",
    "fit_scene_anisothermality",
    "fit_strip_anisothermality",
    "refl_chunk_aniso",
    "refl_window_aniso",
]

log = logging.getLogger(__name__)

#: Windows (nm) protected from the fit: never used as thermal anchors.
EXCLUDED_WINDOWS = ((3300.0, 3520.0), (3850.0, 4050.0))
#: Bulk-thermal anchor (nm) for the per-pixel T0, past the protected windows.
FAR_ANCHOR = (4400.0, 4900.0)
#: Mid-wavelength continuum anchors (nm) for the global (f, dT) fit.
MID_ANCHOR = ((3600.0, 3850.0), (4050.0, 4350.0))
#: Thermal-free, reflection-dominated region (nm) for the nominal reflectance.
REFL_NOM = (1700.0, 2100.0)
#: Flag R where |S - E| < CROSSOVER_GUARD * S (reflected ~ emitted null).
CROSSOVER_GUARD = 0.20


def band_mask(wl_nm, ranges) -> np.ndarray:
    """Boolean mask of bands inside one ``(lo, hi)`` range or a tuple of ranges (nm)."""
    wl_nm = np.asarray(wl_nm)
    m = np.zeros(len(wl_nm), bool)
    for lo, hi in (ranges if isinstance(ranges[0], (tuple, list)) else [ranges]):
        m |= (wl_nm >= lo) & (wl_nm <= hi)
    return m


def emission(wl_um, T0, dT, f):
    """Two-component areal-mixture emission ``(1-f) B(T0) + f B(T0 + dT)``.

    ``wl_um`` in um, temperatures in K; broadcasts ``(nb, ...)`` over pixels.
    Result in mW cm-2 sr-1 um-1.
    """
    return (1 - f) * planck_mwcm(wl_um, T0) + f * planck_mwcm(wl_um, T0 + dT)


def _solve_R(L, S, E):
    """Closed-form Kirchhoff reflectance ``(L - E) / (S - E)``.

    NaN near the reflected == emitted null and outside physical bounds:
    unbounded tails can reach |R| ~ 12 near 2.5 um, and a single such pixel
    shifts a pooled or binned spectrum by more than the weak band depths one
    would look for.
    """
    den = S - E
    R = (L - E) / den
    R[np.abs(den) < CROSSOVER_GUARD * S] = np.nan
    R[(R < -0.05) | (R > 1.5)] = np.nan
    return R


def _nominal_R(L, wl_um, S, good):
    """Per-pixel nominal reflectance from the reflection-dominated 1.7-2.1 um region.

    ``R ~ L / S`` there (thermal negligible); clipped to [0.02, 0.6]. Weakly
    enters the far-band T0 anchor. ``L`` is ``(nb, npx)``, ``S`` ``(nb,)``;
    returns ``(npx,)``.
    """
    ci = np.where(good & band_mask(wl_um * 1000, REFL_NOM))[0]
    r = np.nanmedian(L[ci] / S[ci][:, None], axis=0)
    return np.clip(r, 0.02, 0.6)


def _planck_invT(wl_um, B):
    """Brightness temperature (K) from radiance ``B`` (mW cm-2 sr-1 um-1) at ``wl_um``."""
    lm = wl_um * 1e-6
    c1 = 2 * H * C * C / lm ** 5
    x = np.log1p(np.clip(0.1 * c1 * 1e-6 / np.clip(B, 1e-9, None), 1e-9, None))
    return (H * C / (lm * K)) / x


def _solve_T0(L, wl_um, S, far_idx, Rnom, dT, f, iters=6):
    """Per-pixel mean temperature T0 from the absolute far-band radiance.

    Solves ``L_far = Rnom * S + (1 - Rnom) * E(T0)`` for T0 by vectorised
    Gauss-Newton (thermal emission dominates the far bands, so the reflected
    term barely matters). ``L`` is ``(nb, npx)``, ``Rnom`` ``(npx,)``; returns
    ``(npx,)`` in K, clipped to 140-410 K.
    """
    Lf, Sf, wf = L[far_idx], S[far_idx][:, None], wl_um[far_idx][:, None]
    om = (1 - Rnom)[None, :]
    # initial guess: per-band brightness temperature of the implied emission, medianed
    Btar = np.clip((Lf - Rnom[None, :] * Sf) / om, 1e-6, None)
    T0 = np.clip(np.nanmedian(_planck_invT(wf, Btar), axis=0), 150.0, 400.0)
    for _ in range(iters):
        E = emission(wf, T0[None, :], dT, f)
        dE = 0.5 * (emission(wf, (T0 + 1)[None, :], dT, f)
                    - emission(wf, (T0 - 1)[None, :], dT, f))
        resid = Lf - (Rnom[None, :] * Sf + om * E)
        dmodel = om * dE
        num = np.nansum(resid * dmodel, axis=0)
        den = np.nansum(dmodel * dmodel, axis=0)
        step = np.where(den > 0, num / den, 0.0)
        T0 = np.clip(T0 + np.clip(step, -60, 60), 140.0, 410.0)
    return T0


def _curvature(R, wl_um, idx):
    """RMS deviation of R from a straight line in log(wavelength) over anchor bands ``idx``.

    Measures residual thermal curvature; with T0 anchored absolutely this is
    non-degenerate. Returns inf when fewer than 5 pixels are fully finite.
    """
    x = np.log(wl_um[idx])
    y = R[idx]
    ok = np.isfinite(y).all(axis=0)
    if ok.sum() < 5:
        return np.inf
    y = y[:, ok]
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(np.sqrt(np.mean((y - A @ coef) ** 2)))


def solve_reflectance(L, S, wl_um, good, f: float, dT: float) -> tuple[np.ndarray, np.ndarray]:
    """Two-temperature reflectance for a pixel sample with known ``(f, dT)``.

    Parameters
    ----------
    L
        ``(nb, npx)`` radiance in mW cm-2 sr-1 um-1.
    S
        ``(nb,)`` reflected-solar transfer ``F_sun cos(i) / (pi d^2)`` in the
        same unit.
    wl_um
        ``(nb,)`` band centres in um.
    good
        ``(nb,)`` boolean mask of usable bands.
    f, dT
        Hot-fraction (0-1) and temperature contrast (K) of the emission model;
        ``(0, 0)`` is the isothermal case.

    Returns
    -------
    R
        ``(nb, npx)`` reflectance (NaN where unreliable).
    T0
        ``(npx,)`` mean temperature in K.
    """
    far = np.where(good & band_mask(wl_um * 1000, FAR_ANCHOR))[0]
    Rnom = _nominal_R(L, wl_um, S, good)
    T0 = _solve_T0(L, wl_um, S, far, Rnom, dT, f)
    E = emission(wl_um[:, None], T0[None], dT, f)
    R = _solve_R(L, S[:, None], E)
    return R, T0


def fit_scene_anisothermality(L, wl_um, S, good, verbose: bool = True):
    """Fit the global ``(f, dT)`` on a pixel sample.

    T0 is pinned per pixel by the absolute far-band radiance (non-degenerate);
    ``(f, dT)`` then minimise the mid + far continuum curvature on a coarse
    grid (f 0.02-0.40, dT 20-140 K) followed by a local refinement. The
    isothermal solution is kept unless the two-temperature curvature is below
    90 % of the isothermal one.

    Parameters
    ----------
    L
        ``(nb, npx)`` radiance, mW cm-2 sr-1 um-1.
    wl_um
        ``(nb,)`` band centres, um.
    S
        ``(nb,)`` reflected-solar transfer, mW cm-2 sr-1 um-1.
    good
        ``(nb,)`` usable-band mask (OSF bands excluded).
    verbose
        Log the outcome at INFO level.

    Returns
    -------
    (f, dT, iso_err, aniso_err)
        Hot fraction, contrast in K, and the curvature residuals of the
        isothermal and the chosen solution.
    """
    wl_nm = wl_um * 1000.0
    far = np.where(good & band_mask(wl_nm, FAR_ANCHOR))[0]
    mid = np.where(good & band_mask(wl_nm, MID_ANCHOR))[0]
    if len(far) < 4 or len(mid) < 4:
        raise ValueError("too few clean anchor bands (far=%d mid=%d)" % (len(far), len(mid)))
    anc = np.concatenate([mid, far])
    Rnom = _nominal_R(L, wl_um, S, good)

    def objective(f, dT):
        T0 = _solve_T0(L, wl_um, S, far, Rnom, dT, f)
        E = emission(wl_um[:, None], T0[None, :], dT, f)
        R = _solve_R(L, S[:, None], E)
        return _curvature(R, wl_um, anc)

    iso_err = objective(0.0, 0.0)
    best = (0.0, 0.0, iso_err)
    for f in np.linspace(0.02, 0.40, 10):
        for dT in np.linspace(20, 140, 10):
            e = objective(f, dT)
            if e < best[2]:
                best = (f, dT, e)
    f0, dT0, _ = best
    for f in np.clip(f0 + np.linspace(-0.04, 0.04, 5), 0.0, 0.6):
        for dT in np.clip(dT0 + np.linspace(-12, 12, 5), 0.0, 200):
            e = objective(f, dT)
            if e < best[2]:
                best = (f, dT, e)
    f, dT, aniso_err = best
    if aniso_err > 0.9 * iso_err:
        if verbose:
            log.info("anisothermality not significant (%.4g vs iso %.4g) -> isothermal",
                     aniso_err, iso_err)
        return 0.0, 0.0, iso_err, iso_err
    if verbose:
        log.info("global anisothermality: f=%.3f dT=%.1f K  curvature %.4g -> %.4g",
                 f, dT, iso_err, aniso_err)
    return float(f), float(dT), iso_err, aniso_err


def _transfer(product: IirsProduct, flux, line0: int, line1: int,
              incidence_deg=None, d_au=None):
    """Reflected-solar transfer ``S(band, line)`` and ``d_au`` for a line window."""
    if d_au is None:
        d_au = float(np.mean(sun_moon_distance_au(product.line_times()[line0:line1])))
    if incidence_deg is None:
        incidence_deg = product.line_incidence()[line0:line1]
    inc = np.broadcast_to(np.asarray(incidence_deg, dtype=np.float64), (line1 - line0,))
    cosi = np.clip(np.cos(np.deg2rad(inc)), MIN_COS_INCIDENCE, None)
    S = flux[:, None] * cosi[None, :] / (np.pi * d_au ** 2)      # (nb, nl)
    return S, cosi, float(d_au)


def fit_strip_anisothermality(product: IirsProduct, solar_flux, n_sample: int = 4000,
                              verbose: bool = True, incidence_deg=None, d_au=None):
    """Fit the global ``(f, dT)`` once on a pixel sample spanning the whole strip.

    Lines are sampled uniformly along the strip (varied illumination), all
    samples per line. Requires exposure ``e1``.

    Parameters
    ----------
    product
        L1 (``nci``) product.
    solar_flux
        Solar irradiance at 1 AU on the product bands (mW cm-2 um-1): path,
        ``(nbands, 2)`` table or ``(nbands,)`` array.
    n_sample
        Approximate number of pixels in the sample.
    incidence_deg, d_au
        Optional overrides for the per-line incidence (deg, scalar or
        ``(nlines,)``) and the Sun-Moon distance (AU).

    Returns
    -------
    (f, dT, iso_err, aniso_err)
        See :func:`fit_scene_anisothermality`.
    """
    if str(product.label.get("exposure", "")).lower() != "e1":
        raise ValueError("exposure %r saturates the thermal bands; e1 only"
                         % product.label.get("exposure"))
    wl_um = product.wl / 1000.0
    good = product.good_bands(include_osf=False)
    flux = as_solar_flux(solar_flux, product.wl)
    nl = product.nlines
    lsel = np.unique(np.linspace(0, nl - 1, max(n_sample // product.nsamples, 8)).astype(int))
    L = np.asarray(product.cube[:, lsel, :], dtype=np.float64) * RAD_SCALE_MW   # (nb, k, ns)
    S, _, _ = _transfer(product, flux, 0, nl, incidence_deg, d_au)
    Ssamp = S[:, lsel].mean(axis=1)                                             # (nb,)
    Lf = L.reshape(len(wl_um), -1)
    return fit_scene_anisothermality(Lf, wl_um, Ssamp, good, verbose=verbose)


def refl_chunk_aniso(product: IirsProduct, line0: int, line1: int, f: float, dT: float,
                     solar_flux, incidence_deg=None, d_au=None) -> tuple[np.ndarray, dict]:
    """Two-temperature reflectance for a line window using a pre-fitted ``(f, dT)``.

    Parameters
    ----------
    product
        L1 (``nci``) product.
    line0, line1
        Line window (0-based, end exclusive).
    f, dT
        Scene anisothermality from :func:`fit_strip_anisothermality`.
    solar_flux
        As in :func:`fit_strip_anisothermality`.
    incidence_deg, d_au
        Optional overrides (deg, scalar or ``(line1 - line0,)``; AU).

    Returns
    -------
    R
        ``(nbands, nlines, nsamples)`` float32 reflectance, NaN where unreliable.
    aux
        ``{"T0": (nlines, nsamples) float32 K, "reliable": (nlines, nsamples)
        bool (all far-anchor bands finite), "cosi": (nlines,) float32,
        "d_au": float}``.
    """
    wl_um = product.wl / 1000.0
    good = product.good_bands(include_osf=False)
    flux = as_solar_flux(solar_flux, product.wl)
    L = np.asarray(product.cube[:, line0:line1, :], dtype=np.float64) * RAD_SCALE_MW  # (nb,nl,ns)
    nb, nl, ns = L.shape
    S2, cosi, d_au = _transfer(product, flux, line0, line1, incidence_deg, d_au)   # S2 (nb,nl)
    S = S2[:, :, None]                                                          # (nb,nl,1)
    Lflat = L.reshape(nb, nl * ns)
    # S is constant across samples, so the per-line mean equals the flat mean
    # (avoids materialising a large broadcast copy)
    Smean = S2.mean(axis=1)
    far = np.where(good & band_mask(product.wl, FAR_ANCHOR))[0]
    Rnom = _nominal_R(Lflat, wl_um, Smean, good)
    T0 = _solve_T0(Lflat, wl_um, Smean, far, Rnom, dT, f).reshape(nl, ns)
    E = emission(wl_um[:, None, None], T0[None], dT, f)
    R = _solve_R(L, np.broadcast_to(S, (nb, nl, ns)), E).astype(np.float32)
    aux = {"T0": T0.astype(np.float32),
           "reliable": np.isfinite(R[far]).all(axis=0),
           "cosi": cosi.astype(np.float32), "d_au": d_au}
    return R, aux


def refl_window_aniso(product: IirsProduct, line0: int, line1: int, solar_flux,
                      verbose: bool = True, n_sample: int = 4000,
                      incidence_deg=None, d_au=None) -> tuple[np.ndarray, dict]:
    """Fit ``(f, dT)`` on the whole strip, then apply it to one line window.

    Convenience wrapper around :func:`fit_strip_anisothermality` and
    :func:`refl_chunk_aniso`. ``incidence_deg`` overrides, when given, must
    cover the whole strip (``(nlines,)`` or scalar). The returned ``aux`` adds
    ``f``, ``dT``, ``iso_resid``, ``aniso_resid`` and ``reliable_frac``.
    """
    f, dT, iso_r, ani_r = fit_strip_anisothermality(product, solar_flux, n_sample=n_sample,
                                                    verbose=verbose,
                                                    incidence_deg=incidence_deg, d_au=d_au)
    inc_win = None
    if incidence_deg is not None:
        inc_full = np.broadcast_to(np.asarray(incidence_deg, dtype=np.float64), (product.nlines,))
        inc_win = inc_full[line0:line1]
    R, aux = refl_chunk_aniso(product, line0, line1, f, dT, solar_flux,
                              incidence_deg=inc_win, d_au=d_au)
    aux.update({"f": f, "dT": dT, "iso_resid": iso_r, "aniso_resid": ani_r,
                "reliable_frac": float(aux["reliable"].mean())})
    if verbose:
        T0 = aux["T0"]
        log.info("T0 %.1f..%.1f K (median %.1f) | reliable pixel fraction %.2f",
                 np.nanmin(T0), np.nanmax(T0), np.nanmedian(T0), aux["reliable_frac"])
    return R, aux
