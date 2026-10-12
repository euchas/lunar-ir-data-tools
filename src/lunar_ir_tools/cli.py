"""Command-line interface: ``lunar-ir-tools <sub-command> ...``.

Sub-commands: ``sp-read``, ``sp-search``, ``iirs-read``, ``iirs-reflectance``,
``iirs-thermal``, ``mi-sample``. Each is a thin wrapper around the library
functions; run ``lunar-ir-tools <sub-command> -h`` for the options.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys

import numpy as np

from ._version import __version__


def _parse_pair(text: str, sep: str) -> tuple[float, float]:
    parts = text.split(sep)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected two numbers separated by %r" % sep)
    return float(parts[0]), float(parts[1])


# ----------------------------------------------------------------- sp-read
def cmd_sp_read(args) -> int:
    from .kaguya_sp import load_sp, select_footprint, simple_band_depth

    p = load_sp(args.lbl, spc_path=args.spc)
    wav, ref1, ref2, geo = p.wav, p.ref1, p.ref2, p.geo
    inc, lat, lon = geo["INCIDENCE_ANGLE"], geo["CENTER_LATITUDE"], geo["CENTER_LONGITUDE"]

    print("FILE           :", os.path.basename(p.lbl_path))
    print("footprints (N) :", p.nrows)
    print("bands          : %d  wavelength range %.1f-%.1f nm" % (p.nbands, wav.min(), wav.max()))
    print("incidence deg  : min %.2f  max %.2f  (sunlit if <90)" % (np.nanmin(inc), np.nanmax(inc)))
    print("center lat     : %.3f .. %.3f" % (np.nanmin(lat), np.nanmax(lat)))
    print("center lon     : %.3f .. %.3f" % (np.nanmin(lon), np.nanmax(lon)))

    near = _parse_pair(args.near, ",") if args.near else None
    idx = select_footprint(p, near=near)
    print("\nCHOSEN footprint row %d" % idx)
    print("  center lat/lon : %.3f , %.3f" % (lat[idx], lon[idx]))
    print("  incidence/emission/phase : %.2f / %.2f / %.2f deg"
          % (inc[idx], geo["EMISSION_ANGLE"][idx], geo["PHASE_ANGLE"][idx]))

    lo, hi = args.window
    sel = np.where((wav >= lo) & (wav <= hi))[0]
    print("\n  wavelength(nm)  REF1       REF2")
    for i in sel:
        print("  %8.1f     %8.5f  %8.5f" % (wav[i], ref1[idx, i], ref2[idx, i]))

    print("\n  example diagnostic (REF1): nearest bands to 660 / 750 / 850 / 950 nm")
    for target in (660.0, 750.0, 850.0, 950.0):
        i, w = p.nearest_band(target)
        print("    %4.0f nm band -> %.1f nm  R=%.5f" % (target, w, ref1[idx, i]))
    bd = simple_band_depth(wav, ref1[idx], 850.0, 750.0, 950.0)
    if np.isfinite(bd):
        print("    850 nm band depth (1 - R850 / mean(R750, R950)) = %+.4f "
              "(positive = absorption)" % bd)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["wavelength_nm", "ref1", "ref2", "qa"])
            for i in range(p.nbands):
                w.writerow([wav[i], ref1[idx, i], ref2[idx, i], int(p.qa[idx, i])])
        print("\nwrote", args.csv)
    return 0


# --------------------------------------------------------------- sp-search
def cmd_sp_search(args) -> int:
    from .kaguya_sp_search import PdapClient, search_footprints, write_manifest

    lat_lo, lat_hi = _parse_pair(args.lat, "/")
    lon_lo, lon_hi = _parse_pair(args.lon, "/")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = search_footprints(lat_lo, lat_hi, lon_lo, lon_hi, dataset_id=args.dataset,
                             client=PdapClient(min_interval_s=args.interval))
    write_manifest(rows, args.out)
    print("WROTE %d unique products -> %s" % (len(rows), args.out))
    return 0


# --------------------------------------------------------------- iirs-read
def cmd_iirs_read(args) -> int:
    from .iirs import IirsProduct, RAD_SCALE_MW

    p = IirsProduct(args.product)
    lab = p.label
    print("product        :", p.product_id)
    print("level          :", lab.get("level", "?"), " dtype:", lab.get("dtype"))
    print("shape          : bands %d x lines %d x samples %d (axes %s)"
          % (p.nbands, p.nlines, p.nsamples, "/".join(lab.get("axes", []))))
    print("wavelength     : %.1f-%.1f nm (%d bands)" % (p.wl.min(), p.wl.max(), p.wl.size))
    print("start / stop   : %s / %s" % (lab.get("t0"), lab.get("t1")))
    for k in ("exposure", "gain", "orbit_limb_direction", "pixel_resolution",
              "spacecraft_altitude"):
        if k in lab:
            print("%-15s: %s" % (k, lab[k]))
    print("usable bands   : %d of %d (OSF excluded)" % (p.good_bands().sum(), p.nbands))
    if args.spectrum:
        line, sample = args.spectrum
        spec = p.spectrum(line, sample)
        scale = RAD_SCALE_MW if p.product_id.startswith("ch2_iir_nci") else 1.0
        unit = "mW/cm2/sr/um" if scale != 1.0 else "stored value"
        print("\nspectrum at line %d sample %d (%s):" % (line, sample, unit))
        rows = [(p.wl[i], spec[i] * scale, bool(p.good_bands()[i])) for i in range(p.nbands)]
        if args.csv:
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["wavelength_nm", "value", "good_band"])
                w.writerows(rows)
            print("wrote", args.csv)
        else:
            for wl, v, g in rows:
                print("  %8.1f  %12.6g  %s" % (wl, v, "" if g else "(bad band)"))
    return 0


# -------------------------------------------------------- iirs-reflectance
def cmd_iirs_reflectance(args) -> int:
    from .iirs import IirsProduct
    from .iirs_reflectance import reflectance_window

    p = IirsProduct(args.product)
    l0, l1 = args.lines
    refl, aux = reflectance_window(p, l0, l1, args.solar_flux, thermal=not args.no_thermal,
                                   emissivity=args.emissivity)
    np.savez_compressed(args.out, refl=refl, wl=p.wl, line0=l0, line1=l1,
                        good_bands=p.good_bands(), **aux)
    print("saved ->", args.out, refl.shape)
    if "T" in aux:
        print("T range: %.1f .. %.1f K (median %.1f)"
              % (np.nanmin(aux["T"]), np.nanmax(aux["T"]), np.nanmedian(aux["T"])))
    return 0


# ------------------------------------------------------------ iirs-thermal
def cmd_iirs_thermal(args) -> int:
    from .iirs import IirsProduct
    from .iirs_thermal import refl_window_aniso

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = IirsProduct(args.product)
    l0, l1 = args.lines
    R, aux = refl_window_aniso(p, l0, l1, args.solar_flux, n_sample=args.n_sample)
    np.savez_compressed(args.out, refl=R, wl=p.wl, line0=l0, line1=l1,
                        good_bands=p.good_bands(include_osf=False), **aux)
    print("saved ->", args.out, R.shape)
    print("f=%.3f dT=%.1f K  curvature iso %.4g -> aniso %.4g  reliable px %.2f"
          % (aux["f"], aux["dT"], aux["iso_resid"], aux["aniso_resid"], aux["reliable_frac"]))
    return 0


# --------------------------------------------------------------- mi-sample
def cmd_mi_sample(args) -> int:
    from .kaguya_mi import (mineral_map_url, read_geojson_polygons, sample_points,
                            sample_polygons, MINERAL_MAP_PARAMETERS)

    mosaic = args.mosaic
    if mosaic.lower() in MINERAL_MAP_PARAMETERS:
        mosaic = mineral_map_url(mosaic)
    if not args.points and not args.polygons:
        print("give --points LON,LAT ... and/or --polygons FILE.geojson", file=sys.stderr)
        return 2
    rows = []
    if args.points:
        pts = [_parse_pair(t, ",") for t in args.points]
        vals = sample_points(mosaic, [x for x, _ in pts], [y for _, y in pts])
        for (x, y), v in zip(pts, vals):
            rows.append(["point", "%g,%g" % (x, y), 1 if np.isfinite(v) else 0, v, "", "", "", ""])
    if args.polygons:
        polys = read_geojson_polygons(args.polygons, id_property=args.id_property)
        stats = sample_polygons(mosaic, polys, band_height=args.band_height)
        for pid, _ in polys:
            s = stats.get(pid)
            if s is None:
                rows.append(["polygon", pid, 0, "", "", "", "", ""])
            else:
                rows.append(["polygon", pid, s.count, s.median, s.p25, s.p75, s.mean, s.std])
    header = ["kind", "id", "count", "median", "p25", "p75", "mean", "std"]
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print("wrote %d rows -> %s" % (len(rows), args.out))
    else:
        print(",".join(header))
        for r in rows:
            print(",".join("" if v == "" else ("%g" % v if isinstance(v, float) else str(v)) for v in r))
    return 0


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lunar-ir-tools",
                                 description="Readers and corrections for Kaguya SP, Kaguya MI "
                                             "and Chandrayaan-2 IIRS data.")
    ap.add_argument("--version", action="version", version="lunar-ir-data-tools " + __version__)
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sp-read", help="summarise a Kaguya SP L2C product (.lbl + .spc)")
    s.add_argument("lbl", help="path to the PDS3 .lbl file")
    s.add_argument("--spc", default=None, help=".spc path if not next to the label")
    s.add_argument("--near", default=None, metavar="LAT,LON",
                   help="pick the sunlit footprint nearest this point (degrees)")
    s.add_argument("--window", nargs=2, type=float, default=(600.0, 1000.0),
                   metavar=("LO_NM", "HI_NM"), help="wavelength window to print")
    s.add_argument("--csv", default=None, help="write the chosen footprint's spectrum to CSV")
    s.set_defaults(func=cmd_sp_read)

    s = sub.add_parser("sp-search", help="find SP products in a lat/lon box via JAXA DARTS PDAP")
    s.add_argument("--lat", required=True, metavar="LO/HI", help="latitude bounds, e.g. 83/87")
    s.add_argument("--lon", required=True, metavar="LO/HI",
                   help="longitude bounds in degrees east 0-360, e.g. 190/215")
    s.add_argument("--dataset", default="SLN-L-SP-4-LEVEL2C-V3.0",
                   help="DATA_SET_ID (default L2C; L2B1 = SLN-L-SP-3-LEVEL2B1-V2.0)")
    s.add_argument("--interval", type=float, default=1.0,
                   help="minimum seconds between requests (>= 1)")
    s.add_argument("--out", required=True, help="output CSV manifest")
    s.set_defaults(func=cmd_sp_search)

    s = sub.add_parser("iirs-read", help="summarise a Chandrayaan-2 IIRS PDS4 product")
    s.add_argument("product", help="bundle directory, .qub or .xml path")
    s.add_argument("--spectrum", nargs=2, type=int, metavar=("LINE", "SAMPLE"),
                   help="print (or write with --csv) the spectrum at one pixel")
    s.add_argument("--csv", default=None, help="CSV output for --spectrum")
    s.set_defaults(func=cmd_iirs_read)

    s = sub.add_parser("iirs-reflectance",
                       help="IIRS L1 radiance -> reflectance (single-temperature thermal removal)")
    s.add_argument("product", help="L1 (nci) bundle directory, .qub or .xml path")
    s.add_argument("--lines", nargs=2, type=int, required=True, metavar=("L0", "L1"))
    s.add_argument("--solar-flux", required=True,
                   help="two-column table wavelength_nm irradiance(mW/cm2/um at 1 AU)")
    s.add_argument("--no-thermal", action="store_true", help="skip the thermal removal")
    s.add_argument("--emissivity", type=float, default=0.95)
    s.add_argument("--out", required=True, help="output .npz")
    s.set_defaults(func=cmd_iirs_reflectance)

    s = sub.add_parser("iirs-thermal",
                       help="IIRS L1 radiance -> reflectance with the two-temperature model")
    s.add_argument("product", help="L1 (nci) bundle directory, .qub or .xml path")
    s.add_argument("--lines", nargs=2, type=int, required=True, metavar=("L0", "L1"))
    s.add_argument("--solar-flux", required=True,
                   help="two-column table wavelength_nm irradiance(mW/cm2/um at 1 AU)")
    s.add_argument("--n-sample", type=int, default=4000, help="pixels used for the (f, dT) fit")
    s.add_argument("--out", required=True, help="output .npz")
    s.set_defaults(func=cmd_iirs_thermal)

    s = sub.add_parser("mi-sample", help="sample a Kaguya MI mosaic at points or polygons")
    s.add_argument("mosaic", help="GeoTIFF path/URL, or a mineral-map key: omat feo cpx opx ol plag")
    s.add_argument("--points", nargs="*", metavar="LON,LAT", help="points in degrees")
    s.add_argument("--polygons", default=None, help="GeoJSON FeatureCollection of polygons")
    s.add_argument("--id-property", default=None, help="feature property holding the polygon id")
    s.add_argument("--band-height", type=int, default=2500, help="rows per latitude-band read")
    s.add_argument("--out", default=None, help="CSV output (default: stdout)")
    s.set_defaults(func=cmd_mi_sample)
    return ap


def main(argv=None) -> int:
    """Entry point of the ``lunar-ir-tools`` console script."""
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
