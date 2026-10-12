"""Chandrayaan-2 Imaging Infrared Spectrometer (IIRS) PDS4 product reader.

Reads IIRS Level 0 (``nri``, raw DN), Level 1 (``nci``, calibrated radiance)
and Level 2 (``ndi``, derived reflectance) products as distributed in PDS4
bundles by the ISRO Science Data Archive / PRADAN portal of ISSDC
(https://pradan.issdc.gov.in/; free registration required). Each bundle
unpacks to::

    data/<level>/<YYYYMMDD>/ch2_iir_<typ>_<time>_d_<prd>_<stn>.qub   ENVI-style BSQ cube
    data/<level>/<YYYYMMDD>/ch2_iir_<typ>_<time>_d_<prd>_<stn>.xml   PDS4 label
    data/<level>/<YYYYMMDD>/ch2_iir_<typ>_<time>_d_<prd>_<stn>.hdr   minimal ENVI header
    geometry/<level>/<YYYYMMDD>/*_g_grd_*.csv                        sparse lat/lon grid (L1)
    miscellaneous/<level>/<YYYYMMDD>/*.oat, *.spm, *.lbr             orbit/attitude, Sun
                                                                     parameters (fixed-width ASCII)

Product naming (IIRS PDS4 Archive SIS, Table 23): ``typ`` is ``nri`` for L0
raw counts (14-bit DN in uint16), ``nci`` for L1 calibrated radiance
(float32, stored value x :data:`RAD_SCALE_MW` -> mW cm-2 sr-1 um-1; SIS App. J)
and ``ndi`` for L2 reflectance. The cube is a PDS4 ``Array_3D_Spectrum`` with
axes BAND (256) x LINE x SAMPLE (250), last index fastest (BSQ). Band centres
come from the label's ``Band_Bin_Set`` (712-5010 nm, ~17-25 nm steps).

Bad bands follow SIS Appendix H for the nominal E1/G2 mode: detector ends and
order-sorting-filter (OSF) junctions near 1.2, 1.9 and 3.5 um; higher
exposure settings saturate the thermal end (see :data:`EXPOSURE_SAT_FROM_1B`).

Geolocation offered here is first order only: per-line boresight points from
the OAT file (:meth:`IirsProduct.line_latlon`) or bilinear interpolation of the
sparse geometry grid shipped with L1 products (:meth:`IirsProduct.pixel_latlon`).

Usage::

    from lunar_ir_tools import IirsProduct
    p = IirsProduct("path/to/extracted_bundle")   # or the .qub / .xml path
    p.wl          # (256,) band centres, nm
    p.cube        # memmap (bands, lines, samples)
    p.spectrum(line, sample)
"""
from __future__ import annotations

import glob
import os
import re
import warnings
import xml.etree.ElementTree as ET
from typing import Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator

__all__ = [
    "IirsProduct",
    "read_label",
    "grid_latlon",
    "DTYPES",
    "INVALID_BANDS_1B",
    "OSF_BANDS_1B",
    "EXPOSURE_SAT_FROM_1B",
    "RAD_SCALE_MW",
    "LINE_PERIOD_MS",
]

#: PDS4 data_type -> numpy dtype.
DTYPES = {"UnsignedLSB2": "<u2", "SignedLSB2": "<i2",
          "IEEE754LSBSingle": "<f4", "UnsignedByte": "u1",
          "SignedLSB4": "<i4", "IEEE754LSBDouble": "<f8"}

#: SIS App. H, 1-based band numbers, nominal E1/G2 mode: detector ends.
INVALID_BANDS_1B = set(range(1, 7)) | set(range(252, 257))
#: Order-sorting-filter junctions (~1.2, 1.9, 3.5 um), 1-based, low confidence.
OSF_BANDS_1B = set(range(29, 35)) | set(range(69, 76)) | set(range(162, 172))
#: Exposure setting -> first saturated band (1-based) at the thermal end.
EXPOSURE_SAT_FROM_1B = {"e2": 162, "e3": 147, "e4": 38}
#: L1 (nci) stored float x this -> mW cm-2 sr-1 um-1 (SIS App. J).
RAD_SCALE_MW = 1e-3
#: Nominal line period in ms (label ``line_exposure_duration``).
LINE_PERIOD_MS = 53.060

_STAMP_RE = re.compile(r"_(\d{8}T\d{10})_")


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def grid_latlon(g: np.ndarray, lines, samples) -> np.ndarray:
    """Interpolate the sparse L1 geometry grid to (line, sample) positions.

    Bilinear interpolation of a structured array with fields ``Scan``,
    ``Pixel``, ``Latitude``, ``Longitude`` (the ``*_g_grd_*.csv`` lattice), done
    in unit-sphere Cartesian space. Interpolating ``(cos lat cos lon,
    cos lat sin lon, sin lat)`` is wrap-free everywhere including the pole,
    whereas unwrapping longitude column by column can leave adjacent columns
    offset by 360 deg wherever a grid row has a real >180 deg longitude step.

    Parameters
    ----------
    g
        Structured array from :meth:`IirsProduct.geometry_grid`.
    lines, samples
        Array-likes of equal length; values are clipped to the grid extent.

    Returns
    -------
    numpy.ndarray
        ``(n, 2)`` columns ``latitude`` (deg) and ``longitude`` (deg, 0-360 E).
    """
    scans = np.unique(g["Scan"])
    pixels = np.unique(g["Pixel"])
    lat = np.full((len(scans), len(pixels)), np.nan)
    lon = np.full_like(lat, np.nan)
    si = np.searchsorted(scans, g["Scan"])
    pi = np.searchsorted(pixels, g["Pixel"])
    lat[si, pi] = g["Latitude"]
    lon[si, pi] = g["Longitude"]
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    xyz = (np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la))
    pts = np.column_stack([np.clip(lines, scans[0], scans[-1]),
                           np.clip(samples, pixels[0], pixels[-1])])
    vx, vy, vz = (RegularGridInterpolator((scans, pixels), c)(pts) for c in xyz)
    norm = np.sqrt(vx * vx + vy * vy + vz * vz)
    ilat = np.rad2deg(np.arcsin(np.clip(vz / norm, -1.0, 1.0)))
    ilon = np.rad2deg(np.arctan2(vy, vx)) % 360.0
    return np.column_stack([ilat, ilon])


def read_label(xml_path: str) -> dict:
    """Parse an IIRS PDS4 label.

    Returns
    -------
    dict
        Keys: ``xml`` (path), ``dtype`` (numpy dtype string), ``level``
        (``processing_level``), ``t0``/``t1`` (start/stop ISO strings),
        ``lid`` (logical identifier), ``shape`` (tuple in sequence-number
        order, e.g. ``(256, 13250, 250)``), ``axes`` (axis names in that
        order), ``wl`` (band centres, nm), ``bw`` (band widths, nm) and, when
        present, ``gain``, ``exposure``, ``orbit_limb_direction``,
        ``pixel_resolution``, ``spacecraft_altitude``, ``line_exposure_duration``
        as strings.
    """
    root = ET.parse(xml_path).getroot()
    info: dict = {"xml": xml_path}
    axes: dict = {}
    for el in root.iter():
        t = _strip_ns(el.tag)
        if t == "Axis_Array":
            name = elements = seq = None
            for c in el:
                ct = _strip_ns(c.tag)
                if ct == "axis_name":
                    name = c.text.strip()
                elif ct == "elements":
                    elements = int(c.text)
                elif ct == "sequence_number":
                    seq = int(c.text)
            axes[name] = (seq, elements)
        elif t == "data_type" and "dtype" not in info:
            info["dtype"] = DTYPES.get(el.text.strip(), el.text.strip())
        elif t == "processing_level":
            info["level"] = el.text.strip()
        elif t == "start_date_time":
            info["t0"] = el.text.strip()
        elif t == "stop_date_time":
            info["t1"] = el.text.strip()
        elif t == "logical_identifier":
            info["lid"] = el.text.strip()
        elif t in ("gain", "exposure", "orbit_limb_direction",
                   "pixel_resolution", "spacecraft_altitude",
                   "line_exposure_duration") and t not in info:
            info[t] = el.text.strip()
    wl, bw = [], []
    for bb in root.iter():
        if _strip_ns(bb.tag) == "Band_Bin":
            d = {_strip_ns(c.tag): c.text for c in bb}
            wl.append(float(d["center_wavelength"]))
            bw.append(float(d.get("band_width", "nan")))
    order = sorted(axes.values())                     # by sequence_number
    info["shape"] = tuple(n for _, n in order)        # e.g. (256, 13250, 250)
    info["axes"] = [k for k, _ in sorted(axes.items(), key=lambda kv: kv[1][0])]
    info["wl"] = np.array(wl)
    info["bw"] = np.array(bw)
    return info


def _parse_fixed_width(path: str, widths: Sequence[int], marker: str = "ORBTATTD",
                       what: str = "records") -> np.ndarray:
    """Parse fixed-width ASCII records (marker + numeric fields) into a float array.

    Whitespace splitting is unsafe for these files because adjacent fields can
    abut without a separator (e.g. the block length against the year), so the
    fields are sliced by the column widths. The leading marker field is skipped.
    """
    cols = np.cumsum([0] + list(widths))
    recs, n_bad = [], 0
    with open(path, "r", errors="replace") as f:
        for line in f:
            if not line.startswith(marker) or len(line) < cols[-1]:
                continue
            try:
                recs.append([float(line[a:b]) for a, b in zip(cols[1:-1], cols[2:])])
            except ValueError:
                n_bad += 1
    if n_bad:
        warnings.warn("%s: skipped %d unparseable %s" % (path, n_bad, what))
    if not recs:
        raise ValueError("no parseable %s in %s" % (what, path))
    return np.array(recs)


def _utc_column_times(a: np.ndarray, first_col: int = 2) -> np.ndarray:
    """datetime64[ms] from seven integer columns (Y, M, D, h, m, s, ms)."""
    y, mo, d, h, mi, sec, ms = [a[:, first_col + i].astype(int) for i in range(7)]
    t = (np.array([np.datetime64("%04d-%02d-%02dT%02d:%02d:%02d" % tup, "ms")
                   for tup in zip(y, mo, d, h, mi, sec)])
         + ms.astype("timedelta64[ms]"))
    return t


#: OAT record field widths (characters) after the 8-character marker:
#: record number, block length, 7 x UTC, 6 x 20 (Moon and spacecraft position),
#: 3 x 12 (velocity), 12 x 14 (three quaternions), 6 x 14 (sub-satellite
#: lat/lon, solar az/el, boresight lat/lon), then altitude, roll-velocity
#: angle, eclipse flag, emission, phase, yaw-nadir, slant range, orbit, solar
#: zenith, FoV-velocity angle and yaw/roll/pitch. Follows the bundle's
#: ``miscellaneous/readme.txt``; 628 bytes per record including a 41-byte spare.
OAT_WIDTHS: list[int] = ([8, 6, 4] + [4] * 7 + [20] * 6 + [12] * 3 + [14] * 12 + [14] * 6
                         + [12, 12, 1, 9, 9, 9, 10, 5, 9, 9, 16, 16, 16])

#: SPM (Sun parameter) record field widths after the marker. The four
#: Sun-angle fields (phase, Sun aspect, Sun azimuth, Sun elevation) are read
#: as 16-character F16.8 fields: the bundle readme documents F9.3, but the
#: files actually distributed use 16-character fields (verified against real
#: SPM records). This override is listed in the README's limitations section.
SPM_WIDTHS: list[int] = ([8, 6, 4] + [4] * 7 + [20] * 3 + [12] * 3 + [16] * 4 + [1])


class IirsProduct:
    """One IIRS product: an extracted bundle directory, or a ``.qub``/``.xml`` path.

    Parameters
    ----------
    root
        Bundle directory (searched for ``data/*/*/*.qub``, then ``**/*.qub``),
        or the path of the ``.qub`` or ``.xml`` file itself.

    Attributes
    ----------
    label
        Parsed label dict (see :func:`read_label`).
    wl
        ``(nbands,)`` band centres in nm.
    product_id
        File stem, e.g. ``ch2_iir_nci_20210123T0924038577_d_img_d32``.
    """

    def __init__(self, root):
        root = os.fspath(root)
        low = root.lower()
        if low.endswith(".qub") or low.endswith(".xml"):
            self.qub_path = root[:-4] + ".qub"
            self.xml_path = root[:-4] + ".xml"
            self.root = os.path.dirname(root)
        else:
            qubs = (glob.glob(os.path.join(root, "data", "*", "*", "*.qub"))
                    or glob.glob(os.path.join(root, "**", "*.qub"), recursive=True))
            if not qubs:
                raise FileNotFoundError("no .qub under " + root)
            self.qub_path = sorted(qubs)[0]
            self.xml_path = self.qub_path[:-4] + ".xml"
            self.root = root
        self.label = read_label(self.xml_path)
        self.product_id = os.path.basename(self.qub_path)[:-4]
        self.wl = self.label["wl"]
        self._cube = None
        self._oat = None

    # ------------------------------------------------------------------ basics
    @property
    def nbands(self) -> int:
        """Number of bands (first axis)."""
        return self.label["shape"][0]

    @property
    def nlines(self) -> int:
        """Number of along-track lines (second axis)."""
        return self.label["shape"][1]

    @property
    def nsamples(self) -> int:
        """Number of cross-track samples (third axis)."""
        return self.label["shape"][2]

    @property
    def level(self) -> str:
        """``processing_level`` from the label (``Raw``, ``Calibrated``, ``Derived``)."""
        return self.label.get("level", "")

    @property
    def cube(self) -> np.memmap:
        """Read-only memmap ``(bands, lines, samples)`` of stored values.

        Stored values are DN for ``nri``, radiance x 1000 for ``nci`` (multiply
        by :data:`RAD_SCALE_MW` for mW cm-2 sr-1 um-1) and reflectance for
        ``ndi``. BSQ layout, so ``cube[b]`` is a contiguous band image.
        """
        if self._cube is None:
            shape = self.label["shape"]
            expected = int(np.prod(shape)) * np.dtype(self.label["dtype"]).itemsize
            actual = os.path.getsize(self.qub_path)
            if actual != expected:
                raise ValueError("qub size %d != label %d for %s"
                                 % (actual, expected, self.qub_path))
            self._cube = np.memmap(self.qub_path, dtype=self.label["dtype"],
                                   mode="r", shape=shape)
        return self._cube

    def band(self, wavelength_nm: float) -> int:
        """0-based band index nearest a wavelength in nm."""
        return int(np.argmin(np.abs(self.wl - wavelength_nm)))

    def good_bands(self, include_osf: bool = False) -> np.ndarray:
        """Boolean mask ``(nbands,)`` of usable bands for this product's exposure.

        Excludes :data:`INVALID_BANDS_1B`, the bands saturated for the label's
        exposure setting (:data:`EXPOSURE_SAT_FROM_1B`) and, unless
        ``include_osf`` is True, the OSF junction bands :data:`OSF_BANDS_1B`.
        """
        bad = set(INVALID_BANDS_1B)
        if not include_osf:
            bad |= OSF_BANDS_1B
        exp = str(self.label.get("exposure", "e1")).lower()
        sat_from = EXPOSURE_SAT_FROM_1B.get(exp)
        if sat_from:
            bad |= set(range(sat_from, self.nbands + 1))
        m = np.ones(self.nbands, bool)
        m[[b - 1 for b in bad if b <= self.nbands]] = False
        return m

    def is_descending(self) -> bool:
        """True for descending strips (acquired upside-down: flip along-track)."""
        return str(self.label.get("orbit_limb_direction", "")).lower().startswith("desc")

    def spectrum(self, line: int, sample: int) -> np.ndarray:
        """Stored values ``(nbands,)`` at one pixel as float64.

        DN for ``nri``; for ``nci`` multiply by :data:`RAD_SCALE_MW` to get
        mW cm-2 sr-1 um-1.
        """
        return np.asarray(self.cube[:, line, sample], dtype=np.float64)

    # --------------------------------------------------------------- geometry
    def geometry_grid(self) -> np.ndarray | None:
        """Sparse lat/lon grid shipped with L1 (``nci``) products, or None.

        Reads ``geometry/**/*_g_grd_*.csv`` (columns ``Longitude`` 0-360 E,
        ``Latitude``, ``Pixel``, ``Scan``; roughly every 50 pixels / 50 lines)
        as a structured array.
        """
        csvs = (glob.glob(os.path.join(self.root, "geometry", "**", "*_g_grd_*.csv"),
                          recursive=True)
                if os.path.isdir(getattr(self, "root", "")) else [])
        if not csvs:
            return None
        return np.genfromtxt(sorted(csvs)[0], delimiter=",", names=True)

    def pixel_latlon(self, lines, samples) -> np.ndarray:
        """Per-pixel ``(lat, lon)`` in degrees (lon 0-360 E) from the L1 geometry grid.

        Bilinear interpolation in unit-sphere Cartesian space (:func:`grid_latlon`).
        ``lines`` / ``samples`` are array-likes of equal length. Raises
        ``ValueError`` when no geometry grid is present (L0 products).
        """
        g = self.geometry_grid()
        if g is None:
            raise ValueError("no geometry grid (L0 product?); use line_latlon()")
        return grid_latlon(g, lines, samples)

    def _misc_files(self, ext: str) -> list[str]:
        """Files with extension ``ext`` under miscellaneous/, nearest this product first."""
        qub_dir = os.path.dirname(os.path.normpath(self.qub_path))
        misc_dir = qub_dir.replace(os.sep + "data" + os.sep, os.sep + "miscellaneous" + os.sep)
        found = glob.glob(os.path.join(misc_dir, "*" + ext))
        if not found and os.path.isdir(getattr(self, "root", "")):
            found = glob.glob(os.path.join(self.root, "**", "*" + ext), recursive=True)
        if not found:
            raise FileNotFoundError("no %s file under %s" % (ext, self.root))
        m = _STAMP_RE.search(self.product_id)
        stamped = [p for p in sorted(found) if m and m.group(1) in os.path.basename(p)]
        return stamped or sorted(found)

    def oat(self) -> dict:
        """Parse the ``.oat`` orbit + attitude file (fixed-width ASCII, one record per 512 ms).

        Returns
        -------
        dict
            ``t`` (datetime64[ms]), ``moon_pos`` / ``sat_pos`` (km, J2000,
            Moon-centred in the lunar mission phase), ``sat_vel`` (km/s),
            ``q_body``, ``q_earth_fixed``, ``q_lunar_fixed`` (quaternions),
            ``subsat`` (lat, lon deg), ``sol_azel`` (deg), ``latlon`` (boresight
            ground point, deg), ``alt_km``, ``emission`` and ``phase`` (deg),
            ``slant_km``, ``orbit`` (int), ``sol_zen`` (deg), ``ypr_deg`` and
            ``raw`` (the full numeric record array).
        """
        if self._oat is not None:
            return self._oat
        path = self._misc_files(".oat")[0]
        a = _parse_fixed_width(path, OAT_WIDTHS, what="OAT records")
        # columns (post-marker): 0 recno, 1 blocklen, 2..8 UTC(7I4), 9..11 moon_pos,
        # 12..14 sat_pos, 15..17 sat_vel, 18..21 q_body, 22..25 q_earth_fixed, 26..29
        # q_lunar_fixed, 30/31 subsat lat/lon, 32/33 solar az/el, 34/35 boresight
        # lat/lon, 36 alt_km, 37 roll-vel angle, 38 eclipse, 39 emission, 40 phase,
        # 41 yaw-nadir, 42 slant_km, 43 orbit, 44 solar zenith, 45 FoV-vel,
        # 46..48 yaw/roll/pitch (deg)
        self._oat = {
            "t": _utc_column_times(a),
            "moon_pos": a[:, 9:12], "sat_pos": a[:, 12:15], "sat_vel": a[:, 15:18],
            "q_body": a[:, 18:22], "q_earth_fixed": a[:, 22:26], "q_lunar_fixed": a[:, 26:30],
            "subsat": a[:, 30:32], "sol_azel": a[:, 32:34], "latlon": a[:, 34:36],
            "alt_km": a[:, 36], "emission": a[:, 39], "phase": a[:, 40],
            "slant_km": a[:, 42], "orbit": a[:, 43].astype(int), "sol_zen": a[:, 44],
            "ypr_deg": a[:, 46:49], "raw": a,
        }
        return self._oat

    def spm(self) -> dict:
        """Parse the ``.spm`` Sun-parameter file (fixed-width ASCII records).

        Returns
        -------
        dict
            ``t`` (datetime64[ms]), ``phase``, ``sun_aspect``, ``sun_az``,
            ``sun_el`` (all degrees). Solar incidence angle on a flat surface is
            ``90 - sun_el``. See :data:`SPM_WIDTHS` for the field-width override.
        """
        path = self._misc_files(".spm")[0]
        a = _parse_fixed_width(path, SPM_WIDTHS, what="SPM records")
        return {"t": _utc_column_times(a), "phase": a[:, 15], "sun_aspect": a[:, 16],
                "sun_az": a[:, 17], "sun_el": a[:, 18]}

    def line_times(self) -> np.ndarray:
        """``datetime64[ms]`` per image line, linear between label start and stop.

        Multiplies before dividing: a truncated per-line step would accumulate
        to seconds of error over ~13 000 lines.
        """
        t0 = np.datetime64(self.label["t0"].rstrip("Z"), "ms")
        t1 = np.datetime64(self.label["t1"].rstrip("Z"), "ms")
        return t0 + (np.arange(self.nlines) * (t1 - t0)) / max(self.nlines - 1, 1)

    def line_incidence(self) -> np.ndarray:
        """Per-line solar incidence angle in degrees, ``90 - SPM Sun elevation``.

        Interpolated onto the line times; flat-surface incidence without
        topography.
        """
        sp = self.spm()
        ot = (sp["t"] - sp["t"][0]) / np.timedelta64(1, "s")
        lt = (self.line_times() - sp["t"][0]) / np.timedelta64(1, "s")
        return 90.0 - np.interp(lt, ot, sp["sun_el"])

    def line_latlon(self) -> np.ndarray:
        """Approximate per-line boresight ``(lat, lon)`` in degrees (lon -180..180).

        Interpolates the OAT ground points onto the line times. First-order
        geolocation only (no cross-track solution). Raises ``ValueError`` when
        the OAT span does not cover the line times (2 s tolerance).
        """
        o = self.oat()
        ot = (o["t"] - o["t"][0]) / np.timedelta64(1, "s")
        lt = (self.line_times() - o["t"][0]) / np.timedelta64(1, "s")
        if lt[0] < ot[0] - 2.0 or lt[-1] > ot[-1] + 2.0:
            raise ValueError(
                "OAT span [%s..%s] does not cover line times [%s..%s] for %s"
                % (o["t"][0], o["t"][-1], self.line_times()[0],
                   self.line_times()[-1], self.product_id))
        lat = np.interp(lt, ot, o["latlon"][:, 0])
        lon = np.interp(lt, ot, np.unwrap(np.deg2rad(o["latlon"][:, 1])) * 180 / np.pi)
        return np.column_stack([lat, (lon + 180) % 360 - 180])
