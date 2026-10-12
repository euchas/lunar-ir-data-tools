"""Kaguya (SELENE) Spectral Profiler (SP) Level 2C reader.

Reads one SP L2C point-spectrum product (data set ``SLN-L-SP-4-LEVEL2C-V3.0``,
file pair ``SP_2C_02_<orbit>_<lat>_<lon>.lbl`` + ``.spc``) as distributed by
the SELENE Data Archive at JAXA DARTS (https://darts.isas.jaxa.jp/planet/pdap/selene/).
Pure Python plus numpy: no ISIS, no pandas.

Product layout (every byte offset, dimension and scaling factor is read from
the detached PDS3 label; nothing is hard-coded):

* one product is one ~30 s track segment of N footprints, each a 296-band
  point spectrum covering ~512-2587 nm with three detectors:
  VIS 482.6-980.6 nm (84 bands), NIR1 894.4-1688.9 nm (100 bands),
  NIR2 1694.9-2583.1 nm (112 bands);
* binary objects in the ``.spc`` file, all big-endian (MSB) 16-bit unsigned:

  ==============================  =============  ==================================
  object                          shape          content after SCALING_FACTOR
  ==============================  =============  ==================================
  ANCILLARY_AND_SUPPLEMENT_DATA   N x ROW_BYTES  per-footprint geometry table
                                                 (incidence/emission/phase angles,
                                                 centre latitude/longitude, ...)
  SP_SPECTRUM_WAV                 1 x 296        wavelength (x0.1 -> nm)
  SP_SPECTRUM_RAW                 N x 296        raw DN
  SP_SPECTRUM_REF2                N x 296        reflectance, corrected (x1e-4)
  SP_SPECTRUM_RAD                 N x 296        radiance (x1e-2; unit per label)
  SP_SPECTRUM_REF1                N x 296        reflectance, standard I/F (x1e-4)
  SP_SPECTRUM_QA                  N x 296        per-band quality flags
  ==============================  =============  ==================================

Units: wavelengths in nm, angles in degrees, latitude/longitude in degrees
(longitude as stored in the label, 0-360 E), reflectances dimensionless,
radiance in the unit named by the label's ``UNIT`` keyword.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "SpProduct",
    "parse_label",
    "read_spectrum_grid",
    "read_ancillary",
    "load_sp",
    "nearest_band",
    "select_footprint",
    "simple_band_depth",
    "DEFAULT_ANCILLARY_COLUMNS",
    "SPECTRUM_OBJECTS",
]

_POINTER_RE = re.compile(r'^\^(\w+)\s*=\s*\(\s*"([^"]+)"\s*,\s*(\d+)\s*<BYTES>\s*\)')
_OBJECT_RE = re.compile(r"^OBJECT\s*=\s*(\S+)")
_END_OBJECT_RE = re.compile(r"^END_OBJECT\b")

#: Ancillary-table columns decoded by default (all floats; angles and
#: coordinates in degrees).
DEFAULT_ANCILLARY_COLUMNS: tuple[str, ...] = (
    "INCIDENCE_ANGLE",
    "EMISSION_ANGLE",
    "PHASE_ANGLE",
    "CENTER_LATITUDE",
    "CENTER_LONGITUDE",
    "SOLAR_AZIMUTH_ANGLE",
    "SPACECRAFT_AZIMUTH",
)

#: Spectrum grids in an SP L2C product, in label order.
SPECTRUM_OBJECTS: tuple[str, ...] = (
    "SP_SPECTRUM_WAV",
    "SP_SPECTRUM_RAW",
    "SP_SPECTRUM_REF2",
    "SP_SPECTRUM_RAD",
    "SP_SPECTRUM_REF1",
    "SP_SPECTRUM_QA",
)

#: PDS3 (DATA_TYPE, BYTES) -> numpy dtype for the ancillary table (big-endian).
COL_DTYPES = {
    ("IEEE_REAL", 8): ">f8",
    ("IEEE_REAL", 4): ">f4",
    ("MSB_INTEGER", 1): ">i1",
    ("MSB_INTEGER", 2): ">i2",
    ("MSB_INTEGER", 4): ">i4",
    ("MSB_UNSIGNED_INTEGER", 1): ">u1",
    ("MSB_UNSIGNED_INTEGER", 2): ">u2",
    ("MSB_UNSIGNED_INTEGER", 4): ">u4",
}


def parse_label(lbl_path: str) -> tuple[dict[str, int], dict[str, dict[str, str]], list[dict]]:
    """Parse a PDS3 SP label.

    Parameters
    ----------
    lbl_path
        Path to the detached ``.lbl`` file.

    Returns
    -------
    pointers
        ``{object_name: byte_offset}`` with 0-based byte offsets into the
        data file (the label's ``<BYTES>`` pointers are 1-based).
    objects
        ``{object_name: {keyword: value}}`` for every non-COLUMN ``OBJECT``
        block, values as raw strings with surrounding quotes removed.
    ancillary_cols
        List of ``{"name", "dtype", "start0", "bytes"}`` dicts, one per
        ``COLUMN`` of the ancillary table, in label order (``start0`` is the
        0-based byte offset of the column inside a row).
    """
    pointers: dict[str, int] = {}
    objects: dict[str, dict[str, str]] = {}
    ancillary_cols: list[dict] = []
    stack: list[str] = []
    cur: dict[str, str] | None = None
    col: dict | None = None
    with open(lbl_path, "r", encoding="latin-1") as f:
        for raw in f:
            line = raw.strip()
            m = _POINTER_RE.match(line)
            if m:
                pointers[m.group(1)] = int(m.group(3)) - 1
                continue
            m = _OBJECT_RE.match(line)
            if m:
                name = m.group(1)
                stack.append(name)
                if name == "COLUMN":
                    col = {}
                else:
                    cur = {}
                    objects[name] = cur
                continue
            if _END_OBJECT_RE.match(line):
                ended = stack.pop() if stack else None
                if ended == "COLUMN":
                    if col is not None:
                        ancillary_cols.append(col)
                    col = None
                else:
                    cur = None
                continue
            if "=" not in line:
                continue
            key, val = (x.strip() for x in line.split("=", 1))
            val = val.strip('"')
            if col is not None:  # inside a COLUMN
                if key == "NAME":
                    col["name"] = val
                elif key == "DATA_TYPE":
                    col["dtype"] = val
                elif key == "START_BYTE":
                    col["start0"] = int(val) - 1
                elif key == "BYTES":
                    col["bytes"] = int(val)
            elif cur is not None:  # inside a top-level OBJECT
                cur[key] = val
    return pointers, objects, ancillary_cols


def _num(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def read_spectrum_grid(spc: bytes, byte_offset: int, nlines: int, nsamples: int,
                       scale: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """Decode one big-endian uint16 grid and apply ``value * scale + offset``.

    Parameters
    ----------
    spc
        Whole ``.spc`` file content.
    byte_offset
        0-based byte offset of the grid inside ``spc``.
    nlines, nsamples
        Grid shape (``LINES`` x ``LINE_SAMPLES`` from the label).
    scale, offset
        ``SCALING_FACTOR`` and ``OFFSET`` from the label.

    Returns
    -------
    numpy.ndarray
        ``(nlines, nsamples)`` float64 array in physical units.
    """
    nbytes = nlines * nsamples * 2
    buf = spc[byte_offset:byte_offset + nbytes]
    if len(buf) != nbytes:
        raise ValueError("data file too short for a %d x %d grid at byte %d"
                         % (nlines, nsamples, byte_offset))
    a = np.frombuffer(buf, dtype=">u2").astype(np.float64).reshape(nlines, nsamples)
    return a * scale + offset


def read_ancillary(spc: bytes, byte_offset: int, nrows: int, row_bytes: int,
                   cols: Sequence[dict], wanted: Sequence[str]) -> dict[str, np.ndarray]:
    """Decode selected columns of the fixed-width binary ancillary table.

    Parameters
    ----------
    spc
        Whole ``.spc`` file content.
    byte_offset
        0-based byte offset of the table.
    nrows, row_bytes
        Table shape (``ROWS`` x ``ROW_BYTES``).
    cols
        Column descriptions as returned by :func:`parse_label`.
    wanted
        Column names to decode.

    Returns
    -------
    dict
        ``{name: float64 array (nrows,)}``; columns that are absent from the
        label or have an unsupported ``DATA_TYPE`` are returned as NaN.
    """
    out = {w: np.full(nrows, np.nan) for w in wanted}
    cmap = {c.get("name"): c for c in cols}
    for w in wanted:
        c = cmap.get(w)
        if not c:
            continue
        dt = COL_DTYPES.get((c.get("dtype"), c.get("bytes")))
        if not dt:
            continue
        start = byte_offset + c["start0"]
        end = start + (nrows - 1) * row_bytes + c["bytes"]
        if end > len(spc):
            raise ValueError("data file too short for ancillary column %s" % w)
        vals = np.ndarray(shape=(nrows,), dtype=dt, buffer=spc, offset=start,
                          strides=(row_bytes,))
        out[w] = vals.astype(np.float64)
    return out


@dataclass
class SpProduct:
    """One decoded SP L2C product.

    Attributes
    ----------
    lbl_path, spc_path
        Source files.
    wav
        ``(nbands,)`` band-centre wavelengths in nm.
    ref1
        ``(N, nbands)`` standard I/F reflectance (dimensionless).
    ref2
        ``(N, nbands)`` corrected reflectance (dimensionless).
    qa
        ``(N, nbands)`` per-band quality flags (raw integer values as floats).
    geo
        ``{column: (N,) float64}`` ancillary geometry: angles and coordinates
        in degrees (longitude 0-360 E as stored).
    rad
        ``(N, nbands)`` radiance in the label's ``UNIT`` (None if absent).
    raw
        ``(N, nbands)`` raw DN (None if absent).
    objects
        Raw label keyword dictionaries per object (for ``UNIT`` etc.).
    """

    lbl_path: str
    spc_path: str
    wav: np.ndarray
    ref1: np.ndarray
    ref2: np.ndarray
    qa: np.ndarray
    geo: dict[str, np.ndarray]
    rad: np.ndarray | None = None
    raw: np.ndarray | None = None
    objects: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def nrows(self) -> int:
        """Number of footprints N."""
        return int(self.ref1.shape[0])

    @property
    def nbands(self) -> int:
        """Number of spectral bands."""
        return int(self.wav.size)

    def nearest_band(self, target_nm: float) -> tuple[int, float]:
        """Index and wavelength (nm) of the band closest to ``target_nm``."""
        return nearest_band(self.wav, target_nm)


def _resolve_spc(lbl_path: str, spc_path: str | None) -> str:
    if spc_path:
        return spc_path
    stem = os.path.splitext(lbl_path)[0]
    for cand in (stem + ".spc", stem + ".SPC"):
        if os.path.exists(cand):
            return cand
    return stem + ".spc"


def load_sp(lbl_path: str, spc_path: str | None = None,
            ancillary_columns: Sequence[str] = DEFAULT_ANCILLARY_COLUMNS) -> SpProduct:
    """Load an SP L2C product from its label.

    Parameters
    ----------
    lbl_path
        Path to the ``.lbl`` file.
    spc_path
        Path to the ``.spc`` data file; defaults to the label path with the
        extension replaced by ``.spc``.
    ancillary_columns
        Ancillary-table columns to decode (see :data:`DEFAULT_ANCILLARY_COLUMNS`).

    Returns
    -------
    SpProduct
        Wavelengths in nm, reflectances dimensionless, angles in degrees.
    """
    pointers, objects, cols = parse_label(lbl_path)
    spc_path = _resolve_spc(lbl_path, spc_path)
    with open(spc_path, "rb") as f:
        spc = f.read()

    def grid(name: str) -> np.ndarray:
        o = objects[name]
        nl = int(_num(o.get("LINES")))
        ns = int(_num(o.get("LINE_SAMPLES")))
        bits = int(_num(o.get("SAMPLE_BITS"), 16))
        if bits != 16:
            raise ValueError("%s: only 16-bit samples are supported (label says %d)"
                             % (name, bits))
        sc = _num(o.get("SCALING_FACTOR"), 1.0) or 1.0
        off = _num(o.get("OFFSET"), 0.0) or 0.0
        return read_spectrum_grid(spc, pointers[name], nl, ns, sc, off)

    def optional(name: str) -> np.ndarray | None:
        if name in objects and name in pointers:
            return grid(name)
        return None

    wav = grid("SP_SPECTRUM_WAV")[0]
    ref1 = grid("SP_SPECTRUM_REF1")
    ref2 = grid("SP_SPECTRUM_REF2")
    qa = grid("SP_SPECTRUM_QA")
    rad = optional("SP_SPECTRUM_RAD")
    raw = optional("SP_SPECTRUM_RAW")
    nrows = ref1.shape[0]

    anc = objects["ANCILLARY_AND_SUPPLEMENT_DATA"]
    geo = read_ancillary(spc, pointers["ANCILLARY_AND_SUPPLEMENT_DATA"], nrows,
                         int(_num(anc["ROW_BYTES"])), cols, list(ancillary_columns))
    return SpProduct(lbl_path=lbl_path, spc_path=spc_path, wav=wav, ref1=ref1,
                     ref2=ref2, qa=qa, geo=geo, rad=rad, raw=raw, objects=objects)


def nearest_band(wav: np.ndarray, target_nm: float) -> tuple[int, float]:
    """Return ``(index, wavelength_nm)`` of the band closest to ``target_nm``."""
    i = int(np.argmin(np.abs(np.asarray(wav) - target_nm)))
    return i, float(wav[i])


def select_footprint(product: SpProduct, near: tuple[float, float] | None = None,
                     vis_window_nm: tuple[float, float] = (600.0, 900.0),
                     min_vis_reflectance: float = 0.005,
                     max_incidence_deg: float = 90.0) -> int:
    """Pick one footprint row from a product.

    A footprint is *valid* when its incidence angle is below
    ``max_incidence_deg`` (sunlit) and the median REF1 reflectance inside
    ``vis_window_nm`` is finite and above ``min_vis_reflectance``.

    Parameters
    ----------
    product
        Loaded product.
    near
        ``(lat_deg, lon_deg)``: return the valid footprint nearest this point
        (planar distance with longitude scaled by cos(lat)). When None, return
        the valid footprint with the smallest incidence angle, or the overall
        smallest-incidence footprint if none is valid.

    Returns
    -------
    int
        Row index into ``product.ref1`` etc.
    """
    wav, ref1, geo = product.wav, product.ref1, product.geo
    inc = geo["INCIDENCE_ANGLE"]
    lat = geo["CENTER_LATITUDE"]
    lon = geo["CENTER_LONGITUDE"]
    lo, hi = vis_window_nm
    vis_med = np.nanmedian(ref1[:, (wav > lo) & (wav < hi)], axis=1)
    valid = (inc < max_incidence_deg) & np.isfinite(vis_med) & (vis_med > min_vis_reflectance)
    if near is not None:
        tlat, tlon = float(near[0]), float(near[1])
        dd = np.hypot(lat - tlat,
                      ((lon - tlon + 180) % 360 - 180) * np.cos(np.radians(tlat)))
        dd = np.where(valid, dd, 1e9)
        return int(np.argmin(dd))
    cand = np.where(valid)[0]
    if cand.size:
        return int(cand[np.argmin(inc[cand])])
    return int(np.argmin(inc))


def simple_band_depth(wav: np.ndarray, spectrum: np.ndarray, center_nm: float = 850.0,
                      left_nm: float = 750.0, right_nm: float = 950.0) -> float:
    """Two-shoulder band depth ``1 - R(center) / mean(R(left), R(right))``.

    Uses the bands nearest to the three requested wavelengths (nm). Positive
    values indicate an absorption. Returns NaN when the continuum is not
    positive.
    """
    ic, _ = nearest_band(wav, center_nm)
    il, _ = nearest_band(wav, left_nm)
    ir, _ = nearest_band(wav, right_nm)
    cont = 0.5 * (spectrum[il] + spectrum[ir])
    if not cont > 0:
        return float("nan")
    return float(1.0 - spectrum[ic] / cont)
