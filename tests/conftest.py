"""Synthetic-data generators shared by the tests. No network, no mission data."""
from __future__ import annotations

import os
import struct

import numpy as np
import pytest

from lunar_ir_tools.iirs_reflectance import planck_mwcm

#: IIRS band grid (nm): the label's Band_Bin_Set is exactly linear.
IIRS_WL = 712.3339889 + 16.8523621 * np.arange(256)
IIRS_BW = np.full(256, 20.0)


def synthetic_solar_flux(wl_nm) -> np.ndarray:
    """Stand-in solar irradiance at 1 AU in mW cm-2 um-1: a 5772 K Planck
    spectrum diluted by (R_sun / AU)^2. Realistic to ~10 %; only a test input."""
    r_sun, au = 6.957e8, 1.495978707e11
    return np.pi * planck_mwcm(np.asarray(wl_nm, dtype=np.float64) / 1000.0, 5772.0) * (r_sun / au) ** 2


# ---------------------------------------------------------------- Kaguya SP
SP_WAV_NM = np.array([500.0, 600.0, 660.0, 750.0, 850.0, 950.0, 1000.0, 1500.0, 2000.0, 2500.0])

SP_COLUMNS = [
    ("SP_INDEX", "MSB_INTEGER", 4),
    ("INCIDENCE_ANGLE", "IEEE_REAL", 8),
    ("EMISSION_ANGLE", "IEEE_REAL", 8),
    ("PHASE_ANGLE", "IEEE_REAL", 8),
    ("CENTER_LATITUDE", "IEEE_REAL", 8),
    ("CENTER_LONGITUDE", "IEEE_REAL", 8),
    ("SOLAR_AZIMUTH_ANGLE", "IEEE_REAL", 4),
    ("SPACECRAFT_AZIMUTH", "IEEE_REAL", 4),
    ("SP_TIME", "CHARACTER", 12),
]
SP_ROW_BYTES = 64


def default_sp_arrays():
    """Three footprints x ten bands with a known 850 nm dip in row 0."""
    nb = SP_WAV_NM.size
    ref1 = np.full((3, nb), 0.10)
    ref1[0, 4] = 0.08                     # 850 nm absorption in the first footprint
    ref1[1, :] = 0.001                    # too dark -> invalid
    ref1[2, :] = 0.12                     # night-side -> invalid (incidence 95)
    ref2 = ref1 * 1.1
    raw = (np.arange(3 * nb) % 4000).reshape(3, nb)
    rad = np.linspace(1.0, 50.0, 3 * nb).reshape(3, nb)
    qa = np.zeros((3, nb))
    qa[0, 0] = 3
    geo = {
        "INCIDENCE_ANGLE": np.array([40.0, 20.0, 95.0]),
        "EMISSION_ANGLE": np.array([1.0, 2.0, 3.0]),
        "PHASE_ANGLE": np.array([41.0, 22.0, 98.0]),
        "CENTER_LATITUDE": np.array([10.0, 10.1, 10.2]),
        "CENTER_LONGITUDE": np.array([20.0, 20.0, 20.0]),
        "SOLAR_AZIMUTH_ANGLE": np.array([100.0, 101.0, 102.0]),
        "SPACECRAFT_AZIMUTH": np.array([5.0, 6.0, 7.0]),
    }
    return dict(wav=SP_WAV_NM, ref1=ref1, ref2=ref2, raw=raw, rad=rad, qa=qa, geo=geo)


def write_sp_product(dirpath, stem="SP_2C_02_00001_N001_E0010", arrays=None,
                     include_rad=True, include_raw=True) -> str:
    """Write a fake SP L2C .lbl/.spc pair; returns the label path."""
    a = arrays or default_sp_arrays()
    wav, ref1, ref2, raw, rad, qa, geo = (a[k] for k in ("wav", "ref1", "ref2", "raw", "rad", "qa", "geo"))
    nrows, nb = ref1.shape
    fmt = {("MSB_INTEGER", 4): ">i", ("IEEE_REAL", 8): ">d", ("IEEE_REAL", 4): ">f"}
    starts, s = [], 1
    for _, _, b in SP_COLUMNS:
        starts.append(s)
        s += b
    table = bytearray(nrows * SP_ROW_BYTES)
    for r in range(nrows):
        for (name, dtype, nbytes), st in zip(SP_COLUMNS, starts):
            off = r * SP_ROW_BYTES + st - 1
            if dtype == "CHARACTER":
                table[off:off + nbytes] = b"2008-01-01T0"[:nbytes]
            elif name == "SP_INDEX":
                struct.pack_into(">i", table, off, r + 1)
            else:
                struct.pack_into(fmt[(dtype, nbytes)], table, off, float(geo[name][r]))

    def u16(arr):
        return np.asarray(arr).astype(">u2").tobytes()

    grids = [("SP_SPECTRUM_WAV", np.round(wav * 10), 1, 0.1, "nm")]
    if include_raw:
        grids.append(("SP_SPECTRUM_RAW", raw, nrows, 1.0, "DN"))
    grids.append(("SP_SPECTRUM_REF2", np.round(ref2 * 1e4), nrows, 1e-4, "ND"))
    if include_rad:
        grids.append(("SP_SPECTRUM_RAD", np.round(rad * 100), nrows, 1e-2, "W/m**2/sr/um"))
    grids.append(("SP_SPECTRUM_REF1", np.round(ref1 * 1e4), nrows, 1e-4, "ND"))
    grids.append(("SP_SPECTRUM_QA", qa, nrows, 1.0, "ND"))

    blob = bytearray(table)
    pointers = {"ANCILLARY_AND_SUPPLEMENT_DATA": 1}
    for name, arr, nl, _, _ in grids:
        pointers[name] = len(blob) + 1          # 1-based byte pointer
        blob += u16(np.asarray(arr).reshape(nl, nb))

    spc_name = stem + ".spc"
    L = ["PDS_VERSION_ID                       = PDS3",
         "RECORD_TYPE                          = UNDEFINED",
         'FILE_NAME                            = "%s"' % spc_name,
         'DATA_SET_ID                          = "SLN-L-SP-4-LEVEL2C-V3.0"',
         'PRODUCT_ID                           = "%s"' % stem,
         '^ANCILLARY_AND_SUPPLEMENT_DATA       = ("%s", %d <BYTES>)' % (spc_name, pointers["ANCILLARY_AND_SUPPLEMENT_DATA"])]
    for name, *_ in grids:
        L.append('^%-32s = ("%s", %d <BYTES>)' % (name, spc_name, pointers[name]))
    L += ["OBJECT                               = ANCILLARY_AND_SUPPLEMENT_DATA",
          "  INTERCHANGE_FORMAT                 = BINARY",
          "  ROWS                               = %d" % nrows,
          "  COLUMNS                            = %d" % len(SP_COLUMNS),
          "  ROW_BYTES                          = %d" % SP_ROW_BYTES]
    for (name, dtype, nbytes), st in zip(SP_COLUMNS, starts):
        L += ["  OBJECT                             = COLUMN",
              "    NAME                             = %s" % name,
              "    DATA_TYPE                        = %s" % dtype,
              "    START_BYTE                       = %d" % st,
              "    BYTES                            = %d" % nbytes,
              '    UNIT                             = "deg"',
              "  END_OBJECT                         = COLUMN"]
    L.append("END_OBJECT                           = ANCILLARY_AND_SUPPLEMENT_DATA")
    for name, _, nl, sc, unit in grids:
        L += ["OBJECT                               = %s" % name,
              "  LINES                              = %d" % nl,
              "  LINE_SAMPLES                       = %d" % nb,
              "  SAMPLE_TYPE                        = MSB_UNSIGNED_INTEGER",
              "  SAMPLE_BITS                        = 16",
              '  UNIT                               = "%s"' % unit,
              "  SCALING_FACTOR                     = %g" % sc,
              "  OFFSET                             = 0.0",
              "END_OBJECT                           = %s" % name]
    L.append("END")
    lbl = os.path.join(dirpath, stem + ".lbl")
    with open(lbl, "w", encoding="latin-1", newline="\r\n") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(dirpath, spc_name), "wb") as f:
        f.write(bytes(blob))
    return lbl


@pytest.fixture
def sp_label(tmp_path):
    return write_sp_product(str(tmp_path))


# --------------------------------------------------------------------- IIRS
SPM_WIDTHS_AFTER_MARKER = [6, 4] + [4] * 7 + [20] * 3 + [12] * 3 + [16] * 4 + [1]
OAT_WIDTHS_AFTER_MARKER = ([6, 4] + [4] * 7 + [20] * 6 + [12] * 3 + [14] * 12 + [14] * 6
                           + [12, 12, 1, 9, 9, 9, 10, 5, 9, 9, 16, 16, 16])


def _fw(width: int, value) -> str:
    if isinstance(value, (int, np.integer)):
        s = str(int(value)).rjust(width)
    else:
        p = min(8, max(0, width - 7))
        s = "{:{w}.{p}f}".format(float(value), w=width, p=p)
    assert len(s) == width, (width, value, s)
    return s


def _utc_fields(t: np.datetime64) -> list[int]:
    s = str(np.datetime64(t, "ms"))              # YYYY-MM-DDThh:mm:ss.mmm
    return [int(s[0:4]), int(s[5:7]), int(s[8:10]), int(s[11:13]), int(s[14:16]),
            int(s[17:19]), int(s[20:23])]


def _record(widths, values, total_len) -> str:
    assert len(widths) == len(values)
    body = "ORBTATTD" + "".join(_fw(w, v) for w, v in zip(widths, values))
    assert len(body) <= total_len - 1
    return body.ljust(total_len - 1) + "\n"


def write_spm(path, times, sun_el, sun_az=90.0, phase=40.0, aspect=30.0) -> None:
    """Fixed-width SPM records with the F16.8 Sun-angle fields (249 B/record)."""
    sun_el = np.broadcast_to(np.asarray(sun_el, dtype=float), (len(times),))
    with open(path, "w", newline="") as f:
        for k, t in enumerate(times):
            vals = ([k + 1, 249] + _utc_fields(t) + [1000.0 * (k + 1), 2000.0, 3000.0]
                    + [1.0, 2.0, 3.0] + [phase, aspect, sun_az, float(sun_el[k])] + [0])
            f.write(_record(SPM_WIDTHS_AFTER_MARKER, vals, 249))


def write_oat(path, times, lat, lon, orbit=1234, alt_km=100.0) -> None:
    """Fixed-width OAT records (628 B/record incl. the spare)."""
    lat = np.broadcast_to(np.asarray(lat, dtype=float), (len(times),))
    lon = np.broadcast_to(np.asarray(lon, dtype=float), (len(times),))
    with open(path, "w", newline="") as f:
        for k, t in enumerate(times):
            vals = ([k + 1, 628] + _utc_fields(t)
                    + [1.0, 2.0, 3.0, 1800.0, 10.0, 20.0]          # moon_pos, sat_pos
                    + [1.5, 0.1, 0.2]                               # sat_vel
                    + [0.0, 0.0, 0.0, 1.0] * 3                      # three quaternions
                    + [lat[k] - 0.01, lon[k] + 0.01, 120.0, 45.0, lat[k], lon[k]]
                    + [alt_km, 0.1, 0, 2.0, 40.0, 0.5, 100.5, orbit, 50.0, 0.2,
                       0.01, 0.02, 0.03])
            f.write(_record(OAT_WIDTHS_AFTER_MARKER, vals, 628))


def iirs_label_xml(product_id, shape, dtype_name, level, wl, bw, t0, t1,
                   exposure="e1", gain="g2", limb="Ascending") -> str:
    bands = "".join(
        "<Band_Bin><band_number>%d</band_number><center_wavelength unit=\"nm\">%.7f"
        "</center_wavelength><band_width unit=\"nm\">%.3f</band_width></Band_Bin>\n"
        % (i + 1, w, b) for i, (w, b) in enumerate(zip(wl, bw)))
    return """<?xml version="1.0" encoding="UTF-8"?>
<Product_Observational xmlns="http://pds.nasa.gov/pds4/pds/v1"
    xmlns:isda="https://isda.issdc.gov.in/pds4/isda/v1">
  <Identification_Area>
    <logical_identifier>urn:isro:isda:ch2_iir:%(pid)s</logical_identifier>
    <version_id>1.0</version_id>
    <title>synthetic IIRS test product</title>
    <information_model_version>1.11.0.0</information_model_version>
    <product_class>Product_Observational</product_class>
  </Identification_Area>
  <Observation_Area>
    <Time_Coordinates>
      <start_date_time>%(t0)s</start_date_time>
      <stop_date_time>%(t1)s</stop_date_time>
    </Time_Coordinates>
    <Primary_Result_Summary>
      <purpose>Science</purpose>
      <processing_level>%(level)s</processing_level>
    </Primary_Result_Summary>
    <Mission_Area>
      <isda:Mission_Specific_Parameters>
        <isda:gain>%(gain)s</isda:gain>
        <isda:exposure>%(exposure)s</isda:exposure>
        <isda:orbit_limb_direction>%(limb)s</isda:orbit_limb_direction>
        <isda:line_exposure_duration unit="ms">53.060</isda:line_exposure_duration>
        <isda:pixel_resolution unit="m">80</isda:pixel_resolution>
        <isda:spacecraft_altitude unit="km">100</isda:spacecraft_altitude>
      </isda:Mission_Specific_Parameters>
    </Mission_Area>
    <Discipline_Area>
      <Band_Bin_Set>
%(bands)s      </Band_Bin_Set>
    </Discipline_Area>
  </Observation_Area>
  <File_Area_Observational>
    <File><file_name>%(pid)s.qub</file_name></File>
    <Array_3D_Spectrum>
      <local_identifier>cube</local_identifier>
      <offset unit="byte">0</offset>
      <axes>3</axes>
      <axis_index_order>Last Index Fastest</axis_index_order>
      <Element_Array><data_type>%(dtype)s</data_type></Element_Array>
      <Axis_Array><axis_name>Band</axis_name><elements>%(nb)d</elements><sequence_number>1</sequence_number></Axis_Array>
      <Axis_Array><axis_name>Line</axis_name><elements>%(nl)d</elements><sequence_number>2</sequence_number></Axis_Array>
      <Axis_Array><axis_name>Sample</axis_name><elements>%(ns)d</elements><sequence_number>3</sequence_number></Axis_Array>
    </Array_3D_Spectrum>
  </File_Area_Observational>
</Product_Observational>
""" % dict(pid=product_id, t0=t0, t1=t1, level=level, gain=gain, exposure=exposure,
           limb=limb, bands=bands, dtype=dtype_name, nb=shape[0], nl=shape[1], ns=shape[2])


IIRS_T0 = np.datetime64("2021-01-23T09:24:03.857", "ms")
IIRS_STAMP = "20210123T0924038577"


def write_iirs_product(root, cube, *, typ="nci", level="Calibrated",
                       dtype_name="IEEE754LSBSingle", wl=IIRS_WL, bw=IIRS_BW,
                       exposure="e1", gain="g2", limb="Ascending", t0=IIRS_T0,
                       line_period_ms=53.06, sun_el=50.0, with_spm=True, with_oat=True,
                       with_geometry=False, geometry=None) -> str:
    """Write a fake IIRS PDS4 bundle under ``root``; returns the product id.

    ``cube`` is ``(nb, nl, ns)`` of *stored* values (DN or radiance x 1000).
    """
    cube = np.asarray(cube)
    nb, nl, ns = cube.shape
    pid = "ch2_iir_%s_%s_d_img_d32" % (typ, IIRS_STAMP)
    date = "20210123"
    sub = {"nri": "raw", "nci": "calibrated", "ndi": "derived"}[typ]
    data_dir = os.path.join(root, "data", sub, date)
    os.makedirs(data_dir, exist_ok=True)
    t1 = t0 + np.timedelta64(int(round((nl - 1) * line_period_ms)), "ms")
    iso = lambda t: str(np.datetime64(t, "ms")) + "Z"  # noqa: E731
    with open(os.path.join(data_dir, pid + ".xml"), "w", encoding="utf-8") as f:
        f.write(iirs_label_xml(pid, (nb, nl, ns), dtype_name, level, wl, bw, iso(t0), iso(t1),
                               exposure=exposure, gain=gain, limb=limb))
    np_dtype = {"UnsignedLSB2": "<u2", "IEEE754LSBSingle": "<f4"}[dtype_name]
    cube.astype(np_dtype).tofile(os.path.join(data_dir, pid + ".qub"))
    with open(os.path.join(data_dir, pid + ".hdr"), "w") as f:
        f.write("ENVI\nsamples = %d\nlines = %d\nbands = %d\ninterleave = bsq\n" % (ns, nl, nb))
    if with_spm or with_oat:
        misc_dir = os.path.join(root, "miscellaneous", sub, date)
        os.makedirs(misc_dir, exist_ok=True)
        times = [t0 - np.timedelta64(1024, "ms") + np.timedelta64(512 * k, "ms")
                 for k in range(int((t1 - t0) / np.timedelta64(512, "ms")) + 6)]
        if with_spm:
            write_spm(os.path.join(misc_dir, "ch2_iir_spm_%s_d_img_d32.spm" % IIRS_STAMP),
                      times, sun_el)
        if with_oat:
            lat = 10.0 + 0.001 * np.arange(len(times))
            lon = 179.9 + 0.05 * np.arange(len(times))       # crosses 180 -> tests unwrap
            write_oat(os.path.join(misc_dir, "ch2_iir_oat_%s_d_img_d32.oat" % IIRS_STAMP),
                      times, lat, lon)
    if with_geometry:
        geo_dir = os.path.join(root, "geometry", sub, date)
        os.makedirs(geo_dir, exist_ok=True)
        g = geometry if geometry is not None else default_geometry_grid()
        with open(os.path.join(geo_dir, pid.replace("_d_img_", "_g_grd_") + ".csv"), "w") as f:
            f.write("Longitude,Latitude,Pixel,Scan\n")
            for lon, lat, px, sc in g:
                f.write("%.6f,%.6f,%d,%d\n" % (lon, lat, px, sc))
    return pid


def default_geometry_grid():
    """Lattice Pixel in {0, 2, 4}, Scan in {0, 2, 4}; longitude crosses 360 deg."""
    rows = []
    for sc in (0, 2, 4):
        for px in (0, 2, 4):
            rows.append((355.0 + 4.0 * px, 10.0 + 0.5 * sc, px, sc))   # lon 355, 363->3, 371->11
    return [((lon % 360.0), lat, px, sc) for lon, lat, px, sc in rows]


@pytest.fixture
def iirs_l0_bundle(tmp_path):
    """Tiny L0 (nri) bundle: 256 bands x 5 lines x 3 samples of uint16 DN."""
    nb, nl, ns = 256, 5, 3
    cube = (np.arange(nb)[:, None, None] * 10 + np.arange(nl)[None, :, None] * 2
            + np.arange(ns)[None, None, :]).astype("<u2")
    root = str(tmp_path / "l0")
    pid = write_iirs_product(root, cube, typ="nri", level="Raw", dtype_name="UnsignedLSB2",
                             with_geometry=False)
    return root, pid, cube
