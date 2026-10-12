# lunar-ir-data-tools

Small, dependency-light Python readers and correction routines for lunar
infrared spectrometer data:

| Module | Reads | Archive |
|---|---|---|
| `lunar_ir_tools.kaguya_sp` | Kaguya/SELENE Spectral Profiler (SP) Level 2C point spectra, PDS3 `.lbl` + big-endian `.spc` (data set `SLN-L-SP-4-LEVEL2C-V3.0`) | SELENE Data Archive, JAXA DARTS |
| `lunar_ir_tools.kaguya_sp_search` | SP footprint search (L2C or L2B1) through the JAXA DARTS PDAP service | JAXA DARTS |
| `lunar_ir_tools.kaguya_mi` | Kaguya Multiband Imager (MI) mosaics in geographic GeoTIFF form, e.g. the MI mineral maps of Lemelin et al. (2019) | USGS Astrogeology / SELENE Data Archive |
| `lunar_ir_tools.iirs` | Chandrayaan-2 Imaging Infrared Spectrometer (IIRS) Level 0 (`nri`), Level 1 (`nci`) and Level 2 (`ndi`) PDS4 bundles | ISSDC PRADAN |
| `lunar_ir_tools.iirs_reflectance` | IIRS L1 radiance to reflectance with single-temperature thermal removal (Verma et al. 2022) | - |
| `lunar_ir_tools.iirs_thermal` | IIRS L1 radiance to reflectance with a two-temperature (anisothermal) model | - |

Everything is plain numpy/scipy; `requests` is used only by the PDAP search
and `rasterio` only by the MI sampler. The package was consolidated from
research scripts; the algorithms are unchanged from the versions used in that
research, and the assumptions they make are listed below under
[Known limitations and assumptions](#known-limitations-and-assumptions).

## Install

```bash
pip install git+https://github.com/euchas/lunar-ir-data-tools
# optional extras
pip install "lunar-ir-data-tools[mi] @ git+https://github.com/euchas/lunar-ir-data-tools"   # rasterio for kaguya_mi
pip install "lunar-ir-data-tools[plot] @ git+https://github.com/euchas/lunar-ir-data-tools" # matplotlib for the example quick-look
```

Requires Python 3.10 or newer. For development: `git clone`, then
`pip install -e .[test,mi,plot]` and `pytest -q`.

The command line entry point is `lunar-ir-tools` with the sub-commands
`sp-read`, `sp-search`, `iirs-read`, `iirs-reflectance`, `iirs-thermal` and
`mi-sample` (`lunar-ir-tools <sub-command> -h` lists the options).

## Quickstart

### Kaguya SP Level 2C reader

Product: SP L2C point spectra, data set `SLN-L-SP-4-LEVEL2C-V3.0`, delivered as
a PDS3 label plus binary pair such as `SP_2C_02_03346_S087_E0170.lbl` /
`.spc` (one ~30 s track segment, N footprints x 296 bands, 512-2587 nm).
Download from the SELENE Data Archive at JAXA DARTS,
https://darts.isas.jaxa.jp/planet/pdap/selene/ (product search and direct
file access; the `sp-search` tool below returns direct download URLs).

```bash
lunar-ir-tools sp-read SP_2C_02_03346_S087_E0170.lbl --near -86.5,17.0 --csv footprint.csv
```

```python
from lunar_ir_tools import load_sp, select_footprint, simple_band_depth

p = load_sp("SP_2C_02_03346_S087_E0170.lbl")   # .spc found next to the label
p.wav            # (296,) band centres, nm
p.ref1, p.ref2   # (N, 296) standard I/F and corrected reflectance
p.rad, p.raw     # (N, 296) radiance (label UNIT) and raw DN, when present
p.geo["INCIDENCE_ANGLE"], p.geo["CENTER_LATITUDE"]   # (N,) degrees
i = select_footprint(p, near=(-86.5, 17.0))           # sunlit, nearest footprint
simple_band_depth(p.wav, p.ref1[i], 850, 750, 950)   # 1 - R850 / mean(R750, R950)
```

All byte offsets, dimensions and scaling factors are taken from the label.

### SP footprint search (JAXA DARTS PDAP)

Finds SP products whose footprint centre lies in a latitude/longitude box and
writes a CSV manifest with centre coordinates, times and the `.spc` / `.lbl`
download URLs.

```bash
lunar-ir-tools sp-search --lat 83/87 --lon 190/215 --out sp_manifest.csv
lunar-ir-tools sp-search --lat -10/-5 --lon 30/35 --dataset SLN-L-SP-3-LEVEL2B1-V2.0 --out radiance.csv
```

```python
from lunar_ir_tools import search_footprints, write_manifest

rows = search_footprints(83, 87, 190, 215)        # longitude in degrees east, 0-360
write_manifest(rows, "sp_manifest.csv")
rows[0]["PRODUCT.DATA_ACCESS_REFERENCE"]          # direct .spc URL
```

Please read before using it at scale:

* The PDAP endpoint
  `https://darts.isas.jaxa.jp/app/pdap/selene/1.0/search_exec.php` is **not
  formally documented by JAXA**; its parameters and limits were observed in
  July 2026 and may change without notice.
* The client identifies itself with the User-Agent
  `lunar-ir-data-tools/0.1.0 (+https://github.com/euchas/lunar-ir-data-tools)`,
  waits at least 1 s between requests and retries with exponential back-off
  only on HTTP 5xx responses, timeouts and connection errors.
* The server returns at most 50 records per request and caps a query at
  1000 records; larger boxes are tiled automatically. **Avoid bulk enumeration
  of the whole archive**: query the regions you need and keep the manifest.

### Kaguya MI mosaic sampler

Products: any geographic (longitude/latitude) GeoTIFF mosaic derived from the
Kaguya MI, in particular the MI mineral and maturity maps of Lemelin et al.
(2019) (`Lunar_Kaguya_MIMap_MineralDeconv_<Parameter>_50N50S.tif` for
OpticalMaturityIndex, FeOWeightPercent, ClinopyroxenePercent,
OrthopyroxenePercent, OlivinePercent, PlagioclasePercent), available from USGS
Astrogeology (https://astrogeology.usgs.gov/search, search "Kaguya MI") and
readable directly from the USGS PDS Astrogeology S3 bucket via
`mineral_map_url()`. MI MAP reflectance mosaics from the SELENE Data Archive
work once converted to GeoTIFF. Requires the `mi` extra (rasterio).

```bash
lunar-ir-tools mi-sample feo --points 23.5,-12.0 24.0,-12.5          # remote mosaic by key
lunar-ir-tools mi-sample FeO_local.tif --polygons craters.geojson --id-property name --out feo.csv
```

```python
from lunar_ir_tools import mineral_map_url, sample_points, sample_polygons, read_geojson_polygons

url = mineral_map_url("omat")                        # /vsicurl/https://... (OpticalMaturityIndex)
sample_points(url, lons=[23.5, 24.0], lats=[-12.0, -12.5])
stats = sample_polygons(url, read_geojson_polygons("craters.geojson"), band_height=2500)
stats["crater_1"].median, stats["crater_1"].count
```

The mosaics are tens of GB and untiled, so `sample_polygons` reads each
mosaic once in full-width latitude bands and samples every polygon whose
centroid falls in a band (polygons are clipped to their centroid's band; use
`band_height` larger than your tallest polygon).

### Chandrayaan-2 IIRS reader

Products: IIRS PDS4 bundles from the ISRO Science Data Archive / PRADAN portal
of ISSDC, https://pradan.issdc.gov.in/ (free registration; browse
Chandrayaan-2 > IIRS). Product types are `ch2_iir_nri_*` (Level 0 raw DN),
`ch2_iir_nci_*` (Level 1 calibrated radiance) and `ch2_iir_ndi_*` (Level 2
reflectance). Unpack a bundle and point the reader at the directory (or at the
`.qub` / `.xml` file).

```bash
lunar-ir-tools iirs-read ch2_iir_nci_20210123T0924038577_d_img_d32/
lunar-ir-tools iirs-read ch2_iir_nci_20210123T0924038577_d_img_d32/ --spectrum 4000 125 --csv px.csv
```

```python
from lunar_ir_tools import IirsProduct, RAD_SCALE_MW

p = IirsProduct("ch2_iir_nci_20210123T0924038577_d_img_d32")
p.level, p.nbands, p.nlines, p.nsamples      # 'Calibrated', 256, ~13000, 250
p.wl                                         # (256,) band centres, nm
p.cube                                       # memmap (bands, lines, samples), BSQ
p.spectrum(4000, 125) * RAD_SCALE_MW         # L1: mW cm-2 sr-1 um-1
p.good_bands()                               # SIS App. H bad bands + exposure saturation
p.line_times(), p.line_incidence()           # per line, from the label and the SPM file
p.line_latlon()                              # boresight lat/lon per line from the OAT file
p.pixel_latlon(lines, samples)               # L1 only: interpolated geometry grid
```

### IIRS L1 radiance to reflectance (single temperature)

Implements the Verma, Chauhan and Chauhan (2022) scheme: brightness
temperature from 4.5-4.9 um with emissivity 0.95, subtraction of the thermal
term, then I/F with the solar flux, the Sun-Moon distance and the per-line
incidence angle. Needs an L1 (`nci`) product with exposure `e1` and a
solar-flux table (see [Solar-flux table](#solar-flux-table)).

```bash
lunar-ir-tools iirs-reflectance ch2_iir_nci_.../ --lines 4000 5000 --solar-flux iirs_solar_flux.txt --out refl.npz
```

```python
from lunar_ir_tools import IirsProduct, reflectance_window

p = IirsProduct("ch2_iir_nci_20210123T0924038577_d_img_d32")
refl, aux = reflectance_window(p, 4000, 5000, "iirs_solar_flux.txt")
refl.shape, aux["T"].shape      # (256, 1000, 250) reflectance, (1000, 250) K
```

### IIRS two-temperature thermal removal

A two-component emission model (hot fraction `f` at `T0 + dT`, the rest at
`T0`) that removes the spurious continuum upturn a single temperature leaves
between 3.5 and 4.2 um. `(f, dT)` are fitted once per strip on a pixel sample;
`T0` is solved per pixel from the 4.4-4.9 um radiance. Two wavelength windows
(3300-3520 and 3850-4050 nm) are never used as anchors so that absorption
features there are not fitted away.

```bash
lunar-ir-tools iirs-thermal ch2_iir_nci_.../ --lines 4000 5000 --solar-flux iirs_solar_flux.txt --out refl_aniso.npz
```

```python
from lunar_ir_tools import IirsProduct, fit_strip_anisothermality, refl_chunk_aniso

p = IirsProduct("ch2_iir_nci_20210123T0924038577_d_img_d32")
f, dT, iso_err, aniso_err = fit_strip_anisothermality(p, "iirs_solar_flux.txt")
R, aux = refl_chunk_aniso(p, 4000, 5000, f, dT, "iirs_solar_flux.txt")
aux["reliable"]                 # per pixel: far-anchor bands finite (no reflected/emitted null)
```

The correction is meant for warm scenes (roughly 270-330 K). Where the
reflected and emitted radiances cross (|S - E| < 0.2 S, typically 220-260 K
at moderate incidence) the reflectance is ill-posed and set to NaN; `aux["reliable"]`
flags pixels whose far-anchor bands survived.

## Solar-flux table

The reflectance steps need the solar spectral irradiance at 1 AU on the 256
IIRS bands, in mW cm-2 um-1. **No table is shipped with this package**: the
file used in the original research carried no header, source or licence
information, so its provenance could not be established and it is not
redistributed here. Provide your own as a plain-text file with one band per
row and two whitespace-separated columns (`#` comments allowed):

```
# wavelength_nm   irradiance_mW_cm-2_um-1 (1 AU)
712.3339889       136.12
729.1863510       129.88
...
```

The wavelength column must match the product's band centres (`IirsProduct.wl`)
to within 1 nm. Public sources:

* build one from a standard reference spectrum, e.g. ASTM E-490 (2000),
  Gueymard (2004) or Thuillier et al. (2003), with
  `lunar_ir_tools.resample_solar_flux(ref_wl_nm, ref_irradiance, p.wl, p.label["bw"])`
  (boxcar average per band; convert W m-2 nm-1 to mW cm-2 um-1 by multiplying
  by 100);
* the IIRS team's CH2IIRS QGIS plugin (https://github.com/prabhakaralok/CH2IIRS)
  distributes a 256-row `Solar flux.txt` on the IIRS grid; check its licence
  before use.

## Known limitations and assumptions

| Assumption | Value | Where |
|---|---|---|
| Surface emissivity for the thermal term | 0.95 | `src/lunar_ir_tools/iirs_reflectance.py`: `EMISSIVITY`, used in `brightness_temperature()` and `reflectance_window()` (an `emissivity` argument overrides it) |
| Brightness-temperature bands | 4500-4900 nm, temperatures outside 100-450 K ignored, OSF bands included | `iirs_reflectance.py`: `T_EST_RANGE_NM`, `T_VALID_RANGE_K`, `reflectance_window()` |
| Sun-Moon distance | Earth-orbit formula `1.00014 - 0.01671 cos(2 pi (doy - 3.5) / 365.25)` AU from the fractional day of year; the Moon's +/-0.0026 AU excursion is ignored | `iirs_reflectance.py`: `sun_moon_distance_au()` (a `d_au` argument overrides it) |
| Solar incidence | flat-surface incidence `90 - Sun elevation` from the SPM file, interpolated to line times, no topography; cos(i) clipped at 0.05 | `src/lunar_ir_tools/iirs.py`: `IirsProduct.line_incidence()`; `iirs_reflectance.py`: `MIN_COS_INCIDENCE` (an `incidence_deg` argument overrides it) |
| Two-temperature anchor windows | T0 anchor 4400-4900 nm (`FAR_ANCHOR`); curvature anchors 3600-3850 and 4050-4350 nm (`MID_ANCHOR`); nominal reflectance from 1700-2100 nm (`REFL_NOM`); protected windows 3300-3520 and 3850-4050 nm (`EXCLUDED_WINDOWS`) never used as anchors | `src/lunar_ir_tools/iirs_thermal.py`: module constants, `fit_scene_anisothermality()`, `_solve_T0()`, `_nominal_R()` |
| Two-temperature fit and guards | grid f 0.02-0.40, dT 20-140 K plus local refinement; `(f, dT)` fixed per strip, T0 per pixel clipped to 140-410 K; isothermal fallback unless the curvature drops below 90 %; R set to NaN where \|S - E\| < 0.2 S or R outside (-0.05, 1.5); `(f, dT, T0)` are partly degenerate, only the reflectance is well constrained | `iirs_thermal.py`: `fit_scene_anisothermality()`, `_solve_T0()`, `_solve_R()`, `CROSSOVER_GUARD` |
| L1 radiance unit | stored float x 1e-3 = mW cm-2 sr-1 um-1 (SIS App. J) | `iirs.py`: `RAD_SCALE_MW` |
| Bad bands | SIS App. H for the nominal E1/G2 mode: bands 1-6 and 252-256 invalid, OSF junctions 29-34, 69-75, 162-171; e2/e3/e4 saturated from band 162/147/38 | `iirs.py`: `INVALID_BANDS_1B`, `OSF_BANDS_1B`, `EXPOSURE_SAT_FROM_1B`, `IirsProduct.good_bands()` |
| SPM column widths override the official readme | the four Sun-angle fields (phase, Sun aspect, Sun azimuth, Sun elevation) are read as 16-character F16.8 fields; the bundle `readme.txt` documents F9.3 but the distributed files use 16 characters (verified on real SPM records) | `iirs.py`: `SPM_WIDTHS`, `IirsProduct.spm()` |
| OAT column widths | 628-byte records parsed by fixed column widths following the bundle readme; whitespace splitting is unsafe because fields can abut | `iirs.py`: `OAT_WIDTHS`, `IirsProduct.oat()` |
| Line timing and geolocation | line times linear between the label start and stop; `line_latlon()` is the OAT boresight point per line (first order, no cross-track solution); `pixel_latlon()` interpolates the sparse L1 geometry grid on the unit sphere | `iirs.py`: `line_times()`, `line_latlon()`, `grid_latlon()` |
| Exposure | thermal removal only for exposure `e1` products (others saturate the 4.5-4.9 um bands) | `iirs_reflectance.py`: `reflectance_window()`; `iirs_thermal.py`: `fit_strip_anisothermality()` |
| SP products | 16-bit big-endian samples only; the ancillary table is decoded from the label's COLUMN objects; `select_footprint()` calls a footprint sunlit when incidence < 90 deg and valid when the 600-900 nm median REF1 exceeds 0.005 | `src/lunar_ir_tools/kaguya_sp.py`: `load_sp()`, `select_footprint()` |
| MI mosaics | geographic (longitude/latitude) rasters only; polygons are clipped to the latitude band of their centroid | `src/lunar_ir_tools/kaguya_mi.py`: `sample_polygons()` |
| PDAP service | undocumented endpoint, observed limits (50 per page, 1000 per query) hard-coded; tiling stops at `max_depth` with a warning | `src/lunar_ir_tools/kaguya_sp_search.py` |

## Tests

```bash
pip install -e .[test]
pytest -q
```

The tests generate their own data: a tiny SP `.lbl`/`.spc` pair, a tiny IIRS
PDS4 bundle (label, cube, SPM/OAT records, geometry grid), synthetic radiance
scenes for the reflectance and thermal steps, and a small GeoTIFF for the MI
sampler (skipped when rasterio is not installed). No network access and no
mission data are needed.

## Citing

If this software is useful in your work, please cite it (see `CITATION.cff`):

> Valantinas, A. (2026). lunar-ir-data-tools: readers and corrections for
> Kaguya SP, Kaguya MI and Chandrayaan-2 IIRS data (version 0.1.0).
> https://github.com/euchas/lunar-ir-data-tools

and the data and methods it builds on:

* Kaguya SP: Matsunaga, T. et al. (2008), Geophysical Research Letters 35,
  L23201, https://doi.org/10.1029/2008GL035868; the SP L2C products and their
  photometric correction, Yokota, Y. et al. (2011), Icarus 215, 639-660,
  https://doi.org/10.1016/j.icarus.2011.07.028; data from the SELENE Data
  Archive, JAXA DARTS.
* Kaguya MI: Ohtake, M. et al. (2008), Earth, Planets and Space 60, 257-264,
  https://doi.org/10.1186/BF03352789; mineral maps, Lemelin, M. et al. (2019),
  Planetary and Space Science 165, 230-243, https://doi.org/10.1016/j.pss.2018.10.003.
* Chandrayaan-2 IIRS: Chowdhury, A. R. et al. (2020), Current Science 118(3),
  368-375, https://doi.org/10.18520/cs/v118/i3/368-375; thermal correction,
  Verma, P. K., Chauhan, M. and Chauhan, P. (2022), Icarus 383, 115075,
  https://doi.org/10.1016/j.icarus.2022.115075; data from ISSDC PRADAN.

## Acknowledgement

Much of the code was written with an AI coding assistant (Claude, Anthropic)
under the direction of the author, who reviewed the code and results.

## License

MIT, see `LICENSE`.
