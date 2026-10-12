# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-12

First public release, consolidating research scripts into one package.

### Added

- `lunar_ir_tools.kaguya_sp`: Kaguya/SELENE Spectral Profiler L2C reader
  (PDS3 `.lbl` + big-endian `.spc`), footprint selection and a simple
  band-depth helper.
- `lunar_ir_tools.kaguya_sp_search`: SP footprint search through the JAXA
  DARTS PDAP service with https, a descriptive User-Agent, a 1 s minimum
  request interval, exponential back-off on 5xx/timeouts and recursive tiling
  of the 1000-record cap.
- `lunar_ir_tools.kaguya_mi`: Kaguya Multiband Imager mosaic sampler
  (points and GeoJSON polygons, latitude-band reads; requires rasterio).
- `lunar_ir_tools.iirs`: Chandrayaan-2 IIRS L0/L1/L2 PDS4 reader (label,
  memmapped cube, bad-band masks, geometry grid, OAT/SPM parsers, line times,
  first-order geolocation).
- `lunar_ir_tools.iirs_reflectance`: L1 radiance to reflectance with
  single-temperature thermal removal after Verma et al. (2022); the solar-flux
  table is a required user input (see README).
- `lunar_ir_tools.iirs_thermal`: two-temperature (anisothermal) thermal
  removal with protected feature windows.
- `lunar-ir-tools` command line with `sp-read`, `sp-search`, `iirs-read`,
  `iirs-reflectance`, `iirs-thermal` and `mi-sample`.
- pytest suite on synthetic data (no network, no mission data required).
