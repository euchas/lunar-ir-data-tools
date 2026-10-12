"""lunar-ir-data-tools: readers and corrections for lunar infrared spectrometer data.

* :mod:`~lunar_ir_tools.kaguya_sp` - Kaguya/SELENE Spectral Profiler L2C reader
  (PDS3 ``.lbl`` + ``.spc``, SELENE Data Archive at JAXA DARTS)
* :mod:`~lunar_ir_tools.kaguya_sp_search` - SP footprint search via the JAXA
  DARTS PDAP service
* :mod:`~lunar_ir_tools.kaguya_mi` - Kaguya Multiband Imager mosaic sampler
  (rasterio; optional ``mi`` extra)
* :mod:`~lunar_ir_tools.iirs` - Chandrayaan-2 IIRS L0/L1/L2 PDS4 reader
  (ISSDC PRADAN bundles)
* :mod:`~lunar_ir_tools.iirs_reflectance` - IIRS L1 radiance to reflectance,
  single-temperature thermal removal
* :mod:`~lunar_ir_tools.iirs_thermal` - two-temperature thermal removal
* :mod:`~lunar_ir_tools.cli` - the ``lunar-ir-tools`` command line
"""
from ._version import __version__
from .kaguya_sp import (SpProduct, load_sp, parse_label, nearest_band, select_footprint,
                        simple_band_depth)
from .kaguya_sp_search import (PdapClient, parse_votable, search_footprints, write_manifest,
                               DATASET_L2C, DATASET_L2B1)
from .kaguya_mi import (mineral_map_url, read_geojson_polygons, sample_points, sample_polygons,
                        PolygonStats, MINERAL_MAP_PARAMETERS)
from .iirs import IirsProduct, read_label, grid_latlon, RAD_SCALE_MW
from .iirs_reflectance import (planck_mwcm, brightness_temperature, sun_moon_distance_au,
                               load_solar_flux, resample_solar_flux, reflectance_window,
                               EMISSIVITY)
from .iirs_thermal import (fit_scene_anisothermality, fit_strip_anisothermality,
                           refl_chunk_aniso, refl_window_aniso, solve_reflectance,
                           EXCLUDED_WINDOWS, FAR_ANCHOR, MID_ANCHOR)

__all__ = [
    "__version__",
    # Kaguya SP
    "SpProduct", "load_sp", "parse_label", "nearest_band", "select_footprint",
    "simple_band_depth",
    # SP search
    "PdapClient", "parse_votable", "search_footprints", "write_manifest",
    "DATASET_L2C", "DATASET_L2B1",
    # Kaguya MI
    "mineral_map_url", "read_geojson_polygons", "sample_points", "sample_polygons",
    "PolygonStats", "MINERAL_MAP_PARAMETERS",
    # IIRS
    "IirsProduct", "read_label", "grid_latlon", "RAD_SCALE_MW",
    "planck_mwcm", "brightness_temperature", "sun_moon_distance_au",
    "load_solar_flux", "resample_solar_flux", "reflectance_window", "EMISSIVITY",
    "fit_scene_anisothermality", "fit_strip_anisothermality", "refl_chunk_aniso",
    "refl_window_aniso", "solve_reflectance", "EXCLUDED_WINDOWS", "FAR_ANCHOR",
    "MID_ANCHOR",
]
