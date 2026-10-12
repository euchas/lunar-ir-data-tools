"""Kaguya (SELENE) Multiband Imager (MI) mosaic sampler.

Samples points and polygons from large geographic (longitude/latitude)
GeoTIFF mosaics derived from the Kaguya MI, for example:

* the Kaguya MI mineral and maturity maps of Lemelin et al. (2019; FeO wt%,
  optical maturity OMAT, and clinopyroxene / orthopyroxene / olivine /
  plagioclase abundances, 50N-50S) distributed by USGS Astrogeology
  (https://astrogeology.usgs.gov/search, "Kaguya MI") and mirrored on the
  USGS PDS Astrogeology S3 bucket, see :func:`mineral_map_url`;
* MI MAP reflectance mosaics from the SELENE Data Archive at JAXA DARTS
  (https://darts.isas.jaxa.jp/planet/pdap/selene/), once converted to GeoTIFF.

Requires ``rasterio`` (install the ``mi`` extra). rasterio is imported lazily
so the rest of the package works without it.

The mineral-map mosaics are tens of GB, untiled and stored one row per block,
so random windowed reads over HTTP are slow. :func:`sample_polygons`
therefore reads each mosaic once in full-width latitude bands of
``band_height`` rows and samples every polygon whose centroid falls inside the
band. Polygons that straddle a band boundary are clipped to the band of their
centroid, so choose ``band_height`` larger than the tallest polygon.

Coordinates: longitudes in degrees (either -180..180 or 0..360; they are
wrapped to the mosaic's convention automatically), latitudes in degrees.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "MINERAL_MAP_BASE_URL",
    "MINERAL_MAP_PARAMETERS",
    "DEFAULT_GDAL_ENV",
    "PolygonStats",
    "mineral_map_url",
    "read_geojson_polygons",
    "sample_points",
    "sample_polygons",
]

#: Base URL of the Lemelin et al. (2019) Kaguya MI mineral maps on the USGS
#: PDS Astrogeology S3 bucket; the file name is
#: ``Lunar_Kaguya_MIMap_MineralDeconv_<Parameter>_50N50S.tif``.
MINERAL_MAP_BASE_URL = ("https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/"
                        "Lunar_MI_mineral_maps/Lunar_Kaguya_MIMap_MineralDeconv_")
MINERAL_MAP_SUFFIX = "_50N50S.tif"

#: Short key -> parameter name in the mineral-map file names.
MINERAL_MAP_PARAMETERS = {
    "omat": "OpticalMaturityIndex",
    "feo": "FeOWeightPercent",
    "cpx": "ClinopyroxenePercent",
    "opx": "OrthopyroxenePercent",
    "ol": "OlivinePercent",
    "plag": "PlagioclasePercent",
}

#: GDAL settings that make /vsicurl reads of the big mosaics tolerable.
DEFAULT_GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_MULTIRANGE="YES",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    VSI_CACHE="TRUE",
    VSI_CACHE_SIZE="2000000000",
    GDAL_HTTP_TIMEOUT="300",
    GDAL_HTTP_MAX_RETRY="3",
    GDAL_HTTP_RETRY_DELAY="3",
    CPL_VSIL_CURL_USE_HEAD="YES",
)


@dataclass
class PolygonStats:
    """Statistics of the valid mosaic pixels inside one polygon."""

    count: int
    median: float
    p25: float
    p75: float
    mean: float
    std: float

    def as_tuple(self) -> tuple:
        return (self.count, self.median, self.p25, self.p75, self.mean, self.std)


def mineral_map_url(parameter: str, vsicurl: bool = True) -> str:
    """URL of one Lemelin et al. (2019) MI mineral-map mosaic.

    Parameters
    ----------
    parameter
        A key of :data:`MINERAL_MAP_PARAMETERS` (``omat``, ``feo``, ``cpx``,
        ``opx``, ``ol``, ``plag``) or a full parameter name such as
        ``FeOWeightPercent``.
    vsicurl
        Prefix with ``/vsicurl/`` so rasterio/GDAL can read it remotely.
    """
    name = MINERAL_MAP_PARAMETERS.get(parameter.lower(), parameter)
    url = MINERAL_MAP_BASE_URL + name + MINERAL_MAP_SUFFIX
    return "/vsicurl/" + url if vsicurl else url


def _rasterio():
    try:
        import rasterio  # noqa: F401
        import rasterio.features  # noqa: F401
        from rasterio.windows import Window  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without rasterio
        raise ImportError("kaguya_mi needs rasterio: pip install 'lunar-ir-data-tools[mi]'") from exc
    return rasterio


def _open_path(mosaic: str) -> str:
    mosaic = os.fspath(mosaic)
    if mosaic.startswith(("http://", "https://")):
        return "/vsicurl/" + mosaic
    return mosaic


def _env_for(mosaic: str, env: dict | None) -> dict:
    if env is not None:
        return env
    return dict(DEFAULT_GDAL_ENV) if "/vsicurl/" in mosaic or mosaic.startswith("http") else {}


def _check_geographic(src) -> None:
    crs = src.crs
    if crs is not None and crs.is_projected:
        raise ValueError("expected a geographic (longitude/latitude, degrees) mosaic; "
                         "got a projected CRS: %s" % crs)


def _geom_coords(geom: dict):
    """Yield (lon, lat) pairs of a GeoJSON Polygon / MultiPolygon."""
    t = geom.get("type")
    if t == "Polygon":
        for ring in geom["coordinates"]:
            for xy in ring:
                yield xy[0], xy[1]
    elif t == "MultiPolygon":
        for poly in geom["coordinates"]:
            for ring in poly:
                for xy in ring:
                    yield xy[0], xy[1]
    else:
        raise ValueError("unsupported geometry type %r (Polygon or MultiPolygon only)" % t)


def _wrap_geom(geom: dict, to360: bool) -> dict:
    """Return a copy of a GeoJSON geometry with longitudes wrapped to 0..360 (or -180..180)."""
    def wrap(x):
        return x % 360.0 if to360 else ((x + 180.0) % 360.0) - 180.0

    def ring(r):
        return [[wrap(float(p[0])), float(p[1])] for p in r]

    if geom["type"] == "Polygon":
        return {"type": "Polygon", "coordinates": [ring(r) for r in geom["coordinates"]]}
    return {"type": "MultiPolygon",
            "coordinates": [[ring(r) for r in poly] for poly in geom["coordinates"]]}


def read_geojson_polygons(path: str, id_property: str | None = None) -> list[tuple[str, dict]]:
    """Read ``(id, geometry)`` pairs from a GeoJSON FeatureCollection.

    The id is the feature's ``id`` member, else ``properties[id_property]``
    (when given), else the 0-based feature index as a string. Only Polygon and
    MultiPolygon geometries are kept.
    """
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj.get("features", [gj] if gj.get("type") == "Feature" else [])
    out: list[tuple[str, dict]] = []
    for i, ft in enumerate(feats):
        geom = ft.get("geometry") or {}
        if geom.get("type") not in ("Polygon", "MultiPolygon"):
            continue
        pid = ft.get("id")
        if pid is None and id_property:
            pid = (ft.get("properties") or {}).get(id_property)
        out.append((str(pid if pid is not None else i), geom))
    return out


def sample_points(mosaic: str, lons: Sequence[float], lats: Sequence[float],
                  env: dict | None = None) -> np.ndarray:
    """Sample a mosaic at points (nearest pixel).

    Parameters
    ----------
    mosaic
        Local path, ``/vsicurl/`` path or http(s) URL of a geographic GeoTIFF.
    lons, lats
        Point coordinates in degrees (longitude in either convention).
    env
        GDAL configuration passed to ``rasterio.Env``; defaults to
        :data:`DEFAULT_GDAL_ENV` for remote mosaics.

    Returns
    -------
    numpy.ndarray
        ``(n,)`` float64 values; NaN for nodata or points outside the mosaic.
    """
    rasterio = _rasterio()
    from rasterio.windows import Window

    path = _open_path(mosaic)
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)
    out = np.full(lons.shape, np.nan)
    with rasterio.Env(**_env_for(path, env)):
        with rasterio.open(path) as src:
            _check_geographic(src)
            b = src.bounds
            to360 = b.right > 180 + 1e-6
            xs = lons % 360.0 if to360 else ((lons + 180.0) % 360.0) - 180.0
            nod = src.nodata
            for k, (x, y) in enumerate(zip(xs, lats)):
                if not (b.left <= x <= b.right and b.bottom <= y <= b.top):
                    continue
                r, c = src.index(x, y)
                if not (0 <= r < src.height and 0 <= c < src.width):
                    continue
                v = float(src.read(1, window=Window(c, r, 1, 1))[0, 0])
                if nod is not None and v == nod:
                    continue
                out[k] = v
    return out


def sample_polygons(mosaic: str, polygons: Iterable[tuple[str, dict]],
                    band_height: int = 2500, all_touched: bool = True,
                    env: dict | None = None) -> dict[str, PolygonStats]:
    """Sample every polygon of a set from one mosaic, reading it in latitude bands.

    Parameters
    ----------
    mosaic
        Local path, ``/vsicurl/`` path or http(s) URL of a geographic GeoTIFF.
    polygons
        ``(id, geometry)`` pairs with GeoJSON Polygon / MultiPolygon
        geometries in degrees (see :func:`read_geojson_polygons`).
    band_height
        Rows per full-width read; polygons are assigned to the band of their
        bounding-box centre and clipped to it.
    all_touched
        Include every pixel touched by the polygon outline (rasterio
        ``geometry_mask`` semantics); False keeps pixels whose centre is inside.
    env
        GDAL configuration for ``rasterio.Env``.

    Returns
    -------
    dict
        ``{id: PolygonStats}`` for polygons with at least one valid pixel.
    """
    rasterio = _rasterio()
    from rasterio.features import geometry_mask
    from rasterio.windows import Window

    path = _open_path(mosaic)
    feats = []
    for pid, geom in polygons:
        xy = np.array(list(_geom_coords(geom)), dtype=np.float64)
        if xy.size == 0:
            continue
        feats.append({"pid": str(pid), "geom": geom,
                      "bbox": (xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max())})

    res: dict[str, PolygonStats] = {}
    with rasterio.Env(**_env_for(path, env)):
        with rasterio.open(path) as src:
            _check_geographic(src)
            tr = src.transform
            H, W, nod = src.height, src.width, src.nodata
            b = src.bounds
            to360 = b.right > 180 + 1e-6
            a, e, c, f = tr.a, tr.e, tr.c, tr.f

            def row_of(lat):
                return int((lat - f) / e)

            bands: dict[int, list] = {}
            for ft in feats:
                x0, y0, x1, y1 = ft["bbox"]
                if y1 < b.bottom or y0 > b.top:
                    continue
                clat = 0.5 * (y0 + y1)
                bands.setdefault(max(0, row_of(clat) // band_height), []).append(ft)

            for bi in sorted(bands):
                r0 = bi * band_height
                h = min(band_height, H - r0)
                if h <= 0:
                    continue
                data = src.read(1, window=Window(0, r0, W, h)).astype("float32")
                if nod is not None:
                    data[data == nod] = np.nan
                btr = src.window_transform(Window(0, r0, W, h))
                for ft in bands[bi]:
                    x0, y0, x1, y1 = ft["bbox"]
                    gx0, gx1 = (x0 % 360, x1 % 360) if to360 else (
                        ((x0 + 180) % 360) - 180, ((x1 + 180) % 360) - 180)
                    cc0 = max(0, int((min(gx0, gx1) - c) / a) - 2)
                    cc1 = min(W, int((max(gx0, gx1) - c) / a) + 3)
                    rr0 = max(0, int((max(y0, y1) - f) / e) - 2 - r0)
                    rr1 = min(h, int((min(y0, y1) - f) / e) + 3 - r0)
                    if cc1 <= cc0 or rr1 <= rr0:
                        continue
                    sub = data[rr0:rr1, cc0:cc1]
                    subtr = btr * rasterio.Affine.translation(cc0, rr0)
                    geom = _wrap_geom(ft["geom"], to360)
                    try:
                        m = geometry_mask([geom], out_shape=sub.shape, transform=subtr,
                                          invert=True, all_touched=all_touched)
                    except Exception:
                        continue
                    v = sub[m]
                    v = v[np.isfinite(v)]
                    if v.size:
                        res[ft["pid"]] = PolygonStats(
                            int(v.size), float(np.median(v)), float(np.percentile(v, 25)),
                            float(np.percentile(v, 75)), float(np.mean(v)), float(np.std(v)))
    return res
