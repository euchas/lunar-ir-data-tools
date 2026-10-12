import json

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from lunar_ir_tools import kaguya_mi as mi  # noqa: E402


def _write_mosaic(path, lon0, nodata=-1.0):
    """36 x 18 pixel, 10 deg/pixel geographic mosaic; value = row * 100 + col."""
    H, W = 18, 36
    data = (np.arange(H)[:, None] * 100 + np.arange(W)[None, :]).astype("float32")
    data[0, 0] = nodata
    transform = rasterio.transform.from_origin(lon0, 90.0, 10.0, 10.0)
    with rasterio.open(str(path), "w", driver="GTiff", height=H, width=W, count=1,
                       dtype="float32", crs="EPSG:4326", transform=transform,
                       nodata=nodata) as dst:
        dst.write(data, 1)
    return data


def test_mineral_map_url():
    u = mi.mineral_map_url("feo")
    assert u.startswith("/vsicurl/https://") and u.endswith("FeOWeightPercent_50N50S.tif")
    assert mi.mineral_map_url("OlivinePercent", vsicurl=False).startswith("https://")


def test_sample_points_minus180_and_0_360(tmp_path):
    for lon0 in (-180.0, 0.0):
        path = tmp_path / ("m%d.tif" % int(lon0))
        data = _write_mosaic(path, lon0)
        v = mi.sample_points(str(path), [-175.0, 5.0, 185.0, 500.0], [85.0, -5.0, 85.0, 0.0])
        if lon0 == -180.0:                                  # mosaic spans -180..180
            assert np.isnan(v[0])                           # lon -175, lat 85 -> (0, 0) = nodata
            assert v[1] == 9 * 100 + 18                     # lon 5 -> col 18, lat -5 -> row 9
            assert np.isnan(v[2])                           # 185 wraps to -175 -> nodata pixel
            assert v[3] == 9 * 100 + 32                     # 500 wraps to 140 -> col 32
        else:                                               # mosaic spans 0..360
            assert v[0] == 0 * 100 + 18                     # -175 wraps to 185 -> col 18
            assert v[1] == 9 * 100 + 0                      # lon 5 -> col 0
            assert v[2] == 0 * 100 + 18
            assert v[3] == 9 * 100 + 14                     # 500 wraps to 140 -> col 14
        # a point outside the latitude range is NaN
        assert np.isnan(mi.sample_points(str(path), [0.0], [95.0])[0])


def test_sample_polygons_latitude_bands(tmp_path):
    path = tmp_path / "m.tif"
    data = _write_mosaic(path, -180.0)
    poly = {"type": "Polygon", "coordinates": [[[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]]]}
    res = mi.sample_polygons(str(path), [("a", poly)], band_height=5, all_touched=False)
    s = res["a"]
    expected = np.array([data[7, 18], data[7, 19], data[8, 18], data[8, 19]])
    assert s.count == 4
    assert s.median == pytest.approx(np.median(expected))
    assert s.mean == pytest.approx(expected.mean())
    assert s.std == pytest.approx(expected.std())
    assert s.p25 == pytest.approx(np.percentile(expected, 25))
    # all_touched includes at least those pixels
    res_t = mi.sample_polygons(str(path), [("a", poly)], band_height=5, all_touched=True)
    assert res_t["a"].count >= 4
    # a polygon outside the mosaic yields nothing
    far = {"type": "Polygon", "coordinates": [[[0, 95], [1, 95], [1, 96], [0, 96], [0, 95]]]}
    assert mi.sample_polygons(str(path), [("far", far)]) == {}


def test_read_geojson_polygons_and_wrap(tmp_path):
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "id": "p1", "properties": {"name": "x"},
         "geometry": {"type": "Polygon", "coordinates": [[[-10, 0], [10, 0], [10, 5], [-10, 5], [-10, 0]]]}},
        {"type": "Feature", "properties": {"name": "y"},
         "geometry": {"type": "Point", "coordinates": [0, 0]}},
        {"type": "Feature", "properties": {"name": "z"},
         "geometry": {"type": "MultiPolygon", "coordinates": [[[[0, 0], [1, 0], [1, 1], [0, 0]]]]}},
    ]}
    path = tmp_path / "f.geojson"
    path.write_text(json.dumps(gj))
    polys = mi.read_geojson_polygons(str(path), id_property="name")
    assert [pid for pid, _ in polys] == ["p1", "z"]
    wrapped = mi._wrap_geom(polys[0][1], to360=True)
    assert wrapped["coordinates"][0][0][0] == 350.0
    mosaic = tmp_path / "m.tif"
    _write_mosaic(mosaic, 0.0)
    res = mi.sample_polygons(str(mosaic), polys, all_touched=False)
    assert "p1" in res and res["p1"].count > 0
