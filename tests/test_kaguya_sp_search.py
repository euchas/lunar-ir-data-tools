"""PDAP search tests with a fake HTTP session: no network is used."""
import csv

import pytest
import requests

from lunar_ir_tools import kaguya_sp_search as s

VOTABLE = """<?xml version="1.0"?>
<VOTABLE xmlns="http://www.ivoa.net/xml/VOTable/v1.1" version="1.1">
<RESOURCE>
<INFO name="QUERY_STATUS" value="{status}"/>
<PARAM name="TOTAL_RECORDS" datatype="int" value="{total}"/>
<TABLE>
<FIELD ID="PRODUCT.PRODUCT_ID" datatype="char"/>
<FIELD ID="PRODUCT.CENTER_LATITUDE" datatype="double"/>
<FIELD ID="PRODUCT.CENTER_LONGITUDE" datatype="double"/>
<FIELD ID="PRODUCT.DATA_ACCESS_REFERENCE" datatype="char"/>
<DATA><TABLEDATA>
{rows}
</TABLEDATA></DATA>
</TABLE></RESOURCE></VOTABLE>
"""


def votable(rows, status="OK", total=None):
    trs = "\n".join("<TR><TD>%s</TD><TD>%s</TD><TD>%s</TD><TD>%s</TD></TR>" % r for r in rows)
    return VOTABLE.format(status=status, total=len(rows) if total is None else total,
                          rows=trs).encode()


class FakeResponse:
    def __init__(self, status_code=200, content=b""):
        self.status_code = status_code
        self.content = content

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise requests.HTTPError("HTTP %d" % self.status_code, response=self)


class FakeSession:
    """Replays a script of responses / exceptions and records the calls."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(dict(url=url, params=params, headers=headers, timeout=timeout))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClock:
    def __init__(self):
        self.t = 1000.0
        self.sleeps = []

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.sleeps.append(dt)
        self.t += dt


def make_client(script, **kw):
    clock = FakeClock()
    sess = FakeSession(script)
    c = s.PdapClient(session=sess, sleep=clock.sleep, clock=clock, **kw)
    return c, sess, clock


def test_parse_votable():
    status, total, rows = s.parse_votable(votable([("P1", "1.5", "20.0", "https://x/P1.spc")],
                                                  total=7))
    assert status == "OK" and total == 7
    assert rows == [{"PRODUCT.PRODUCT_ID": "P1", "PRODUCT.CENTER_LATITUDE": "1.5",
                     "PRODUCT.CENTER_LONGITUDE": "20.0",
                     "PRODUCT.DATA_ACCESS_REFERENCE": "https://x/P1.spc"}]


def test_constants_https_and_user_agent():
    assert s.ENDPOINT.startswith("https://darts.isas.jaxa.jp/")
    assert s.USER_AGENT.startswith("lunar-ir-data-tools/0.1.0 (+https://github.com/")
    assert "@" not in s.USER_AGENT
    assert s.MIN_INTERVAL_S >= 1.0


def test_request_headers_params_and_url():
    c, sess, clock = make_client([FakeResponse(200, votable([]))])
    c.query_page(s.DATASET_L2C, 83, 87, 190, 215, 1)
    call = sess.calls[0]
    assert call["url"] == s.ENDPOINT and call["url"].startswith("https://")
    assert call["headers"]["User-Agent"] == s.USER_AGENT
    assert call["params"]["LATITUDE"] == "83/87" and call["params"]["LONGITUDE"] == "190/215"
    assert call["params"]["PAGE_SIZE"] == "50" and call["params"]["RETURN_TYPE"] == "VOTABLE"
    assert call["timeout"] == 60.0


def test_throttle_waits_at_least_one_second_between_requests():
    c, sess, clock = make_client([FakeResponse(200, votable([])), FakeResponse(200, votable([]))])
    c.query_page(s.DATASET_L2C, 0, 1, 0, 1, 1)
    c.query_page(s.DATASET_L2C, 0, 1, 0, 1, 2)
    assert len(sess.calls) == 2
    assert clock.sleeps and clock.sleeps[0] == pytest.approx(1.0)


def test_retry_on_timeout_then_5xx_with_exponential_backoff():
    body = votable([("P1", "1", "2", "u")])
    c, sess, clock = make_client([requests.Timeout("t"), FakeResponse(503), FakeResponse(200, body)],
                                 backoff_base_s=2.0)
    out = c.query_page(s.DATASET_L2C, 0, 1, 0, 1, 1)
    assert out == body
    assert len(sess.calls) == 3 and c.n_requests == 3
    backoffs = [d for d in clock.sleeps if d >= 2.0]
    assert backoffs == [2.0, 4.0]


def test_gives_up_after_max_retries():
    c, sess, clock = make_client([requests.ConnectionError("x")] * 3, max_retries=2)
    with pytest.raises(requests.ConnectionError):
        c.query_page(s.DATASET_L2C, 0, 1, 0, 1, 1)
    assert len(sess.calls) == 3


def test_4xx_is_not_retried():
    c, sess, clock = make_client([FakeResponse(404)] * 3)
    with pytest.raises(requests.HTTPError):
        c.query_page(s.DATASET_L2C, 0, 1, 0, 1, 1)
    assert len(sess.calls) == 1


def test_client_rejects_fast_polling_and_plain_http():
    with pytest.raises(ValueError):
        s.PdapClient(session=FakeSession([]), min_interval_s=0.1)
    with pytest.raises(ValueError):
        s.PdapClient(session=FakeSession([]), endpoint="http://darts.isas.jaxa.jp/x")
    with pytest.raises(ValueError):
        s.PdapClient(session=FakeSession([])).query_page(s.DATASET_L2C, 0, 1, 0, 1, 21)


class TilingClient:
    """Fake client: a tile reports 1000 records while wider than 10 deg in
    latitude; smaller tiles return a few products, with one duplicate."""

    def __init__(self):
        self.calls = []

    def query_page(self, dataset_id, lat_lo, lat_hi, lon_lo, lon_hi, page):
        self.calls.append((lat_lo, lat_hi, lon_lo, lon_hi, page))
        if lat_hi - lat_lo > 10:
            return votable([("CAPPED", "0", "0", "u")] * 50, total=1000)
        rows = [("P_%g_%g_%d" % (lat_lo, lon_lo, k), str(lat_lo), str(lon_lo), "u") for k in range(3)]
        rows.append(("DUP", "0", "0", "u"))
        return votable(rows)


def test_search_tiles_and_deduplicates(tmp_path):
    client = TilingClient()
    rows = s.search_footprints(-20, 20, 100, 120, client=client)
    # 40 deg -> 20 deg -> 10 deg: four leaf tiles, each 3 unique + a shared DUP
    pids = [r["PRODUCT.PRODUCT_ID"] for r in rows]
    assert len(pids) == len(set(pids)) == 13
    assert "CAPPED" not in pids
    assert len(client.calls) == 1 + 2 + 4
    out = tmp_path / "m.csv"
    s.write_manifest(rows, str(out))
    with open(out, newline="") as f:
        rd = list(csv.reader(f))
    assert rd[0][0] == "PRODUCT_ID" and rd[0][-1] == "FOOTPRINT"
    assert len(rd) == 14


def test_search_validates_box():
    with pytest.raises(ValueError):
        s.search_footprints(10, 5, 0, 10, client=TilingClient())
    with pytest.raises(ValueError):
        s.search_footprints(0, 5, -10, 10, client=TilingClient())


def test_search_max_depth_warns():
    class AlwaysCapped(TilingClient):
        def query_page(self, *a):
            self.calls.append(a)
            return votable([("X%d" % len(self.calls), "0", "0", "u")] * 50, total=1000)

    with pytest.warns(UserWarning):
        rows = s.search_footprints(0, 1, 0, 1, client=AlwaysCapped(), max_depth=1)
    assert rows
