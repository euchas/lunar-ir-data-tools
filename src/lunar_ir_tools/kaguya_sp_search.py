"""Footprint search for Kaguya (SELENE) Spectral Profiler (SP) products via JAXA DARTS PDAP.

Finds SP point-spectrum products (L2C reflectance, data set
``SLN-L-SP-4-LEVEL2C-V3.0``, or L2B1 radiance, ``SLN-L-SP-3-LEVEL2B1-V2.0``)
whose footprint centre lies inside a latitude/longitude box, and returns the
product IDs together with the direct ``.spc`` / ``.lbl`` download references
served by the SELENE Data Archive at JAXA DARTS.

Why PDAP rather than the archive file tree: the data-set root has no
cumulative index and the per-date index tables carry no latitude/longitude
columns, so the PDAP (Planetary Data Access Protocol) service is the only
server-side spatial filter. It returns centre lat/lon, a footprint polygon and
the download references for every matching product.

Service behaviour as observed in July 2026 (the endpoint is **not formally
documented by JAXA** and may change without notice):

* endpoint ``https://darts.isas.jaxa.jp/app/pdap/selene/1.0/search_exec.php``
* parameters ``RESOURCE_CLASS=PRODUCT``, ``DATA_SET_ID=...``,
  ``RETURN_TYPE=VOTABLE``, ``LATITUDE=<lo>/<hi>`` (deg, -90..90),
  ``LONGITUDE=<lo>/<hi>`` (deg, 0..360 east, matching the ``E####`` part of
  SP file names), ``PAGE_SIZE`` (at most 50; larger values give
  ``QUERY_STATUS=NG``), ``PAGE_NUMBER`` (1..20; ``TOTAL_RECORDS`` caps at 1000
  and record 1001+ gives ``NG``);
* therefore any box with 1000 or more matches must be tiled, which
  :func:`search_footprints` does recursively.

Responsible use. This is a shared, public, undocumented service:

* every request carries the descriptive User-Agent
  ``lunar-ir-data-tools/<version> (+https://github.com/euchas/lunar-ir-data-tools)``;
* the client waits at least :data:`MIN_INTERVAL_S` (1 s) between requests and
  retries with exponential back-off only on HTTP 5xx responses, timeouts and
  connection errors (4xx responses are raised immediately);
* **bulk enumeration of the whole archive should be avoided**: query the
  regions you actually need and keep the resulting manifest instead of
  re-querying. One query returns at most 50 records, so a global search would
  need tens of thousands of requests.

Only ``requests`` and the standard library are used.
"""
from __future__ import annotations

import csv
import logging
import time
import warnings
import xml.etree.ElementTree as ET
from typing import Callable, Sequence

import requests

from ._version import __version__

__all__ = [
    "ENDPOINT",
    "USER_AGENT",
    "DATASET_L2C",
    "DATASET_L2B1",
    "PAGE_SIZE",
    "MAX_PAGES",
    "RECORD_CAP",
    "MIN_INTERVAL_S",
    "MANIFEST_FIELDS",
    "PdapClient",
    "parse_votable",
    "search_footprints",
    "write_manifest",
]

log = logging.getLogger(__name__)

ENDPOINT = "https://darts.isas.jaxa.jp/app/pdap/selene/1.0/search_exec.php"
USER_AGENT = "lunar-ir-data-tools/%s (+https://github.com/euchas/lunar-ir-data-tools)" % __version__

DATASET_L2C = "SLN-L-SP-4-LEVEL2C-V3.0"   #: SP Level 2C reflectance
DATASET_L2B1 = "SLN-L-SP-3-LEVEL2B1-V2.0"  #: SP Level 2B1 radiance

PAGE_SIZE = 50                 #: server hard maximum per page
MAX_PAGES = 20                 #: server hard ceiling (20 x 50 = 1000 records)
RECORD_CAP = MAX_PAGES * PAGE_SIZE
MIN_INTERVAL_S = 1.0           #: minimum spacing between requests, seconds

#: VOTABLE FIELD IDs written to the manifest (header uses the part after the dot).
MANIFEST_FIELDS: list[str] = [
    "PRODUCT.PRODUCT_ID",
    "PRODUCT.CENTER_LATITUDE",
    "PRODUCT.CENTER_LONGITUDE",
    "PRODUCT.START_TIME",
    "PRODUCT.STOP_TIME",
    "PRODUCT.DATA_ACCESS_REFERENCE",
    "PRODUCT.LABEL_ACCESS_REFERENCE",
    "PRODUCT.ICON_ACCESS_REFERENCE",
    "PRODUCT.SUB_SPACECRAFT_LATITUDE",
    "PRODUCT.SUB_SPACECRAFT_LONGITUDE",
    "PRODUCT.TARGET_PHASE_ANGLE",
    "FOOTPRINT",
]


def _local(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


class PdapClient:
    """Throttled, retrying HTTP client for the DARTS PDAP search endpoint.

    Parameters
    ----------
    session
        A ``requests.Session``-like object with a ``get`` method; a new
        session is created when None.
    endpoint, user_agent
        Service URL and User-Agent header.
    min_interval_s
        Minimum spacing between consecutive requests (seconds). Values below
        :data:`MIN_INTERVAL_S` are rejected.
    timeout_s
        Per-request timeout (seconds).
    max_retries
        Number of retries after the first attempt on 5xx / timeout /
        connection error. Delays are ``backoff_base_s * 2**k`` seconds
        (2, 4, 8, 16 s by default).
    sleep, clock
        Injection points for tests (default ``time.sleep`` / ``time.monotonic``).
    """

    def __init__(self, session=None, *, endpoint: str = ENDPOINT,
                 user_agent: str = USER_AGENT, min_interval_s: float = MIN_INTERVAL_S,
                 timeout_s: float = 60.0, max_retries: int = 4,
                 backoff_base_s: float = 2.0,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        if min_interval_s < MIN_INTERVAL_S:
            raise ValueError("min_interval_s must be >= %.1f s" % MIN_INTERVAL_S)
        if not endpoint.lower().startswith("https://"):
            raise ValueError("endpoint must use https")
        self.session = session if session is not None else requests.Session()
        self.endpoint = endpoint
        self.user_agent = user_agent
        self.min_interval_s = float(min_interval_s)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.backoff_base_s = float(backoff_base_s)
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None
        self.n_requests = 0  #: attempts made (including failed ones)

    def _throttle(self) -> None:
        now = self._clock()
        if self._last_request is not None:
            wait = self.min_interval_s - (now - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def query_page(self, dataset_id: str, lat_lo: float, lat_hi: float,
                   lon_lo: float, lon_hi: float, page: int) -> bytes:
        """Fetch one VOTABLE page of results.

        Parameters
        ----------
        dataset_id
            PDS data-set ID (:data:`DATASET_L2C` or :data:`DATASET_L2B1`).
        lat_lo, lat_hi
            Latitude bounds in degrees (-90..90).
        lon_lo, lon_hi
            Longitude bounds in degrees east (0..360).
        page
            1-based page number (1..20).

        Returns
        -------
        bytes
            Raw VOTABLE XML.

        Raises
        ------
        requests.HTTPError
            On a 4xx response, or on a 5xx response after all retries.
        requests.Timeout, requests.ConnectionError
            After all retries.
        """
        if not 1 <= page <= MAX_PAGES:
            raise ValueError("page must be in 1..%d" % MAX_PAGES)
        params = {
            "RESOURCE_CLASS": "PRODUCT",
            "RETURN_TYPE": "VOTABLE",
            "DATA_SET_ID": dataset_id,
            "LATITUDE": "%g/%g" % (lat_lo, lat_hi),
            "LONGITUDE": "%g/%g" % (lon_lo, lon_hi),
            "PAGE_SIZE": str(PAGE_SIZE),
            "PAGE_NUMBER": str(page),
        }
        headers = {"User-Agent": self.user_agent,
                   "Accept": "application/x-votable+xml, text/xml, */*"}
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                delay = self.backoff_base_s * 2 ** (attempt - 1)
                log.warning("PDAP request failed (%s); retry %d/%d in %.0f s",
                            last_err, attempt, self.max_retries, delay)
                self._sleep(delay)
            self._throttle()
            self.n_requests += 1
            try:
                resp = self.session.get(self.endpoint, params=params, headers=headers,
                                        timeout=self.timeout_s)
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_err = exc
                continue
            if resp.status_code >= 500:
                last_err = requests.HTTPError("PDAP returned HTTP %d" % resp.status_code,
                                              response=resp)
                continue
            resp.raise_for_status()
            return resp.content
        assert last_err is not None
        raise last_err


def parse_votable(xml_bytes: bytes) -> tuple[str | None, int | None, list[dict[str, str]]]:
    """Parse a PDAP VOTABLE response.

    Returns
    -------
    status
        ``QUERY_STATUS`` (``"OK"`` or ``"NG"``), or None if absent.
    total
        ``TOTAL_RECORDS`` as reported by the server (capped at 1000), or None.
    rows
        One ``{FIELD_ID: text}`` dict per ``TR`` row.
    """
    root = ET.fromstring(xml_bytes)
    status, total = None, None
    fields: list[str] = []
    rows: list[dict[str, str]] = []
    for el in root.iter():
        t = _local(el.tag)
        if t == "INFO" and el.get("name") == "QUERY_STATUS":
            status = el.get("value")
        elif t == "PARAM" and el.get("name") == "TOTAL_RECORDS":
            try:
                total = int(el.get("VALUE") or el.get("value"))
            except (TypeError, ValueError):
                total = None
        elif t == "FIELD":
            fields.append(el.get("ID"))
        elif t == "TR":
            tds = [c for c in el if _local(c.tag) == "TD"]
            vals = [(c.text or "").strip() for c in tds]
            rows.append(dict(zip(fields, vals)))
    return status, total, rows


def _fetch_tile(client: PdapClient, dataset_id: str, lat_lo: float, lat_hi: float,
                lon_lo: float, lon_hi: float, out: list, seen: set,
                depth: int, max_depth: int) -> None:
    """Recursively subdivide a box until each tile has < RECORD_CAP records, then page it."""
    status, total, rows = parse_votable(
        client.query_page(dataset_id, lat_lo, lat_hi, lon_lo, lon_hi, 1))
    if status == "NG":
        log.warning("[NG] tile lat %.4f/%.4f lon %.4f/%.4f skipped",
                    lat_lo, lat_hi, lon_lo, lon_hi)
        return
    if total is None:
        total = len(rows)
    log.info("%stile lat %.4f/%.4f lon %.4f/%.4f -> TOTAL_RECORDS=%d",
             "  " * depth, lat_lo, lat_hi, lon_lo, lon_hi, total)

    if total >= RECORD_CAP and depth < max_depth:
        # capped -> subdivide along the longer side
        if (lat_hi - lat_lo) >= (lon_hi - lon_lo):
            mid = (lat_lo + lat_hi) / 2.0
            _fetch_tile(client, dataset_id, lat_lo, mid, lon_lo, lon_hi, out, seen, depth + 1, max_depth)
            _fetch_tile(client, dataset_id, mid, lat_hi, lon_lo, lon_hi, out, seen, depth + 1, max_depth)
        else:
            mid = (lon_lo + lon_hi) / 2.0
            _fetch_tile(client, dataset_id, lat_lo, lat_hi, lon_lo, mid, out, seen, depth + 1, max_depth)
            _fetch_tile(client, dataset_id, lat_lo, lat_hi, mid, lon_hi, out, seen, depth + 1, max_depth)
        return
    if total >= RECORD_CAP:
        warnings.warn("tile lat %.4f/%.4f lon %.4f/%.4f still reports %d records at "
                      "max_depth=%d; only the first %d are returned"
                      % (lat_lo, lat_hi, lon_lo, lon_hi, total, max_depth, RECORD_CAP))
        total = RECORD_CAP

    npages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    for p in range(1, npages + 1):
        page_rows = rows if p == 1 else parse_votable(
            client.query_page(dataset_id, lat_lo, lat_hi, lon_lo, lon_hi, p))[2]
        for row in page_rows:
            pid = row.get("PRODUCT.PRODUCT_ID", "")
            if pid and pid not in seen:
                seen.add(pid)
                out.append(row)


def search_footprints(lat_lo: float, lat_hi: float, lon_lo: float, lon_hi: float,
                      dataset_id: str = DATASET_L2C, client: PdapClient | None = None,
                      max_depth: int = 12) -> list[dict[str, str]]:
    """Find all SP products whose footprint centre lies in a lat/lon box.

    The box is tiled recursively whenever the server reports the 1000-record
    cap, so the result is complete for any box size, at the cost of one
    request per 50 records plus one per tile. Please keep boxes small and do
    not enumerate the whole archive (see the module docstring).

    Parameters
    ----------
    lat_lo, lat_hi
        Latitude bounds, degrees (-90..90).
    lon_lo, lon_hi
        Longitude bounds, degrees east (0..360).
    dataset_id
        :data:`DATASET_L2C` (default) or :data:`DATASET_L2B1`.
    client
        A :class:`PdapClient`; a default one is created when None.
    max_depth
        Maximum subdivision depth (each level halves the longer side).

    Returns
    -------
    list of dict
        Unique products (by ``PRODUCT.PRODUCT_ID``), each a ``{FIELD_ID: text}``
        dict with centre lat/lon (degrees), times (ISO strings) and the
        ``DATA_ACCESS_REFERENCE`` / ``LABEL_ACCESS_REFERENCE`` download URLs.
    """
    if not (-90 <= lat_lo < lat_hi <= 90):
        raise ValueError("need -90 <= lat_lo < lat_hi <= 90")
    if not (0 <= lon_lo < lon_hi <= 360):
        raise ValueError("need 0 <= lon_lo < lon_hi <= 360 (degrees east)")
    client = client if client is not None else PdapClient()
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    _fetch_tile(client, dataset_id, lat_lo, lat_hi, lon_lo, lon_hi, out, seen, 0, max_depth)
    return out


def write_manifest(rows: Sequence[dict[str, str]], path: str,
                   fields: Sequence[str] = MANIFEST_FIELDS) -> None:
    """Write search results to a CSV manifest.

    The header uses the short field names (the part after the last dot, e.g.
    ``PRODUCT_ID``, ``DATA_ACCESS_REFERENCE``); missing fields are left empty.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([c.split(".")[-1] for c in fields])
        for row in rows:
            w.writerow([row.get(c, "") for c in fields])
