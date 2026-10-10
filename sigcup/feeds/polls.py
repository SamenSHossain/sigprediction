"""Ratings, generic ballot, FEC totals and the manual ratings/polls loader. Writes Rating / GenericBallot /
Poll rows (and FEC totals into the KV store). Collector contract (shared by every feeds module)::

    async def run_once(session_factory, cfg, secrets) -> int      # rows written

Sources (all best-effort HTML; none offers an official API; every network failure is logged and skipped):

* Sabato's Crystal Ball: the 2026 rating-changes log (Date | Race | Old | New) and the per-chamber
  ratings pages (category heading + list/table of races).
* Cook Political Report public ratings pages (Senate / Governor / House): per-category tables or one
  table with a Rating column.
* RacetotheWH: Race | Rating (| Dem win %) tables.
* 270toWin: consensus tables with one column per rater (Cook, Inside Elections, Sabato, Split Ticket,
  RacetotheWH, Consensus) and the single-rater mirrors (Split Ticket, Inside Elections).
* Generic ballot: RealClearPolling and FiftyPlusOne pages (the "Average" row of a Dem/Rep table, else a
  ``D+8.0`` / ``Democrats 51.1% ... Republicans 43.3%`` text pattern) -> GenericBallot rows.
* FEC: ``/v1/candidates/totals/`` (key from secrets.fec_api_key; never logged) -> per-race, per-party
  receipts / cash on hand, stored under KV key ``fec_totals`` (there is no FEC table).
* Manual: config/ratings_manual.yaml (``ratings:`` and optional ``polls:``), the day-one fallback.

Design: every parser is a pure function over an HTML/JSON *string* so it is testable from the synthetic
fixtures in tests/fixtures/feeds/. The HTML parsers do not depend on exact page markup: they extract
every table and list with its preceding heading, pick the column whose cells look like races, the
column(s) whose cells look like ratings (or the heading when a whole block is one category) and map
rating strings through sigcup.feeds.universe.normalize_rating. Rating strings are stored canonically
("Lean D", "Toss-up") with prob_dem from cfg.fairvalue.ratings_prob.

Idempotency: a Rating row is written only when the latest stored rating for (race_id, source) is older
and either differs or is stale by >= RATING_REFRESH_DAYS; a GenericBallot row only when the margin
changed or a day passed; Poll rows are unique on (race_id, pollster, end_date, margin).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import httpx
import yaml
from dateutil import parser as dateparser
from sqlalchemy import select

from sigcup.common.time import ensure_utc, utc_now
from sigcup.db.models import GenericBallot, Poll, Race, Rating
from sigcup.db.repo import kv_get, kv_set
from sigcup.feeds.universe import (
    canonical_rating,
    format_district,
    load_races_yaml,
    normalize_rating,
    parse_race_id,
    race_id_from_text,
    race_row_values,
    rating_prob_dem,
)
from sigcup.settings import CONFIG_DIR, Secrets, StrategyConfig

log = logging.getLogger(__name__)

RATINGS_MANUAL_YAML = CONFIG_DIR / "ratings_manual.yaml"
HTTP_TIMEOUT_S = 20.0
USER_AGENT = "Mozilla/5.0 (compatible; sigcup/0.1; research feed)"
RATING_REFRESH_DAYS = 7  # re-write an unchanged rating after this long so it does not look stale
FEC_BASE = "https://api.open.fec.gov/v1"
FEC_CYCLE = 2026
FEC_MIN_INTERVAL = timedelta(hours=20)
FEC_PAGES = {"S": 4, "H": 10}  # per office, 100 candidates per page sorted by receipts
KV_FEC_TOTALS = "fec_totals"
KV_FEC_LAST_RUN = "fec_last_run"

# header words that identify a rater column (270toWin consensus tables) -> Rating.source
SOURCE_ALIASES: dict[str, str] = {
    "cook": "cook",
    "cook political": "cook",
    "sabato": "sabato",
    "crystal ball": "sabato",
    "inside elections": "inside",
    "inside": "inside",
    "ie": "inside",
    "split ticket": "split",
    "split-ticket": "split",
    "racetothewh": "rttwh",
    "race to the wh": "rttwh",
    "rttwh": "rttwh",
    "consensus": "270towin",
    "270towin": "270towin",
    "ddhq": "ddhq",
    "decision desk": "ddhq",
}
_EXCLUDED_HEADER_WORDS = ("previous", "old", "prior", "from", "was", "last")
_DEM_HEADER_RE = re.compile(r"\b(democrat(?:s|ic)?|dem|d)\b", re.IGNORECASE)
_REP_HEADER_RE = re.compile(r"\b(republican(?:s)?|rep|gop|r)\b", re.IGNORECASE)
_AVG_RE = re.compile(r"\b(average|avg|aggregate)\b", re.IGNORECASE)
_DATE_RANGE_RE = re.compile(r"\d{1,2}/\d{1,2}\s*[-–]\s*\d{1,2}/\d{1,2}")


# ---------------------------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------------------------


@dataclass
class RatingRow:
    """One (race, source) rating ready to be stored."""

    race_id: str
    source: str
    rating: str  # canonical: "Lean D", "Likely R", "Toss-up"
    prob_dem: float | None
    ts: datetime
    raw: str = ""

    def to_row(self) -> Rating:
        assert self.prob_dem is not None
        return Rating(race_id=self.race_id, source=self.source, rating=self.rating, prob_dem=self.prob_dem, ts=self.ts)


@dataclass
class GenericBallotRow:
    source: str
    ts: datetime
    margin: float  # dem - rep, points
    dem: float | None = None
    rep: float | None = None

    def to_row(self) -> GenericBallot:
        return GenericBallot(source=self.source, ts=self.ts, margin=self.margin)


@dataclass
class PollRow:
    race_id: str
    pollster: str
    end_date: datetime
    margin: float
    start_date: datetime | None = None
    sample_size: int | None = None
    population: str | None = None
    dem: float | None = None
    rep: float | None = None
    grade: str | None = None
    source_url: str | None = None

    def to_row(self, fetched_at: datetime) -> Poll:
        return Poll(
            race_id=self.race_id,
            pollster=self.pollster,
            start_date=self.start_date,
            end_date=self.end_date,
            sample_size=self.sample_size,
            population=self.population,
            dem=self.dem,
            rep=self.rep,
            margin=self.margin,
            grade=self.grade,
            source_url=self.source_url,
            fetched_at=fetched_at,
        )


@dataclass
class FecTotal:
    candidate_id: str
    name: str
    party: str | None  # D | R | I | other
    state: str | None
    district: str | None
    office: str | None  # S | H | P
    receipts: float | None
    disbursements: float | None
    cash_on_hand: float | None
    cycle: int | None
    race_id: str | None


# ---------------------------------------------------------------------------------------------
# HTML blocks (tables and lists with their preceding heading), stdlib only
# ---------------------------------------------------------------------------------------------


@dataclass
class HtmlRow:
    cells: list[str]
    is_header: bool = False


@dataclass
class HtmlTable:
    heading: str | None
    caption: str | None
    rows: list[HtmlRow] = field(default_factory=list)

    @property
    def header(self) -> list[str] | None:
        for r in self.rows:
            if r.is_header and r.cells:
                return r.cells
        return None

    @property
    def body(self) -> list[HtmlRow]:
        return [r for r in self.rows if not r.is_header and r.cells]

    @property
    def ncols(self) -> int:
        return max((len(r.cells) for r in self.rows), default=0)


@dataclass
class HtmlList:
    heading: str | None
    items: list[str] = field(default_factory=list)


_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


class _BlockParser(HTMLParser):
    """Collects tables and lists in document order, each tagged with the last heading seen before it."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[HtmlTable | HtmlList] = []
        self.text: list[str] = []
        self._heading: str | None = None
        self._heading_buf: list[str] | None = None
        self._caption_buf: list[str] | None = None
        self._tables: list[HtmlTable] = []
        self._rows: list[HtmlRow | None] = []
        self._cells: list[list[str] | None] = []
        self._cell_th: list[bool] = []
        self._lists: list[HtmlList] = []
        self._items: list[list[str] | None] = []
        self._skip = 0

    # -- helpers --
    @staticmethod
    def _text(buf: list[str]) -> str:
        return re.sub(r"\s+", " ", "".join(buf)).strip()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "noscript", "svg"):
            self._skip += 1
            return
        if self._skip:
            return
        if tag in _HEADING_TAGS:
            self._heading_buf = []
        elif tag == "table":
            self._tables.append(HtmlTable(heading=self._heading, caption=None))
            self._rows.append(None)
            self._cells.append(None)
            self._cell_th.append(False)
        elif tag == "caption" and self._tables:
            self._caption_buf = []
        elif tag == "tr" and self._tables:
            self._rows[-1] = HtmlRow(cells=[], is_header=True)
        elif tag in ("td", "th") and self._tables:
            if self._rows[-1] is None:
                self._rows[-1] = HtmlRow(cells=[], is_header=True)
            self._cells[-1] = []
            self._cell_th[-1] = tag == "th"
        elif tag in ("ul", "ol"):
            self._lists.append(HtmlList(heading=self._heading))
            self._items.append(None)
        elif tag == "li" and self._lists:
            self._items[-1] = []
        elif tag == "br":
            self.handle_data(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript", "svg"):
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag in _HEADING_TAGS and self._heading_buf is not None:
            self._heading = self._text(self._heading_buf) or self._heading
            self._heading_buf = None
        elif tag == "caption" and self._caption_buf is not None and self._tables:
            self._tables[-1].caption = self._text(self._caption_buf) or None
            self._caption_buf = None
        elif tag in ("td", "th") and self._tables and self._cells[-1] is not None:
            row = self._rows[-1]
            if row is not None:
                row.cells.append(self._text(self._cells[-1]))
                if not self._cell_th[-1]:
                    row.is_header = False
            self._cells[-1] = None
        elif tag == "tr" and self._tables:
            self._finish_row()
        elif tag == "table" and self._tables:
            self._finish_row()
            table = self._tables.pop()
            self._rows.pop()
            self._cells.pop()
            self._cell_th.pop()
            self.blocks.append(table)
        elif tag == "li" and self._lists and self._items[-1] is not None:
            item = self._text(self._items[-1])
            if item:
                self._lists[-1].items.append(item)
            self._items[-1] = None
        elif tag in ("ul", "ol") and self._lists:
            lst = self._lists.pop()
            self._items.pop()
            self.blocks.append(lst)

    def _finish_row(self) -> None:
        row = self._rows[-1]
        if row is not None:
            if self._cells[-1] is not None:  # unterminated cell
                row.cells.append(self._text(self._cells[-1]))
                if not self._cell_th[-1]:
                    row.is_header = False
                self._cells[-1] = None
            if row.cells:
                self._tables[-1].rows.append(row)
            self._rows[-1] = None

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._heading_buf is not None:
            self._heading_buf.append(data)
        if self._caption_buf is not None:
            self._caption_buf.append(data)
        if self._tables and self._cells[-1] is not None:
            self._cells[-1].append(data)
        if self._lists and self._items[-1] is not None:
            self._items[-1].append(data)
        self.text.append(data)


def extract_blocks(html: str) -> list[HtmlTable | HtmlList]:
    """Every <table> and <ul>/<ol> in the document, in order, with the nearest preceding heading."""
    p = _BlockParser()
    try:
        p.feed(html or "")
        p.close()
    except Exception as e:  # noqa: BLE001  (never let malformed HTML kill a collector)
        log.warning("html parse error: %s", type(e).__name__)
    return p.blocks


def extract_tables(html: str) -> list[HtmlTable]:
    return [b for b in extract_blocks(html) if isinstance(b, HtmlTable)]


def html_text(html: str) -> str:
    """Visible text of the document (scripts/styles removed, whitespace collapsed)."""
    p = _BlockParser()
    try:
        p.feed(html or "")
        p.close()
    except Exception:  # noqa: BLE001
        return ""
    return re.sub(r"\s+", " ", " ".join(p.text)).strip()


# ---------------------------------------------------------------------------------------------
# Ratings tables -> RatingRow (generic; the site wrappers only set source / chamber / url)
# ---------------------------------------------------------------------------------------------


def parse_date(text: str | None, default: datetime | None = None) -> datetime | None:
    """'10/6/2026', 'October 6, 2026', '2026-10-06' -> aware UTC datetime; `default` when unparseable."""
    if not text:
        return default
    t = text.strip()
    if not t or _DATE_RANGE_RE.search(t):
        return default  # a date range like '9/16 - 10/5' is not a single date
    try:
        dt = dateparser.parse(t, default=datetime(2026, 1, 1))
    except (ValueError, OverflowError, TypeError):
        return default
    if dt is None:
        return default
    return ensure_utc(dt)


def source_from_header(header: str | None) -> str | None:
    """Column header -> Rating.source when it names a rater (None otherwise)."""
    if not header:
        return None
    h = header.strip().lower()
    for alias, src in sorted(SOURCE_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if alias in h:
            return src
    return None


def _header_excluded(header: str | None) -> bool:
    if not header:
        return False
    h = header.lower()
    return any(re.search(rf"\b{w}\b", h) for w in _EXCLUDED_HEADER_WORDS)


def _column(table: HtmlTable, i: int) -> list[str]:
    return [r.cells[i] if i < len(r.cells) else "" for r in table.body]


def _race_column(table: HtmlTable, chamber: str | None) -> int | None:
    body = table.body
    if not body:
        return None
    best, best_n = None, 0
    for i in range(table.ncols):
        n = sum(1 for c in _column(table, i) if race_id_from_text(c, chamber))
        if n > best_n:
            best, best_n = i, n
    if best is None or best_n < max(1, len(body) // 3):
        return None
    return best


def _rating_columns(table: HtmlTable, race_col: int, default_source: str) -> list[tuple[int, str]]:
    """(column index, source) for every column whose cells mostly parse as ratings."""
    body = table.body
    header = table.header or []
    out: list[tuple[int, str]] = []
    for i in range(table.ncols):
        if i == race_col:
            continue
        h = header[i] if i < len(header) else None
        if _header_excluded(h):
            continue
        n = sum(1 for c in _column(table, i) if normalize_rating(c))
        if n >= max(1, (len(body) + 1) // 2):
            out.append((i, source_from_header(h) or default_source))
    return out


def _date_column(table: HtmlTable) -> int | None:
    header = table.header or []
    for i, h in enumerate(header):
        if re.search(r"\bdate\b|\bupdated\b|\bchanged\b", h, re.IGNORECASE):
            return i
    return None


def _make_row(
    race_id: str, source: str, rating_text: str, ts: datetime, ratings_prob: Mapping[str, float]
) -> RatingRow | None:
    norm = normalize_rating(rating_text)
    if norm is None:
        return None
    level, party = norm
    prob = rating_prob_dem(level, party, ratings_prob)
    if prob is None:
        return None  # independent-favoured ratings cannot be expressed as P(D)
    return RatingRow(race_id=race_id, source=source, rating=canonical_rating(level, party), prob_dem=round(prob, 6), ts=ts, raw=rating_text)


def parse_ratings_html(
    html: str,
    *,
    source: str,
    chamber: str | None,
    ts: datetime,
    ratings_prob: Mapping[str, float],
    races: Iterable[str] | None = None,
) -> list[RatingRow]:
    """Generic ratings-page parser: tables with a race column and rating column(s) (or a rating heading)
    and lists under a rating heading. Keeps the newest row per (race, source)."""
    allowed = set(races) if races is not None else None
    found: dict[tuple[str, str], RatingRow] = {}

    def keep(row: RatingRow | None) -> None:
        if row is None or (allowed is not None and row.race_id not in allowed):
            return
        k = (row.race_id, row.source)
        prev = found.get(k)
        if prev is None or row.ts > prev.ts:
            found[k] = row

    for block in extract_blocks(html):
        if isinstance(block, HtmlList):
            norm = normalize_rating(block.heading)
            if norm is None:
                continue
            for item in block.items:
                rid = race_id_from_text(item, chamber)
                if rid:
                    keep(_make_row(rid, source, block.heading or "", ts, ratings_prob))
            continue
        race_col = _race_column(block, chamber)
        if race_col is None:
            continue
        cols = _rating_columns(block, race_col, source)
        date_col = _date_column(block)
        heading_rating = block.heading if normalize_rating(block.heading) else (block.caption if normalize_rating(block.caption) else None)
        if not cols and heading_rating is None:
            continue
        for r in block.body:
            if race_col >= len(r.cells):
                continue
            rid = race_id_from_text(r.cells[race_col], chamber)
            if not rid:
                continue
            row_ts = parse_date(r.cells[date_col], ts) if date_col is not None and date_col < len(r.cells) else ts
            if cols:
                for ci, src in cols:
                    if ci < len(r.cells):
                        keep(_make_row(rid, src, r.cells[ci], row_ts or ts, ratings_prob))
            else:
                keep(_make_row(rid, source, heading_rating or "", row_ts or ts, ratings_prob))
    return list(found.values())


# -- site wrappers (thin: they fix source / chamber; the fixtures exercise the generic logic) ----


def parse_sabato_rating_changes(html: str, *, ts: datetime, ratings_prob: Mapping[str, float]) -> list[RatingRow]:
    """Crystal Ball '2026 rating changes' log: Date | Race | Old Rating | New Rating (newest first)."""
    return parse_ratings_html(html, source="sabato", chamber=None, ts=ts, ratings_prob=ratings_prob)


def parse_sabato_ratings(html: str, chamber: str, *, ts: datetime, ratings_prob: Mapping[str, float]) -> list[RatingRow]:
    """Crystal Ball per-chamber ratings page (category heading followed by a list or table of races)."""
    return parse_ratings_html(html, source="sabato", chamber=chamber, ts=ts, ratings_prob=ratings_prob)


def parse_cook_ratings(html: str, chamber: str, *, ts: datetime, ratings_prob: Mapping[str, float]) -> list[RatingRow]:
    """Cook public ratings page: per-category tables ('Toss Up' heading) or a table with a Rating column."""
    return parse_ratings_html(html, source="cook", chamber=chamber, ts=ts, ratings_prob=ratings_prob)


def parse_rttwh_ratings(html: str, chamber: str, *, ts: datetime, ratings_prob: Mapping[str, float]) -> list[RatingRow]:
    """RacetotheWH: Race | Rating (| Dem win %) tables."""
    return parse_ratings_html(html, source="rttwh", chamber=chamber, ts=ts, ratings_prob=ratings_prob)


def parse_270towin_table(
    html: str, chamber: str, *, ts: datetime, ratings_prob: Mapping[str, float], source: str = "270towin"
) -> list[RatingRow]:
    """270toWin consensus table (one column per rater -> one RatingRow per rater) or a single-rater mirror
    page (pass ``source='split'`` / ``'inside'`` so the Rating column is attributed to that rater)."""
    return parse_ratings_html(html, source=source, chamber=chamber, ts=ts, ratings_prob=ratings_prob)


# ---------------------------------------------------------------------------------------------
# Generic ballot
# ---------------------------------------------------------------------------------------------

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_D_PLUS_RE = re.compile(r"\b(?:D|Dem|Dems|Democrats?)\s*\+\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_R_PLUS_RE = re.compile(r"\b(?:R|Rep|GOP|Republicans?)\s*\+\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_DEM_PCT_RE = re.compile(r"\bDemocrats?\b[^0-9%]{0,40}?(\d{1,2}(?:\.\d+)?)\s*%", re.IGNORECASE)
_REP_PCT_RE = re.compile(r"\bRepublicans?\b[^0-9%]{0,40}?(\d{1,2}(?:\.\d+)?)\s*%", re.IGNORECASE)


def _num(text: str) -> float | None:
    m = _NUM_RE.search(text or "")
    return float(m.group(0)) if m else None


def parse_generic_ballot(html: str, *, source: str, ts: datetime) -> GenericBallotRow | None:
    """The site's generic-ballot *average*, in order of preference: the 'Average' row of a table with
    Democrat and Republican columns (RealClearPolling), a ``D+8.0`` / ``R+2`` text pattern (FiftyPlusOne's
    summary card), the first row of such a table, else a ``Democrats 51.1% ... Republicans 43.3%`` pattern."""
    first_row: GenericBallotRow | None = None
    for table in extract_tables(html):
        header = table.header
        if not header:
            continue
        dem_col = next((i for i, h in enumerate(header) if _DEM_HEADER_RE.search(h)), None)
        rep_col = next((i for i, h in enumerate(header) if _REP_HEADER_RE.search(h) and i != dem_col), None)
        if dem_col is None or rep_col is None:
            continue
        date_col = _date_column(table)
        for r in table.body:
            if max(dem_col, rep_col) >= len(r.cells):
                continue
            dem, rep = _num(r.cells[dem_col]), _num(r.cells[rep_col])
            if dem is None or rep is None:
                continue
            row_ts = parse_date(r.cells[date_col], ts) if date_col is not None and date_col < len(r.cells) else ts
            row = GenericBallotRow(source=source, ts=row_ts or ts, margin=round(dem - rep, 2), dem=dem, rep=rep)
            if _AVG_RE.search(r.cells[0]):
                return row
            if first_row is None:
                first_row = row
    text = html_text(html)
    dm, rm = _DEM_PCT_RE.search(text), _REP_PCT_RE.search(text)
    dem = float(dm.group(1)) if dm else None
    rep = float(rm.group(1)) if rm else None
    m = _D_PLUS_RE.search(text)
    if m:
        return GenericBallotRow(source=source, ts=ts, margin=float(m.group(1)), dem=dem, rep=rep)
    m = _R_PLUS_RE.search(text)
    if m:
        return GenericBallotRow(source=source, ts=ts, margin=-float(m.group(1)), dem=dem, rep=rep)
    if first_row is not None:
        return first_row
    if dem is not None and rep is not None:
        return GenericBallotRow(source=source, ts=ts, margin=round(dem - rep, 2), dem=dem, rep=rep)
    return None


# ---------------------------------------------------------------------------------------------
# FEC candidate totals
# ---------------------------------------------------------------------------------------------

_FEC_PARTY = {"DEM": "D", "DFL": "D", "REP": "R", "GOP": "R", "IND": "I", "NPA": "I", "NNE": "I", "UNK": None}


def _pick(d: Mapping[str, Any] | None, *keys: str) -> Any:
    if not isinstance(d, Mapping):
        return None
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _f(v: Any) -> float | None:
    try:
        return float(v) if v is not None and v != "" else None
    except (TypeError, ValueError):
        return None


def fec_race_id(office: str | None, state: str | None, district: str | int | None) -> str | None:
    """('S', 'TX', None) -> SEN-TX; ('H', 'PA', '08') -> HOUSE-PA-08; ('H', 'AK', '00') -> HOUSE-AK-AL."""
    if not office or not state:
        return None
    o = str(office).upper()[:1]
    st = str(state).upper()
    if o == "S":
        return f"SEN-{st}"
    if o == "H":
        d = format_district(district) if district is not None and str(district).strip() != "" else "AL"
        return f"HOUSE-{st}-{d}"
    return None


def parse_fec_totals(payload: Any) -> list[FecTotal]:
    """``/candidates/totals/`` response (``{"results": [...]}`` or a bare list) -> FecTotal list."""
    rows = payload.get("results") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        return []
    out: list[FecTotal] = []
    for r in rows:
        if not isinstance(r, Mapping):
            continue
        cid = _pick(r, "candidate_id", "candidateId", "id")
        if not cid:
            continue
        party_raw = str(_pick(r, "party", "party_full") or "").upper()
        party = _FEC_PARTY.get(party_raw[:3], None) if party_raw else None
        if party is None and party_raw:
            party = "D" if party_raw.startswith("DEM") else "R" if party_raw.startswith("REP") else "I" if "IND" in party_raw else None
        office = _pick(r, "office", "office_sought")
        state = _pick(r, "state", "candidate_state")
        district = _pick(r, "district", "district_number")
        out.append(
            FecTotal(
                candidate_id=str(cid),
                name=str(_pick(r, "name", "candidate_name") or ""),
                party=party,
                state=str(state).upper() if state else None,
                district=str(district) if district is not None else None,
                office=str(office).upper()[:1] if office else None,
                receipts=_f(_pick(r, "receipts", "total_receipts")),
                disbursements=_f(_pick(r, "disbursements", "total_disbursements")),
                cash_on_hand=_f(_pick(r, "cash_on_hand_end_period", "last_cash_on_hand_end_period", "cash_on_hand")),
                cycle=int(_pick(r, "cycle", "election_year") or 0) or None,
                race_id=fec_race_id(office, state, district),
            )
        )
    return out


def fec_totals_by_race(totals: Iterable[FecTotal], races: Iterable[str] | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    """{race_id: {party: {name, candidate_id, receipts, disbursements, cash_on_hand}}} keeping the
    best-funded candidate per party (filtered to `races` when given)."""
    allowed = set(races) if races is not None else None
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for t in totals:
        if not t.race_id or not t.party or (allowed is not None and t.race_id not in allowed):
            continue
        cur = out.setdefault(t.race_id, {}).get(t.party)
        if cur is None or (t.receipts or 0.0) > (cur.get("receipts") or 0.0):
            out[t.race_id][t.party] = {
                "name": t.name,
                "candidate_id": t.candidate_id,
                "receipts": t.receipts,
                "disbursements": t.disbursements,
                "cash_on_hand": t.cash_on_hand,
            }
    return out


# ---------------------------------------------------------------------------------------------
# Manual ratings / polls (config/ratings_manual.yaml)
# ---------------------------------------------------------------------------------------------


def _as_of(value: Any, default: datetime | None) -> datetime | None:
    """YAML date / datetime / string -> aware UTC datetime (``default`` when absent or unparseable)."""
    if isinstance(value, datetime):
        return ensure_utc(value)
    if value is None or value == "":
        return default
    if hasattr(value, "year") and not isinstance(value, str):  # datetime.date from YAML
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    return parse_date(str(value), default)


def load_manual_ratings(
    path: Path | None = None, *, ratings_prob: Mapping[str, float], now: datetime | None = None
) -> list[RatingRow]:
    """config/ratings_manual.yaml ``ratings:`` entries -> RatingRow list (explicit ``prob_dem`` wins)."""
    p = path or RATINGS_MANUAL_YAML
    if not p.exists():
        return []
    try:
        data = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as e:
        log.warning("ratings_manual.yaml is invalid: %s", e)
        return []
    if not isinstance(data, dict):
        return []
    default_ts = _as_of(data.get("as_of"), now or utc_now()) or (now or utc_now())
    out: list[RatingRow] = []
    for entry in data.get("ratings") or []:
        if not isinstance(entry, Mapping):
            continue
        rid = str(entry.get("race_id") or "").strip().upper()
        if parse_race_id(rid) is None:
            log.warning("manual rating with bad race_id skipped: %r", rid)
            continue
        source = str(entry.get("source") or "manual").strip().lower()
        ts = _as_of(entry.get("as_of"), default_ts) or default_ts
        rating_text = str(entry.get("rating") or "")
        norm = normalize_rating(rating_text)
        explicit = _f(entry.get("prob_dem"))
        if norm is None and explicit is None:
            log.warning("manual rating for %s has neither a parsable rating nor prob_dem", rid)
            continue
        prob = explicit if explicit is not None else rating_prob_dem(norm[0], norm[1], ratings_prob)
        if prob is None or not 0.0 <= prob <= 1.0:
            log.warning("manual rating for %s cannot be mapped to P(D): %r", rid, rating_text)
            continue
        # An explicit prob_dem is stored under a label without level/party words ("explicit 0.63"):
        # sigcup.trader.fairvalue parses the label first and only falls back to prob_dem when it cannot.
        label = f"explicit {prob:.2f}" if explicit is not None else canonical_rating(*norm)
        out.append(RatingRow(race_id=rid, source=source, rating=label[:32], prob_dem=float(prob), ts=ts, raw=rating_text))
    return out


def load_manual_polls(path: Path | None = None) -> list[PollRow]:
    """Optional ``polls:`` entries (race_id, pollster, end_date, dem, rep | margin, ...) -> PollRow list."""
    p = path or RATINGS_MANUAL_YAML
    if not p.exists():
        return []
    try:
        data = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError:
        return []
    if not isinstance(data, dict):
        return []
    out: list[PollRow] = []
    for e in data.get("polls") or []:
        if not isinstance(e, Mapping):
            continue
        rid = str(e.get("race_id") or "").strip().upper()
        end = _as_of(e.get("end_date"), None)
        if parse_race_id(rid) is None or end is None:
            continue
        dem, rep, margin = _f(e.get("dem")), _f(e.get("rep")), _f(e.get("margin"))
        if margin is None and dem is not None and rep is not None:
            margin = round(dem - rep, 2)
        if margin is None:
            continue
        start = _as_of(e.get("start_date"), None)
        try:
            n = int(e["sample_size"]) if e.get("sample_size") is not None else None
        except (TypeError, ValueError):
            n = None
        out.append(
            PollRow(
                race_id=rid,
                pollster=str(e.get("pollster") or "manual")[:128],
                end_date=end,
                margin=float(margin),
                start_date=start,
                sample_size=n,
                population=(str(e["population"])[:8] if e.get("population") else None),
                dem=dem,
                rep=rep,
                grade=(str(e["grade"])[:8] if e.get("grade") else None),
                source_url=(str(e["source_url"])[:512] if e.get("source_url") else None),
            )
        )
    return out


# ---------------------------------------------------------------------------------------------
# Writers (idempotent)
# ---------------------------------------------------------------------------------------------

SessionFactory = Callable[[], Any]


def ensure_race_rows(session_factory: SessionFactory, data: Mapping[str, Any] | None = None) -> int:
    """Insert Race rows from config/races.yaml for race ids missing in the DB (never updates existing
    rows: the bootstrap owns those). Returns rows inserted. Lets ratings land before bootstrap runs."""
    data = data if data is not None else load_races_yaml()
    races = data.get("races") or {}
    if not isinstance(races, Mapping):
        return 0
    n = 0
    with session_factory() as s:
        existing = set(s.execute(select(Race.race_id)).scalars().all())
        for rid, entry in races.items():
            if rid in existing:
                continue
            vals = race_row_values(str(rid), entry if isinstance(entry, Mapping) else None)
            if vals is None:
                continue
            s.add(Race(**vals))
            existing.add(rid)
            n += 1
    return n


def known_race_ids(session_factory: SessionFactory) -> set[str]:
    with session_factory() as s:
        return set(s.execute(select(Race.race_id)).scalars().all())


def write_ratings(session_factory: SessionFactory, rows: Iterable[RatingRow], *, known_races: set[str] | None = None) -> int:
    """Store ratings for known races; skip unchanged/older ones (see the module docstring)."""
    rows = [r for r in rows if r.prob_dem is not None]
    if not rows:
        return 0
    known = known_races if known_races is not None else known_race_ids(session_factory)
    n = 0
    skipped_unknown: set[str] = set()
    with session_factory() as s:
        for r in rows:
            if r.race_id not in known:
                skipped_unknown.add(r.race_id)
                continue
            latest = s.execute(
                select(Rating)
                .where(Rating.race_id == r.race_id, Rating.source == r.source)
                .order_by(Rating.ts.desc(), Rating.id.desc())
                .limit(1)
            ).scalar_one_or_none()
            if latest is not None:
                lts = ensure_utc(latest.ts)
                if lts >= r.ts:
                    continue
                if latest.rating == r.rating and abs(float(latest.prob_dem) - float(r.prob_dem)) < 1e-9:
                    if r.ts - lts < timedelta(days=RATING_REFRESH_DAYS):
                        continue
            s.add(r.to_row())
            s.flush()
            n += 1
    if skipped_unknown:
        log.info("ratings for %d unknown race ids skipped (e.g. %s)", len(skipped_unknown), sorted(skipped_unknown)[:5])
    return n


def write_generic_ballot(session_factory: SessionFactory, rows: Iterable[GenericBallotRow]) -> int:
    n = 0
    with session_factory() as s:
        for r in rows:
            latest = s.execute(
                select(GenericBallot).where(GenericBallot.source == r.source).order_by(GenericBallot.ts.desc(), GenericBallot.id.desc()).limit(1)
            ).scalar_one_or_none()
            if latest is not None:
                lts = ensure_utc(latest.ts)
                if lts >= r.ts:
                    continue
                if abs(float(latest.margin) - r.margin) < 1e-9 and r.ts - lts < timedelta(days=1):
                    continue
            s.add(r.to_row())
            s.flush()
            n += 1
    return n


def write_polls(session_factory: SessionFactory, rows: Iterable[PollRow], *, known_races: set[str] | None = None, now: datetime | None = None) -> int:
    known = known_races if known_races is not None else known_race_ids(session_factory)
    fetched = now or utc_now()
    n = 0
    with session_factory() as s:
        for r in rows:
            if r.race_id not in known:
                continue
            dup = s.execute(
                select(Poll.id).where(
                    Poll.race_id == r.race_id, Poll.pollster == r.pollster, Poll.end_date == r.end_date, Poll.margin == r.margin
                )
            ).first()
            if dup is not None:
                continue
            s.add(r.to_row(fetched))
            s.flush()
            n += 1
    return n


def write_fec_totals(session_factory: SessionFactory, by_race: Mapping[str, Any], *, now: datetime | None = None) -> int:
    """Store the per-race FEC totals under KV ``fec_totals``; returns the number of races stored."""
    ts = now or utc_now()
    with session_factory() as s:
        kv_set(s, KV_FEC_TOTALS, {"as_of": ts.isoformat(), "cycle": FEC_CYCLE, "races": dict(by_race)})
        kv_set(s, KV_FEC_LAST_RUN, {"ts": ts.isoformat()})
    return len(by_race)


def fec_due(session_factory: SessionFactory, now: datetime) -> bool:
    with session_factory() as s:
        last = kv_get(s, KV_FEC_LAST_RUN)
    if not isinstance(last, Mapping) or not last.get("ts"):
        return True
    try:
        return now - ensure_utc(datetime.fromisoformat(str(last["ts"]))) >= FEC_MIN_INTERVAL
    except ValueError:
        return True


# ---------------------------------------------------------------------------------------------
# Sources and fetchers
# ---------------------------------------------------------------------------------------------

RatingParser = Callable[..., list[RatingRow]]


@dataclass(frozen=True)
class RatingSource:
    name: str
    url: str
    source: str  # Rating.source
    chamber: str | None  # senate | governor | house | None (mixed, e.g. a changes log)


RATING_SOURCES: tuple[RatingSource, ...] = (
    RatingSource("sabato_changes", "https://centerforpolitics.org/crystalball/2026-rating-changes/", "sabato", None),
    RatingSource("sabato_senate", "https://centerforpolitics.org/crystalball/2026-senate/", "sabato", "senate"),
    RatingSource("sabato_governor", "https://centerforpolitics.org/crystalball/2026-governor/", "sabato", "governor"),
    RatingSource("sabato_house", "https://centerforpolitics.org/crystalball/2026-house/", "sabato", "house"),
    RatingSource("cook_senate", "https://www.cookpolitical.com/ratings/senate-race-ratings", "cook", "senate"),
    RatingSource("cook_governor", "https://www.cookpolitical.com/ratings/governor-race-ratings", "cook", "governor"),
    RatingSource("cook_house", "https://www.cookpolitical.com/print/ratings/races/house", "cook", "house"),
    RatingSource("rttwh_senate", "https://www.racetothewh.com/senate/26", "rttwh", "senate"),
    RatingSource("rttwh_governor", "https://www.racetothewh.com/governor/26", "rttwh", "governor"),
    RatingSource("rttwh_house", "https://www.racetothewh.com/house", "rttwh", "house"),
    RatingSource("270towin_senate", "https://www.270towin.com/2026-senate-election/consensus-2026-senate-forecast", "270towin", "senate"),
    RatingSource("270towin_governor", "https://www.270towin.com/2026-governor-election/consensus-2026-governor-forecast", "270towin", "governor"),
    RatingSource("270towin_house", "https://www.270towin.com/2026-house-election/consensus-2026-house-forecast", "270towin", "house"),
    RatingSource("split_senate", "https://www.270towin.com/2026-senate-election/split-ticket-2026-senate-ratings", "split", "senate"),
    RatingSource("inside_senate", "https://www.270towin.com/2026-senate-election/inside-elections-2026-senate-ratings", "inside", "senate"),
)

GENERIC_BALLOT_SOURCES: tuple[tuple[str, str], ...] = (
    ("rcp", "https://www.realclearpolling.com/polls/state-of-the-union/generic-congressional-vote"),
    ("fiftyplusone", "https://www.fiftyplusone.news/polls/generic-ballot/generic-ballot"),
)


def _client(http: httpx.AsyncClient | None) -> tuple[httpx.AsyncClient, bool]:
    if http is not None:
        return http, False
    return httpx.AsyncClient(timeout=HTTP_TIMEOUT_S, headers={"User-Agent": USER_AGENT}, follow_redirects=True), True


async def fetch_text(http: httpx.AsyncClient, url: str) -> str | None:
    """GET -> body text; None on any failure (logged without query parameters)."""
    try:
        resp = await http.get(url)
    except httpx.HTTPError as e:
        log.warning("GET %s failed: %s", url.split("?")[0], type(e).__name__)
        return None
    if resp.status_code != 200:
        log.warning("GET %s -> %s", url.split("?")[0], resp.status_code)
        return None
    return resp.text


async def fetch_json(http: httpx.AsyncClient, url: str, params: Mapping[str, Any] | None = None) -> Any:
    try:
        resp = await http.get(url, params=params)
    except httpx.HTTPError as e:
        log.warning("GET %s failed: %s", url, type(e).__name__)
        return None
    if resp.status_code != 200:
        log.warning("GET %s -> %s", url, resp.status_code)
        return None
    try:
        return resp.json()
    except ValueError:
        return None


async def fetch_fec_totals(
    http: httpx.AsyncClient, api_key: str, *, base: str = FEC_BASE, cycle: int = FEC_CYCLE, pages: Mapping[str, int] | None = None
) -> list[FecTotal]:
    """Top candidates by receipts for the Senate and House this cycle (the key travels as a query
    parameter and is never logged)."""
    pages = pages or FEC_PAGES
    out: list[FecTotal] = []
    for office, npages in pages.items():
        for page in range(1, npages + 1):
            payload = await fetch_json(
                http,
                f"{base.rstrip('/')}/candidates/totals/",
                {
                    "api_key": api_key,
                    "cycle": cycle,
                    "election_year": cycle,
                    "office": office,
                    "is_active_candidate": "true",
                    "sort": "-receipts",
                    "per_page": 100,
                    "page": page,
                },
            )
            if payload is None:
                break
            rows = parse_fec_totals(payload)
            out.extend(rows)
            if len(rows) < 100:
                break
    return out


def parse_rating_source(src: RatingSource, html: str, *, ts: datetime, ratings_prob: Mapping[str, float]) -> list[RatingRow]:
    return parse_ratings_html(html, source=src.source, chamber=src.chamber, ts=ts, ratings_prob=ratings_prob)


# ---------------------------------------------------------------------------------------------
# Collector entry point
# ---------------------------------------------------------------------------------------------


async def run_once(
    session_factory: SessionFactory,
    cfg: StrategyConfig,
    secrets: Secrets | None = None,
    *,
    http: httpx.AsyncClient | None = None,
    now: datetime | None = None,
    rating_sources: Sequence[RatingSource] | None = None,
    generic_sources: Sequence[tuple[str, str]] | None = None,
    fetch_web: bool = True,
    fetch_fec: bool = True,
    manual_path: Path | None = None,
) -> int:
    """One pass: manual ratings/polls, then every web ratings page, the generic-ballot pages and (at most
    once per FEC_MIN_INTERVAL) FEC totals. Returns rows written (ratings + generic ballot + polls + FEC races)."""
    now = now or utc_now()
    ratings_prob = cfg.fairvalue.ratings_prob
    total = 0
    try:
        inserted = ensure_race_rows(session_factory)
        if inserted:
            log.info("polls: inserted %d Race rows from races.yaml", inserted)
    except Exception as e:  # noqa: BLE001
        log.warning("polls: could not seed Race rows: %s", e)
    known = known_race_ids(session_factory)

    manual = load_manual_ratings(manual_path, ratings_prob=ratings_prob, now=now)
    n = write_ratings(session_factory, manual, known_races=known)
    total += n
    log.info("polls: manual ratings %d parsed, %d written", len(manual), n)
    total += write_polls(session_factory, load_manual_polls(manual_path), known_races=known, now=now)

    if not fetch_web and not fetch_fec:
        return total
    client, owns = _client(http)
    try:
        if fetch_web:
            for src in rating_sources if rating_sources is not None else RATING_SOURCES:
                html = await fetch_text(client, src.url)
                if not html:
                    continue
                try:
                    rows = parse_rating_source(src, html, ts=now, ratings_prob=ratings_prob)
                except Exception as e:  # noqa: BLE001
                    log.warning("polls: %s parse failed: %s", src.name, e)
                    continue
                n = write_ratings(session_factory, rows, known_races=known)
                total += n
                log.info("polls: %s -> %d ratings parsed, %d written", src.name, len(rows), n)
            gb_rows: list[GenericBallotRow] = []
            for source, url in generic_sources if generic_sources is not None else GENERIC_BALLOT_SOURCES:
                html = await fetch_text(client, url)
                if not html:
                    continue
                try:
                    gb = parse_generic_ballot(html, source=source, ts=now)
                except Exception as e:  # noqa: BLE001
                    log.warning("polls: generic ballot %s parse failed: %s", source, e)
                    continue
                if gb is not None:
                    gb_rows.append(gb)
            if gb_rows:
                n = write_generic_ballot(session_factory, gb_rows)
                total += n
                log.info("polls: generic ballot %s -> %d written", [(g.source, g.margin) for g in gb_rows], n)
        if fetch_fec:
            key = (secrets.fec_api_key if secrets is not None else "") or ""
            if key and fec_due(session_factory, now):
                totals = await fetch_fec_totals(client, key)
                by_race = fec_totals_by_race(totals, known)
                if by_race:
                    total += write_fec_totals(session_factory, by_race, now=now)
                    log.info("polls: FEC totals for %d races", len(by_race))
    finally:
        if owns:
            await client.aclose()
    return total
