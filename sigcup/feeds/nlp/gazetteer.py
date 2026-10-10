"""Entity linking post -> race_ids from config/races.yaml candidates (names, aliases, handles, hashtags,
race codes). Surname-only matches require a co-occurring state/race token or handle.

Also the shared ingest helpers every collector uses (``PostDraft``, ``write_posts``, ``parse_ts``,
``pick``, ``strip_html``): they live here because the gazetteer is the one module every collector
already imports, and the ingest step *is* "link, bot-score, persist".

Linking rules (confidence in parentheses):

* full candidate name, allowing up to two middle tokens         (0.95)
* ``@handle`` in the text, or the post author *is* the candidate (0.95 / 0.90)
* campaign hashtag from races.yaml                              (0.90)
* alias from races.yaml                                         (0.85)
* surname only, **only if** the text also carries a race token
  for that race (state name, ``#GASen``, ``GA-Sen``, ``PA-08``,
  "Georgia's 8th district" ...) or a handle of a candidate
  in that race                                                  (0.60)
* race code only (``#NCSen``, ``AZ-01``), no candidate: party is
  ``None`` and the stance classifier picks the party            (0.50)

Collisions: a surname shared by candidates in several races is linked only to the race whose state
token co-occurs; same-state collisions additionally need a chamber token (Senate / Governor /
district). Surnames that are common English words ("Young", "Hill") never match on their own.
"""

from __future__ import annotations

import html
import logging
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from time import struct_time
from typing import Any

from sqlalchemy import func, select

from sigcup.common.ids import text_hash
from sigcup.common.time import ensure_utc, utc_now
from sigcup.db.models import Post, Race
from sigcup.feeds.nlp import botfilter

log = logging.getLogger(__name__)

Link = tuple[str, str | None, float]
"""(race_id, party or None for race-code-only matches, confidence in [0, 1])."""

STATE_NAMES: dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
    "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "DC": "District of Columbia",
}

# Surnames that are ordinary English words: never linked on the surname alone, even with a race token.
COMMON_WORD_SURNAMES: frozenset[str] = frozenset(
    {
        "young", "brown", "green", "hill", "price", "king", "rich", "long", "wood", "field", "fields",
        "gray", "grey", "white", "black", "rose", "love", "moore", "bell", "hall", "lane", "park",
        "ford", "stone", "wells", "banks", "bush", "gold", "cook", "baker", "may", "day", "ray",
        "west", "north", "south", "east", "page", "post", "reed", "read", "fox", "wolf", "hunt",
        "law", "good", "best", "strong", "small", "little", "sharp", "swift", "hope", "grant",
        "cross", "ball", "bird", "fisher", "miller", "carpenter", "mason", "smith", "frost", "snow",
        "summers", "winters", "cotton", "rice", "berry", "lamb", "hart", "star", "peace", "power",
        "pence", "sanders", "walker", "porter", "hunter", "marsh", "lake", "rivers", "brooks", "burns",
    }
)

PARTY_ALIASES = {"D": "D", "DEM": "D", "DEMOCRAT": "D", "DEMOCRATIC": "D", "R": "R", "REP": "R", "GOP": "R",
                 "REPUBLICAN": "R", "I": "I", "IND": "I", "INDEPENDENT": "I"}

CONF_FULL_NAME = 0.95
CONF_HANDLE = 0.95
CONF_AUTHOR = 0.90
CONF_HASHTAG = 0.90
CONF_ALIAS = 0.85
CONF_SURNAME = 0.60
CONF_RACE_CODE = 0.50

_ORDINAL = r"(?:st|nd|rd|th)"
_DAY = timedelta(days=1)


# ---------------------------------------------------------------------------------------------
# Lenient parsing helpers shared by the collectors
# ---------------------------------------------------------------------------------------------


def pick(d: Any, *keys: str, default: Any = None) -> Any:
    """First present, non-None value among candidate keys (dotted paths allowed; attribute access tolerated)."""
    if d is None:
        return default
    for key in keys:
        cur: Any = d
        ok = True
        for part in key.split("."):
            if isinstance(cur, Mapping):
                if part in cur:
                    cur = cur[part]
                else:
                    ok = False
                    break
            elif hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok and cur is not None and cur != "":
            return cur
    return default


_TWITTER_TS = re.compile(r"^[A-Z][a-z]{2} [A-Z][a-z]{2} \d{1,2} \d{2}:\d{2}:\d{2} [+-]\d{4} \d{4}$")
_GDELT_TS = re.compile(r"^(\d{4})(\d{2})(\d{2})T?(\d{2})(\d{2})(\d{2})Z?$")


def parse_ts(value: Any) -> datetime | None:
    """Parse the timestamp shapes the feeds emit into a tz-aware UTC datetime (None if hopeless).

    Accepts ISO-8601 (with ``Z``), GDELT ``20261010T120000Z``, Twitter ``Tue Dec 10 07:00:30 +0000 2024``,
    RFC-2822, ``time.struct_time`` (feedparser) and epoch seconds / milliseconds / microseconds.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return ensure_utc(value)
    if isinstance(value, struct_time):
        try:
            return datetime(*value[:6], tzinfo=UTC)
        except (TypeError, ValueError):
            return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        v = float(value)
        if v > 1e15:  # microseconds
            v /= 1e6
        elif v > 1e12:  # milliseconds
            v /= 1e3
        try:
            return datetime.fromtimestamp(v, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    s = str(value).strip()
    if not s:
        return None
    if s.isdigit() and len(s) >= 9 and not _GDELT_TS.match(s):
        return parse_ts(int(s))
    m = _GDELT_TS.match(s)
    if m:
        try:
            return datetime(*(int(g) for g in m.groups()), tzinfo=UTC)
        except ValueError:
            return None
    try:
        return ensure_utc(datetime.fromisoformat(s.replace("Z", "+00:00")))
    except ValueError:
        pass
    if _TWITTER_TS.match(s):
        try:
            return ensure_utc(datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y"))
        except ValueError:
            return None
    try:
        return ensure_utc(parsedate_to_datetime(s))
    except (TypeError, ValueError, IndexError):
        return None


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def strip_html(s: Any) -> str:
    """Remove tags and collapse whitespace; tolerant of None / non-strings."""
    if s is None:
        return ""
    txt = html.unescape(_TAG_RE.sub(" ", str(s)))
    return _WS_RE.sub(" ", txt).strip()


# ---------------------------------------------------------------------------------------------
# Gazetteer
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    race_id: str
    party: str
    name: str
    aliases: tuple[str, ...] = ()
    handles: tuple[str, ...] = ()
    hashtags: tuple[str, ...] = ()

    @property
    def surname(self) -> str:
        toks = [t for t in re.split(r"\s+", self.name.strip()) if t]
        return toks[-1] if toks else ""


@dataclass(frozen=True)
class RaceMeta:
    race_id: str
    chamber: str
    state: str | None
    district: str | None
    label: str


def _norm_handle(h: str) -> str:
    return str(h).strip().lstrip("@").lower()


def _norm_tag(t: str) -> str:
    return str(t).strip().lstrip("#").lower()


def _norm_party(p: Any) -> str:
    return PARTY_ALIASES.get(str(p).strip().upper(), str(p).strip().upper()[:8] or "other")


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    try:
        return [str(x) for x in v if x is not None and str(x).strip()]
    except TypeError:
        return []


def _phrase_regex(phrase: str) -> re.Pattern[str]:
    """Case-insensitive whole-phrase match with flexible whitespace and word boundaries."""
    toks = [re.escape(t) for t in re.split(r"\s+", phrase.strip()) if t]
    body = r"\W{1,3}".join(toks)
    return re.compile(rf"(?<![\w@#]){body}(?!\w)", re.IGNORECASE)


def _full_name_regex(name: str) -> re.Pattern[str] | None:
    toks = [re.escape(t) for t in re.split(r"\s+", name.strip()) if t]
    if len(toks) < 2:
        return None
    first, last = toks[0], toks[-1]
    return re.compile(rf"(?<![\w@#]){first}(?:\W{{1,3}}\w+\.?){{0,2}}\W{{1,3}}{last}(?!\w)", re.IGNORECASE)


def _race_token_regexes(meta: RaceMeta) -> tuple[list[re.Pattern[str]], list[re.Pattern[str]]]:
    """``(evidence, codes)``: *evidence* patterns identify the race's state (enough to disambiguate a
    surname); *codes* identify this specific race and chamber (``#GASen``, ``GA-Gov``, ``PA-08``)."""
    st = (meta.state or "").upper()
    if not st:
        return [], []
    evidence: list[re.Pattern[str]] = []
    codes: list[re.Pattern[str]] = []
    name = STATE_NAMES.get(st)
    if name:
        evidence.append(re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE))
    evidence.append(re.compile(rf"(?<![\w@])#{st}(?:pol|politics)(?!\w)", re.IGNORECASE))
    if meta.chamber == "senate":
        codes.append(re.compile(rf"(?<![\w@])#{st}sen(?:ate)?(?!\w)", re.IGNORECASE))
        codes.append(re.compile(rf"(?<![\w@#]){st}[\s-]?(?:sen|senate)(?!\w)", re.IGNORECASE))
        if name:
            codes.append(re.compile(rf"(?<!\w){re.escape(name)}\s+(?:senate|senator|u\.?s\.? senate)(?!\w)", re.IGNORECASE))
    elif meta.chamber == "governor":
        codes.append(re.compile(rf"(?<![\w@])#{st}gov(?:ernor)?(?!\w)", re.IGNORECASE))
        codes.append(re.compile(rf"(?<![\w@#]){st}[\s-]?(?:gov|governor)(?!\w)", re.IGNORECASE))
        if name:
            codes.append(re.compile(rf"(?<!\w){re.escape(name)}\s+(?:governor|gubernatorial)(?!\w)", re.IGNORECASE))
    elif meta.chamber == "house":
        d = (meta.district or "").upper().lstrip("0") or "AL"
        if d == "AL":
            codes.append(re.compile(rf"(?<![\w@#]){st}[\s-]?AL(?!\w)", re.IGNORECASE))
        else:
            codes.append(re.compile(rf"(?<![\w@#]){st}[\s-]?0?{d}(?!\d)", re.IGNORECASE))
            codes.append(re.compile(rf"(?<![\w@])#{st}0?{d}(?!\d)", re.IGNORECASE))
            codes.append(re.compile(rf"(?<!\w){d}{_ORDINAL}\s+(?:congressional\s+)?district(?!\w)", re.IGNORECASE))
            evidence.append(re.compile(rf"(?<!\w)district\s+0?{d}(?!\d)", re.IGNORECASE))
    return evidence, codes


def _chamber_token_regex(meta: RaceMeta) -> re.Pattern[str] | None:
    if meta.chamber == "senate":
        return re.compile(r"(?<!\w)(?:senate|senator|sen\.?|#\w*sen)(?!\w)", re.IGNORECASE)
    if meta.chamber == "governor":
        return re.compile(r"(?<!\w)(?:governor|gubernatorial|gov\.?|#\w*gov)(?!\w)", re.IGNORECASE)
    if meta.chamber == "house":
        d = (meta.district or "").lstrip("0")
        pat = r"(?:house|congress|congressional|district|rep\.?|representative"
        if d and d.upper() != "AL":
            pat += rf"|{d}{_ORDINAL}|-0?{d}\b"
        return re.compile(rf"(?<!\w){pat})(?!\w)", re.IGNORECASE)
    return None


class Gazetteer:
    """Links free text to (race_id, party, confidence) triples; also serves search terms to collectors."""

    def __init__(self, candidates: Iterable[Candidate], races: Mapping[str, RaceMeta] | None = None) -> None:
        self.candidates: list[Candidate] = list(candidates)
        self.races: dict[str, RaceMeta] = dict(races or {})
        for c in self.candidates:
            self.races.setdefault(c.race_id, _meta_from_race_id(c.race_id))
        self._compile()

    # -- construction -----------------------------------------------------------------------

    @classmethod
    def from_races(cls, races: Mapping[str, Mapping[str, Any]]) -> Gazetteer:
        """Build from ``{race_id: {chamber, state, district, label, candidates: {party: {...}}}}``.

        The inner shape is the races.yaml / Race.candidates shape; missing fields are tolerated and a
        bare ``{party: "Full Name"}`` entry is accepted too.
        """
        cands: list[Candidate] = []
        metas: dict[str, RaceMeta] = {}
        for race_id, spec in (races or {}).items():
            if not isinstance(spec, Mapping):
                continue
            rid = str(race_id)
            inferred = _meta_from_race_id(rid)
            metas[rid] = RaceMeta(
                race_id=rid,
                chamber=str(pick(spec, "chamber", default=inferred.chamber)).lower(),
                state=(str(pick(spec, "state", default=inferred.state or "")).upper() or None),
                district=(str(pick(spec, "district", default=inferred.district or "")) or None),
                label=str(pick(spec, "label", default=inferred.label)),
            )
            for party, c in (pick(spec, "candidates", default={}) or {}).items():
                if isinstance(c, str):
                    c = {"name": c}
                if not isinstance(c, Mapping):
                    continue
                name = str(pick(c, "name", "full_name", default="")).strip()
                if not name:
                    continue
                cands.append(
                    Candidate(
                        race_id=rid,
                        party=_norm_party(party),
                        name=name,
                        aliases=tuple(_as_list(pick(c, "aliases", "nicknames", default=[]))),
                        handles=tuple(_norm_handle(h) for h in _as_list(pick(c, "handles", "handle", default=[]))),
                        hashtags=tuple(_norm_tag(t) for t in _as_list(pick(c, "hashtags", "hashtag", default=[]))),
                    )
                )
        return cls(cands, metas)

    @classmethod
    def from_db(cls, session_factory: Callable[[], Any]) -> Gazetteer:
        """Build from Race rows; falls back to config/races.yaml when the table has no candidates."""
        races: dict[str, dict[str, Any]] = {}
        try:
            with session_factory() as s:
                for r in s.execute(select(Race)).scalars().all():
                    races[r.race_id] = {
                        "chamber": r.chamber,
                        "state": r.state,
                        "district": r.district,
                        "label": r.label,
                        "candidates": r.candidates or {},
                    }
        except Exception:  # noqa: BLE001 - a missing table must not kill a collector
            log.exception("gazetteer: could not read races table")
        if not any(v.get("candidates") for v in races.values()):
            from sigcup.settings import load_yaml

            raw = load_yaml("races") or {}
            yaml_races = raw.get("races", raw) if isinstance(raw, Mapping) else {}
            if isinstance(yaml_races, Mapping) and yaml_races:
                log.info("gazetteer: no candidates in DB, using config/races.yaml (%d races)", len(yaml_races))
                races = dict(yaml_races)
        return cls.from_races(races)

    # -- compiled patterns ------------------------------------------------------------------

    def _compile(self) -> None:
        self._full: list[tuple[re.Pattern[str], Candidate]] = []
        self._alias: list[tuple[re.Pattern[str], Candidate]] = []
        self._handles: dict[str, Candidate] = {}
        self._tags: dict[str, Candidate] = {}
        self._surname: dict[str, list[Candidate]] = {}
        self._race_tokens: dict[str, list[re.Pattern[str]]] = {}
        self._chamber_tokens: dict[str, re.Pattern[str] | None] = {}
        self._race_handles: dict[str, set[str]] = {}
        for c in self.candidates:
            fr = _full_name_regex(c.name)
            if fr is not None:
                self._full.append((fr, c))
            for a in c.aliases:
                if a.strip():
                    self._alias.append((_phrase_regex(a), c))
            for h in c.handles:
                if h:
                    self._handles[h] = c
                    self._race_handles.setdefault(c.race_id, set()).add(h)
            for t in c.hashtags:
                if t:
                    self._tags[t] = c
            sn = c.surname.lower()
            if sn and len(sn) >= 3:
                self._surname.setdefault(sn, []).append(c)
        self._race_codes: dict[str, list[re.Pattern[str]]] = {}
        for rid, meta in self.races.items():
            evidence, codes = _race_token_regexes(meta)
            self._race_tokens[rid] = evidence + codes
            self._race_codes[rid] = codes
            self._chamber_tokens[rid] = _chamber_token_regex(meta)
        self._surname_re: dict[str, re.Pattern[str]] = {
            sn: re.compile(rf"(?<![\w@#]){re.escape(sn)}(?!\w)", re.IGNORECASE) for sn in self._surname
        }
        self._handle_re = re.compile(r"(?<!\w)@([A-Za-z0-9_]{1,30})")
        self._tag_re = re.compile(r"(?<!\w)#([A-Za-z0-9_]{1,60})")

    # -- queries ----------------------------------------------------------------------------

    def race_ids(self) -> list[str]:
        return sorted(self.races)

    def candidates_for(self, race_id: str) -> list[Candidate]:
        return [c for c in self.candidates if c.race_id == race_id]

    def search_terms(self, race_id: str) -> list[str]:
        """Full names (plus multi-word aliases) for keyword queries to GDELT / X / Bluesky search."""
        out: list[str] = []
        for c in self.candidates_for(race_id):
            if len(c.name.split()) >= 2:
                out.append(c.name)
            out.extend(a for a in c.aliases if len(a.split()) >= 2)
        return list(dict.fromkeys(out))

    def race_context(self, race_id: str) -> dict[str, Any]:
        """Compact description for LLM prompts: label, state, chamber, candidates by party."""
        meta = self.races.get(race_id) or _meta_from_race_id(race_id)
        return {
            "race_id": race_id,
            "label": meta.label,
            "state": meta.state,
            "chamber": meta.chamber,
            "candidates": {c.party: c.name for c in self.candidates_for(race_id)},
        }

    def has_race_token(self, race_id: str, text: str) -> bool:
        return any(p.search(text) for p in self._race_tokens.get(race_id, ()))

    def link(self, text: str, author_handle: str | None = None) -> list[Link]:
        """Link ``text`` to races; see the module docstring for the rules. Sorted by confidence desc."""
        if not text:
            return []
        best: dict[tuple[str, str | None], float] = {}

        def hit(c: Candidate | None, race_id: str, conf: float, party: str | None = None) -> None:
            key = (race_id, c.party if c is not None else party)
            if conf > best.get(key, 0.0):
                best[key] = conf

        for rx, c in self._full:
            if rx.search(text):
                hit(c, c.race_id, CONF_FULL_NAME)
        for rx, c in self._alias:
            if rx.search(text):
                hit(c, c.race_id, CONF_ALIAS)
        for m in self._handle_re.finditer(text):
            c = self._handles.get(m.group(1).lower())
            if c is not None:
                hit(c, c.race_id, CONF_HANDLE)
        if author_handle:
            c = self._handles.get(_norm_handle(author_handle))
            if c is not None:
                hit(c, c.race_id, CONF_AUTHOR)
        for m in self._tag_re.finditer(text):
            c = self._tags.get(m.group(1).lower())
            if c is not None:
                hit(c, c.race_id, CONF_HASHTAG)

        # Surname-only: needs a race token (or a race handle) for the specific race.
        for sn, cands in self._surname.items():
            if sn in COMMON_WORD_SURNAMES or not self._surname_re[sn].search(text):
                continue
            eligible = [c for c in cands if (c.race_id, c.party) not in best]
            eligible = [c for c in eligible if self._race_evidence(c.race_id, text, author_handle)]
            if len({c.race_id for c in eligible}) > 1:
                # same-state collision -> need a chamber token too
                eligible = [c for c in eligible if self._chamber_evidence(c.race_id, text)]
                if len({c.race_id for c in eligible}) > 1:
                    continue
            for c in eligible:
                hit(c, c.race_id, CONF_SURNAME)

        # Race-code-only matches (e.g. "#NCSen", "AZ-01"): party unknown. State names alone are too weak.
        linked_races = {rid for rid, _ in best}
        for rid, pats in self._race_codes.items():
            if rid in linked_races:
                continue
            if any(p.search(text) for p in pats):
                hit(None, rid, CONF_RACE_CODE, None)

        return sorted(((rid, party, conf) for (rid, party), conf in best.items()), key=lambda t: (-t[2], t[0]))

    def link_race_ids(self, text: str, author_handle: str | None = None, min_conf: float = 0.0) -> list[str]:
        """Distinct race_ids from :meth:`link`, best first."""
        out: list[str] = []
        for rid, _party, conf in self.link(text, author_handle):
            if conf >= min_conf and rid not in out:
                out.append(rid)
        return out

    def _race_evidence(self, race_id: str, text: str, author_handle: str | None) -> bool:
        if self.has_race_token(race_id, text):
            return True
        handles = self._race_handles.get(race_id, set())
        if author_handle and _norm_handle(author_handle) in handles:
            return True
        return any(m.group(1).lower() in handles for m in self._handle_re.finditer(text))

    def _chamber_evidence(self, race_id: str, text: str) -> bool:
        rx = self._chamber_tokens.get(race_id)
        return bool(rx and rx.search(text))


def _meta_from_race_id(race_id: str) -> RaceMeta:
    parts = race_id.upper().split("-")
    chamber = {"SEN": "senate", "GOV": "governor", "HOUSE": "house", "CONTROL": "control"}.get(parts[0], parts[0].lower())
    state = parts[1] if len(parts) > 1 and len(parts[1]) == 2 and chamber != "control" else None
    district = parts[2] if chamber == "house" and len(parts) > 2 else None
    label = race_id
    if state:
        label = f"{STATE_NAMES.get(state, state)} {chamber}" + (f" {district}" if district else "")
    return RaceMeta(race_id=race_id, chamber=chamber, state=state, district=district, label=label)


def load_gazetteer(session_factory: Callable[[], Any] | None) -> Gazetteer:
    """Gazetteer from the DB (yaml fallback); an empty gazetteer when no session factory is given."""
    if session_factory is None:
        return Gazetteer([])
    return Gazetteer.from_db(session_factory)


# ---------------------------------------------------------------------------------------------
# Post ingest
# ---------------------------------------------------------------------------------------------


@dataclass
class PostDraft:
    """A normalised post before linking / bot scoring. ``post_id`` must already carry the platform prefix."""

    post_id: str
    platform: str
    ts: datetime
    text: str
    author_id: str | None = None
    author_handle: str | None = None
    author_created_at: datetime | None = None
    url: str | None = None
    race_ids: list[str] = field(default_factory=list)
    link_text: str | None = None  # extra context for the gazetteer (video title, feed name) not stored as text


def write_posts(
    session_factory: Callable[[], Any],
    drafts: Iterable[PostDraft],
    gazetteer: Gazetteer | None = None,
    *,
    require_link: bool = True,
    now: datetime | None = None,
) -> int:
    """Link, bot-score and persist drafts; returns the number of NEW Post rows (existing ids are skipped).

    ``require_link=True`` drops posts that neither the draft nor the gazetteer tie to a race.
    """
    now = now or utc_now()
    written = 0
    seen_batch: dict[str, int] = {}
    with session_factory() as s:
        for d in drafts:
            if not d.post_id or not d.text or not d.text.strip():
                continue
            pid = d.post_id[:128]
            if s.get(Post, pid) is not None:
                continue
            race_ids = list(dict.fromkeys(d.race_ids or []))
            if gazetteer is not None:
                link_src = d.text if not d.link_text else f"{d.text}\n{d.link_text}"
                for rid in gazetteer.link_race_ids(link_src, d.author_handle):
                    if rid not in race_ids:
                        race_ids.append(rid)
            if require_link and not race_ids:
                continue
            th = text_hash(d.text)
            dup_db = s.execute(select(func.count()).select_from(Post).where(Post.text_hash == th)).scalar() or 0
            dup = int(dup_db) + seen_batch.get(th, 0)
            seen_batch[th] = seen_batch.get(th, 0) + 1
            rate = 0
            if d.author_id:
                rate = int(
                    s.execute(
                        select(func.count())
                        .select_from(Post)
                        .where(Post.platform == d.platform, Post.author_id == d.author_id, Post.ts >= now - _DAY)
                    ).scalar()
                    or 0
                )
            score = botfilter.score(d, dup_count=dup, author_posts_24h=rate, now=now)
            s.add(
                Post(
                    post_id=pid,
                    platform=d.platform[:16],
                    ts=ensure_utc(d.ts),
                    author_id=(d.author_id or None) and str(d.author_id)[:128],
                    author_handle=(d.author_handle or None) and str(d.author_handle)[:128],
                    author_created_at=ensure_utc(d.author_created_at) if d.author_created_at else None,
                    text=d.text,
                    text_hash=th,
                    url=(d.url or None) and str(d.url)[:512],
                    race_ids=race_ids,
                    bot_score=score,
                    classified=False,
                    fetched_at=now,
                )
            )
            written += 1
    return written

