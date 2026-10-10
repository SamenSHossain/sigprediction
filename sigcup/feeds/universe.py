"""The race universe: state table, race_id conventions, SIG market-title parsing, Kalshi / Polymarket
crosswalk patterns, rating-string normalisation and config/races.yaml load/save.

Conventions (shared with the DB and the trader):

* race_id: ``SEN-XX`` (Senate, incl. specials), ``GOV-XX`` (governor), ``HOUSE-XX-NN`` (House district,
  two-digit zero-padded; ``AL`` for at-large), ``CONTROL-SENATE`` / ``CONTROL-HOUSE``.
* party legs: ``D``, ``R``, ``I`` (independent legs: NE-Sen Osborn, MT-Sen, RI-Gov), ``L``/``G`` tolerated.
* ``exclusive`` / ``exhaustive`` on a race come from the exchange relationships API only, never from titles.

Everything in this module is pure (no network, no DB) so it is testable from fixtures; the bootstrap
script and the collectors import it.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sigcup.settings import CONFIG_DIR

RACES_YAML = CONFIG_DIR / "races.yaml"

# ---------------------------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------------------------

STATES: dict[str, str] = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "DC": "District of Columbia",
}
STATE_BY_NAME: dict[str, str] = {v.lower(): k for k, v in STATES.items()}
# Longest names first so "West Virginia" wins over "Virginia" and "North Carolina" over "Carolina".
_STATE_NAMES_DESC: list[str] = sorted(STATE_BY_NAME, key=len, reverse=True)
_STATE_NAME_RE = re.compile(
    r"\b(" + "|".join(re.escape(n) for n in _STATE_NAMES_DESC) + r")\b", re.IGNORECASE
)
_STATE_CODE_RE = re.compile(r"\b(" + "|".join(STATES) + r")\b")

CHAMBERS = ("senate", "governor", "house", "control")
PARTIES = ("D", "R", "I")

_PARTY_WORDS: dict[str, str] = {
    "democratic": "D",
    "democrat": "D",
    "democrats": "D",
    "dem": "D",
    "dems": "D",
    "d": "D",
    "republican": "R",
    "republicans": "R",
    "gop": "R",
    "rep": "R",
    "r": "R",
    "independent": "I",
    "independents": "I",
    "ind": "I",
    "i": "I",
    "libertarian": "L",
    "green": "G",
}
# Independent candidates known to carry their own SIG leg (lower-case surname -> (race_id, party)).
KNOWN_INDEPENDENTS: dict[str, tuple[str, str]] = {
    "osborn": ("SEN-NE", "I"),
}

_ORDINAL_RE = re.compile(r"(\d{1,2})(?:st|nd|rd|th)?", re.IGNORECASE)


def state_name(code: str) -> str:
    """'TX' -> 'Texas' (the code itself when unknown)."""
    return STATES.get(code.upper(), code)


def find_state(text: str) -> str | None:
    """First full state name in `text` (longest names first), else a bare two-letter code, else None."""
    m = _STATE_NAME_RE.search(text)
    if m:
        return STATE_BY_NAME[m.group(1).lower()]
    m = _STATE_CODE_RE.search(text)
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------------------------
# race ids
# ---------------------------------------------------------------------------------------------


def make_race_id(chamber: str, state: str | None, district: str | int | None = None) -> str:
    """Build a race_id from its parts (``HOUSE-PA-08``; districts are zero-padded; ``AL`` = at-large)."""
    ch = chamber.lower()
    if ch == "control":
        if state is None:
            raise ValueError("control races need 'senate' or 'house' in place of the state")
        return f"CONTROL-{state.upper()}"
    if state is None:
        raise ValueError("state is required")
    st = state.upper()
    if ch == "senate":
        return f"SEN-{st}"
    if ch == "governor":
        return f"GOV-{st}"
    if ch == "house":
        if district is None:
            raise ValueError("house races need a district")
        return f"HOUSE-{st}-{format_district(district)}"
    raise ValueError(f"unknown chamber {chamber!r}")


def format_district(district: str | int) -> str:
    """'8' / 8 / '08' -> '08'; 'AL' / 'at-large' -> 'AL'."""
    if isinstance(district, int):
        return f"{district:02d}"
    d = str(district).strip().upper().replace("AT-LARGE", "AL").replace("AT LARGE", "AL")
    if d in ("AL", "0", "00"):
        return "AL"
    m = _ORDINAL_RE.fullmatch(d)
    if m:
        return f"{int(m.group(1)):02d}"
    return d


@dataclass(frozen=True)
class RaceKey:
    chamber: str  # senate | governor | house | control
    state: str | None  # two-letter code; for control: None
    district: str | None  # '08' | 'AL' | None
    control_of: str | None = None  # 'senate' | 'house' for control races

    @property
    def race_id(self) -> str:
        if self.chamber == "control":
            return make_race_id("control", self.control_of)
        return make_race_id(self.chamber, self.state, self.district)

    @property
    def label(self) -> str:
        if self.chamber == "control":
            return f"{(self.control_of or '').title()} control"
        st = state_name(self.state or "")
        if self.chamber == "senate":
            return f"{st} Senate"
        if self.chamber == "governor":
            return f"{st} Governor"
        return f"{self.state}-{self.district} House"


def parse_race_id(race_id: str) -> RaceKey | None:
    """Inverse of make_race_id; None for strings that are not race ids."""
    rid = race_id.strip().upper()
    if rid in ("CONTROL-SENATE", "CONTROL-HOUSE"):
        return RaceKey("control", None, None, rid.split("-")[1].lower())
    m = re.fullmatch(r"SEN-([A-Z]{2})", rid)
    if m:
        return RaceKey("senate", m.group(1), None)
    m = re.fullmatch(r"GOV-([A-Z]{2})", rid)
    if m:
        return RaceKey("governor", m.group(1), None)
    m = re.fullmatch(r"HOUSE-([A-Z]{2})-([0-9]{1,2}|AL)", rid)
    if m:
        return RaceKey("house", m.group(1), format_district(m.group(2)))
    return None


def chamber_of(race_id: str) -> str | None:
    key = parse_race_id(race_id)
    return key.chamber if key else None


# ---------------------------------------------------------------------------------------------
# SIG market-title parsing
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedTitle:
    race_id: str
    party: str
    chamber: str
    state: str | None
    district: str | None
    label: str
    special: bool = False
    candidate: str | None = None  # named candidate when the title names one instead of a party


_CONTROL_RE = re.compile(
    r"\b(?:control(?:\s+of)?|majority(?:\s+in|\s+of)?|win(?:\s+control\s+of)?)\s+(?:the\s+)?(?:u\.?\s?s\.?\s+)?"
    r"(senate|house)(?:\s+of\s+representatives)?\b",
    re.IGNORECASE,
)
_DISTRICT_CODE_RE = re.compile(r"\b([A-Z]{2})[-\s]?(\d{1,2}|AL)\b")
_DISTRICT_WORD_RE = re.compile(
    r"(?:'s)?\s+(\d{1,2})(?:st|nd|rd|th)?\s+(?:congressional\s+)?district\b|\bdistrict\s+(\d{1,2})\b|\bat[-\s]large\b",
    re.IGNORECASE,
)
_PARTY_RE = re.compile(
    r"\b(democratic|democrats?|dems?|republicans?|gop|independents?|ind|libertarian|green)\b", re.IGNORECASE
)
_PAREN_PARTY_RE = re.compile(r"\((D|R|I|DEM|REP|IND)\)", re.IGNORECASE)
_CANDIDATE_RE = re.compile(r"^will\s+(.+?)\s+(?:win|carry|take|flip|hold)\b", re.IGNORECASE)


def _clean_title(title: str) -> str:
    t = title.replace("’", "'").replace("–", "-").replace("—", "-")
    t = re.sub(r"\s+", " ", t).strip().rstrip("?").strip()
    return t


def _detect_party(subject: str) -> tuple[str | None, str | None]:
    """(party, candidate_name) from the part of the title before 'win'/'control'."""
    name_part = re.sub(r"^\s*will\s+", "", subject, flags=re.IGNORECASE).strip()
    m = _PAREN_PARTY_RE.search(subject)
    if m:
        party = _PARTY_WORDS.get(m.group(1).lower())
        name = _PAREN_PARTY_RE.sub("", name_part).strip()
        return party, name or None
    m = _PARTY_RE.search(subject)
    if m:
        return _PARTY_WORDS.get(m.group(1).lower()), None
    low = subject.lower()
    for surname, (_rid, party) in KNOWN_INDEPENDENTS.items():
        if surname in low:
            return party, name_part or None
    return None, None


_NOT_A_WINNER_MARKET_RE = re.compile(
    r"\b(primary|primaries|nominee|nomination|runoff|margin|popular vote|turnout|by more than|by less than|"
    r"vote share|debate|endorse|drop out|withdraw|resign|recount|concede|seats)\b",
    re.IGNORECASE,
)


def parse_title(title: str) -> ParsedTitle | None:
    """Map a SIG market title to (race_id, party).

    Handles::

        Will the Democratic Party win the Texas Senate race?           -> SEN-TX, D
        Will the Republican Party win the Georgia Governor race?       -> GOV-GA, R
        Will the Democratic Party win the PA-08 House race?            -> HOUSE-PA-08, D
        Will the Democratic Party control the Senate?                  -> CONTROL-SENATE, D
        Will an Independent candidate win the Nebraska Senate race?    -> SEN-NE, I
        Will Dan Osborn (I) win the Nebraska Senate race?              -> SEN-NE, I
        ... Ohio Senate special election ... / Florida Senate (Special) -> SEN-OH / SEN-FL, special=True

    Returns None when either the race or the party cannot be determined; nothing is guessed.
    """
    if not title:
        return None
    t = _clean_title(title)
    low = t.lower()
    if _NOT_A_WINNER_MARKET_RE.search(low):
        return None  # primaries, margins, nominations ... are not "party wins the race" markets

    # subject = the clause before the verb; the race description is the rest
    m = re.search(r"\b(win|control|carry|take|flip|hold|have)\b", low)
    subject = t[: m.start()] if m else t
    rest = t[m.start() :] if m else t
    party, candidate = _detect_party(subject)
    if party is None:
        party, candidate = _detect_party(t)
    if party is None:
        return None

    special = "special" in low
    # --- chamber control --------------------------------------------------------------------
    cm = _CONTROL_RE.search(rest)
    rest_low = rest.lower()
    has_race_word = bool(re.search(r"\b(race|seat|election|district)\b", rest_low))
    if cm and (not has_race_word or "control" in rest_low or "majority" in rest_low):
        state = find_state(rest)
        if state is None or "control" in rest_low or "majority" in rest_low:
            key = RaceKey("control", None, None, cm.group(1).lower())
            return ParsedTitle(key.race_id, party, "control", None, None, key.label, special, candidate)

    # --- House district ---------------------------------------------------------------------
    dm = _DISTRICT_CODE_RE.search(rest)
    if dm and dm.group(1) in STATES:
        key = RaceKey("house", dm.group(1), format_district(dm.group(2)))
        return ParsedTitle(key.race_id, party, "house", key.state, key.district, key.label, special, candidate)
    wm = _DISTRICT_WORD_RE.search(rest)
    if wm:
        state = find_state(rest)
        if state is not None:
            num = wm.group(1) or wm.group(2)
            district = format_district(num) if num else "AL"
            key = RaceKey("house", state, district)
            return ParsedTitle(
                key.race_id, party, "house", key.state, key.district, key.label, special, candidate
            )

    # --- Senate / Governor ------------------------------------------------------------------
    chamber: str | None = None
    if re.search(r"\b(governor|governor's|gubernatorial|governorship)\b", rest_low):
        chamber = "governor"
    elif re.search(r"\bsenate\b|\bsenator\b", rest_low) and "state senate" not in rest_low:
        chamber = "senate"
    elif re.search(r"\bhouse\b|\bcongressional\b", rest_low):
        chamber = "house"
    if chamber is None:
        return None
    state = find_state(rest)
    if state is None:
        if chamber in ("senate", "house") and cm:
            key = RaceKey("control", None, None, chamber)
            return ParsedTitle(key.race_id, party, "control", None, None, key.label, special, candidate)
        return None
    if chamber == "house":
        return None  # a House title without a district is ambiguous
    key = RaceKey(chamber, state, None)
    return ParsedTitle(key.race_id, party, chamber, state, None, key.label, special, candidate)


# ---------------------------------------------------------------------------------------------
# Crosswalk patterns
# ---------------------------------------------------------------------------------------------


def kalshi_ticker_candidates(race_id: str, party: str) -> list[str]:
    """Candidate Kalshi tickers for a (race, party) leg, most likely first.

    Observed patterns (docs/research/platform.md section 5): ``SENATETX-26-D``, ``GOVPARTYGA-26-R``,
    ``HOUSEPA8-26-D``, ``CONTROLS-2026-D``, ``CONTROLH-2026-D``, ``KXSENATELA-26NOV-D``. Every candidate is
    verified by fetching it before it is written anywhere.
    """
    key = parse_race_id(race_id)
    if key is None:
        return []
    p = party.upper()
    suffixes = [p] if p != "I" else ["I", "IND"]
    out: list[str] = []
    if key.chamber == "control":
        letter = "S" if key.control_of == "senate" else "H"
        for s in suffixes:
            out += [f"CONTROL{letter}-2026-{s}", f"KXCONTROL{letter}-2026-{s}", f"KXCONTROL{letter}-26-{s}"]
    elif key.chamber == "senate":
        st = key.state
        for s in suffixes:
            out += [
                f"SENATE{st}-26-{s}",
                f"KXSENATE{st}-26-{s}",
                f"KXSENATE{st}-26NOV-{s}",
                f"SENATE{st}SPECIAL-26-{s}",
                f"KXSENATE{st}SPECIAL-26-{s}",
                f"KXSENATE{st}SPEC-26-{s}",
            ]
    elif key.chamber == "governor":
        st = key.state
        for s in suffixes:
            out += [f"GOVPARTY{st}-26-{s}", f"KXGOVPARTY{st}-26-{s}", f"GOV{st}-26-{s}", f"KXGOV{st}-26-{s}"]
    elif key.chamber == "house":
        st = key.state
        d = key.district or ""
        nums = [str(int(d))] if d.isdigit() else ["AL", "1"]
        if d.isdigit():
            nums.append(d)  # zero-padded variant
        for s in suffixes:
            for n in dict.fromkeys(nums):
                out += [f"HOUSE{st}{n}-26-{s}", f"KXHOUSE{st}{n}-26-{s}"]
    return list(dict.fromkeys(out))


_PARTY_SLUG = {"D": ["democratic-party", "democrats", "democrat"], "R": ["republican-party", "republicans", "republican"]}


def polymarket_slug_candidates(race_id: str, party: str) -> list[str]:
    """Candidate Gamma slugs (``will-the-{party}-win-the-{state}-{race}-race-in-2026``)."""
    key = parse_race_id(race_id)
    if key is None:
        return []
    p = party.upper()
    parties = _PARTY_SLUG.get(p, ["an-independent", "independent"])
    out: list[str] = []
    if key.chamber == "control":
        which = key.control_of or "senate"
        for pp in parties:
            out += [
                f"will-{pp}-control-the-{which}-after-the-2026-midterms",
                f"will-the-{pp}-control-the-{which}-after-the-2026-midterms",
                f"will-{pp}-win-the-{which}-in-2026",
                f"which-party-will-control-the-{which}-after-the-2026-midterms-{pp}",
            ]
        return out
    st = state_name(key.state or "").lower().replace(" ", "-")
    if key.chamber == "house":
        d = key.district or ""
        code = f"{(key.state or '').lower()}-{d}"
        ords = [f"{st}-{_ordinal(d)}-district", code, f"{(key.state or '').lower()}-{int(d) if d.isdigit() else d}"]
        for pp in parties:
            for o in ords:
                out += [f"will-the-{pp}-win-the-{o}-house-race-in-2026", f"will-the-{pp}-win-{o}-in-2026"]
        return list(dict.fromkeys(out))
    race = "senate" if key.chamber == "senate" else "governor"
    for pp in parties:
        out += [
            f"will-the-{pp}-win-the-{st}-{race}-race-in-2026",
            f"will-the-{pp}-win-the-{st}-{race}-race-2026",
            f"will-{pp}-win-the-{st}-{race}-race-in-2026",
            f"will-a-{pp.rstrip('s')}-win-the-{st}-{race}-race-in-2026",
        ]
        if race == "governor":
            out.append(f"will-the-{pp}-win-the-{st}-gubernatorial-election-in-2026")
        if race == "senate":
            out.append(f"will-the-{pp}-win-the-{st}-senate-special-election-in-2026")
    return list(dict.fromkeys(out))


def _ordinal(d: str) -> str:
    if not d.isdigit():
        return "at-large"
    n = int(d)
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def polymarket_search_query(race_id: str) -> str:
    """Free-text query for Gamma /public-search."""
    key = parse_race_id(race_id)
    if key is None:
        return race_id
    if key.chamber == "control":
        return f"{(key.control_of or '').title()} control 2026"
    if key.chamber == "house":
        return f"{state_name(key.state or '')} {_ordinal(key.district or '')} district House 2026"
    return f"{state_name(key.state or '')} {'Senate' if key.chamber == 'senate' else 'Governor'} 2026"


_EXCLUDE_WORDS = (
    "primary",
    "nominee",
    "margin",
    "popular vote",
    "turnout",
    "seats",
    "by more than",
    "by less than",
    "spread",
    "vote share",
    "debate",
    "endorse",
    "drop out",
    "withdraw",
    "resign",
    "recount",
    "concede",
)
_PARTY_TITLE_WORDS = {
    "D": ("democrat", "democratic", "democrats"),
    "R": ("republican", "republicans", "gop"),
    "I": ("independent", "osborn"),
}


def title_similarity(a: str, b: str) -> float:
    """Case-insensitive SequenceMatcher ratio in [0, 1]."""
    return difflib.SequenceMatcher(None, _clean_title(a).lower(), _clean_title(b).lower()).ratio()


def expected_title(race_id: str, party: str) -> str:
    """The canonical SIG-style title for a leg, used as the similarity anchor."""
    key = parse_race_id(race_id)
    pname = {"D": "Democratic Party", "R": "Republican Party", "I": "Independent"}.get(party.upper(), party)
    if key is None:
        return f"Will the {pname} win {race_id}?"
    if key.chamber == "control":
        return f"Will the {pname} control the {(key.control_of or '').title()}?"
    if key.chamber == "house":
        return f"Will the {pname} win the {key.state}-{key.district} House race?"
    which = "Senate" if key.chamber == "senate" else "Governor"
    return f"Will the {pname} win the {state_name(key.state or '')} {which} race?"


def crosswalk_match_score(race_id: str, party: str, title: str) -> float:
    """Score an external market title as a match for (race_id, party) in [0, 1]; 0 = structural mismatch.

    Requires the state (name or district token), the chamber word, the party word and a 2026 token
    (or an unambiguous 2026 phrase) to be present, rejects primary/margin/etc. markets, then uses the
    SequenceMatcher ratio against the expected title as a tie-breaker.
    """
    key = parse_race_id(race_id)
    if key is None or not title:
        return 0.0
    low = _clean_title(title).lower()
    if any(w in low for w in _EXCLUDE_WORDS):
        return 0.0
    if not any(w in low for w in _PARTY_TITLE_WORDS.get(party.upper(), (party.lower(),))):
        return 0.0
    if key.chamber == "control":
        which = key.control_of or ""
        if which not in low or not re.search(r"control|majority|win the (senate|house)", low):
            return 0.0
        return 0.5 + 0.5 * title_similarity(expected_title(race_id, party), title)
    st_name = state_name(key.state or "").lower()
    st_code = (key.state or "").lower()
    if key.chamber == "house":
        d = key.district or ""
        dnum = str(int(d)) if d.isdigit() else "at-large"
        has_state = st_name in low or re.search(rf"\b{st_code}[-\s]?0?{dnum}\b", low) is not None
        has_district = (
            re.search(rf"\b{st_code}[-\s]?0?{dnum}\b", low) is not None
            or re.search(rf"\b{dnum}(?:st|nd|rd|th)?\s+(?:congressional\s+)?district\b", low) is not None
            or re.search(rf"\bdistrict\s+{dnum}\b", low) is not None
            or (dnum == "at-large" and "at-large" in low)
        )
        if not (has_state and has_district):
            return 0.0
        if not re.search(r"\bhouse\b|\bcongressional\b|\bdistrict\b", low):
            return 0.0
    else:
        if st_name not in low:
            return 0.0
        if key.chamber == "senate" and not re.search(r"\bsenate\b|\bsenator\b", low):
            return 0.0
        if key.chamber == "governor" and not re.search(r"\bgovernor|\bgubernatorial", low):
            return 0.0
        if key.chamber == "senate" and "governor" in low:
            return 0.0
        if key.chamber == "governor" and "senate" in low:
            return 0.0
    if "2026" not in low and "midterm" not in low:
        return 0.0
    return 0.5 + 0.5 * title_similarity(expected_title(race_id, party), title)


# ---------------------------------------------------------------------------------------------
# Rating strings
# ---------------------------------------------------------------------------------------------

RATING_LEVELS = ("safe", "likely", "lean", "tilt", "tossup")
_LEVEL_WORDS: dict[str, str] = {
    "safe": "safe",
    "solid": "safe",
    "strong": "safe",
    "likely": "likely",
    "lean": "lean",
    "leans": "lean",
    "leaning": "lean",
    "tilt": "tilt",
    "tilts": "tilt",
    "tilting": "tilt",
    "tossup": "tossup",
    "toss-up": "tossup",
    "toss up": "tossup",
    "toss-ups": "tossup",
    "tossups": "tossup",
    "toss ups": "tossup",
    "pure toss-up": "tossup",
    "pure tossup": "tossup",
    "tu": "tossup",
}
_RATING_RE = re.compile(
    r"\b(solid|safe|strong|likely|leans?|leaning|tilts?|tilting|toss[\s-]?ups?|tu)\b"
    r"(?:[\s-]*(democratic|democrat|dem|d|republican|rep|gop|r|independent|ind|i))?\b",
    re.IGNORECASE,
)


def normalize_rating(text: str | None) -> tuple[str, str | None] | None:
    """'Leans Democratic' -> ('lean', 'D'); 'Toss Up' -> ('tossup', None); None when not a rating."""
    if not text:
        return None
    t = text.strip().lower().replace("‑", "-").replace("–", "-")
    if not t:
        return None
    m = _RATING_RE.search(t)
    if not m:
        return None
    level_word = re.sub(r"\s+", " ", m.group(1))
    level = _LEVEL_WORDS.get(level_word) or _LEVEL_WORDS.get(level_word.replace(" ", "-")) or (
        "tossup" if level_word.startswith("toss") else None
    )
    if level is None:
        return None
    party = _PARTY_WORDS.get(m.group(2).lower()) if m.group(2) else None
    if level == "tossup":
        return ("tossup", None)
    if party is None:
        # "Lean D" written as "D Lean"? try a party token anywhere
        pm = re.search(r"\b(democratic|democrat|dem|republican|gop|rep|independent)\b|\b([DRI])\b", t)
        if pm:
            party = _PARTY_WORDS.get((pm.group(1) or pm.group(2)).lower())
    if party is None:
        return None
    return (level, party)


def canonical_rating(level: str, party: str | None) -> str:
    """('lean', 'D') -> 'Lean D'; ('tossup', None) -> 'Toss-up'."""
    if level == "tossup":
        return "Toss-up"
    return f"{level.title()} {party or ''}".strip()


def rating_prob_dem(level: str, party: str | None, ratings_prob: dict[str, float]) -> float | None:
    """P(Democrat wins) implied by a rating given cfg.fairvalue.ratings_prob; None for independent leans."""
    if level == "tossup":
        return float(ratings_prob.get("tossup", 0.5))
    p = ratings_prob.get(level)
    if p is None:
        return None
    if party == "D":
        return float(p)
    if party == "R":
        return float(1.0 - p)
    return None


# ---------------------------------------------------------------------------------------------
# Race references in ratings tables ("Alaska", "AK (Sullivan)", "PA-08", "Pennsylvania 8th")
# ---------------------------------------------------------------------------------------------

_REF_DISTRICT_RE = re.compile(r"\b([A-Z]{2})[-\s]?(\d{1,2}|AL)\b")
_REF_DISTRICT_WORDS_RE = re.compile(
    r"(?:'s)?\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s+(?:congressional\s+)?district)?\b|\bdistrict\s+(\d{1,2})\b|\bat[-\s]large\b",
    re.IGNORECASE,
)


def race_id_from_text(text: str, chamber: str | None = None) -> str | None:
    """Best-effort race id from a ratings-table cell. `chamber` disambiguates Senate vs Governor tables.

    Examples: ('Alaska', 'senate') -> SEN-AK; ('Georgia (Special)', 'senate') -> SEN-GA;
    ('PA-08', None) -> HOUSE-PA-08; ('Pennsylvania 8th', 'house') -> HOUSE-PA-08; ('AK Gov', None) -> GOV-AK.
    """
    if not text:
        return None
    t = re.sub(r"\s+", " ", text.replace("’", "'")).strip()
    if not t:
        return None
    low = t.lower()
    ch = chamber.lower() if chamber else None
    if ch is None:
        if re.search(r"\bgov(?:ernor)?\b", low):
            ch = "governor"
        elif re.search(r"\bsen(?:ate)?\b", low):
            ch = "senate"
        elif re.search(r"\bhouse\b|district|\b[A-Z]{2}-\d{1,2}\b", t):
            ch = "house"
    dm = _REF_DISTRICT_RE.search(t)
    if dm and dm.group(1) in STATES:
        # an explicit hyphenated code ('PA-08') is a House district whatever the table's chamber hint;
        # a whitespace form ('PA 8') only counts on House (or unhinted) tables
        hyphenated = "-" in dm.group(0)
        if ch in (None, "house") or (hyphenated and dm.group(2).isdigit()):
            return make_race_id("house", dm.group(1), format_district(dm.group(2)))
    state = find_state(t)
    if state is None:
        return None
    if ch == "house":
        wm = _REF_DISTRICT_WORDS_RE.search(t)
        if wm:
            num = wm.group(1) or wm.group(2)
            return make_race_id("house", state, format_district(num) if num else "AL")
        return None
    if ch == "governor":
        return make_race_id("governor", state)
    if ch == "senate":
        return make_race_id("senate", state)
    return None


# ---------------------------------------------------------------------------------------------
# config/races.yaml
# ---------------------------------------------------------------------------------------------

RACES_YAML_HEADER = """\
# config/races.yaml -- the race universe for SIG's 2026 Midterm Elections Competition.
#
# HAND-EDITABLE. scripts/bootstrap_races.py merges the live SIG market list into this file and never
# overwrites a non-empty hand-set value except the exchange-owned fields (sig_market_id, sig_exchange_id,
# title, status, settlement_date) and the API-derived `exclusive` / `exhaustive` flags. Body comments
# are NOT preserved by the bootstrap rewrite: keep notes in `note:` / `settlement_note:` fields.
#
# race_id conventions: SEN-XX, GOV-XX, HOUSE-XX-NN (zero-padded), CONTROL-SENATE, CONTROL-HOUSE.
# Fields per race:
#   chamber            senate | governor | house | control
#   state / district   two-letter code / 'NN' or 'AL'
#   label              human label
#   special            true for the FL / OH Senate specials
#   incumbent_party    D | R | I | null (party holding the seat before the election)
#   pvi                Cook PVI in points, POSITIVE = Republican lean, null = unknown. Seed values are the
#                      2025 Cook PVI as remembered at seed time (+-2 pts): verify before relying on them.
#   exclusive / exhaustive   ONLY from the exchange relationships API (bootstrap). null = no evidence.
#   settlement_note    runoff / fusion / independent / ranked-choice caveats (read before trading)
#   candidates         {party: {name, aliases[], handles[], hashtags[]}} for the gazetteer
#   markets            {party: {sig_market_id, sig_exchange_id, title, status, settlement_date,
#                      kalshi_ticker, polymarket_condition_id, polymarket_token_id, polymarket_slug,
#                      polymarket_gamma_id, predictit_id}}; empty strings are filled by bootstrap.
"""

MARKET_FIELDS: tuple[str, ...] = (
    "sig_market_id",
    "sig_exchange_id",
    "title",
    "status",
    "settlement_date",
    "kalshi_ticker",
    "polymarket_condition_id",
    "polymarket_token_id",
    "polymarket_slug",
    "polymarket_gamma_id",
    "predictit_id",
)
EXCHANGE_OWNED_FIELDS: tuple[str, ...] = ("sig_market_id", "sig_exchange_id", "title", "status", "settlement_date")


def empty_market_entry() -> dict[str, Any]:
    return {k: "" for k in MARKET_FIELDS}


def empty_race_entry(key: RaceKey, *, special: bool = False) -> dict[str, Any]:
    return {
        "chamber": key.chamber,
        "state": key.state,
        "district": key.district,
        "label": key.label,
        "special": special,
        "incumbent_party": None,
        "pvi": None,
        "exclusive": None,
        "exhaustive": None,
        "settlement_note": None,
        "candidates": {},
        "markets": {},
    }


def load_races_yaml(path: Path | None = None) -> dict[str, Any]:
    """Load config/races.yaml; a missing or empty file yields ``{"version": 1, "races": {}}``."""
    p = path or RACES_YAML
    data: Any = yaml.safe_load(p.read_text()) if p.exists() else None
    if not isinstance(data, dict):
        data = {}
    data.setdefault("version", 1)
    races = data.get("races")
    if not isinstance(races, dict):
        data["races"] = {}
    return data


def save_races_yaml(data: dict[str, Any], path: Path | None = None) -> None:
    """Write the documented header followed by the YAML body (sorted by race id within chamber order)."""
    p = path or RACES_YAML
    races = data.get("races") or {}
    order = {"control": 0, "senate": 1, "governor": 2, "house": 3}
    sorted_races = dict(
        sorted(races.items(), key=lambda kv: (order.get(str(kv[1].get("chamber")), 9), kv[0]))
    )
    body = {k: v for k, v in data.items() if k != "races"}
    body["races"] = sorted_races
    text = RACES_YAML_HEADER + yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=110)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def race_entry(data: dict[str, Any], race_id: str, *, create: bool = True, special: bool = False) -> dict[str, Any] | None:
    """The entry for a race id, created from the id's parts when absent (and `create`)."""
    races = data.setdefault("races", {})
    entry = races.get(race_id)
    if entry is None and create:
        key = parse_race_id(race_id)
        if key is None:
            return None
        entry = empty_race_entry(key, special=special)
        races[race_id] = entry
    if entry is not None:
        entry.setdefault("markets", {})
        entry.setdefault("candidates", {})
    return entry


def market_entry(entry: dict[str, Any], party: str) -> dict[str, Any]:
    """The market leg dict for a party inside a race entry (created with empty fields when absent)."""
    markets = entry.setdefault("markets", {})
    leg = markets.get(party)
    if not isinstance(leg, dict):
        leg = empty_market_entry()
        markets[party] = leg
    for k in MARKET_FIELDS:
        leg.setdefault(k, "")
    return leg


def market_legs(data: dict[str, Any]) -> list[tuple[str, str, dict[str, Any], dict[str, Any]]]:
    """Flatten the universe to (race_id, party, race_entry, market_leg) tuples."""
    out = []
    for rid, entry in (data.get("races") or {}).items():
        if not isinstance(entry, dict):
            continue
        for party, leg in (entry.get("markets") or {}).items():
            if isinstance(leg, dict):
                out.append((rid, str(party), entry, leg))
    return out


@dataclass
class UniverseStats:
    races: int = 0
    legs: int = 0
    with_sig: int = 0
    with_kalshi: int = 0
    with_polymarket: int = 0
    by_chamber: dict[str, int] = field(default_factory=dict)


def universe_stats(data: dict[str, Any]) -> UniverseStats:
    st = UniverseStats()
    for rid, entry in (data.get("races") or {}).items():
        if not isinstance(entry, dict):
            continue
        st.races += 1
        ch = str(entry.get("chamber") or chamber_of(rid) or "?")
        st.by_chamber[ch] = st.by_chamber.get(ch, 0) + 1
        for _party, leg in (entry.get("markets") or {}).items():
            if not isinstance(leg, dict):
                continue
            st.legs += 1
            st.with_sig += bool(leg.get("sig_exchange_id"))
            st.with_kalshi += bool(leg.get("kalshi_ticker"))
            st.with_polymarket += bool(leg.get("polymarket_condition_id"))
    return st


# ---------------------------------------------------------------------------------------------
# DB row values (pure: builds plain dicts for sigcup.db.models.Race; the callers own the session)
# ---------------------------------------------------------------------------------------------


def race_row_values(race_id: str, entry: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Column values for a ``Race`` row from a races.yaml entry (None when the id is not a race id).

    Only the race-level fields are returned (``exclusive`` / ``exhaustive`` are copied as stored, i.e.
    whatever the bootstrap derived from the relationships API). Markets are handled separately.
    """
    key = parse_race_id(race_id)
    if key is None:
        return None
    e: Mapping[str, Any] = entry if isinstance(entry, Mapping) else {}
    chamber = str(e.get("chamber") or key.chamber)
    state = e.get("state") if e.get("state") is not None else key.state
    district = e.get("district") if e.get("district") is not None else key.district
    pvi = e.get("pvi")
    try:
        pvi_f = float(pvi) if pvi is not None and pvi != "" else None
    except (TypeError, ValueError):
        pvi_f = None
    candidates = e.get("candidates")
    return {
        "race_id": key.race_id,
        "chamber": chamber,
        "state": str(state).upper() if state else None,
        "district": str(district) if district else None,
        "label": str(e.get("label") or key.label),
        "incumbent_party": (str(e["incumbent_party"]).upper() if e.get("incumbent_party") else None),
        "pvi": pvi_f,
        "exclusive": e.get("exclusive") if isinstance(e.get("exclusive"), bool) else None,
        "exhaustive": e.get("exhaustive") if isinstance(e.get("exhaustive"), bool) else None,
        "settlement_note": e.get("settlement_note") or None,
        "candidates": candidates if isinstance(candidates, dict) and candidates else None,
    }


def title_contradicts(race_id: str, party: str, title: str | None) -> bool:
    """True when an external market title clearly describes a *different* contract than (race_id, party).

    Used for crosswalk candidates that were found by id/ticker pattern (Kalshi), whose titles are short
    ("Texas Senate: Democratic") and often lack the year, so the strict ``crosswalk_match_score`` would
    reject them. A title contradicts when it names another state (and not ours), another chamber, the
    other major party, or a non-winner contract (primary, margin ...). An empty title never contradicts.
    """
    if not title:
        return False
    key = parse_race_id(race_id)
    if key is None:
        return True
    low = _clean_title(title).lower()
    if any(w in low for w in _EXCLUDE_WORDS):
        return True
    other_party = {"D": "R", "R": "D"}.get(party.upper())
    if other_party and any(w in low for w in _PARTY_TITLE_WORDS[other_party]):
        if not any(w in low for w in _PARTY_TITLE_WORDS[party.upper()]):
            return True
    if key.chamber == "control":
        which = key.control_of or ""
        other = "house" if which == "senate" else "senate"
        return bool(re.search(rf"\b{other}\b", low)) and not re.search(rf"\b{which}\b", low)
    st_name = state_name(key.state or "").lower()
    found = _STATE_NAME_RE.search(low)
    if found and found.group(1).lower() != st_name and st_name not in low:
        return True
    if key.chamber == "senate" and re.search(r"\bgovernor|\bgubernatorial|\bhouse\b", low) and "senate" not in low:
        return True
    if key.chamber == "governor" and re.search(r"\bsenate\b|\bhouse\b", low) and "governor" not in low:
        return True
    if key.chamber == "house" and re.search(r"\bsenate\b|\bgovernor\b", low) and not re.search(r"\bhouse\b|district", low):
        return True
    if key.chamber == "house" and key.district:
        d = key.district
        dnum = str(int(d)) if d.isdigit() else None
        m = re.search(r"\b([a-z]{2})[-\s]?0?(\d{1,2})\b", low)
        if m and dnum and m.group(1) == (key.state or "").lower() and m.group(2) != dnum:
            return True
    return False
