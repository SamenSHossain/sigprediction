"""scripts/bootstrap_races.py --probe-writes places REAL orders, so it is gated exactly like the CLI: the
probe (and the script run directly, without `sigcup bootstrap`) refuses unless LIVE=1 AND a trade key
(SIG_API_KEY) are set, says why and exits non-zero. Nothing touches the network."""

from __future__ import annotations

import pytest

from sigcup.settings import Secrets
from tests.test_feeds_bootstrap import NOW, B, FakeAdapter

REFUSALS = [
    ({"live": False, "sig_api_key": "trade-key"}, "LIVE is not 1"),
    ({"live": True, "sig_api_key": ""}, "SIG_API_KEY is empty"),
    ({"live": False, "sig_api_key": "", "sig_read_key": "read-key"}, "LIVE is not 1"),
]


def sec(**kw) -> Secrets:
    return Secrets(_env_file=None, **kw)


@pytest.mark.parametrize(("kw", "why"), REFUSALS)
async def test_probe_batch_writes_refuses_without_live_and_trade_key(monkeypatch, kw, why):
    monkeypatch.setattr(B, "load_secrets", lambda: sec(**kw))  # the default: secrets() from env / .env
    fake = FakeAdapter()
    err: Exception | None = None
    try:
        await B.probe_batch_writes(fake, fake.markets, execute=True, now=NOW)
    except Exception as e:  # noqa: BLE001
        err = e
    assert fake.placed == [] and fake.cancelled == [], "no real order without LIVE=1 and a trade key"
    assert isinstance(err, B.ProbeWritesRefused) and why in str(err)
    # an explicit Secrets is held to the same rule
    with pytest.raises(B.ProbeWritesRefused, match=why):
        await B.probe_batch_writes(fake, fake.markets, execute=True, now=NOW, sec=sec(**kw))
    assert fake.placed == []
    # the dry description never needs either
    assert (await B.probe_batch_writes(fake, fake.markets, execute=False, sec=sec(**kw)))["verdict"].startswith("skipped")


@pytest.mark.parametrize(("kw", "why"), REFUSALS)
def test_script_main_refuses_probe_writes_and_exits_nonzero(monkeypatch, capsys, kw, why):
    monkeypatch.setattr(B, "load_secrets", lambda: sec(**kw))

    def no_client(*a, **k):
        raise AssertionError("no exchange client may be built for a refused probe")

    monkeypatch.setattr(B, "make_adapter", no_client)
    code = B.main(["--probe-writes", "--skip-crosswalk", "--dry-run"])
    assert code != 0
    err = capsys.readouterr().err
    assert "refusing --probe-writes" in err and why in err


def test_script_main_runs_the_probe_with_live_and_trade_key(monkeypatch, capsys):
    monkeypatch.setattr(B, "load_secrets", lambda: sec(live=True, sig_api_key="trade-key"))
    fake = FakeAdapter()
    built: list[bool] = []

    def make_adapter(s, cfg, *, trade=False):
        built.append(trade)
        return fake, ""

    monkeypatch.setattr(B, "make_adapter", make_adapter)
    assert B.main(["--probe-writes", "--skip-crosswalk", "--dry-run"]) == 0
    assert built == [True] and len(fake.placed) == 2 and fake.cancelled == ["o-0", "o-1"]
    assert "probe batch_writes" in capsys.readouterr().out
