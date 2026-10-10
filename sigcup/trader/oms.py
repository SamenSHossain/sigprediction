"""Order management: desired orders -> exchange, with an idempotency state machine and a write budget.

Rules (owner: execution workstream):
- Persist OrderIntent(NEW) BEFORE sending; SENT while in flight; ACKED on 2xx; UNCERTAIN on timeout/502;
  on UNCERTAIN re-POST the SAME idempotency key (409 -> wait out the lease, then poll /orders and /fills).
- On startup: reconcile every UNCERTAIN/SENT intent against open orders and fills before quoting anything.
- Reconcile every config.cadence.reconcile_s: fills ledger == positions, intents == open orders,
  reservation <= cash; log and repair discrepancies.
- Write budget per minute: priority 0 arb/election takes (burst), 1 theta, 2 value, 3 mm, 4 housekeeping;
  re-quote only when |new - old| >= value.requote_threshold with a per-market cooldown; prefer letting an
  order expire over cancel+replace (1 write vs 2).
- Every order gets expires_at = now + risk.order_expiry_s; nothing is sent after config.deadline_utc.
- Kill switch: if the kill file exists or the daily-loss limit trips -> cancel_all, mark state, stop.
- Internal cash reservation for resting buys (exchange does not reserve).

State machine per intent (intent.id == idempotency key)::

    NEW --send--> SENT --2xx--> ACKED --fills--> FILLED
                   |                 \\--cancel/expiry--> CANCELLED / EXPIRED
                   |--timeout/502/503--> UNCERTAIN --re-POST same key--> (SENT ...)
                   |--409 lease--> wait lease, poll open orders + fills, else re-POST
                   \\--4xx validation / auth--> REJECTED

The OMS never throttles itself: the adapter owns the limiters. It only *reads*
``budget.writes.available()`` to decide how many actions to attempt this cycle, keeps one slot free for
a kill, and defers the rest to the next sync in priority order.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select

from sigcup.common.ids import new_idempotency_key
from sigcup.common.log import get_logger
from sigcup.common.time import before_deadline, ensure_utc, utc_now
from sigcup.db import engine as dbengine
from sigcup.db.models import AccountSnap, Fill, IntentState, OrderIntent, Position, Strategy
from sigcup.db.repo import kv_get, kv_set
from sigcup.settings import StrategyConfig
from sigcup.sig.adapter import ExchangeAdapter
from sigcup.sig.errors import (
    AuthError,
    OrderStatusUnknown,
    RateLimited,
    RequestInFlight,
    SigError,
    TransientError,
    ValidationError,
)
from sigcup.sig.ratelimit import Budget
from sigcup.sig.schemas import FillInfo, OrderInfo, OrderRequest, OrderResult, PositionInfo, round_tick
from sigcup.trader.types import (
    PRIORITY_ARB,
    PRIORITY_HOUSEKEEPING,
    PRIORITY_MM,
    PRIORITY_THETA,
    PRIORITY_VALUE,
    DesiredOrder,
)

log = get_logger(__name__)

KV_KILLED = "killed"
LEASE_S = 90.0
WORKING_STATES = {IntentState.SENT.value, IntentState.UNCERTAIN.value, IntentState.ACKED.value}
OPEN_STATES = WORKING_STATES | {IntentState.NEW.value}
_UNCERTAIN_EXC: tuple[type[BaseException], ...] = (
    OrderStatusUnknown,
    TransientError,
    TimeoutError,
    httpx.TimeoutException,
    httpx.TransportError,
)

_FILLED_STATUSES = {"filled", "complete", "completed"}

Key = tuple[str, str, str, str]  # (exchange_id, side, action, strategy)


def priority_for(strategy: str | Strategy, take: bool = False) -> int:
    """Default priority for a strategy tag (used when a DesiredOrder is not available)."""
    s = str(getattr(strategy, "value", strategy)).lower()
    if take or s in (Strategy.ARB.value, Strategy.ELECTION.value):
        return PRIORITY_ARB
    if s == Strategy.THETA.value:
        return PRIORITY_THETA
    if s in (Strategy.VALUE.value, Strategy.CONVEX.value, Strategy.FADE.value, Strategy.FOLLOW.value):
        return PRIORITY_VALUE
    if s == Strategy.MM.value:
        return PRIORITY_MM
    return PRIORITY_HOUSEKEEPING


def intent_key(i: OrderIntent) -> Key:
    return (i.exchange_id, str(i.side), str(i.action), str(i.strategy))


def desired_key(d: DesiredOrder) -> Key:
    return (d.exchange_id, str(d.side), str(d.action), str(getattr(d.strategy, "value", d.strategy)))


@dataclass
class SyncReport:
    """What one ``sync`` did."""

    placed: list[str] = field(default_factory=list)  # intent ids sent (incl. groups' legs)
    cancelled: list[str] = field(default_factory=list)
    kept: int = 0
    deferred: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (exchange_id, reason)
    errors: list[str] = field(default_factory=list)
    blocked: str | None = None  # why nothing was sent (killed / deadline / auth)
    killed: str | None = None


@dataclass
class ReconcileReport:
    cash: float = 0.0
    positions_value: float = 0.0
    reserved: float = 0.0
    open_orders: int = 0
    working_intents: int = 0
    new_fills: int = 0
    discrepancies: list[str] = field(default_factory=list)
    repaired: list[str] = field(default_factory=list)


@dataclass
class _Action:
    kind: str  # place | group | cancel | replace
    priority: int
    take: bool
    seq: int
    cost: int
    desired: list[DesiredOrder] = field(default_factory=list)
    intent: OrderIntent | None = None  # the working intent for cancel / replace


class OMS:
    """Order management system. One instance per process; all methods are coroutine-safe via a lock."""

    def __init__(
        self,
        adapter: ExchangeAdapter,
        cfg: StrategyConfig,
        budget: Budget,
        *,
        paper: bool = False,
        kill_path: Path | str | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        max_retries: int = 4,
        backoff_s: float = 1.0,
        lease_s: float = LEASE_S,
        concurrency: int = 8,
        kill_reserve: int = 1,
    ) -> None:
        self.adapter = adapter
        self.cfg = cfg
        self.budget = budget
        self.paper = paper
        self.kill_path = Path(kill_path) if kill_path is not None else None
        self.clock = clock or utc_now
        self.sleep = sleep or asyncio.sleep
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.lease_s = lease_s
        self.concurrency = max(1, concurrency)
        self.kill_reserve = max(0, kill_reserve)

        self.intents: dict[str, OrderIntent] = {}
        self.positions: dict[str, PositionInfo] = {}
        self.cash: float = cfg.bankroll_start
        self.positions_value: float = 0.0
        self.reserved: float = 0.0
        self.killed: str | None = None
        self.auth_failed: bool = False
        self.last_fill_ts: datetime | None = None
        self.last_reconcile: ReconcileReport | None = None
        self._last_quote_at: dict[Key, datetime] = {}
        self._lock = asyncio.Lock()
        self._seen_fills: set[str] = set()

    # -- properties ----------------------------------------------------------------------------
    @property
    def equity(self) -> float:
        return self.cash + self.positions_value

    def working(self) -> list[OrderIntent]:
        """Intents the exchange may be holding (SENT / UNCERTAIN / ACKED)."""
        return [i for i in self.intents.values() if str(i.state) in WORKING_STATES]

    def acked(self) -> list[OrderIntent]:
        return [i for i in self.intents.values() if str(i.state) == IntentState.ACKED.value]

    def may_send(self, now: datetime | None = None) -> bool:
        """True when new orders may be sent: not killed, auth ok, strictly before the deadline."""
        now = now or self.clock()
        return self.killed is None and not self.auth_failed and before_deadline(self.cfg.deadline_utc, now)

    # -- persistence ---------------------------------------------------------------------------
    def _persist(self, intent: OrderIntent) -> None:
        intent.updated_at = self.clock()
        with dbengine.session() as s:
            s.merge(intent)
        self.intents[intent.id] = intent

    def _set_state(self, intent: OrderIntent, state: IntentState, error: str | None = None, **fields: Any) -> None:
        intent.state = state.value
        if error is not None:
            intent.error = error[:500]
        for k, v in fields.items():
            setattr(intent, k, v)
        self._persist(intent)
        log.info(
            "intent %s %s %s %s %s @%.3f x%d -> %s%s",
            intent.id[:8], intent.strategy, intent.exchange_id, intent.action, intent.side,
            intent.price, intent.quantity, state.value, f" ({error})" if error else "",
        )

    def _load_intents(self) -> list[OrderIntent]:
        with dbengine.session() as s:
            rows = s.execute(
                select(OrderIntent).where(OrderIntent.state.in_(sorted(OPEN_STATES)), OrderIntent.paper == self.paper)
            ).scalars().all()
        return list(rows)

    def _new_intent(self, d: DesiredOrder, now: datetime, quantity: int | None = None) -> OrderIntent:
        return OrderIntent(
            id=new_idempotency_key(),
            exchange_id=d.exchange_id,
            side=str(d.side),
            action=str(d.action),
            price=round_tick(d.price),
            quantity=int(quantity if quantity is not None else d.quantity),
            strategy=str(getattr(d.strategy, "value", d.strategy)),
            state=IntentState.NEW.value,
            group_id=d.group_id,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(seconds=self.cfg.risk.order_expiry_s),
            filled_qty=0,
            paper=self.paper,
        )

    def _request(self, intent: OrderIntent) -> OrderRequest:
        return OrderRequest(
            exchange_id=intent.exchange_id,
            side=intent.side,  # type: ignore[arg-type]
            action=intent.action,  # type: ignore[arg-type]
            price=round_tick(intent.price),
            quantity=int(intent.quantity),
            idempotency_key=intent.id,
            expires_at=intent.expires_at,
            strategy=str(intent.strategy),
            tournament_id=getattr(self.adapter, "tournament_id", None),
        )

    # -- reservation ---------------------------------------------------------------------------
    def _cash_required(self, intent: OrderIntent) -> float:
        remaining = max(0, int(intent.quantity) - int(intent.filled_qty or 0))
        if remaining <= 0:
            return 0.0
        if intent.action == "buy":
            return float(intent.price) * remaining
        pos = self.positions.get(intent.exchange_id)
        q = pos.quantity if pos else 0
        owned = max(0, q) if intent.side == "yes" else max(0, -q)
        return (1.0 - float(intent.price)) * max(0, remaining - owned)

    def _recompute_reserved(self) -> float:
        self.reserved = sum(self._cash_required(i) for i in self.working())
        return self.reserved

    # -- the state machine ---------------------------------------------------------------------
    def _apply_result(self, intent: OrderIntent, res: OrderResult) -> None:
        order_id = res.order_id or intent.sig_order_id
        traded = int(res.quantity_traded or 0)
        filled = max(int(intent.filled_qty or 0), traded)
        avg = intent.avg_fill
        if traded > 0:
            if res.fill_price is not None:
                avg = float(res.fill_price)
            elif res.total_cost is not None:
                avg = float(res.total_cost) / traded
        status = (res.status or "").lower()
        if status in {"rejected", "failed", "error"}:
            self._set_state(intent, IntentState.REJECTED, error=f"status={status}", sig_order_id=order_id)
        elif status in {"cancelled", "canceled"}:
            self._set_state(intent, IntentState.CANCELLED, sig_order_id=order_id, filled_qty=filled, avg_fill=avg)
        elif filled >= int(intent.quantity) or status in _FILLED_STATUSES:
            if filled < int(intent.quantity):  # exchange says filled but did not report the quantity
                filled = int(intent.quantity)
            self._set_state(intent, IntentState.FILLED, sig_order_id=order_id, filled_qty=filled, avg_fill=avg)
        elif status == "expired":
            self._set_state(intent, IntentState.EXPIRED, sig_order_id=order_id, filled_qty=filled, avg_fill=avg)
        else:
            self._set_state(intent, IntentState.ACKED, sig_order_id=order_id, filled_qty=filled, avg_fill=avg)
        self._recompute_reserved()

    async def _submit(self, intent: OrderIntent) -> OrderIntent:
        """Drive one intent through NEW/UNCERTAIN -> SENT -> terminal/ACKED with same-key retries."""
        attempt = 0
        while True:
            if not self.may_send():
                if str(intent.state) == IntentState.NEW.value:
                    self._set_state(intent, IntentState.CANCELLED, error="send refused (killed/deadline/auth)")
                return intent
            self._set_state(intent, IntentState.SENT)
            try:
                res = await self.adapter.place_order(self._request(intent))
            except RequestInFlight:
                log.warning("intent %s: 409 in flight; waiting out the lease", intent.id[:8])
                self._set_state(intent, IntentState.UNCERTAIN, error="409 request in flight")
                if await self._resolve_after_lease(intent):
                    return intent
                attempt += 1
            except _UNCERTAIN_EXC as e:
                self._set_state(intent, IntentState.UNCERTAIN, error=f"{type(e).__name__}: {e}")
                attempt += 1
            except RateLimited as e:
                self._set_state(intent, IntentState.REJECTED, error=f"rate limited ({e.retry_after}s)")
                return intent
            except ValidationError as e:
                self._set_state(intent, IntentState.REJECTED, error=f"validation: {e}")
                return intent
            except AuthError as e:
                self.auth_failed = True
                self._set_state(intent, IntentState.REJECTED, error=f"auth: {e}")
                log.critical("authentication failed; no further orders will be sent: %s", e)
                return intent
            except SigError as e:
                if e.status is not None and 400 <= e.status < 500:
                    self._set_state(intent, IntentState.REJECTED, error=f"{e.status} {e.code}: {e}")
                    return intent
                self._set_state(intent, IntentState.UNCERTAIN, error=f"{e.status} {e.code}: {e}")
                attempt += 1
            else:
                if res.uncertain:
                    self._set_state(intent, IntentState.UNCERTAIN, error="order status unknown")
                    attempt += 1
                else:
                    self._apply_result(intent, res)
                    return intent
            if attempt > self.max_retries:
                log.error("intent %s still UNCERTAIN after %d attempts; reconcile will resolve it", intent.id[:8], attempt)
                return intent
            await self.sleep(self.backoff_s * (2 ** (attempt - 1)))

    async def _resolve_after_lease(self, intent: OrderIntent) -> bool:
        """After a 409: wait out the idempotency lease, then look for the order in open orders / fills."""
        await self.sleep(self.lease_s)
        try:
            open_orders = await self.adapter.get_open_orders()
        except Exception as e:  # noqa: BLE001
            log.warning("poll open orders failed: %s", e)
            return False
        if self._adopt_open(intent, open_orders):
            return True
        try:
            fills = await self.adapter.get_fills(since=ensure_utc(intent.created_at) - timedelta(minutes=5))
        except Exception as e:  # noqa: BLE001
            log.warning("poll fills failed: %s", e)
            return False
        return self._adopt_fills(intent, fills)

    def _adopt_open(self, intent: OrderIntent, open_orders: Iterable[OrderInfo]) -> bool:
        """Match an intent to an open order by idempotency key (or known order id) and mark it ACKED."""
        for o in open_orders:
            if (o.idempotency_key and o.idempotency_key == intent.id) or (
                intent.sig_order_id and o.order_id == intent.sig_order_id
            ):
                filled = int(o.filled or intent.filled_qty or 0)
                self._set_state(intent, IntentState.ACKED, sig_order_id=o.order_id, filled_qty=max(filled, int(intent.filled_qty or 0)))
                self._recompute_reserved()
                return True
        return False

    def _adopt_fills(self, intent: OrderIntent, fills: Iterable[FillInfo]) -> bool:
        """Resolve an intent from fills (by order id, else a conservative shape match). True if resolved."""
        matched = [f for f in fills if self._fill_matches(intent, f)]
        if not matched:
            return False
        qty = sum(int(f.quantity) for f in matched)
        cost = sum(float(f.price) * int(f.quantity) for f in matched)
        order_id = intent.sig_order_id or next((f.order_id for f in matched if f.order_id), None)
        filled = max(int(intent.filled_qty or 0), qty)
        avg = cost / qty if qty else intent.avg_fill
        state = IntentState.FILLED if filled >= int(intent.quantity) else IntentState.ACKED
        self._set_state(intent, state, sig_order_id=order_id, filled_qty=filled, avg_fill=avg)
        self._record_fills(matched, intent)
        self._recompute_reserved()
        return True

    def _fill_matches(self, intent: OrderIntent, f: FillInfo) -> bool:
        if intent.sig_order_id and f.order_id:
            return f.order_id == intent.sig_order_id
        if f.exchange_id != intent.exchange_id or str(f.side) != str(intent.side) or str(f.action) != str(intent.action):
            return False
        if abs(float(f.price) - float(intent.price)) > 1e-9:
            return False
        created = ensure_utc(intent.created_at)
        return ensure_utc(f.ts) >= created - timedelta(seconds=5)

    def _record_fills(self, fills: Iterable[FillInfo], intent: OrderIntent | None) -> list[FillInfo]:
        """Insert Fill rows (idempotent by fill_id). Returns the fills that were actually new."""
        new: list[FillInfo] = []
        with dbengine.session() as s:
            for f in fills:
                if f.fill_id in self._seen_fills or s.get(Fill, f.fill_id) is not None:
                    self._seen_fills.add(f.fill_id)
                    continue
                s.add(
                    Fill(
                        fill_id=f.fill_id,
                        sig_order_id=f.order_id or (intent.sig_order_id if intent else None),
                        intent_id=intent.id if intent else None,
                        exchange_id=f.exchange_id,
                        ts=ensure_utc(f.ts),
                        side=str(f.side),
                        action=str(f.action),
                        price=float(f.price),
                        quantity=int(f.quantity),
                        fee=float(f.fee or 0.0),
                        strategy=str(intent.strategy) if intent else None,
                        paper=self.paper,
                    )
                )
                self._seen_fills.add(f.fill_id)
                new.append(f)
                ts = ensure_utc(f.ts)
                if self.last_fill_ts is None or ts > self.last_fill_ts:
                    self.last_fill_ts = ts
        return new

    # -- public: startup -----------------------------------------------------------------------
    async def startup_reconcile(self) -> ReconcileReport:
        """Resolve every NEW/SENT/UNCERTAIN/ACKED intent against the exchange before quoting anything."""
        async with self._lock:
            with dbengine.session() as s:
                flag = kv_get(s, KV_KILLED)
            if flag:
                self.killed = str(flag.get("reason", "killed")) if isinstance(flag, dict) else str(flag)
                log.critical("kill flag is set in KV (%s): trading disabled until cleared", self.killed)
            rows = self._load_intents()
            self.intents = {r.id: r for r in rows}
            with dbengine.session() as s:
                self._seen_fills = set(s.execute(select(Fill.fill_id).where(Fill.paper == self.paper)).scalars().all())
            log.info("startup: %d open intents to reconcile, %d fills already recorded", len(rows), len(self._seen_fills))
            open_orders = await self.adapter.get_open_orders()
            unresolved = [r for r in rows if str(r.state) in WORKING_STATES]
            fills: list[FillInfo] = []
            if unresolved:
                since = min(ensure_utc(r.created_at) for r in unresolved) - timedelta(hours=1)
                fills = list(await self.adapter.get_fills(since=since))
            for intent in rows:
                await self._resolve_intent(intent, open_orders, fills, allow_resend=True)
            self._recompute_reserved()
            return await self._reconcile_locked(open_orders=None)

    async def _resolve_intent(
        self,
        intent: OrderIntent,
        open_orders: list[OrderInfo],
        fills: list[FillInfo],
        *,
        allow_resend: bool,
    ) -> None:
        state = str(intent.state)
        now = self.clock()
        if state == IntentState.NEW.value:
            self._set_state(intent, IntentState.CANCELLED, error="never sent (restart)")
            return
        if self._adopt_open(intent, open_orders):
            return
        if self._adopt_fills(intent, fills):
            return
        expired = intent.expires_at is not None and ensure_utc(intent.expires_at) <= now
        if state in (IntentState.SENT.value, IntentState.UNCERTAIN.value):
            if expired:
                self._set_state(intent, IntentState.EXPIRED, error="unresolved past expiry")
            elif allow_resend and self.may_send(now):
                log.warning("intent %s unresolved (%s): re-POSTing the same idempotency key", intent.id[:8], state)
                await self._submit(intent)
            else:
                log.warning("intent %s left %s (cannot resend now)", intent.id[:8], state)
            return
        # ACKED but gone from the open orders and no fills seen
        if int(intent.filled_qty or 0) >= int(intent.quantity):
            self._set_state(intent, IntentState.FILLED)
        elif expired:
            self._set_state(intent, IntentState.EXPIRED)
        else:
            self._set_state(intent, IntentState.CANCELLED, error="missing from open orders")

    # -- public: sync --------------------------------------------------------------------------
    async def sync(self, desired: list[DesiredOrder]) -> SyncReport:
        """Diff the desired order set against working orders and act within this minute's write budget."""
        async with self._lock:
            rep = SyncReport()
            now = self.clock()
            if self.kill_path is not None and self.kill_path.exists() and self.killed is None:
                await self._kill_locked(f"kill file present: {self.kill_path}")
            if self.killed is not None:
                rep.blocked = rep.killed = self.killed
                return rep
            if self.auth_failed:
                rep.blocked = "auth failed"
                return rep
            if not before_deadline(self.cfg.deadline_utc, now):
                rep.blocked = "deadline passed: no orders may be sent"
                log.warning(rep.blocked)
                return rep
            self._expire_locally(now)
            actions = self._plan(desired, now, rep)
            quota = max(0, self.budget.writes.available() - self.kill_reserve)
            chosen: list[_Action] = []
            for a in actions:
                if a.cost <= quota:
                    chosen.append(a)
                    quota -= a.cost
                else:
                    rep.deferred += 1
            if rep.deferred:
                log.info("sync: %d actions deferred (write budget)", rep.deferred)
            sem = asyncio.Semaphore(self.concurrency)

            async def run(a: _Action) -> None:
                async with sem:
                    try:
                        await self._execute(a, now, rep)
                    except Exception as e:  # noqa: BLE001
                        log.exception("sync action %s failed", a.kind)
                        rep.errors.append(f"{a.kind}: {e}")

            await asyncio.gather(*(run(a) for a in chosen))
            self._recompute_reserved()
            return rep

    def _expire_locally(self, now: datetime) -> None:
        for i in self.working():
            if i.expires_at is not None and ensure_utc(i.expires_at) <= now:
                self._set_state(i, IntentState.EXPIRED)

    def _working_by_key(self) -> dict[Key, OrderIntent]:
        out: dict[Key, OrderIntent] = {}
        for i in self.working():
            out.setdefault(intent_key(i), i)
        return out

    def _plan(self, desired: list[DesiredOrder], now: datetime, rep: SyncReport) -> list[_Action]:
        working = self._working_by_key()
        singles: dict[Key, DesiredOrder] = {}
        groups: dict[str, list[DesiredOrder]] = {}
        for d in desired:
            if d.quantity <= 0:
                continue
            if d.group_id:
                groups.setdefault(d.group_id, []).append(d)
                continue
            k = desired_key(d)
            if k in singles:
                log.debug("sync: duplicate desired order for %s; keeping the first", k)
                continue
            singles[k] = d
        actions: list[_Action] = []
        seq = 0
        for k, d in singles.items():
            w = working.get(k)
            if w is None:
                actions.append(_Action("place", d.priority, d.take, seq, 1, [d]))
            else:
                decision = self._requote_decision(w, d, k, now)
                if decision == "replace":
                    actions.append(_Action("replace", d.priority, d.take, seq, 2, [d], w))
                else:
                    rep.kept += 1
            seq += 1
        desired_keys = set(singles)
        for legs in groups.values():
            keys = {desired_key(d) for d in legs}
            desired_keys |= keys
            if any(k in working for k in keys):
                rep.kept += 1
                continue
            actions.append(_Action("group", min(d.priority for d in legs), any(d.take for d in legs), seq, 1, legs))
            seq += 1
        for k, w in working.items():
            if k in desired_keys or str(w.state) != IntentState.ACKED.value:
                continue
            remaining = (ensure_utc(w.expires_at) - now).total_seconds() if w.expires_at else 0.0
            if remaining > self.cfg.value.requote_cooldown_s:
                actions.append(_Action("cancel", PRIORITY_HOUSEKEEPING, False, seq, 1, [], w))
                seq += 1
        actions.sort(key=lambda a: (a.priority, 0 if a.take else 1, a.seq))
        return actions

    def _requote_decision(self, w: OrderIntent, d: DesiredOrder, k: Key, now: datetime) -> str:
        """keep | wait | replace. Replace only for adverse moves, large moves or takes; otherwise let the
        resting order expire (1 write) rather than cancel+replace (2 writes)."""
        thr = self.cfg.value.requote_threshold
        delta = round_tick(d.price) - float(w.price)
        if abs(delta) < thr - 1e-9:
            return "keep"
        last = self._last_quote_at.get(k)
        if last is not None and (now - last).total_seconds() < self.cfg.value.requote_cooldown_s:
            return "keep"
        adverse = (w.action == "buy" and delta < 0) or (w.action == "sell" and delta > 0)
        if adverse or d.take or d.priority <= PRIORITY_ARB or abs(delta) >= 2 * thr - 1e-9:
            return "replace"
        return "wait"

    async def _execute(self, a: _Action, now: datetime, rep: SyncReport) -> None:
        if a.kind == "cancel":
            assert a.intent is not None
            if await self._cancel(a.intent):
                rep.cancelled.append(a.intent.id)
            return
        if a.kind == "replace":
            assert a.intent is not None
            ok = await self._cancel(a.intent)
            if ok:
                rep.cancelled.append(a.intent.id)
            if str(a.intent.state) in WORKING_STATES:
                rep.skipped.append((a.intent.exchange_id, "cancel failed; not replacing"))
                return
        if a.kind == "group":
            await self._place_group(a.desired, now, rep)
            return
        d = a.desired[0]
        intent = self._prepare(d, now, rep)
        if intent is None:
            return
        self._last_quote_at[desired_key(d)] = now
        await self._submit(intent)
        if str(intent.state) in WORKING_STATES or str(intent.state) == IntentState.FILLED.value:
            rep.placed.append(intent.id)
        else:
            rep.skipped.append((d.exchange_id, f"{intent.state}: {intent.error}"))

    def _prepare(self, d: DesiredOrder, now: datetime, rep: SyncReport) -> OrderIntent | None:
        """Persist a NEW intent after the cash-reservation check (shrinking the size to fit if needed)."""
        intent = self._new_intent(d, now)
        need = self._cash_required(intent)
        free = self.cash - self.reserved
        if need > free + 1e-9:
            per_share = need / max(1, int(intent.quantity))
            fit = int(max(0.0, free) / per_share) if per_share > 0 else 0
            if fit <= 0:
                rep.skipped.append((d.exchange_id, f"insufficient free cash ({free:.0f} < {need:.0f})"))
                return None
            log.info("sync: shrinking %s %s to %d shares to respect the cash reservation", d.exchange_id, d.action, fit)
            intent.quantity = fit
        self._persist(intent)  # NEW is on disk before anything is sent
        self.reserved += self._cash_required(intent)
        return intent

    async def _place_group(self, legs: list[DesiredOrder], now: datetime, rep: SyncReport) -> None:
        intents: list[OrderIntent] = []
        for d in legs:
            i = self._prepare(d, now, rep)
            if i is None:
                for j in intents:
                    self._set_state(j, IntentState.CANCELLED, error="group leg could not be funded")
                self._recompute_reserved()
                return
            intents.append(i)
        if not self.may_send(now):
            for j in intents:
                self._set_state(j, IntentState.CANCELLED, error="send refused")
            return
        for i in intents:
            self._set_state(i, IntentState.SENT)
            self._last_quote_at[intent_key(i)] = now
        try:
            res = await self.adapter.place_multi_leg([self._request(i) for i in intents])
        except (RequestInFlight, *_UNCERTAIN_EXC) as e:
            for i in intents:
                self._set_state(i, IntentState.UNCERTAIN, error=f"{type(e).__name__}: {e}")
            rep.errors.append(f"group: {e}")
            return
        except (ValidationError, RateLimited) as e:
            for i in intents:
                self._set_state(i, IntentState.REJECTED, error=str(e))
            rep.skipped.append((legs[0].exchange_id, f"group rejected: {e}"))
            return
        except AuthError as e:
            self.auth_failed = True
            for i in intents:
                self._set_state(i, IntentState.REJECTED, error=f"auth: {e}")
            return
        results = list(res.results)
        if not res.atomic_ok and not results:
            for i in intents:
                self._set_state(i, IntentState.REJECTED, error="multi-leg not atomic")
            return
        for idx, i in enumerate(intents):
            r = results[idx] if idx < len(results) else OrderResult()
            if r.uncertain:
                self._set_state(i, IntentState.UNCERTAIN, error="order status unknown")
            else:
                self._apply_result(i, r)
            if str(i.state) in WORKING_STATES or str(i.state) == IntentState.FILLED.value:
                rep.placed.append(i.id)

    async def _cancel(self, intent: OrderIntent) -> bool:
        if not intent.sig_order_id:
            self._set_state(intent, IntentState.CANCELLED, error="cancelled before ack")
            return True
        try:
            ok = await self.adapter.cancel_order(intent.sig_order_id)
        except (ValidationError, SigError) as e:
            log.warning("cancel %s failed: %s", intent.id[:8], e)
            return False
        except _UNCERTAIN_EXC as e:
            log.warning("cancel %s uncertain: %s", intent.id[:8], e)
            return False
        if not ok:
            # already gone (filled / expired): reconcile decides which
            log.info("cancel %s: exchange reports no open order; reconcile will classify", intent.id[:8])
        self._set_state(intent, IntentState.CANCELLED, error=None if ok else "cancel returned false")
        self._recompute_reserved()
        return True

    # -- public: reconcile ---------------------------------------------------------------------
    async def reconcile(self) -> ReconcileReport:
        """Pull account / positions / open orders / fills, persist them, repair intents and check invariants."""
        async with self._lock:
            return await self._reconcile_locked()

    async def _reconcile_locked(self, open_orders: list[OrderInfo] | None = None) -> ReconcileReport:
        rep = ReconcileReport()
        now = self.clock()
        account = await self.adapter.get_account()
        self.cash = float(account.cash)
        positions = list(await self.adapter.get_positions())
        self.positions = {p.exchange_id: p for p in positions}
        pv = account.positions_value
        if pv is None:
            pv = sum(_position_value(p) for p in positions)
        self.positions_value = float(pv)
        if open_orders is None:
            open_orders = list(await self.adapter.get_open_orders())
        since = (self.last_fill_ts - timedelta(minutes=2)) if self.last_fill_ts else None
        fills = list(await self.adapter.get_fills(since=since))

        # fills ledger
        by_order: dict[str, OrderIntent] = {i.sig_order_id: i for i in self.intents.values() if i.sig_order_id}
        new_fills: list[FillInfo] = [f for f in fills if f.fill_id not in self._seen_fills]
        touched: dict[str, list[FillInfo]] = {}
        orphan_fills: list[FillInfo] = []
        for f in new_fills:
            i = by_order.get(f.order_id or "")
            if i is None:
                orphan_fills.append(f)
            else:
                touched.setdefault(i.id, []).append(f)
        for iid, fs in touched.items():
            i = self.intents[iid]
            inserted = self._record_fills(fs, i)
            rep.new_fills += len(inserted)
            if inserted:
                self._apply_fill_rows(i, inserted)
        if orphan_fills:
            inserted = self._record_fills(orphan_fills, None)
            rep.new_fills += len(inserted)
            if inserted:
                rep.discrepancies.append(f"{len(inserted)} fills without a known intent")

        # intents == open orders
        by_id = {o.order_id: o for o in open_orders}
        by_key = {o.idempotency_key: o for o in open_orders if o.idempotency_key}
        for i in self.working():
            o = by_id.get(i.sig_order_id or "") or by_key.get(i.id)
            if o is not None:
                if not i.sig_order_id or str(i.state) != IntentState.ACKED.value:
                    self._set_state(i, IntentState.ACKED, sig_order_id=o.order_id)
                    rep.repaired.append(f"{i.id[:8]} -> ACKED")
                if o.filled is not None and int(o.filled) > int(i.filled_qty or 0):
                    i.filled_qty = int(o.filled)
                    self._persist(i)
                continue
            before = str(i.state)
            await self._resolve_intent(i, open_orders, fills, allow_resend=self.may_send(now))
            if str(i.state) != before:
                rep.repaired.append(f"{i.id[:8]} {before} -> {i.state}")
        known_ids = {i.sig_order_id for i in self.working() if i.sig_order_id}
        for o in open_orders:
            if o.order_id in known_ids:
                continue
            rep.discrepancies.append(f"open order {o.order_id} has no intent; adopting as manual")
            self._adopt_orphan(o, now)

        # positions vs fills ledger (information only: the exchange is the source of truth)
        ledger = self._ledger_positions()
        for eid, q in ledger.items():
            have = self.positions[eid].quantity if eid in self.positions else 0
            if q != have:
                rep.discrepancies.append(f"{eid}: fills ledger {q} != exchange position {have}")
        self._write_positions(positions, now)

        # reservation <= cash
        self._recompute_reserved()
        if self.reserved > self.cash + 1e-6:
            rep.discrepancies.append(f"reservation {self.reserved:.0f} > cash {self.cash:.0f}")
            cancelled = await self._shed_buys(limit=3)
            rep.repaired.extend(f"cancelled {iid[:8]} to restore reservation <= cash" for iid in cancelled)
        with dbengine.session() as s:
            s.add(
                AccountSnap(ts=now, cash=self.cash, positions_value=self.positions_value, total=self.equity, reserved=self.reserved)
            )
        rep.cash, rep.positions_value, rep.reserved = self.cash, self.positions_value, self.reserved
        rep.open_orders, rep.working_intents = len(open_orders), len(self.working())
        for d in rep.discrepancies:
            log.warning("reconcile discrepancy: %s", d)
        for r in rep.repaired:
            log.info("reconcile repair: %s", r)
        self.last_reconcile = rep
        return rep

    def _apply_fill_rows(self, i: OrderIntent, fs: list[FillInfo]) -> None:
        qty = sum(int(f.quantity) for f in fs)
        cost = sum(float(f.price) * int(f.quantity) for f in fs)
        prev_q = int(i.filled_qty or 0)
        prev_cost = (i.avg_fill or 0.0) * prev_q
        new_q = prev_q + qty
        i.filled_qty = new_q
        i.avg_fill = (prev_cost + cost) / new_q if new_q else i.avg_fill
        if new_q >= int(i.quantity) and str(i.state) in WORKING_STATES:
            self._set_state(i, IntentState.FILLED)
        else:
            self._persist(i)

    def _adopt_orphan(self, o: OrderInfo, now: datetime) -> None:
        intent = OrderIntent(
            id=o.idempotency_key or new_idempotency_key(),
            exchange_id=o.exchange_id or "?",
            side=str(o.side or "yes"),
            action=str(o.action or "buy"),
            price=float(o.price or 0.5),
            quantity=int(o.quantity or 0),
            strategy=Strategy.MANUAL.value,
            state=IntentState.ACKED.value,
            sig_order_id=o.order_id,
            created_at=ensure_utc(o.created_at) if o.created_at else now,
            updated_at=now,
            expires_at=ensure_utc(o.expires_at) if o.expires_at else None,
            filled_qty=int(o.filled or 0),
            paper=self.paper,
            error="adopted from open orders",
        )
        self._persist(intent)

    def _ledger_positions(self) -> dict[str, int]:
        with dbengine.session() as s:
            rows = s.execute(select(Fill).where(Fill.paper == self.paper)).scalars().all()
        out: dict[str, int] = {}
        for f in rows:
            sign = 1 if (f.side == "yes") == (f.action == "buy") else -1
            out[f.exchange_id] = out.get(f.exchange_id, 0) + sign * int(f.quantity)
        return {k: v for k, v in out.items() if v != 0}

    def _write_positions(self, positions: list[PositionInfo], now: datetime) -> None:
        with dbengine.session() as s:
            seen = set()
            for p in positions:
                seen.add(p.exchange_id)
                row = s.get(Position, p.exchange_id)
                if row is None:
                    s.add(Position(exchange_id=p.exchange_id, quantity=int(p.quantity), avg_price=p.avg_price, mark=p.mark, updated_at=now))
                else:
                    row.quantity, row.avg_price, row.mark, row.updated_at = int(p.quantity), p.avg_price, p.mark, now
            for row in s.execute(select(Position)).scalars().all():
                if row.exchange_id not in seen and row.quantity != 0:
                    row.quantity, row.updated_at = 0, now

    async def _shed_buys(self, limit: int) -> list[str]:
        """Cancel the lowest-priority / newest resting buys until reservation <= cash."""
        cancelled: list[str] = []
        buys = [i for i in self.acked() if self._cash_required(i) > 0]
        # lowest priority (highest number) first, newest first within a priority
        buys.sort(key=lambda i: (priority_for(i.strategy), ensure_utc(i.created_at)), reverse=True)
        for i in buys:
            if self.reserved <= self.cash + 1e-6 or len(cancelled) >= limit:
                break
            if await self._cancel(i):
                cancelled.append(i.id)
            self._recompute_reserved()
        return cancelled

    # -- public: kill / cancel-all -------------------------------------------------------------
    async def kill(self, reason: str) -> None:
        """Cancel everything, mark the KV 'killed' flag and refuse all further sends."""
        async with self._lock:
            await self._kill_locked(reason)

    async def _kill_locked(self, reason: str) -> None:
        if self.killed is not None:
            return
        log.critical("KILL SWITCH: %s", reason)
        self.killed = reason
        try:
            await self._cancel_all_locked()
        except Exception:  # noqa: BLE001
            log.exception("cancel_all failed during kill")
        with dbengine.session() as s:
            kv_set(s, KV_KILLED, {"reason": reason, "ts": self.clock().isoformat(), "paper": self.paper})

    async def cancel_all(self) -> int:
        """Cancel every resting order (used on shutdown and by the kill switch)."""
        async with self._lock:
            return await self._cancel_all_locked()

    async def _cancel_all_locked(self) -> int:
        n = await self.adapter.cancel_all()
        for i in self.working():
            self._set_state(i, IntentState.CANCELLED, error="cancel_all")
        self.reserved = 0.0
        log.info("cancel_all: exchange cancelled %s orders", n)
        return int(n or 0)

    @staticmethod
    def clear_killed() -> None:
        """Clear the persistent kill flag (operator action after investigating)."""
        with dbengine.session() as s:
            kv_set(s, KV_KILLED, None)


def _position_value(p: PositionInfo) -> float:
    mark = p.mark if p.mark is not None else (p.avg_price if p.avg_price is not None else 0.5)
    return p.quantity * mark if p.quantity > 0 else -p.quantity * (1.0 - mark)
