"""Exchange error taxonomy. The OMS decides what to do; the client only classifies."""

from __future__ import annotations


class SigError(Exception):
    def __init__(self, message: str, status: int | None = None, code: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code


class RateLimited(SigError):
    """429 RATE_LIMITED: all writes are frozen for `retry_after` seconds."""

    def __init__(self, retry_after: float = 60.0):
        super().__init__("rate limited", 429, "RATE_LIMITED")
        self.retry_after = retry_after


class RequestInFlight(SigError):
    """409 REQUEST_IN_FLIGHT: same idempotency key holds a ~90 s lease; poll instead of resending."""


class OrderStatusUnknown(SigError):
    """502 ORDER_STATUS_UNKNOWN or a timeout: the order may or may not exist; retry with the SAME key."""


class ValidationError(SigError):
    """4xx VALIDATION_ERROR / insufficient funds / off-tick price: do not retry."""


class TransientError(SigError):
    """503 TX_CONFLICT / SERVICE_UNAVAILABLE: retry with backoff and the same key."""


class AuthError(SigError):
    """401/403: key missing, tournament not joined, or Terms not accepted. Stop and tell the user."""
