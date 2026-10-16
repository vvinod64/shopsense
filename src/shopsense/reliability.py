"""Reliability for order-system calls: seeded fault injection, retries, circuit breaker and a
human-queue fallback.

Only transient errors are retried. LookupError and ValueError are business answers (unknown order,
bad reason code): they are never retried and never count towards opening a breaker.
"""
import functools
import random
import time
from typing import Callable, Optional

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

TRANSIENT = (ConnectionError, TimeoutError)
BUSINESS_ERRORS = (LookupError, ValueError)
HUMAN_QUEUE = "human_queue"
TOOL_CALL_LOG: list[dict] = []


# Fault injection

class FaultConfig:
    def __init__(self, seed: int, fail_rate: float = 0.0, timeout_rate: float = 0.0, malformed_rate: float = 0.0):
        assert fail_rate + timeout_rate + malformed_rate <= 1.0, "rates must not exceed 100%"
        self.seed, self.fail_rate, self.timeout_rate, self.malformed_rate = seed, fail_rate, timeout_rate, malformed_rate

    @classmethod
    def parse(cls, spec: str) -> "FaultConfig":
        """Parse 'seed=42,fail=0.2,timeout=0.1,malformed=0.1' (the SHOPSENSE_FAULTS format)."""
        kv = dict(part.split("=") for part in spec.replace(" ", "").split(",") if part)
        return cls(int(kv.get("seed", 0)), float(kv.get("fail", 0)), float(kv.get("timeout", 0)),
                   float(kv.get("malformed", 0)))

    def __repr__(self):
        return (f"FaultConfig(seed={self.seed}, fail={self.fail_rate}, timeout={self.timeout_rate}, "
                f"malformed={self.malformed_rate})")


def make_flaky(fn: Callable, config: FaultConfig, malformed: Callable = lambda *a, **k: {"malformed": True}):
    """Seeded wrapper: one random draw per call decides the outcome."""
    rng = random.Random(config.seed)
    calls = {"n": 0}

    @functools.wraps(fn)
    def flaky(*args, **kwargs):
        calls["n"] += 1
        roll = rng.random()
        if roll < config.fail_rate:
            raise ConnectionError(f"simulated order-system failure ({fn.__name__})")
        if roll < config.fail_rate + config.timeout_rate:
            raise TimeoutError(f"simulated order-system timeout ({fn.__name__})")
        if roll < config.fail_rate + config.timeout_rate + config.malformed_rate:
            return malformed(*args, **kwargs)
        return fn(*args, **kwargs)
    flaky.calls = calls
    return flaky


# Retries

def make_retrying(fn: Callable, attempts: int = 4, multiplier: float = 0.01, max_wait: float = 0.2):
    return retry(stop=stop_after_attempt(attempts), wait=wait_exponential(multiplier=multiplier, max=max_wait),
                 retry=retry_if_exception_type(TRANSIENT), reraise=True)(fn)


# Circuit breaker

class CircuitOpenError(Exception):
    """Raised when an open breaker short-circuits a call."""

CircuitBreakerOpen = CircuitOpenError     # alias used by the agent


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, reset_timeout: float = 10.0):
        self.failure_threshold, self.reset_timeout = failure_threshold, reset_timeout
        self.failures, self.state, self.opened_at = 0, "closed", None

    def call(self, fn, *args, **kwargs):
        if self.state == "open":
            if time.monotonic() - self.opened_at >= self.reset_timeout:
                self.state = "half_open"
            else:
                raise CircuitOpenError("circuit open - call short-circuited")
        try:
            result = fn(*args, **kwargs)
        except BUSINESS_ERRORS:
            self.failures, self.state = 0, "closed"      # a business error means the system is up
            raise
        except Exception:
            self.failures += 1
            if self.state == "half_open" or self.failures >= self.failure_threshold:
                self.state, self.opened_at = "open", time.monotonic()
            raise
        self.failures, self.state = 0, "closed"
        return result


BREAKERS: dict[str, CircuitBreaker] = {}


def breaker(name: str, failure_threshold: int = 3, reset_timeout: float = 30.0) -> CircuitBreaker:
    return BREAKERS.setdefault(name, CircuitBreaker(failure_threshold, reset_timeout))


def reset_breakers():
    BREAKERS.clear()


# Breaker + retries + fallback

def make_robust(fn: Callable, name: str, validate: Optional[Callable[[object], bool]] = None,
                cb: Optional[CircuitBreaker] = None, attempts: int = 4):
    """Breaker, then retries, then the call. Returns (value, status) where status is "ok" or "fallback"
    with value {"fallback": "human_queue", "reason": ...}. Business errors are raised to the caller.
    """
    from shopsense.tracing import span
    counter = {"n": 0}

    def counted(*args, **kwargs):
        counter["n"] += 1
        return fn(*args, **kwargs)
    retrying = make_retrying(counted, attempts=attempts)
    cb = cb or breaker(name)

    @functools.wraps(fn)
    def robust(*args, **kwargs):
        counter["n"] = 0
        entry = {"tool": name, "args": args, "ts": time.time()}
        with span(f"tool:{name}", as_type="tool", input={"args": [str(a) for a in args]}) as s:
            reason = None
            try:
                value = cb.call(retrying, *args, **kwargs)
                if validate and not validate(value):
                    reason = "malformed response"
            except BUSINESS_ERRORS as e:
                entry.update(status="business_error", error=str(e), attempts=counter["n"])
                TOOL_CALL_LOG.append(entry)
                s.update(output={"error": str(e)}, metadata={"status": "business_error", "attempts": counter["n"]})
                raise
            except CircuitOpenError:
                reason = "circuit open"
            except TRANSIENT as e:
                reason = f"retries exhausted ({type(e).__name__})"
            status = "fallback" if reason else "ok"
            if reason:
                value = {"fallback": HUMAN_QUEUE, "reason": reason, "tool": name}
            meta = {"status": status, "attempts": counter["n"], "breaker": cb.state, "reason": reason}
            entry.update(meta)
            TOOL_CALL_LOG.append(entry)
            s.update(output=value if status == "fallback" else {"ok": True}, metadata=meta,
                     level="WARNING" if status == "fallback" else "DEFAULT")
            return value, status
    robust.breaker = cb
    return robust


def make_robust_tool(fn: Callable, name: str, attempts: int = 3):
    """Tool variant: returns the value and raises when the order system is unavailable."""
    from shopsense.tracing import span
    retrying = make_retrying(fn, attempts=attempts)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        cb = breaker(name)
        entry = {"tool": name, "args": args, "ts": time.time()}
        with span(f"tool:{name}", as_type="tool", input={"args": [str(a) for a in args]}) as s:
            try:
                result = cb.call(retrying, *args, **kwargs)
            except Exception as e:
                entry.update(result=None, error=f"{type(e).__name__}: {e}")
                TOOL_CALL_LOG.append(entry)
                s.update(output={"error": entry["error"]}, metadata={"breaker": cb.state})
                raise
            entry.update(result=result, error=None)
            TOOL_CALL_LOG.append(entry)
            s.update(output={"ok": True}, metadata={"breaker": cb.state})
            return result
    return wrapper


# Self-check

BASE_CONFIG = FaultConfig(seed=42, fail_rate=0.2, timeout_rate=0.1, malformed_rate=0.1)


def run_fault_experiment(fn: Callable, args_cycle: list, validate: Callable, n: int = 200,
                         config: FaultConfig = BASE_CONFIG) -> dict:
    """Baseline, retries, and retries with fallback over n calls, each with a fresh flaky wrapper."""
    malformed = lambda *a, **k: {"malformed": True}
    out = {}
    flaky = make_flaky(fn, config, malformed)
    ok = conn = timeout = bad = 0
    for i in range(n):
        try:
            ok, bad = (ok + 1, bad) if validate(flaky(args_cycle[i % len(args_cycle)])) else (ok, bad + 1)
        except ConnectionError:
            conn += 1
        except TimeoutError:
            timeout += 1
    out["baseline"] = {"ok": ok, "conn": conn, "timeout": timeout, "malformed": bad, "success_rate": ok / n}

    retrying = make_retrying(make_flaky(fn, config, malformed))
    ok = bad = failed = 0
    for i in range(n):
        try:
            ok, bad = (ok + 1, bad) if validate(retrying(args_cycle[i % len(args_cycle)])) else (ok, bad + 1)
        except TRANSIENT:
            failed += 1
    out["retries"] = {"ok": ok, "malformed": bad, "failed": failed, "success_rate": ok / n}

    robust = make_robust(make_flaky(fn, config, malformed), "experiment", validate,
                         cb=CircuitBreaker(failure_threshold=3, reset_timeout=10.0))
    fallbacks = sum(1 for i in range(n) if robust(args_cycle[i % len(args_cycle)])[1] == "fallback")
    out["fallback"] = {"ok": n, "fallback_used": fallbacks, "success_rate": 1.0}

    down = make_flaky(fn, FaultConfig(seed=7, fail_rate=1.0), malformed)
    cb, real, short = CircuitBreaker(failure_threshold=3, reset_timeout=10.0), 0, 0
    for _ in range(10):
        try:
            cb.call(down, args_cycle[0])
        except ConnectionError:
            real += 1
        except CircuitOpenError:
            short += 1
    out["breaker"] = {"real_attempts": real, "short_circuited": short, "calls_that_hit_the_dependency": down.calls["n"]}
    return out


def self_check():
    import os
    os.environ.setdefault("SHOPSENSE_TRACING", "0")
    lookup = lambda ref: {"order_ref": ref, "order_value_inr": 999, "customer_ref": "KW-C-1"}
    valid = lambda r: isinstance(r, dict) and {"order_ref", "order_value_inr", "customer_ref"} <= set(r)
    r = run_fault_experiment(lookup, ["A", "B", "C", "D"], valid)
    # Fixed seed and rates give reproducible numbers.
    assert r["baseline"] == {"ok": 114, "conn": 38, "timeout": 32, "malformed": 16, "success_rate": 0.57}, r
    assert r["retries"] == {"ok": 179, "malformed": 20, "failed": 1, "success_rate": 0.895}, r
    assert r["fallback"] == {"ok": 200, "fallback_used": 21, "success_rate": 1.0}, r
    assert r["breaker"] == {"real_attempts": 3, "short_circuited": 7, "calls_that_hit_the_dependency": 3}, r

    # half-open: one trial call after the cooldown; a failure re-opens
    cb = CircuitBreaker(failure_threshold=1, reset_timeout=0.05)
    def boom(_):
        raise ConnectionError("down")
    for _ in range(2):
        try:
            cb.call(boom, 1)
        except (ConnectionError, CircuitOpenError):
            pass
    assert cb.state == "open"
    time.sleep(0.06)
    try:
        cb.call(boom, 1)
    except ConnectionError:
        pass
    assert cb.state == "open", "a failed half-open trial must re-open the breaker"
    time.sleep(0.06)
    assert cb.call(lambda x: x, 5) == 5 and cb.state == "closed", "a successful trial closes it"

    # business errors are neither retried nor counted
    n = {"calls": 0}
    def not_found(_):
        n["calls"] += 1
        raise LookupError("order not found")
    nf = make_robust(not_found, "nf", cb=CircuitBreaker(3, 10))
    for _ in range(5):
        try:
            nf("x")
        except LookupError:
            pass
    assert n["calls"] == 5 and nf.breaker.state == "closed"
    print("reliability self-check: PASS - seeded fault experiment reproduced exactly (57% -> 89.5% -> 100% with "
          "21 fallbacks; breaker 3 real / 7 short-circuited), half-open re-opens on failure, "
          "business errors never retried.")


if __name__ == "__main__":
    self_check()
