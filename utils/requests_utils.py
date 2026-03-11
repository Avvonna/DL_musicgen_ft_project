import logging
import time
from typing import Callable, Optional, TypeVar

from openai import RateLimitError

logger = logging.getLogger(__name__)

T = TypeVar("T")

def retry_call(fn: Callable[[], T], retries: int = 3, base_delay: float = 1.0) -> T:
    """Повтор с откатом (учитывает rate limit и X-RateLimit-Reset)"""
    delay = base_delay
    last_exc: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            return fn()
        except RateLimitError as e:
            last_exc = e
            wait = delay
            try:
                resp = getattr(e, "response", None)
                ts = (
                    resp.headers.get("X-RateLimit-Reset")
                    if resp and hasattr(resp, "headers")
                    else None
                )
                if ts:
                    ts = float(ts) / 1000.0  # мс -> сек
                    wait = max(0.0, ts - time.time())
            except Exception:
                pass

            if attempt < retries:
                logger.warning(
                    f"Rate limit (attempt {attempt}/{retries}); sleep {wait:.1f}s"
                )
                time.sleep(wait)
                delay *= 2
            else:
                raise
        except Exception as e:
            last_exc = e
            if attempt < retries:
                logger.warning(f"{e} (attempt {attempt}/{retries}); sleep {delay:.1f}s")
                time.sleep(delay)
                delay *= 2
            else:
                raise last_exc
    raise
