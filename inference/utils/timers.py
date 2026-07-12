"""
Timing utility.

Unlike the other modules in this package, this is **not** a consolidation of
existing notebook code -- no reusable timer class existed. The notebook
measured elapsed time with the same inline pattern repeated at every call
site:

    t0 = time.perf_counter()
    ...
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

and, for whole-stage wall-clock timing:

    start_time = time.time()
    ...
    elapsed = time.time() - start_time

This module provides both as a small reusable context manager, per the
explicit request to consolidate a "timers" utility. It is additive
infrastructure for future modules (e.g. deployment/benchmarking.py) to use
if they choose to -- it does not retroactively change any existing logic,
and when the benchmarking stage is migrated the original inline
``time.perf_counter()`` measurements will be preserved verbatim wherever
the exact original numeric behaviour matters, to guarantee no output changes.
"""
from __future__ import annotations

import time
from types import TracebackType
from typing import Optional, Type


class Timer:
    """Context manager measuring wall-clock elapsed time.

    Example:
        with Timer() as t:
            do_work()
        print(t.elapsed_ms)   # float, milliseconds
        print(t.elapsed_s)    # float, seconds

    Uses ``time.perf_counter()`` (matching the notebook's sub-operation
    timing calls) rather than ``time.time()`` (matching its whole-stage
    timing calls) since perf_counter is monotonic and the better default for
    interval measurement; callers needing wall-clock timestamps should use
    ``time.time()`` directly, as the notebook did for stage start/finish logs.
    """

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed_s(self) -> float:
        """Elapsed seconds. Valid after the ``with`` block exits."""
        if self._start is None or self._end is None:
            raise RuntimeError("Timer has not completed a full 'with' block yet.")
        return self._end - self._start

    @property
    def elapsed_ms(self) -> float:
        """Elapsed milliseconds. Valid after the ``with`` block exits."""
        return self.elapsed_s * 1000.0
