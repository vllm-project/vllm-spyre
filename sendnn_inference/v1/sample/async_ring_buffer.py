# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the sendnn-inference project

import contextlib
import queue
import threading
from abc import ABC, abstractmethod
from collections.abc import Generator

import torch


class AsyncRingBuffer(ABC):
    """Pre-generates data rows on a background thread via a ring buffer.

    Maintains a contiguous ``(S, V)`` tensor (``S = scale * max_batch_size``)
    and two shared counters:

    * ``_read_pos`` — next row index the consumer will read from.
    * ``_tail`` — upper bound (in unwrapped space) up to which the consumer
      may read without stalling.

    On init the buffer is fully filled and ``_tail = S``.  The consumer
    advances ``_read_pos`` after each call; when it approaches the end of the
    buffer it wraps back to 0.  Each consumed segment is enqueued for the
    background thread to refill, which increments ``_tail`` once done.

    Args:
        vocab_size: Number of columns ``V``.
        max_batch_size: Maximum rows per :meth:`get_rows` call ``B``.
        scale: Buffer depth multiplier; ``S = scale * B``.  Must be >= 2 so
            there is always at least one full batch of pre-filled rows ahead
            of the consumer.
    """

    def __init__(
        self,
        vocab_size: int,
        max_batch_size: int,
        scale: int = 4,
    ) -> None:
        assert scale >= 2, "scale must be >= 2"
        self._V = vocab_size
        self._B = max_batch_size
        self._S = scale * max_batch_size

        # buffer allocation
        self._buf = torch.empty(self._S, self._V, dtype=torch.float32)

        # first-time buffer initialization
        self._refill_slice(0, self._S)
        self._tail: int = self._S
        self._read_pos: int = 0

        # _tail and _read_pos are guarded by _cond.
        self._cond = threading.Condition(threading.Lock())

        # Refill requests: (start, end, wrap)
        self._refill_q: queue.Queue[tuple[int, int, bool] | None] = queue.Queue()

        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    @abstractmethod
    def _refill_slice(self, start: int, end: int) -> None:
        """Fill ``self._buf[start:end]`` with fresh values in-place."""
        ...

    @property
    def vocab_size(self) -> int:
        return self._V

    @contextlib.contextmanager
    def borrow_rows(self, n: int) -> Generator[torch.Tensor, None, None]:
        """Context manager that yields a zero-copy ``(n, V)`` view.

        The backing rows are released for refill automatically when the
        ``with`` block exits, even if an exception is raised.  The view
        must not be used after the block.

        Args:
            n: Number of rows to borrow.  Must satisfy ``1 <= n <= B``.

        Raises:
            ValueError: If ``n`` is outside the valid range.

        Example::

            with buf.borrow_rows(batch_size) as noise:
                tokens = probs.div(noise).argmax(dim=-1)
        """
        if n > self._B or n < 1:
            raise ValueError(f"n (got {n}) must satisfy 1 <= n <= {self._B} (max_batch_size)")

        start = self._read_pos
        end = start + n

        # wait for the consumer to fill up at least n many values ahead
        with self._cond:
            self._cond.wait_for(lambda: self._tail >= end)

        # get view (zero-copy)
        view = self._buf[start:end]

        wrap: bool = end > self._S - self._B
        if wrap:
            with self._cond:
                self._tail -= self._S

            self._read_pos = 0
        else:
            self._read_pos = end

        try:
            # yield view to outside consumer
            yield view
        finally:
            # issue refill request once view has been consumed and returned
            self._refill_q.put((start, end, wrap))

    def _produce(self) -> None:
        while True:
            req = self._refill_q.get()

            # handle termination signal
            if req is None:
                break

            # refill buffer
            start, end, wrap = req
            self._refill_slice(start, end)

            increment = (self._S - start) if wrap else (end - start)
            with self._cond:
                self._tail += increment
                self._cond.notify_all()

    def shutdown(self) -> None:
        """Signal the background thread to stop and wait for it to exit."""
        self._refill_q.put(None)
        self._thread.join()


class AsyncExponential_RingBuffer(AsyncRingBuffer):
    """Ring buffer that pre-generates exponential log noise via ``exponential_().log_()``."""

    def _refill_slice(self, start: int, end: int) -> None:
        self._buf[start:end].exponential_().log_()


class _AsyncCounterRingBuffer(AsyncRingBuffer):
    """Ring buffer that fills each row with the cumulative row index.

    Used in tests to verify that consumers receive the correct rows in order
    without repeating any.
    """

    def __init__(self, vocab_size: int, max_batch_size: int, scale: int = 4) -> None:
        self._total_generated: int = 0
        super().__init__(vocab_size, max_batch_size, scale)

    def _refill_slice(self, start: int, end: int) -> None:
        n = end - start
        for i in range(n):
            self._buf[start + i].fill_(self._total_generated)
            self._total_generated += 1
