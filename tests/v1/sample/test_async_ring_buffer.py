# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the sendnn-inference project

import pytest

from sendnn_inference.v1.sample.async_ring_buffer import (
    AsyncExponential_RingBuffer,
    AsyncRingBuffer,
    _AsyncCounterRingBuffer,
)


class TestAsyncRingBuffer:
    """Tests for AsyncRingBuffer, AsyncExponentialRingBuffer, and
    _AsyncCounterRingBuffer."""

    def test_abc_cannot_instantiate(self):
        """AsyncRingBuffer is abstract and must not be instantiated directly."""
        with pytest.raises(TypeError):
            AsyncRingBuffer(vocab_size=10, max_batch_size=4)  # type: ignore[abstract]

    def test_counter_no_duplicates(self):
        """Every row index must be yielded at most once across all borrows.

        The counter buffer does NOT guarantee strict global sequentiality:
        rows near the wrap boundary can be skipped.  The invariant it *does*
        guarantee is that no row index is ever handed to the consumer twice.
        """
        V, B, scale = 3, 4, 4
        buf = _AsyncCounterRingBuffer(vocab_size=V, max_batch_size=B, scale=scale)
        total_steps = 20
        seen: list[int] = []
        try:
            for _ in range(total_steps):
                with buf.borrow_rows(1) as rows:
                    assert rows.shape == (1, V)
                    # All columns of a row share the same counter value.
                    val = int(rows[0, 0].item())
                    seen.append(val)
        finally:
            buf.shutdown()

        assert len(seen) == len(set(seen)), f"duplicate row indices returned: {seen}"

    def test_counter_variable_batch_sizes(self):
        """No row index must appear twice across borrows of varying size."""
        V, B, scale = 2, 4, 4
        buf = _AsyncCounterRingBuffer(vocab_size=V, max_batch_size=B, scale=scale)
        batch_sizes = [1, 2, 3, 4, 1, 4, 2, 3]
        seen: list[int] = []
        try:
            for b in batch_sizes:
                with buf.borrow_rows(b) as rows:
                    assert rows.shape == (b, V)
                    for i in range(b):
                        seen.append(int(rows[i, 0].item()))
        finally:
            buf.shutdown()

        assert len(seen) == len(set(seen)), f"duplicate row indices returned: {seen}"

    def test_counter_wrap_around(self):
        """No row index is repeated when the buffer wraps multiple times."""
        V, B, scale = 2, 2, 4  # S = 8
        buf = _AsyncCounterRingBuffer(vocab_size=V, max_batch_size=B, scale=scale)
        n_steps = 5 * scale
        seen: list[int] = []
        try:
            for _ in range(n_steps):
                with buf.borrow_rows(B) as rows:
                    for i in range(B):
                        seen.append(int(rows[i, 0].item()))
        finally:
            buf.shutdown()

        assert len(seen) == len(set(seen)), f"duplicate row indices returned: {seen}"

    def test_exponential_shape_and_positivity(self):
        """AsyncExponentialRingBuffer returns correctly shaped positive tensors."""
        V, B = 16, 4
        buf = AsyncExponential_RingBuffer(vocab_size=V, max_batch_size=B)
        try:
            for b in [1, 2, B]:
                with buf.borrow_rows(b) as rows:
                    assert rows.shape == (b, V)
                    assert (rows > 0).all(), "exponential values must be positive"
        finally:
            buf.shutdown()

    def test_borrow_is_zero_copy(self):
        """borrow_rows must yield a view into the backing buffer, not a copy."""
        V, B = 8, 4
        buf = AsyncExponential_RingBuffer(vocab_size=V, max_batch_size=B)
        try:
            with buf.borrow_rows(B) as rows:
                assert rows.untyped_storage().data_ptr() == buf._buf.untyped_storage().data_ptr()
        finally:
            buf.shutdown()

    def test_release_on_exception(self):
        """Release must occur even when the consumer body raises."""
        V, B = 4, 2
        buf = _AsyncCounterRingBuffer(vocab_size=V, max_batch_size=B)
        try:
            with pytest.raises(RuntimeError, match="intentional") as _:
                with buf.borrow_rows(B):
                    raise RuntimeError("intentional")
        finally:
            # If release happened, a second borrow must succeed.
            with buf.borrow_rows(B) as rows:
                assert rows.shape == (B, V)
            buf.shutdown()

    def test_stop_joins_thread(self):
        """stop() must cause the background thread to finish."""
        buf = AsyncExponential_RingBuffer(vocab_size=4, max_batch_size=2)
        assert buf._thread.is_alive()
        buf.shutdown()
        assert not buf._thread.is_alive()

    @pytest.mark.parametrize("scale", [2, 3, 4, 8])
    def test_scale_invariance(self, scale: int):
        """Different scale values must all produce no duplicate row indices."""
        V, B = 4, 3
        buf = _AsyncCounterRingBuffer(vocab_size=V, max_batch_size=B, scale=scale)
        seen: list[int] = []
        try:
            for _ in range(scale * 3):
                with buf.borrow_rows(B) as rows:
                    for i in range(B):
                        seen.append(int(rows[i, 0].item()))
        finally:
            buf.shutdown()

        assert len(seen) == len(set(seen)), f"duplicate row indices returned: {seen}"
