"""Regression tests: Executor results must stay aligned with submitted jobs.

After `Executor.cancel()`, `results()` used to return only the jobs that had
finished, compacted into a shorter list. `results[k]` then no longer belonged
to the k-th submitted job, so scores were silently attributed to the wrong
sample/metric. These tests use only local synthetic coroutines.
"""

import asyncio
import math

import pytest

from ragas.executor import Executor


@pytest.mark.parametrize("batch_size", [None, 3])
def test_cancelled_results_keep_job_positions(batch_size):
    executor = Executor(show_progress=False, batch_size=batch_size)

    async def slow_job():
        # Never finishes before cancellation; it is cancelled by the executor.
        await asyncio.sleep(30)
        return "slow"

    async def fast_job():
        return "fast"

    async def cancelling_job():
        await asyncio.sleep(0.01)
        executor.cancel()
        return "cancelling"

    executor.submit(slow_job, name="job-0")
    executor.submit(fast_job, name="job-1")
    executor.submit(cancelling_job, name="job-2")

    results = executor.results()

    assert executor.is_cancelled()
    # One entry per submitted job, in submission order.
    assert len(results) == 3
    assert isinstance(results[0], float) and math.isnan(results[0])
    assert results[1] == "fast"
    assert results[2] == "cancelling"


def test_uncancelled_results_unchanged():
    executor = Executor(show_progress=False)

    async def echo(value):
        await asyncio.sleep(0.001 * (3 - value))
        return value

    for i in range(3):
        executor.submit(echo, i)

    assert executor.results() == [0, 1, 2]


def test_second_run_after_cancel_indexes_new_jobs():
    """Job indices keep counting across runs; the second run must map correctly."""
    executor = Executor(show_progress=False)

    async def echo(value):
        return value

    executor.submit(echo, "a")
    executor.submit(echo, "b")
    assert executor.results() == ["a", "b"]

    executor.submit(echo, "c")
    assert executor.results() == ["c"]
