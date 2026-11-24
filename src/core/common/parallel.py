"""
Ordered-window parallelism for chunk translation.

The translation pipelines are sequential by design: a chunk's checkpoint pointer
(`current_chunk_index`) must advance contiguously so resume always restarts at
`current_chunk_index + 1`. Naive fan-out with out-of-order completion would let
a late-finishing early chunk be skipped on resume.

The ordered-concurrent pattern preserves that invariant while still issuing
several LLM requests at once: up to `workers` requests are in flight, and
results are handed back strictly in input order so the caller can persist them
(and advance the checkpoint pointer) exactly as it would sequentially.

With one worker this degrades to plain sequential iteration, byte-for-byte
identical to the legacy loops.
"""
import asyncio

try:  # Python >= 3.10
    from contextlib import aclosing
except ImportError:  # pragma: no cover - Python 3.8/3.9 fallback
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def aclosing(thing):
        """Backport of `contextlib.aclosing` (added in Python 3.10)."""
        try:
            yield thing
        finally:
            await thing.aclose()


async def iter_ordered_concurrent(items, workers, worker, should_interrupt=None):
    """Yield ``(item, result_or_exception)`` in input order while keeping up to
    ``workers`` ``worker(item)`` coroutines in flight at all times.

    The
    moment one request finishes, the next is launched, so the pool stays full
    and throughput approaches ``workers`` x even when per-item latency varies
    (the common case for cloud LLMs). Results are still surfaced strictly in
    input order, so the caller can persist them and advance a checkpoint pointer
    exactly as it would sequentially.

    With ``workers == 1`` the next item is only launched after the previous one
    has been yielded and processed, so a caller that mutates shared state between
    yields (e.g. previous-translation context) sees it reflected in the next
    call — byte-for-byte identical to a plain sequential loop.

    Args:
        items: ordered work items (e.g. chunk indices).
        workers: maximum number of concurrent ``worker`` coroutines (>= 1).
        worker: coroutine factory called once per item.
        should_interrupt: optional callable; when it returns True, no new work is
            launched (already in-flight items still complete and are yielded).
            The caller is responsible for detecting the interruption itself and
            persisting partial state once iteration ends.

    Callers that may ``break`` out of the ``async for`` (e.g. on a rate-limit
    result) MUST wrap the iteration in :func:`aclosing`::

        async with aclosing(iter_ordered_concurrent(...)) as stream:
            async for item, result in stream:
                ...

    A bare ``break`` only suspends the generator: its ``finally`` block — and
    therefore the cancellation of the still-running tasks — is deferred to
    garbage collection, so the abandoned LLM requests keep running (and keep
    being billed) for an unbounded time. ``aclosing`` closes the generator at
    block exit, which cancels every in-flight task immediately.

    Because results are yielded in order, every item before the one that
    triggered the break has already been yielded (and committed by the caller).
    """
    limit = max(1, int(workers))
    n = len(items)
    pending_tasks = {}      # asyncio.Task -> position
    results = {}            # position -> result or Exception
    next_launch = 0
    next_yield = 0

    try:
        while next_yield < n:
            # Fill the pool, unless interruption was requested.
            while (next_launch < n
                   and len(pending_tasks) < limit
                   and not (should_interrupt and should_interrupt())):
                pos = next_launch
                task = asyncio.ensure_future(worker(items[pos]))
                pending_tasks[task] = pos
                next_launch += 1

            if not pending_tasks:
                # Nothing in flight and nothing new to launch (interrupted or
                # done): surface whatever contiguous results remain, then stop.
                break

            done, _ = await asyncio.wait(
                list(pending_tasks.keys()), return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                pos = pending_tasks.pop(task)
                try:
                    results[pos] = task.result()
                except Exception as exc:  # noqa: BLE001 - surfaced to caller in order
                    results[pos] = exc

            while next_yield in results:
                yield items[next_yield], results.pop(next_yield)
                next_yield += 1
    finally:
        for task in pending_tasks:
            task.cancel()
        if pending_tasks:
            await asyncio.gather(*pending_tasks.keys(), return_exceptions=True)
