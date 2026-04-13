"""Dilation schedules for TCN blocks.

Each function takes ``n_blocks`` and returns a list of integer dilation values,
one per block.  Pass the result as the ``dilations`` argument to ``TCN``.

Schedules
---------
exponential
    Classic doubling schedule: 1, 2, 4, 8, …  Maximises receptive field growth
    per block.  This is the TCN default.

linear
    Uniformly increasing dilations: 1, 2, 3, 4, …  Slower RF growth than
    exponential; neighbouring blocks share more context overlap.

cyclic
    Repeats a base cycle: [1, 2, 4, 1, 2, 4, …]  Lets deeper blocks
    re-use small dilations, which can help when n_blocks is large.  The
    cycle is configurable.

fibonacci
    Fibonacci sequence starting at 1: 1, 1, 2, 3, 5, 8, …  Growth rate sits
    between linear and exponential.
"""

from __future__ import annotations


def exponential(n_blocks: int) -> list[int]:
    """Doubling dilations: 1, 2, 4, 8, …, 2^(n_blocks-1)."""
    return [2 ** i for i in range(n_blocks)]


def linear(n_blocks: int) -> list[int]:
    """Linearly increasing dilations: 1, 2, 3, …, n_blocks."""
    return list(range(1, n_blocks + 1))


def cyclic(n_blocks: int, cycle: list[int] | None = None) -> list[int]:
    """Repeat ``cycle`` to fill ``n_blocks`` positions.

    Parameters
    ----------
    n_blocks:
        Number of TCN blocks.
    cycle:
        Base pattern to repeat.  Defaults to ``[1, 2, 4]``.

    Examples
    --------
    >>> cyclic(7)
    [1, 2, 4, 1, 2, 4, 1]
    >>> cyclic(5, cycle=[1, 2])
    [1, 2, 1, 2, 1]
    """
    if cycle is None:
        cycle = [1, 2, 4]
    return [cycle[i % len(cycle)] for i in range(n_blocks)]


def fibonacci(n_blocks: int) -> list[int]:
    """Fibonacci dilations: 1, 1, 2, 3, 5, 8, …

    Examples
    --------
    >>> fibonacci(6)
    [1, 1, 2, 3, 5, 8]
    """
    if n_blocks == 0:
        return []
    seq = [1, 1]
    while len(seq) < n_blocks:
        seq.append(seq[-1] + seq[-2])
    return seq[:n_blocks]
