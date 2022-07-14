from __future__ import annotations

from dials.algorithms.symmetry import origin
from dials.array_family import flex


def test_origin_offset_miller_indices():
    mi = flex.miller_index(
        [(h, k, l) for h in range(5) for k in range(5) for l in range(5)]
    )
    omi = origin.offset_miller_indices(mi, (0, 1, -1))
    ref = flex.miller_index(
        [(h, k, l) for h in range(5) for k in range(1, 6) for l in range(-1, 4)]
    )

    assert ref == omi
