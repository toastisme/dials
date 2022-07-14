from __future__ import annotations

import procrunner

import iotbx.mtz
from cctbx import uctbx
from libtbx.test_utils import approx_equal


def test(dials_data, tmpdir):
    g = sorted(f for f in dials_data("x4wide", pathlib=True).glob("*"))
    assert len(g) == 90

    commands = [
        ["dials.import"] + g,
        ["dials.slice_sequence", "imported.expt", "image_range=80,90"],
        ["dials.find_spots", "imported_80_90.expt", "nproc=1"],
        ["dials.index", "imported_80_90.expt", "strong.refl", "space_group=P41212"],
        ["dials.refine", "indexed.expt", "indexed.refl", "scan_varying=True"],
        ["dials.integrate", "refined.expt", "indexed.refl", "nproc=1"],
        [
            "dials.export",
            "refined.expt",
            "integrated.refl",
            "partiality_threshold=0.99",
        ],
    ]

    for cmd in commands:
        # print cmd
        result = procrunner.run(cmd, working_directory=tmpdir)
        assert not result.returncode and not result.stderr

    integrated_mtz = tmpdir.join("integrated.mtz")
    assert integrated_mtz.check(file=1)

    mtz_object = iotbx.mtz.object(file_name=integrated_mtz.strpath)
    assert mtz_object.column_labels()[:14] == [
        "H",
        "K",
        "L",
        "M_ISYM",
        "BATCH",
        "IPR",
        "SIGIPR",
        "I",
        "SIGI",
        "BG",
        "SIGBG",
        "FRACTIONCALC",
        "XDET",
        "YDET",
    ]

    assert len(mtz_object.batches()) == 11
    batch = mtz_object.batches()[0]
    expected_unit_cell = uctbx.unit_cell((42.5787, 42.5787, 40.2983, 90, 90, 90))
    assert expected_unit_cell.is_similar_to(uctbx.unit_cell(list(batch.cell()))), (
        expected_unit_cell.parameters(),
        list(batch.cell()),
    )
    assert mtz_object.space_group().type().lookup_symbol() == "P 41 21 2"
    assert approx_equal(mtz_object.n_reflections(), 7446, eps=2e3)
