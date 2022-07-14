from __future__ import annotations

from dxtbx.model.experiment_list import ExperimentListFactory
from scitbx import matrix

from dials.algorithms.integration import Corrections, CorrectionsMulti
from dials.array_family import flex


def test_run(dials_data):
    filename = dials_data("centroid_test_data", pathlib=True) / "experiments.json"

    exlist = ExperimentListFactory.from_json_file(filename)
    assert len(exlist) == 1

    rlist = flex.reflection_table.from_predictions_multi(exlist)

    corrector = CorrectionsMulti()
    for experiment in exlist:
        corrector.append(
            Corrections(experiment.beam, experiment.goniometer, experiment.detector)
        )

    lp1 = corrector.lp(rlist["id"], rlist["s1"])

    lp2 = flex.double(
        [LP_calculations(exlist[i], s1) for i, s1 in zip(rlist["id"], rlist["s1"])]
    )

    diff = flex.abs(lp1 - lp2)
    assert diff.all_lt(1e-7)


def LP_calculations(experiment, s1):
    """See Kabsch, J. Appl. Cryst 1988 21 916-924."""

    tpl_n = experiment.beam.get_polarization_normal()
    tpl_s0 = experiment.beam.get_s0()
    tpl_m2 = experiment.goniometer.get_rotation_axis()
    tpl_s1 = s1
    p = experiment.beam.get_polarization_fraction()

    n = matrix.col(tpl_n)
    s0 = matrix.col(tpl_s0)
    u = matrix.col(tpl_m2)
    s = matrix.col(tpl_s1)

    L_f = abs(s.dot(u.cross(s0))) / (s.length() * s0.length())

    P_f = (1 - 2 * p) * (1 - (n.dot(s) / s.length()) ** 2.0) + p * (
        1 + (s.dot(s0) / (s.length() * s0.length())) ** 2.0
    )

    return L_f / P_f
