"""Test that compute_delta_cchalf returns required values"""
from __future__ import annotations

import json
from unittest import mock

from dxtbx.model import Crystal, Experiment, Scan
from dxtbx.model.experiment_list import ExperimentList
from libtbx import phil

from dials.algorithms.scaling.model.model import KBScalingModel
from dials.algorithms.scaling.scale_and_filter import AnalysisResults, log_cycle_results
from dials.algorithms.statistics.cc_half_algorithm import CCHalfFromDials, DeltaCCHalf
from dials.array_family import flex
from dials.util.options import ArgumentParser


def generate_test_reflections(n=2):
    """Generate data for testing."""
    reflections = flex.reflection_table()
    for id_ in range(0, n):
        r = flex.reflection_table()
        r["id"] = flex.int(10, id_)
        r["xyzobs.px.value"] = flex.vec3_double([(0, 0, i + 0.5) for i in range(0, 10)])
        r.experiment_identifiers()[id_] = str(id_)
        r.set_flags(flex.bool(10, True), r.flags.integrated)
        r.set_flags(flex.bool(10, True), r.flags.scaled)
        reflections.extend(r)
    return reflections


def generated_params():
    """Generate a param phil scope."""
    phil_scope = phil.parse(
        """
      include scope dials.algorithms.scaling.scaling_options.phil_scope
      include scope dials.algorithms.scaling.model.model.model_phil_scope
      include scope dials.algorithms.scaling.scaling_refiner.scaling_refinery_phil_scope
  """,
        process_includes=True,
    )
    parser = ArgumentParser(phil=phil_scope, check_format=False)
    parameters, _ = parser.parse_args(args=[], quick_parse=True, show_diff_phil=False)
    parameters.model = "KB"
    return parameters


def get_scaling_model():
    """Make a KB Scaling model instance"""
    return KBScalingModel.from_data(generated_params(), [], [])


def generate_test_experiments(n=2):
    """Make a test experiment list"""
    experiments = ExperimentList()
    exp_dict = {
        "__id__": "crystal",
        "real_space_a": [1.0, 0.0, 0.0],
        "real_space_b": [0.0, 1.0, 0.0],
        "real_space_c": [0.0, 0.0, 2.0],
        "space_group_hall_symbol": " C 2y",
    }
    for i in range(n):
        experiments.append(
            Experiment(
                crystal=Crystal.from_dict(exp_dict),
                sequence=Scan(image_range=[1, 10], oscillation=[0.0, 1.0]),
                scaling_model=get_scaling_model(),
                identifier=str(i),
            )
        )
    return experiments


def test_scale_and_filter_results_logging():
    """Test ScaleAndFilter.log_cycle_results method."""
    results = AnalysisResults()
    scaling_script = mock.Mock()
    scaling_script.merging_statistics_result = "stats_results"
    scaling_script.scaled_miller_array.size.return_value = 1000

    filter_script = mock.Mock()
    filter_script.results_summary = {
        "dataset_removal": {
            "mode": "image_group",
            "image_ranges_removed": [[(6, 10), 0]],
            "experiments_fully_removed": [],
            "experiment_ids_fully_removed": [],
            "n_reflections_removed": 50,
        },
        "mean_cc_half": 80.0,
        "per_dataset_delta_cc_half_values": {
            "delta_cc_half_values": [-0.1, 0.1, -0.2, 0.2]
        },
    }

    def _parse_side_effect(*args):
        return args[0]

    with mock.patch.object(
        results, "_parse_merging_stats", side_effect=_parse_side_effect
    ):
        res = log_cycle_results(results, scaling_script, filter_script)
        # test things have been logged correctly
        cycle_results = res.get_cycle_results()
        assert len(cycle_results) == 1
        assert cycle_results[0]["cumul_percent_removed"] == 100 * 50.0 / 1000.0
        assert cycle_results[0]["n_removed"] == 50
        assert cycle_results[0]["image_ranges_removed"] == [[(6, 10), 0]]
        assert cycle_results[0]["removed_datasets"] == []
        assert cycle_results[0]["delta_cc_half_values"] == [-0.1, 0.1, -0.2, 0.2]
        assert res.get_merging_stats()[0] == "stats_results"
        assert res.initial_n_reflections == 1000

    # add another cycle of results
    with mock.patch.object(
        results, "_parse_merging_stats", side_effect=_parse_side_effect
    ):
        res = log_cycle_results(res, scaling_script, filter_script)
        cycle_results = res.get_cycle_results()
        assert len(cycle_results) == 2
        assert cycle_results[1]["cumul_percent_removed"] == 100 * 2 * 50.0 / 1000.0
        assert cycle_results[1]["n_removed"] == 50
        assert cycle_results[1]["image_ranges_removed"] == [[(6, 10), 0]]
        assert cycle_results[1]["removed_datasets"] == []
        assert cycle_results[0]["delta_cc_half_values"] == [-0.1, 0.1, -0.2, 0.2]
        assert res.get_merging_stats()[1] == "stats_results"
        assert res.initial_n_reflections == 1000


def test_compute_delta_cchalf_returned_results():
    """Test that delta cchalf return necessary values for scale_and_filter."""

    # Check for correct recording of
    # results_summary['per_dataset_delta_cc_half_values']['delta_cc_half_values']
    summary = {}
    delta_cc = {0: -4, 1: 2, 2: -3, 3: -5, 4: 1}
    sorted_data, sorted_ccs = DeltaCCHalf.sort_deltacchalf_values(delta_cc, summary)
    expected_data_order = [3, 0, 2, 4, 1]
    expected_cc_order = [-5, -4, -3, 1, 2]
    assert list(sorted_data) == expected_data_order
    assert list(sorted_ccs) == expected_cc_order
    assert (
        summary["per_dataset_delta_cc_half_values"]["delta_cc_half_values"]
        == expected_cc_order
    )

    # Check for correct recording for dataset mode
    exp = generate_test_experiments(2)
    refls = generate_test_reflections(2)
    ids_to_remove = [0]
    results_summary = {"dataset_removal": {}}
    _ = CCHalfFromDials.remove_datasets_below_cutoff(
        exp, refls, ids_to_remove, results_summary
    )
    assert "experiments_fully_removed" in results_summary["dataset_removal"]
    assert "n_reflections_removed" in results_summary["dataset_removal"]
    assert results_summary["dataset_removal"]["experiments_fully_removed"] == ["0"]
    assert results_summary["dataset_removal"]["n_reflections_removed"] == 10

    # Check for correct recording for image group mode.
    exp = generate_test_experiments(2)
    refls = generate_test_reflections(2)
    ids_to_remove = [0, 1]
    image_group_to_expid_and_range = {
        0: ("0", (1, 5)),
        1: ("0", (6, 10)),
        2: ("1", (1, 5)),
        3: ("1", (6, 10)),
    }
    expids_to_image_groups = {"0": [0, 1], "1": [2, 3]}
    results_summary = {"dataset_removal": {}}
    _ = CCHalfFromDials.remove_image_ranges_below_cutoff(
        exp,
        refls,
        ids_to_remove,
        image_group_to_expid_and_range,
        expids_to_image_groups,
        results_summary,
    )
    assert "experiments_fully_removed" in results_summary["dataset_removal"]
    assert "n_reflections_removed" in results_summary["dataset_removal"]
    assert "image_ranges_removed" in results_summary["dataset_removal"]
    assert results_summary["dataset_removal"]["experiments_fully_removed"] == ["0"]
    assert results_summary["dataset_removal"]["n_reflections_removed"] == 10
    assert [(6, 10), 0] in results_summary["dataset_removal"]["image_ranges_removed"]
    assert [(1, 5), 0] in results_summary["dataset_removal"]["image_ranges_removed"]
    assert len(results_summary["dataset_removal"]["image_ranges_removed"]) == 2


def test_analysis_results_to_from_dict():
    d = {
        "termination_reason": "made up",
        "initial_expids_and_image_ranges": [["foo", [1, 42]], ["bar", [1, 10]]],
        "expids_and_image_ranges": [["foo", [1, 42]]],
        "cycle_results": {"1": {"some stat": -424242}},
        "initial_n_reflections": 424242,
        "final_stats": "some final stats",
    }
    results = AnalysisResults.from_dict(d)
    # The cycle_results dict output by AnalysisResults has integer keys but after
    # conversion to json has str keys. AnalysisResults.from_dict expects str keys,
    # hence do the comparison after converting to/from json
    assert json.loads(json.dumps(results.to_dict())) == d
