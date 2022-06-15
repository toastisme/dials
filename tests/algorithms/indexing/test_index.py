from __future__ import annotations

import collections
import os
import pathlib

import procrunner
import pytest

from cctbx import uctbx
from dxtbx.model import ExperimentList
from dxtbx.serialize import load

from dials.array_family import flex


def unit_cells_are_similar(
    uc1, uc2, relative_length_tolerance=0.01, absolute_angle_tolerance=1
):
    # see also uctbx.cpp unit_cell::is_similar_to()
    l1 = uc1.parameters()
    l2 = uc2.parameters()
    for i in range(3):
        if abs(min(l1[i], l2[i]) / max(l1[i], l2[i]) - 1) > relative_length_tolerance:
            return False
    for i in range(3, 6):
        if abs(l1[i] - l2[i]) > absolute_angle_tolerance:
            if abs(l1[i] - (180 - l2[i])) > absolute_angle_tolerance:
                return False
    return True


_indexing_result = collections.namedtuple(
    "indexing", ["indexed_reflections", "experiments", "rmsds"]
)


def run_indexing(
    reflections,
    experiment,
    working_directory,
    extra_args,
    expected_unit_cell,
    expected_rmsds,
    expected_hall_symbol,
    n_expected_lattices=1,
    relative_length_tolerance=0.005,
    absolute_angle_tolerance=0.5,
):
    commands = ["dials.index"]
    if isinstance(reflections, list):
        commands.extend(reflections)
    else:
        commands.append(reflections)
    if isinstance(experiment, list):
        commands.extend(experiment)
    else:
        commands.append(experiment)
    commands.extend(extra_args)

    result = procrunner.run(commands, working_directory=working_directory)
    assert not result.returncode and not result.stderr

    out_expts = working_directory / "indexed.expt"
    out_refls = working_directory / "indexed.refl"
    assert out_expts.is_file()
    assert out_refls.is_file()

    experiments_list = load.experiment_list(out_expts, check_format=False)
    assert len(experiments_list.crystals()) == n_expected_lattices
    indexed_reflections = flex.reflection_table.from_file(out_refls)
    indexed_reflections.assert_experiment_identifiers_are_consistent(experiments_list)
    rmsds = None

    for i, experiment in enumerate(experiments_list):
        assert unit_cells_are_similar(
            experiment.crystal.get_unit_cell(),
            expected_unit_cell,
            relative_length_tolerance=relative_length_tolerance,
            absolute_angle_tolerance=absolute_angle_tolerance,
        ), (
            experiment.crystal.get_unit_cell().parameters(),
            expected_unit_cell.parameters(),
        )
        sg = experiment.crystal.get_space_group()
        assert sg.type().hall_symbol() == expected_hall_symbol, (
            sg.type().hall_symbol(),
            expected_hall_symbol,
        )
        reflections = indexed_reflections.select(indexed_reflections["id"] == i)
        mi = reflections["miller_index"]
        assert (mi != (0, 0, 0)).count(False) == 0
        reflections = reflections.select(mi != (0, 0, 0))
        reflections = reflections.select(
            reflections.get_flags(reflections.flags.used_in_refinement)
        )
        assert len(reflections) > 0
        obs_x, obs_y, obs_z = reflections["xyzobs.mm.value"].parts()
        calc_x, calc_y, calc_z = reflections["xyzcal.mm"].parts()
        rmsd_x = flex.mean(flex.pow2(obs_x - calc_x)) ** 0.5
        rmsd_y = flex.mean(flex.pow2(obs_y - calc_y)) ** 0.5
        rmsd_z = flex.mean(flex.pow2(obs_z - calc_z)) ** 0.5
        rmsds = (rmsd_x, rmsd_y, rmsd_z)
        for actual, expected in zip(rmsds, expected_rmsds):
            assert actual <= expected, f"{rmsds} {expected_rmsds}"
        assert experiment.identifier != ""
        expt = ExperimentList()
        expt.append(experiment)
        reflections.assert_experiment_identifiers_are_consistent(expt)

    return _indexing_result(indexed_reflections, experiments_list, rmsds)


def test_index_i04_weak_data_fft3d(dials_regression, tmp_path):
    # thaumatin
    data_dir = os.path.join(dials_regression, "indexing_test_data", "i04_weak_data")
    pickle_path = os.path.join(data_dir, "full.pickle")
    sequence_path = os.path.join(data_dir, "experiments_import.json")
    extra_args = [
        "bin_size_fraction=0.25",
        "image_range=1,20",
        "image_range=250,270",
        "image_range=520,540",
    ]
    expected_unit_cell = uctbx.unit_cell((57.7, 57.7, 149.8, 90, 90, 90))
    expected_rmsds = (0.05, 0.04, 0.0005)
    expected_hall_symbol = " P 1"

    run_indexing(
        pickle_path,
        sequence_path,
        tmp_path,
        extra_args,
        expected_unit_cell,
        expected_rmsds,
        expected_hall_symbol,
    )


def test_index_trypsin_four_lattice_P212121(dials_regression, tmp_path):
    # synthetic trypsin multi-lattice dataset (4 lattices)
    data_dir = pathlib.Path(dials_regression) / "indexing_test_data" / "trypsin"
    pickle_path = data_dir / "P1_X6_1_2_3_4.pickle"
    sequence_path = data_dir / "experiments_P1_X6_1_2_3_4.json"
    extra_args = [
        "indexing.method=real_space_grid_search",
        "reflections_per_degree=10",
        "n_macro_cycles=5",
        "known_symmetry.unit_cell=54.3,58.3,66.5,90,90,90",
        "known_symmetry.space_group=P212121",
        "image_range=0,10",
        "beam.fix=all",
        "detector.fix=all",
        "max_cell=70",
    ]
    expected_unit_cell = uctbx.unit_cell((54.3, 58.3, 66.5, 90, 90, 90))
    expected_rmsds = (0.28, 0.30, 0.006)
    expected_hall_symbol = " P 2ac 2ab"
    n_expected_lattices = 1

    run_indexing(
        pickle_path,
        sequence_path,
        tmp_path,
        extra_args,
        expected_unit_cell,
        expected_rmsds,
        expected_hall_symbol,
        n_expected_lattices=n_expected_lattices,
        relative_length_tolerance=0.02,
        absolute_angle_tolerance=1,
    )


def test_index_i04_weak_data_fft1d(dials_regression, tmp_path):
    # thaumatin
    data_dir = pathlib.Path(dials_regression) / "indexing_test_data" / "i04_weak_data"
    pickle_path = data_dir / "full.pickle"
    sequence_path = data_dir / "experiments_import.json"
    extra_args = [
        "n_macro_cycles=2",
        "indexing.method=fft1d",
        "bin_size_fraction=0.25",
        "image_range=1,20",
        "image_range=250,270",
        "image_range=520,540",
    ]
    expected_unit_cell = uctbx.unit_cell((57.7, 57.7, 149.9, 90, 90, 90))
    expected_rmsds = (0.06, 0.05, 0.0005)
    expected_hall_symbol = " P 1"

    run_indexing(
        pickle_path,
        sequence_path,
        tmp_path,
        extra_args,
        expected_unit_cell,
        expected_rmsds,
        expected_hall_symbol,
    )


def test_index_trypsin_index_assignment_local(dials_regression, tmp_path):
    # synthetic trypsin multi-lattice dataset (3 lattices)
    data_dir = pathlib.Path(dials_regression) / "indexing_test_data" / "trypsin"
    pickle_path = data_dir / "P1_X6_1_2_3.pickle"
    sequence_path = data_dir / "experiments_P1_X6_1_2_3.json"
    extra_args = [
        "indexing.method=real_space_grid_search",
        "d_min_start=3",
        "n_macro_cycles=3",
        "known_symmetry.unit_cell=54.3,58.3,66.5,90,90,90",
        "known_symmetry.space_group=P212121",
        "image_range=0,10",
        "beam.fix=all",
        "detector.fix=all",
        "max_lattices=3",
        "index_assignment.method=local",
        "nearest_neighbours=50",
    ]

    expected_unit_cell = uctbx.unit_cell((54.3, 58.3, 66.5, 90, 90, 90))
    expected_rmsds = (0.33, 0.40, 0.0024)
    expected_hall_symbol = " P 2ac 2ab"
    n_expected_lattices = 3

    run_indexing(
        pickle_path,
        sequence_path,
        tmp_path,
        extra_args,
        expected_unit_cell,
        expected_rmsds,
        expected_hall_symbol,
        n_expected_lattices=n_expected_lattices,
        relative_length_tolerance=0.02,
        absolute_angle_tolerance=1,
    )


def test_index_peak_search_clean(dials_regression, tmp_path):
    # test indexing from single image of i04_weak_data
    data_dir = pathlib.Path(dials_regression) / "indexing_test_data" / "i04_weak_data"
    pickle_path = data_dir / "first_image.pickle"
    sequence_path = data_dir / "experiments_import.json"
    extra_args = [
        "indexing.method=fft3d",
        "known_symmetry.space_group=P4",
        "known_symmetry.unit_cell=57.8,57.8,150,90,90,90",
        "peak_search=clean",
        "min_samples=15",
        "n_macro_cycles=4",
        "reciprocal_space_grid.d_min=4",
    ]

    expected_unit_cell = uctbx.unit_cell((57.8, 57.8, 150, 90, 90, 90))
    expected_rmsds = (0.06, 0.07, 0.003)
    expected_hall_symbol = " P 4"

    run_indexing(
        pickle_path,
        sequence_path,
        tmp_path,
        extra_args,
        expected_unit_cell,
        expected_rmsds,
        expected_hall_symbol,
    )


@pytest.mark.parametrize("specify_unit_cell", [False, True])
def test_index_imosflm_tutorial(dials_regression, tmp_path, specify_unit_cell):
    # test on spots derived from imosflm tutorial data:
    # http://www.ccp4.ac.uk/courses/BCA2005/tutorials/dataproc-tutorial.html
    data_dir = pathlib.Path(dials_regression) / "indexing_test_data" / "imosflm_hg_mar"
    pickle_path = data_dir / "strong.pickle"
    sequence_path = data_dir / "experiments.json"

    unit_cell = uctbx.unit_cell((58.373, 58.373, 155.939, 90, 90, 120))
    hall_symbol = '-R 3 2"'

    extra_args = [
        "bin_size_fraction=0.25",
        'known_symmetry.space_group="Hall: %s"' % hall_symbol.replace('"', '\\"'),
    ]
    if specify_unit_cell:
        extra_args.append(
            'known_symmetry.unit_cell="%s %s %s %s %s %s"' % unit_cell.parameters()
        )

    expected_unit_cell = unit_cell
    expected_hall_symbol = hall_symbol
    expected_rmsds = (0.08, 0.11, 0.004)

    run_indexing(
        pickle_path,
        sequence_path,
        tmp_path,
        extra_args,
        expected_unit_cell,
        expected_rmsds,
        expected_hall_symbol,
    )


@pytest.fixture(scope="session")
def insulin_spotfinding(dials_data, tmp_path_factory):
    """Return experiment and reflection files for 2 images of the insulin dataset"""

    data_dir = dials_data("insulin", pathlib=True)
    tmp_path = tmp_path_factory.mktemp("insulin")

    command = ["dials.import"]
    for i, image_path in enumerate(("insulin_1_001.img", "insulin_1_045.img")):
        command.append(data_dir / image_path)

    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr

    experiment = tmp_path / "imported.expt"
    assert experiment.is_file()

    command = ["dials.find_spots", "nproc=1", experiment]
    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr

    reflections = tmp_path / "strong.refl"
    assert reflections.is_file()

    return experiment, reflections


@pytest.mark.parametrize("method", ["fft3d", "fft1d", "real_space_grid_search"])
def test_index_insulin_multi_sequence(insulin_spotfinding, tmp_path, method):
    experiment, reflections = insulin_spotfinding
    expected_unit_cell = uctbx.unit_cell(
        (78.163, 78.163, 78.163, 90.000, 90.000, 90.000)
    )
    expected_hall_symbol = " I 2 2 3"
    expected_rmsds = (0.05, 0.06, 0.01)
    extra_args = [
        'known_symmetry.unit_cell="%s %s %s %s %s %s"'
        % expected_unit_cell.parameters(),
        f'known_symmetry.space_group="Hall: {expected_hall_symbol}"',
        f"indexing.method={method}",
        "treat_single_image_as_still=False",
    ]
    run_indexing(
        reflections,
        experiment,
        tmp_path,
        extra_args,
        expected_unit_cell,
        expected_rmsds,
        expected_hall_symbol,
    )


@pytest.fixture(scope="session")
def insulin_spotfinding_stills(dials_data, tmp_path_factory):
    """Return experiment and reflection files for 1 image of the insulin
    dataset treated as still image"""

    data_dir = dials_data("insulin", pathlib=True)
    tmp_path = tmp_path_factory.mktemp("insulin")

    command = [
        "dials.import",
        "convert_sequences_to_stills=True",
        data_dir / "insulin_1_001.img",
    ]
    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr

    experiment = tmp_path / "imported.expt"
    assert experiment.is_file()

    command = ["dials.find_spots", "nproc=1", experiment]
    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr

    reflections = tmp_path / "strong.refl"
    assert reflections.is_file()

    return experiment, reflections


@pytest.mark.xfel
@pytest.mark.parametrize("method", ["fft3d", "fft1d", "real_space_grid_search"])
def test_index_insulin_force_stills(insulin_spotfinding_stills, tmp_path, method):
    experiment, reflections = insulin_spotfinding_stills
    expected_unit_cell = uctbx.unit_cell(
        (78.163, 78.163, 78.163, 90.000, 90.000, 90.000)
    )
    expected_hall_symbol = " I 2 2 3"
    expected_rmsds = (0.05, 0.06, 0.01)

    extra_args = [
        "stills.indexer=stills",
        'known_symmetry.unit_cell="%s %s %s %s %s %s"'
        % expected_unit_cell.parameters(),
        f'known_symmetry.space_group="Hall: {expected_hall_symbol}"',
        f"indexing.method={method}",
    ]

    run_indexing(
        reflections,
        experiment,
        tmp_path,
        extra_args,
        expected_unit_cell,
        expected_rmsds,
        expected_hall_symbol,
    )


def test_multiple_experiments(dials_regression, tmp_path):
    # Test indexing 4 lysozyme still shots in a single dials.index job
    #   - the first image doesn't index
    #   - the last three images do index
    data_dir = (
        pathlib.Path(dials_regression) / "indexing_test_data" / "i24_lysozyme_stills"
    )
    pickle_path = data_dir / "strong.pickle"
    experiments_json = data_dir / "imported_experiments.json"

    expected_unit_cell = uctbx.unit_cell((38.06, 78.78, 78.91, 90, 90, 90))
    expected_hall_symbol = " P 1"
    expected_rmsds = (0.1, 0.07, 0.0)

    extra_args = [
        "stills.indexer=sequences",
        "joint_indexing=False",
        "outlier.algorithm=sauter_poon",
    ]

    run_indexing(
        pickle_path,
        experiments_json,
        tmp_path,
        extra_args,
        expected_unit_cell,
        expected_rmsds,
        expected_hall_symbol,
        n_expected_lattices=3,
        relative_length_tolerance=0.01,
    )


def test_index_4rotation(dials_regression, tmp_path):
    data_dir = pathlib.Path(dials_regression) / "indexing_test_data" / "4rotation"
    pickle_path = data_dir / "strong.pickle"
    sequence_path = data_dir / "experiments.json"
    extra_args = [
        "max_refine=10",
        "reflections_per_degree=50",
        "known_symmetry.space_group=R3",
        "n_macro_cycles=3",
    ]
    expected_unit_cell = uctbx.unit_cell((48.397, 48.397, 284.767, 90, 90, 120))
    expected_rmsds = (0.06, 0.08, 0.22)
    expected_hall_symbol = " R 3"

    result = run_indexing(
        pickle_path,
        sequence_path,
        tmp_path,
        extra_args,
        expected_unit_cell,
        expected_rmsds,
        expected_hall_symbol,
    )
    assert len(result.indexed_reflections) > 276800, len(result.indexed_reflections)


def test_index_small_molecule_multi_sequence_4(dials_regression, tmp_path):
    # test for small molecule multi-sequence indexing, 4 sequences with different values
    # of goniometer.fixed_rotation()
    data_dir = pathlib.Path(dials_regression) / "indexing_test_data" / "multi_sweep"
    pickle_paths = [
        sorted((data_dir / f"SWEEP{i + 1}" / "index").glob("*_strong.pickle"))[0]
        for i in range(4)
    ]
    sequence_paths = [
        data_dir / f"SWEEP{i + 1}" / "index" / "experiments.json" for i in range(4)
    ]
    extra_args = ["known_symmetry.space_group=I4", "filter_ice=False"]
    expected_unit_cell = uctbx.unit_cell((7.310, 7.310, 6.820, 90.000, 90.000, 90.000))
    expected_rmsds = (0.10, 0.7, 0.5)
    expected_hall_symbol = " I 4"

    result = run_indexing(
        pickle_paths,
        sequence_paths,
        tmp_path,
        extra_args,
        expected_unit_cell,
        expected_rmsds,
        expected_hall_symbol,
    )
    assert len(result.indexed_reflections) > 1250, len(result.indexed_reflections)


def test_index_small_molecule_multi_sequence_3(dials_regression, tmp_path):
    # test for small molecule multi-sequence indexing, 3 sequences with different values
    # of goniometer setting rotation (i.e. phi scans)
    data_dir = pathlib.Path(dials_regression) / "dials-191"
    print(data_dir)
    pickle_paths = [
        sorted(data_dir.glob(f"*_SWEEP{i + 1}_strong.pickle"))[0] for i in range(3)
    ]
    sequence_paths = [
        sorted(data_dir.glob(f"*_SWEEP{i + 1}_experiments.json"))[0] for i in range(3)
    ]
    extra_args = ["filter_ice=False"]
    expected_unit_cell = uctbx.unit_cell(
        (9.440, 15.313, 17.126, 90.073, 90.106, 79.248)
    )
    expected_rmsds = (0.32, 0.34, 0.005)
    expected_hall_symbol = " P 1"

    result = run_indexing(
        pickle_paths,
        sequence_paths,
        tmp_path,
        extra_args,
        expected_unit_cell,
        expected_rmsds,
        expected_hall_symbol,
    )
    assert len(result.indexed_reflections) > 12000, len(result.indexed_reflections)
    # expect at least indexed 2000 reflections per experiment
    for i in range(3):
        assert (result.indexed_reflections["id"] == i).count(True) > 2000


def test_index_small_molecule_ice_max_cell(dials_regression, tmp_path):
    # test for small molecule indexing: presence of ice rings makes max-cell
    # estimation tricky
    data_dir = os.path.join(dials_regression, "indexing_test_data", "MXSW-904")
    pickle_path = os.path.join(data_dir, "1_SWEEP1_strong.pickle")
    experiments = os.path.join(data_dir, "1_SWEEP1_experiments.json")
    extra_args = ["filter_ice=False"]
    expected_unit_cell = uctbx.unit_cell((11.72, 11.72, 11.74, 109.08, 109.24, 108.99))
    expected_rmsds = (0.06, 0.05, 0.04)
    expected_hall_symbol = " P 1"

    result = run_indexing(
        pickle_path,
        experiments,
        tmp_path,
        extra_args,
        expected_unit_cell,
        expected_rmsds,
        expected_hall_symbol,
    )
    assert len(result.indexed_reflections) > 1300, len(result.indexed_reflections)


@pytest.mark.xfail
def test_refinement_failure_on_max_lattices_a15(dials_regression, tmp_path):
    """Problem: Sometimes there is enough data to index, but not enough to
    refine. If this happens in the (N>1)th crystal of max_lattices, then
    all existing solutions are also dropped."""
    data_dir = os.path.join(dials_regression, "indexing_test_data", "lattice_failures")

    result = procrunner.run(
        [
            "dials.index",
            os.path.join(data_dir, "lpe4-2-a15_strong.pickle"),
            os.path.join(data_dir, "lpe4-2-a15_datablock.json"),
            "max_lattices=3",
        ],
        working_directory=tmp_path,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "indexed.refl").if_file()
    assert (tmp_path / "indexed.expt").is_file()
    experiments_list = load.experiment_list(
        tmp_path / "indexed.expt", check_format=False
    )
    assert len(experiments_list) == 2

    # now try to reindex with existing model
    result = procrunner.run(
        [
            "dials.index",
            tmp_path / "indexed.expt",
            os.path.join(data_dir, "lpe4-2-a15_strong.pickle"),
            "max_lattices=2",
        ],
        working_directory=tmp_path,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "indexed.refl").is_file()
    assert (tmp_path / "indexed.expt").is_file()
    experiments_list = load.experiment_list(
        tmp_path / "indexed.expt", check_format=False
    )
    assert len(experiments_list) == 2


@pytest.mark.xfel
def test_stills_indexer_multi_lattice_bug_MosaicSauter2014(dials_regression, tmp_path):
    """Problem: In stills_indexer, before calling the refine function, the
    experiment list contains a list of dxtbx crystal models (that are not
    MosaicSauter2014 models). The conversion to MosaicSauter2014 is made
    during the refine step when functions from nave_parameters is called.
    If the experiment list contains more than 1 experiment, for eg.
    multiple lattices, only the first crystal gets assigned mosaicity. In
    actuality, all crystal models should be assigned mosaicity. This test
    only compares whether or not all crystal models have been assigned a
    MosaicSauter2014 model."""

    import dxtbx.model
    from dxtbx.model import Crystal
    from dxtbx.model.experiment_list import (
        Experiment,
        ExperimentList,
        ExperimentListFactory,
    )

    from dials.algorithms.indexing.stills_indexer import StillsIndexer
    from dials.array_family import flex
    from dials.command_line.stills_process import (
        phil_scope as stills_process_phil_scope,
    )

    experiment_data = os.path.join(
        dials_regression,
        "refinement_test_data",
        "cspad_refinement",
        "cspad_refined_experiments_step6_level2_300.json",
    )
    reflection_data = os.path.join(
        dials_regression,
        "refinement_test_data",
        "cspad_refinement",
        "cspad_reflections_step7_300.pickle",
    )

    refl = flex.reflection_table.from_file(reflection_data)
    explist = ExperimentListFactory.from_json_file(experiment_data, check_format=False)[
        0:2
    ]
    reflist = refl.select(refl["id"] < 2)  # Only use the first 2 for convenience
    # Construct crystal models that don't have mosaicity. These A,B,C values are the same
    # as read in from the dials_regression folder
    # Crystal-0
    cs0 = Crystal(explist[0].crystal)
    exp0 = Experiment(
        imageset=explist[0].imageset,
        beam=explist[0].beam,
        detector=explist[0].detector,
        goniometer=None,
        sequence=None,
        crystal=cs0,
    )

    # Crystal-1
    cs1 = Crystal(explist[1].crystal)
    exp1 = Experiment(
        imageset=explist[1].imageset,
        beam=explist[1].beam,
        detector=explist[1].detector,
        goniometer=None,
        sequence=None,
        crystal=cs1,
    )
    # Construct a new experiment_list that will be passed on for refinement
    unrefined_explist = ExperimentList([exp0, exp1])
    # Get default params from stills_process and construct StillsIndexer, then run refinement
    params = stills_process_phil_scope.extract()
    SI = StillsIndexer(reflist, unrefined_explist, params=params)
    refined_explist, new_reflist = SI.refine(unrefined_explist, reflist)
    # Now check whether the models have mosaicity after stills_indexer refinement
    # Also check that mosaicity values are within expected limits
    for ii, crys in enumerate(refined_explist.crystals()):
        assert isinstance(crys, dxtbx.model.MosaicCrystalSauter2014)
        if ii == 0:
            assert crys.get_domain_size_ang() == pytest.approx(2242.0, rel=0.1)
        if ii == 1:
            assert crys.get_domain_size_ang() == pytest.approx(2689.0, rel=0.1)


@pytest.mark.parametrize(
    "indexer_type,fix_cell",
    (("sequences", False), pytest.param("stills", True, marks=pytest.mark.xfel)),
)
def test_index_ED_still_low_res_spot_match(
    dials_data, tmp_path, indexer_type, fix_cell
):
    # test indexing from a single simulated lysozyme ED still

    image_path = (
        dials_data("image_examples", pathlib=True) / "simtbx_FormatSMVJHSim_001.img"
    )

    command = ["dials.import", image_path]
    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr

    experiment = tmp_path / "imported.expt"
    assert experiment.is_file()

    command = ["dials.find_spots", "nproc=1", experiment]
    result = procrunner.run(command, working_directory=tmp_path)
    assert not result.returncode and not result.stderr

    reflections = tmp_path / "strong.refl"

    extra_args = [
        "indexing.method=low_res_spot_match",
        "known_symmetry.space_group=P43212",
        "known_symmetry.unit_cell=78.84,78.84,38.29,90,90,90",
        "stills.indexer=" + indexer_type,
        "n_macro_cycles=2",
        "detector.fix_list=Dist",
    ]
    if fix_cell:
        extra_args += ["crystal.fix=cell"]

    expected_unit_cell = uctbx.unit_cell((78.84, 78.84, 38.29, 90, 90, 90))
    expected_rmsds = (0.0065, 0.0065, 0.000)
    expected_hall_symbol = " P 4nw 2abw"

    run_indexing(
        reflections,
        experiment,
        tmp_path,
        extra_args,
        expected_unit_cell,
        expected_rmsds,
        expected_hall_symbol,
    )


@pytest.mark.parametrize(
    "cell_params",
    [
        (44.47, 52.85, 62.23, 115.14, 101.72, 90.01),
        (52.85, 62.23, 44.47, 101.72, 90.01, 115.14),
    ],
)
def test_unconventional_P1_cell(dials_data, tmp_path, cell_params):
    """
    Indexing in P1 should succeed even if the cell parameters are provided in
    a non-conventional setting
    """
    data_dir = dials_data("mpro_x0305_processed", pathlib=True)
    experiment = data_dir / "imported.expt"
    reflections = data_dir / "strong.refl"

    cell_params_str = ",".join([str(x) for x in cell_params])
    extra_args = [
        "indexing.method=fft3d",
        "known_symmetry.space_group=P1",
        "known_symmetry.unit_cell=" + cell_params_str,
    ]
    expected_unit_cell = uctbx.unit_cell(cell_params)
    expected_rmsds = (1, 1, 1)
    expected_hall_symbol = " P 1"

    run_indexing(
        reflections,
        experiment,
        tmp_path,
        extra_args,
        expected_unit_cell,
        expected_rmsds,
        expected_hall_symbol,
    )


def test_real_space_grid_search_no_unit_cell(dials_regression, tmp_path):
    data_dir = pathlib.Path(dials_regression) / "indexing_test_data" / "i04_weak_data"
    experiments_json = data_dir / "experiments_import.json"
    pickle_path = data_dir / "full.pickle"
    commands = [
        "dials.index",
        experiments_json,
        pickle_path,
        "indexing.method=real_space_grid_search",
    ]
    result = procrunner.run(commands, working_directory=tmp_path)
    assert result.stderr
    assert (
        result.stderr.strip()
        == b"Target unit cell must be provided for real_space_grid_search"
    )


def test_index_known_orientation(dials_data, tmp_path):
    data_dir = dials_data("vmxi_proteinase_k_sweeps", pathlib=True)
    experiments_json = data_dir / "experiments_0.json"
    reflections = data_dir / "reflections_0.pickle"

    expected_unit_cell = uctbx.unit_cell((68.395, 68.395, 104, 90, 90, 90))
    expected_rmsds = (0.013, 0.012, 0.008)
    expected_hall_symbol = " P 4"

    run_indexing(
        reflections,
        experiments_json,
        tmp_path,
        [],
        expected_unit_cell,
        expected_rmsds,
        expected_hall_symbol,
    )


def test_all_expt_ids_have_expts(dials_data, tmp_path):
    result = procrunner.run(
        [
            "dials.index",
            dials_data("vmxi_thaumatin_grid_index", pathlib=True) / "split_07602.expt",
            dials_data("vmxi_thaumatin_grid_index", pathlib=True) / "split_07602.refl",
            "stills.indexer=sequences",
            "indexing.method=real_space_grid_search",
            "space_group=P4",
            "unit_cell=58,58,150,90,90,90",
            "max_lattices=8",
            "beam.fix=all",
            "detector.fix=all",
        ],
        working_directory=tmp_path,
    )
    assert not result.returncode and not result.stderr
    assert (tmp_path / "indexed.expt").is_file()
    assert (tmp_path / "indexed.refl").is_file()

    refl = flex.reflection_table.from_file(tmp_path / "indexed.refl")
    expt = ExperimentList.from_file(tmp_path / "indexed.expt", check_format=False)

    assert flex.max(refl["id"]) + 1 == len(expt)
