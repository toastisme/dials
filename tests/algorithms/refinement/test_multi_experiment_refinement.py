"""
A simple test of refinement using two crystals.
"""

from math import pi

from cctbx.sgtbx import space_group, space_group_symbols
from cctbx.uctbx import unit_cell
from dxtbx.model import ScanFactory
from dxtbx.model.experiment_list import Experiment, ExperimentList
from libtbx.phil import parse
from libtbx.test_utils import approx_equal
from rstbx.symmetry.constraints.parameter_reduction import symmetrize_reduce_enlarge
from scitbx import matrix
from scitbx.array_family import flex

import dials.tests.algorithms.refinement.setup_geometry as setup_geometry
from dials.algorithms.refinement.parameterisation.beam_parameters import (
    BeamParameterisation,
)
from dials.algorithms.refinement.parameterisation.crystal_parameters import (
    CrystalOrientationParameterisation,
    CrystalUnitCellParameterisation,
)
from dials.algorithms.refinement.parameterisation.detector_parameters import (
    DetectorParameterisationSinglePanel,
)
from dials.algorithms.refinement.prediction.managed_predictors import (
    ScansExperimentsPredictor,
    ScansRayPredictor,
)
from dials.algorithms.refinement.refiner import RefinerFactory, phil_scope
from dials.algorithms.spot_prediction import IndexGenerator, ray_intersection


def test(args=[]):

    #############################
    # Setup experimental models #
    #############################

    master_phil = parse(
        """
        include scope dials.tests.algorithms.refinement.geometry_phil
        include scope dials.tests.algorithms.refinement.minimiser_phil
        """,
        process_includes=True,
    )

    models = setup_geometry.Extract(
        master_phil,
        cmdline_args=args,
        local_overrides="geometry.parameters.random_seed = 1",
    )

    crystal1 = models.crystal

    models = setup_geometry.Extract(
        master_phil,
        cmdline_args=args,
        local_overrides="geometry.parameters.random_seed = 2",
    )

    mydetector = models.detector
    mygonio = models.goniometer
    crystal2 = models.crystal
    mybeam = models.beam

<<<<<<< HEAD:tests/algorithms/refinement/test_multi_experiment_refinement.py
    # Build a mock scan for an 18 degree sequence
    sf = ScanFactory()
=======
    # Build a mock scan for a 180 degree sequence
    sf = SequenceFactory()
>>>>>>> Name changes for Scan/sequence.:test/algorithms/refinement/test_multi_experiment_refinement.py
    myscan = sf.make_scan(
        image_range=(1, 180),
        exposure_times=0.1,
        oscillation=(0, 0.1),
        epochs=list(range(180)),
        deg=True,
    )
    sequence_range = myscan.get_oscillation_range(deg=False)
    im_width = myscan.get_oscillation(deg=False)[1]
    assert sequence_range == (0.0, pi / 10)
    assert approx_equal(im_width, 0.1 * pi / 180.0)

    # Build an experiment list
    experiments = ExperimentList()
    experiments.append(
        Experiment(
            beam=mybeam,
            detector=mydetector,
            goniometer=mygonio,
            scan=myscan,
            crystal=crystal1,
            imageset=None,
        )
    )
    experiments.append(
        Experiment(
            beam=mybeam,
            detector=mydetector,
            goniometer=mygonio,
            scan=myscan,
            crystal=crystal2,
            imageset=None,
        )
    )

    assert len(experiments.detectors()) == 1

    ##########################################################
    # Parameterise the models (only for perturbing geometry) #
    ##########################################################

    det_param = DetectorParameterisationSinglePanel(mydetector)
    s0_param = BeamParameterisation(mybeam, mygonio)
    xl1o_param = CrystalOrientationParameterisation(crystal1)
    xl1uc_param = CrystalUnitCellParameterisation(crystal1)
    xl2o_param = CrystalOrientationParameterisation(crystal2)
    xl2uc_param = CrystalUnitCellParameterisation(crystal2)

    # Fix beam to the X-Z plane (imgCIF geometry), fix wavelength
    s0_param.set_fixed([True, False, True])

    ################################
    # Apply known parameter shifts #
    ################################

    # shift detector by 1.0 mm each translation and 2 mrad each rotation
    det_p_vals = det_param.get_param_vals()
    p_vals = [a + b for a, b in zip(det_p_vals, [1.0, 1.0, 1.0, 2.0, 2.0, 2.0])]
    det_param.set_param_vals(p_vals)

    # shift beam by 2 mrad in free axis
    s0_p_vals = s0_param.get_param_vals()
    p_vals = list(s0_p_vals)

    p_vals[0] += 2.0
    s0_param.set_param_vals(p_vals)

    # rotate crystal a bit (=2 mrad each rotation)
    xlo_p_vals = []
    for xlo in (xl1o_param, xl2o_param):
        p_vals = xlo.get_param_vals()
        xlo_p_vals.append(p_vals)
        new_p_vals = [a + b for a, b in zip(p_vals, [2.0, 2.0, 2.0])]
        xlo.set_param_vals(new_p_vals)

    # change unit cell a bit (=0.1 Angstrom length upsets, 0.1 degree of
    # gamma angle)
    xluc_p_vals = []
    for xluc, xl in ((xl1uc_param, crystal1), ((xl2uc_param, crystal2))):
        p_vals = xluc.get_param_vals()
        xluc_p_vals.append(p_vals)
        cell_params = xl.get_unit_cell().parameters()
        cell_params = [
            a + b for a, b in zip(cell_params, [0.1, 0.1, 0.1, 0.0, 0.0, 0.1])
        ]
        new_uc = unit_cell(cell_params)
        newB = matrix.sqr(new_uc.fractionalization_matrix()).transpose()
        S = symmetrize_reduce_enlarge(xl.get_space_group())
        S.set_orientation(orientation=newB)
        X = tuple([e * 1.0e5 for e in S.forward_independent_parameters()])
        xluc.set_param_vals(X)

    #############################
    # Generate some reflections #
    #############################

    # All indices in a 2.5 Angstrom sphere for crystal1
    resolution = 2.5
    index_generator = IndexGenerator(
        crystal1.get_unit_cell(),
        space_group(space_group_symbols(1).hall()).type(),
        resolution,
    )
    indices1 = index_generator.to_array()

    # All indices in a 2.5 Angstrom sphere for crystal2
    resolution = 2.5
    index_generator = IndexGenerator(
        crystal2.get_unit_cell(),
        space_group(space_group_symbols(1).hall()).type(),
        resolution,
    )
    indices2 = index_generator.to_array()

    # Predict rays within the sequence range. Set experiment IDs
    ray_predictor = ScansRayPredictor(experiments, sequence_range)
    obs_refs1 = ray_predictor(indices1, experiment_id=0)
    obs_refs1["id"] = flex.int(len(obs_refs1), 0)
    obs_refs2 = ray_predictor(indices2, experiment_id=1)
    obs_refs2["id"] = flex.int(len(obs_refs2), 1)

    # Take only those rays that intersect the detector
    intersects = ray_intersection(mydetector, obs_refs1)
    obs_refs1 = obs_refs1.select(intersects)
    intersects = ray_intersection(mydetector, obs_refs2)
    obs_refs2 = obs_refs2.select(intersects)

    # Make a reflection predictor and re-predict for all these reflections. The
    # result is the same, but we gain also the flags and xyzcal.px columns
    ref_predictor = ScansExperimentsPredictor(experiments)
    obs_refs1 = ref_predictor(obs_refs1)
    obs_refs2 = ref_predictor(obs_refs2)

    # Set 'observed' centroids from the predicted ones
    obs_refs1["xyzobs.mm.value"] = obs_refs1["xyzcal.mm"]
    obs_refs2["xyzobs.mm.value"] = obs_refs2["xyzcal.mm"]

    # Invent some variances for the centroid positions of the simulated data
    im_width = 0.1 * pi / 18.0
    px_size = mydetector[0].get_pixel_size()
    var_x = flex.double(len(obs_refs1), (px_size[0] / 2.0) ** 2)
    var_y = flex.double(len(obs_refs1), (px_size[1] / 2.0) ** 2)
    var_phi = flex.double(len(obs_refs1), (im_width / 2.0) ** 2)
    obs_refs1["xyzobs.mm.variance"] = flex.vec3_double(var_x, var_y, var_phi)
    var_x = flex.double(len(obs_refs2), (px_size[0] / 2.0) ** 2)
    var_y = flex.double(len(obs_refs2), (px_size[1] / 2.0) ** 2)
    var_phi = flex.double(len(obs_refs2), (im_width / 2.0) ** 2)
    obs_refs2["xyzobs.mm.variance"] = flex.vec3_double(var_x, var_y, var_phi)

    # concatenate reflection lists
    obs_refs1.extend(obs_refs2)
    obs_refs = obs_refs1

    ###############################
    # Undo known parameter shifts #
    ###############################

    s0_param.set_param_vals(s0_p_vals)
    det_param.set_param_vals(det_p_vals)
    xl1o_param.set_param_vals(xlo_p_vals[0])
    xl2o_param.set_param_vals(xlo_p_vals[1])
    xl1uc_param.set_param_vals(xluc_p_vals[0])
    xl2uc_param.set_param_vals(xluc_p_vals[1])

    # scan static first
    params = phil_scope.fetch(source=parse("")).extract()
    refiner = RefinerFactory.from_parameters_data_experiments(
        params, obs_refs, experiments
    )
    refiner.run()

    # scan varying
    params.refinement.parameterisation.scan_varying = True
    refiner = RefinerFactory.from_parameters_data_experiments(
        params, obs_refs, experiments
    )
    refiner.run()

    # Ensure all models have scan-varying state set
    # (https://github.com/dials/dials/issues/798)
    refined_experiments = refiner.get_experiments()
    sp = [xl.get_num_scan_points() for xl in refined_experiments.crystals()]

    assert sp.count(181) == 2
