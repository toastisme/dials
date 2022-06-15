from __future__ import annotations

import math

import pytest

from . import geometry_phil


def test():
    from cctbx.sgtbx import space_group, space_group_symbols
    from dxtbx.model.experiment_list import Experiment, ExperimentList
    from libtbx.phil import parse
    from scitbx.array_family import flex

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
    from dials.algorithms.refinement.parameterisation.goniometer_parameters import (
        GoniometerParameterisation,
    )

    #### Import model parameterisations
    from dials.algorithms.refinement.parameterisation.prediction_parameters import (
        XYPhiPredictionParameterisation,
    )
    from dials.algorithms.refinement.prediction.managed_predictors import (
        ScansExperimentsPredictor,
        ScansRayPredictor,
    )

    ##### Imports for reflection prediction
    from dials.algorithms.spot_prediction import IndexGenerator, ray_intersection

    ##### Import model builder
    from .setup_geometry import Extract

    #### Create models

    overrides = """geometry.parameters.crystal.a.length.range = 10 50
  geometry.parameters.crystal.b.length.range = 10 50
  geometry.parameters.crystal.c.length.range = 10 50"""

    master_phil = parse(geometry_phil)

    models = Extract(master_phil, overrides)

    mydetector = models.detector
    mygonio = models.goniometer
    mycrystal = models.crystal
    mybeam = models.beam

    # Build a mock scan for a 72 degree sequence
    sequence_range = (0.0, math.pi / 5.0)
    from dxtbx.model import SequenceFactory

    sf = SequenceFactory()
    myscan = sf.make_scan(
        image_range=(1, 720),
        exposure_times=0.1,
        oscillation=(0, 0.1),
        epochs=list(range(720)),
        deg=True,
    )

    #### Create parameterisations of these models
    det_param = DetectorParameterisationSinglePanel(mydetector)
    s0_param = BeamParameterisation(mybeam, mygonio)
    xlo_param = CrystalOrientationParameterisation(mycrystal)
    xluc_param = CrystalUnitCellParameterisation(mycrystal)
    gon_param = GoniometerParameterisation(mygonio, mybeam)

    # Create an ExperimentList
    experiments = ExperimentList()
    experiments.append(
        Experiment(
            beam=mybeam,
            detector=mydetector,
            goniometer=mygonio,
            sequence=myscan,
            crystal=mycrystal,
            imageset=None,
        )
    )

    #### Unit tests

    # Build a prediction parameterisation
    pred_param = XYPhiPredictionParameterisation(
        experiments,
        detector_parameterisations=[det_param],
        beam_parameterisations=[s0_param],
        xl_orientation_parameterisations=[xlo_param],
        xl_unit_cell_parameterisations=[xluc_param],
        goniometer_parameterisations=[gon_param],
    )

    # Generate reflections
    resolution = 2.0
    index_generator = IndexGenerator(
        mycrystal.get_unit_cell(),
        space_group(space_group_symbols(1).hall()).type(),
        resolution,
    )
    indices = index_generator.to_array()

    # Predict rays within the sequence range
    ray_predictor = ScansRayPredictor(experiments, sequence_range)
    obs_refs = ray_predictor(indices)

    # Take only those rays that intersect the detector
    intersects = ray_intersection(mydetector, obs_refs)
    obs_refs = obs_refs.select(intersects)

    # Make a reflection predictor and re-predict for all these reflections. The
    # result is the same, but we gain also the flags and xyzcal.px columns
    ref_predictor = ScansExperimentsPredictor(experiments)
    obs_refs["id"] = flex.int(len(obs_refs), 0)
    obs_refs = ref_predictor(obs_refs)

    # Set 'observed' centroids from the predicted ones
    obs_refs["xyzobs.mm.value"] = obs_refs["xyzcal.mm"]

    # Invent some variances for the centroid positions of the simulated data
    im_width = 0.1 * math.pi / 180.0
    px_size = mydetector[0].get_pixel_size()
    var_x = flex.double(len(obs_refs), (px_size[0] / 2.0) ** 2)
    var_y = flex.double(len(obs_refs), (px_size[1] / 2.0) ** 2)
    var_phi = flex.double(len(obs_refs), (im_width / 2.0) ** 2)
    obs_refs["xyzobs.mm.variance"] = flex.vec3_double(var_x, var_y, var_phi)

    # use a ReflectionManager to exclude reflections too close to the spindle
    from dials.algorithms.refinement.reflection_manager import ReflectionManager

    refman = ReflectionManager(obs_refs, experiments, outlier_detector=None)
    refman.finalise()

    # Redefine the reflection predictor to use the type expected by the Target class
    ref_predictor = ScansExperimentsPredictor(experiments)

    # keep only those reflections that pass inclusion criteria and have predictions
    reflections = refman.get_matches()

    # get analytical gradients
    an_grads = pred_param.get_gradients(reflections)

    # get finite difference gradients
    p_vals = pred_param.get_param_vals()
    deltas = [1.0e-7] * len(p_vals)

    for i, delta in enumerate(deltas):
        val = p_vals[i]

        p_vals[i] -= delta / 2.0
        pred_param.set_param_vals(p_vals)

        ref_predictor(reflections)

        rev_state = reflections["xyzcal.mm"].deep_copy()

        p_vals[i] += delta
        pred_param.set_param_vals(p_vals)

        ref_predictor(reflections)

        fwd_state = reflections["xyzcal.mm"].deep_copy()
        p_vals[i] = val

        fd = fwd_state - rev_state
        x_grads, y_grads, phi_grads = fd.parts()
        x_grads /= delta
        y_grads /= delta
        phi_grads /= delta

        # compare with analytical calculation
        assert x_grads == pytest.approx(an_grads[i]["dX_dp"], abs=5.0e-6)
        assert y_grads == pytest.approx(an_grads[i]["dY_dp"], abs=5.5e-6)
        assert phi_grads == pytest.approx(an_grads[i]["dphi_dp"], abs=5.0e-6)

    # return to the initial state
    pred_param.set_param_vals(p_vals)
