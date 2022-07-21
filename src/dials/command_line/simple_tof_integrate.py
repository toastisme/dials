# LIBTBX_SET_DISPATCHER_NAME dev.dials.simple_integrate
from __future__ import annotations

import logging

import cctbx.array_family.flex
from dxtbx.model import Goniometer

import dials.util.log
import dials_array_family_flex_ext
from dials.algorithms.integration.report import IntegrationReport, ProfileModelReport
from dials.algorithms.profile_model.gaussian_rs import GaussianRSProfileModeller
from dials.algorithms.profile_model.gaussian_rs import Model as GaussianRSProfileModel
from dials.algorithms.shoebox import MaskCode
from dials.array_family import flex
from dials.command_line.integrate import process_reference
from dials.extensions.simple_background_ext import SimpleBackgroundExt
from dials.extensions.simple_centroid_ext import SimpleCentroidExt
from dials.model.data import make_image
from dials.util.options import ArgumentParser, reflections_and_experiments_from_files
from dials.util.phil import parse
from dials.util.version import dials_version
from dials_algorithms_integration_integrator_ext import ShoeboxProcessor

logger = logging.getLogger("dials.command_line.simple_integrate")

phil_scope = parse(
    """
output {
reflections = 'simple_integrated.refl'
    .type = str
    .help = "The integrated output filename"
phil = 'dials.simple_integrate.phil'
    .type = str
    .help = "The output phil file"
log = 'dials.simple_integrate.log'
    .type = str
    .help = "The log filename"
}
"""
)

"""
Kabsch 2010 refers to
Kabsch W., Integration, scaling, space-group assignment and
post-refinment, Acta Crystallographica Section D, 2010, D66, 133-144
Usage:
$ dev.dials.simple_integrate.py refined.expt refined.refl
"""


def get_reference_profiles_as_reflections(model):
    model_data = []
    for i in range(len(model)):
        if model.valid(i):
            try:
                panel = (model.panel(i)[0],)
            except TypeError:
                panel = model.panel(i)
            coords = model.coord_with_panel(i, panel)
            model_data.append([panel, coords])

    xyz = cctbx.array_family.flex.vec3_double(len(model_data), (0, 0, 0))
    panel_nums = cctbx.array_family.flex.size_t(len(model_data), 0)
    bbox = dials_array_family_flex_ext.int6(len(model_data))
    for i in range(len(model_data)):
        panel_nums[i] = model_data[i][0]
        xyz[i] = model_data[i][1]
        bbox[i] = (
            int(model_data[i][1][0] - 2),
            int(model_data[i][1][0] + 2),
            int(model_data[i][1][1] - 2),
            int(model_data[i][1][1] + 2),
            int(model_data[i][1][2] - 2),
            int(model_data[i][1][2] + 2),
        )
    reflections = flex.reflection_table.empty_standard(len(model_data))
    reflections["xyz.px.value"] = xyz
    reflections["panel"] = panel_nums
    reflections["bbox"] = bbox
    reflections["flags"] = cctbx.array_family.flex.size_t(len(model_data), 32)
    return reflections


def run():

    """
    Input setup
    """

    phil = phil_scope.fetch()

    usage = "usage: dev.dials.simple_integrate.py models.expt reflections.expt"
    parser = ArgumentParser(
        usage=usage,
        phil=phil,
        epilog=__doc__,
        read_experiments=True,
        read_reflections=True,
    )

    params, options = parser.parse_args(args=None, show_diff_phil=False)

    dials.util.log.config(verbosity=options.verbose, logfile=params.output.log)
    logger.info(dials_version())

    """
    Load experiment and reflections
    """

    reflections, experiments = reflections_and_experiments_from_files(
        params.input.reflections, params.input.experiments
    )
    reflections = reflections[0]

    reflections["id"] = cctbx.array_family.flex.int(len(reflections), 0)
    reflections["imageset_id"] = cctbx.array_family.flex.int(len(reflections), 0)

    integrated_reflections = run_simple_integrate(params, experiments, reflections)
    integrated_reflections.as_msgpack_file(params.output.reflections)


def run_simple_integrate(params, experiments, reflections):

    experiment = experiments[0]

    # Remove bad reflections (e.g. those not indexed)
    reflections, _ = process_reference(reflections)
    # Mask neighbouring pixels to shoeboxes
    # reflections = filter_reference_pixels(reflections, experiments)

    """
    Predict reflections using experiment crystal
    """

    min_s0_idx = min(
        range(len(reflections["wavelength"])), key=reflections["wavelength"].__getitem__
    )
    min_s0 = reflections["s0"][min_s0_idx]
    dmin = experiment.detector.get_max_resolution(min_s0)
    predicted_reflections = flex.reflection_table.from_predictions(
        experiment, padding=1.0, dmin=dmin
    )
    predicted_reflections["id"] = cctbx.array_family.flex.int(
        len(predicted_reflections), 0
    )
    predicted_reflections["imageset_id"] = cctbx.array_family.flex.int(
        len(predicted_reflections), 0
    )
    # Updates flags to set which reflections to use in generating reference profiles
    matched, reflections, unmatched = predicted_reflections.match_with_reference(
        reflections
    )

    """
    Create profile model and add it to experiment.
    This is used to predict reflection properties.
    """

    # Filter reflections to use to create the model
    used_in_ref = reflections.get_flags(reflections.flags.used_in_refinement)
    model_reflections = reflections.select(used_in_ref)

    # sigma_D in 3.1 of Kabsch 2010
    # sigma_b = ComputeEsdBeamDivergence(
    #    experiment.detector, model_reflections, centroid_definition="s1"
    # ).sigma()

    # sigma_m in 3.1 of Kabsch 2010
    sigma_m = 0.1
    sigma_b = 0.01
    # The Gaussian model given in 2.3 of Kabsch 2010
    experiment.profile = GaussianRSProfileModel(
        params=params, n_sigma=3, sigma_b=sigma_b, sigma_m=sigma_m
    )

    """
    Compute properties for predicted reflections using profile model,
    accessed via experiment.profile_model. These reflection_table
    methods are largely just wrappers for profile_model.compute_bbox etc.
    Note: I do not think all these properties are needed for integration,
    but are all present in the current dials.integrate output.
    """

    predicted_reflections.compute_bbox(experiments)
    predicted_reflections.compute_d(experiments)
    predicted_reflections.compute_partiality(experiments)

    # Shoeboxes
    predicted_reflections["shoebox"] = flex.shoebox(
        predicted_reflections["panel"],
        predicted_reflections["bbox"],
        allocate=False,
        flatten=False,
    )

    # Get actual shoebox values and the reflections for each image
    shoebox_processor = ShoeboxProcessor(
        predicted_reflections,
        len(experiment.detector),
        0,
        len(experiment.imageset),
        False,
    )

    for i in range(len(experiment.imageset)):
        image = experiment.imageset.get_corrected_data(i)
        mask = experiment.imageset.get_mask(i)
        shoebox_processor.next_data_only(make_image(image, mask))

    predicted_reflections.is_overloaded(experiments)
    predicted_reflections.compute_mask(experiments)
    predicted_reflections.contains_invalid_pixels()

    # Background calculated explicitly to expose underlying algorithm
    background_algorithm = SimpleBackgroundExt(params=None, experiments=experiments)
    success = background_algorithm.compute_background(predicted_reflections)
    predicted_reflections.set_flags(
        ~success, predicted_reflections.flags.failed_during_background_modelling
    )

    # Centroids calculated explicitly to expose underlying algorithm
    centroid_algorithm = SimpleCentroidExt(params=None, experiments=experiments)
    centroid_algorithm.compute_centroid(predicted_reflections)

    predicted_reflections.compute_summed_intensity()

    # Filter reflections with a high fraction of masked foreground
    valid_foreground_threshold = 1.0  # DIALS default
    sboxs = predicted_reflections["shoebox"]
    nvalfg = sboxs.count_mask_values(MaskCode.Valid | MaskCode.Foreground)
    nforeg = sboxs.count_mask_values(MaskCode.Foreground)
    fraction_valid = nvalfg.as_double() / nforeg.as_double()
    selection = fraction_valid < valid_foreground_threshold
    predicted_reflections.set_flags(
        selection, predicted_reflections.flags.dont_integrate
    )

    predicted_reflections["num_pixels.valid"] = sboxs.count_mask_values(MaskCode.Valid)
    predicted_reflections["num_pixels.background"] = sboxs.count_mask_values(
        MaskCode.Valid | MaskCode.Background
    )
    predicted_reflections["num_pixels.background_used"] = sboxs.count_mask_values(
        MaskCode.Valid | MaskCode.Background | MaskCode.BackgroundUsed
    )
    predicted_reflections["num_pixels.foreground"] = nvalfg

    """
    Load modeller that will calculate reference profiles and
    do the actual profile fitting integration.
    """

    sel = predicted_reflections.get_flags(predicted_reflections.flags.reference_spot)
    reference_reflections = predicted_reflections.select(sel)
    reference_reflections.as_msgpack_file(
        "/home/davidmcdonagh/work/dials/modules/dials/src/dials/command_line/predicted_reference.refl"
    )
    px, py, pz = reference_reflections["xyzobs.px.value"].parts()
    experiment.sequence.set_image_range((int(min(pz) - 10), int(max(pz) + 10)))

    # Default params when running dials.integrate with C2sum_1_*.cbf.gz
    fit_method = 1  # reciprocal space fitter (called explicitly below)
    grid_method = 2  # regular grid
    grid_size = 5  # Downsampling grid size described in 3.3 of Kabsch 2010
    # Get the number of scan points
    num_scan_points = 72
    n_sigma = 4.5  # multiplier to expand bounding boxes
    fitting_threshold = 0.02
    reference_profile_modeller = GaussianRSProfileModeller(
        experiment.beam,
        experiment.detector,
        Goniometer(),
        experiment.sequence,
        sigma_b,
        sigma_m,
        n_sigma,
        grid_size,
        num_scan_points,
        fitting_threshold,
        grid_method,
        fit_method,
    )

    """
    Calculate grid of reference profiles from predicted reflections
    that matched observed.
    ("Learning phase" of 3.3 in Kabsch 2010)
    """

    sel = reference_reflections.get_flags(reference_reflections.flags.dont_integrate)
    sel = ~sel
    reference_reflections = reference_reflections.select(sel)
    num_reflections = {}
    for i in range(11):
        num_reflections[i] = (reference_reflections["panel"] == i).count(True)
    # sel = reference_reflections["panel"]==8
    # reference_reflections = reference_reflections.select(sel)
    # logger.info(f"Using {len(reference_reflections)} from panel 8")
    reference_profile_modeller.model(reference_reflections)
    reference_profile_modeller.normalize_profiles()

    profile_model_report = ProfileModelReport(
        experiments, [reference_profile_modeller], model_reflections
    )
    logger.info("")
    logger.info(profile_model_report.as_str(prefix=" "))
    logger.info(str(profile_model_report.num_profiles))
    logger.info(str(num_reflections))
    reference_profiles = get_reference_profiles_as_reflections(
        reference_profile_modeller
    )

    reference_profiles.as_msgpack_file(
        "/home/davidmcdonagh/work/dials/modules/dials/src/dials/command_line/reference_profiles.refl"
    )

    """
    Carry out the integration by fitting to reference profiles in 1D.
    (Calculates intensity using 3.4 of Kabsch 2010)
    """

    sel = predicted_reflections.get_flags(predicted_reflections.flags.dont_integrate)
    sel = ~sel
    predicted_reflections = predicted_reflections.select(sel)

    pred_px, pred_py, pred_pz = predicted_reflections["xyzcal.px"].parts()
    sel = pred_pz > min(pz) and pred_pz < max(pz)
    predicted_reflections = predicted_reflections.select(sel)
    reference_profile_modeller.fit_reciprocal_space_tof(predicted_reflections)
    # predicted_reflections.compute_corrections(experiments)

    integration_report = IntegrationReport(experiments, predicted_reflections)
    logger.info("")
    logger.info(integration_report.as_str(prefix=" "))

    """
    Filter for integrated reflections and remove shoeboxes
    """

    # del predicted_reflections["shoebox"]
    sel = predicted_reflections.get_flags(
        predicted_reflections.flags.integrated, all=False
    )
    predicted_reflections = predicted_reflections.select(sel)
    return predicted_reflections


if __name__ == "__main__":
    run()
