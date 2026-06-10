# LIBTBX_SET_DISPATCHER_NAME dials.tof_integrate
from __future__ import annotations

import logging
import multiprocessing
from copy import deepcopy
from math import ceil, erf, floor, sqrt
from typing import Dict, Sequence, Tuple

import numpy as np

import cctbx.array_family.flex
import libtbx
from dxtbx import flumpy
from dxtbx.model import Experiment, ExperimentList, ImageSequence

import dials.util.log
from dials.algorithms.integration.integrator import (
    generate_phil_scope as integrator_phil_scope,
)
from dials.algorithms.integration.overlaps_filter import OverlapsFilter
from dials.algorithms.integration.report import IntegrationReport
from dials.algorithms.shoebox import MaskCode
from dials.algorithms.spot_prediction import TOFReflectionPredictor
from dials.array_family import flex
from dials.command_line.integrate import process_reference
from dials.extensions.simple_background_ext import SimpleBackgroundExt
from dials.util.options import ArgumentParser, reflections_and_experiments_from_files
from dials.util.phil import parse
from dials.util.version import dials_version
from dials_algorithms_tof_integration_ext import (
    TOFProfile1DParams,
    TOFProfile3DGutmannParams,
    TOFProfile3DICParams,
    integrate_reflection_table,
    tof_calculate_ellipse_shoebox_mask,
    tof_calculate_seed_skewness_shoebox_mask,
)
from dials_tof_scaling_ext import (
    TOFAbsorptionParams,
    TOFIncidentSpectrumParams,
    tof_extract_shoeboxes_to_reflection_table,
)

logger = logging.getLogger("dials.command_line.tof_integrate")

phil_scope = parse(
    """
output {
experiments = 'integrated.expt'
    .type = str
    .help = "The experiments output filename"
reflections = 'integrated.refl'
    .type = str
    .help = "The integrated output filename"
phil = 'dials.simple_integrate.phil'
    .type = str
    .help = "The output phil file"
log = 'tof_integrate.log'
    .type = str
    .help = "The log filename"
}
integration_type = *observed calculated
    .type = choice
    .help = "observed integrates only reflections observed during spot finding"
            "calculated integrates all reflections out to calculated.dmin"

background_model = constant2d constant3d linear2d *linear3d
    .type = choice
    .help = Model used to calculate the background of each reflection

calculated{
    dmin = none
    .type = float (value_min=0.5)
    .help = "The resolution spots are integrated to when using integration_type.calculated"

}
method = *summation profile_1d profile_3d_gutmann profile_3d_ic
    .type = choice
    .help = "Integration method: "
            "summation: shoebox summation"
            "profile_1d: https://doi.org/10.1038/srep36628 "
            "profile_3d_gutmann: https://doi.org/10.1016/j.nima.2016.12.026"
            "profile_3d_ic: separable Ikeda-Carpenter x Bivariate-Gaussian 3D profile"

mask = *ellipse seed_skewness
    .type = choice
    .help = "Foreground/background mask method: "
            "seed_skewness: https://doi.org/10.1107/S0021889803021939"
ellipse_mask{
    scale = 3.0
    .type = float (value_min=0.5)
    .help = "Number of standard deviations to use when generating the ellipse mask"
}


corrections{
    incident_run = None
        .type = str
        .help = "Path to incident run to normalize intensities (e.g. Vanadium)."
    empty_run = None
        .type = str
        .help = "Path to empty run to correct empty counts."
    lorentz = False
        .type = bool
        .help = "Apply the Lorentz correction to target spectrum."
    absorption{
        incident_spectrum{
            sample_number_density = 0.0722
                .type = float
                .help = "Sample number density for incident run."
                        "Default is Vanadium used at SXD"
            sample_radius = 0.03
                .type = float
                .help = "Sample radius incident run."
                        "Default is Vanadium used at SXD"
            scattering_x_section = 5.158
                .type = float
                .help = "Sample scattering cross section used for incident run."
                        "Default is Vanadium used at SXD"
            absorption_x_section = 4.4883
                .type = float
                .help = "Sample absorption cross section for incident run."
                        "Default is Vanadium used at SXD"
        }
        target_spectrum{
            sample_number_density = None
                .type = float
                .help = "Sample number density for target run."
            sample_radius = None
                .type = float
                .help = "Sample radius target run."
            scattering_x_section = None
                .type = float
                .help = "Sample scattering cross section used for target run."
            absorption_x_section = None
                .type = float
                .help = "Sample absorption cross section for target run."
        }
    }
}
profile_1d{
    init_alpha = 0.03
        .type = float
        .help = "Initial alpha value before optimization"
    init_beta = 0.03
        .type = float
        .help = "Initial beta value before optimization"
    min_alpha = 0.02
        .type = float
        .help = "Min alpha value for optimization"
    max_alpha = 1.0
        .type = float
        .help = "Max alpha value for optimization"
    min_beta = 0.0
        .type = float
        .help = "Min beta value for optimization"
    max_beta = 1.0
        .type = float
        .help = "Max beta value for optimization"
    n_restarts = 5000
        .type = int(value_min=0)
        .help = "If fit fails, number of additional attempts with perturbed params"

}
profile_3d_gutmann{
    init_alpha = 3.0
        .type = float
        .help = "Initial alpha value before optimization"
    init_beta = 0.5
        .type = float
        .help = "Initial beta value before optimization"
    min_alpha = 0.5
        .type = float
        .help = "Min alpha value for optimization"
    max_alpha = 20.
        .type = float
        .help = "Max alpha value for optimization"
    min_beta = 1e-2
        .type = float
        .help = "Min beta value for optimization"
    max_beta = 5.0
        .type = float
        .help = "Max beta value for optimization"
    n_restarts = 1000
        .type = int(value_min=0)
        .help = "If fit fails, number of additional attempts with perturbed params"
    gradient_method = *forward_difference central_difference
        .type = choice
        .help = "Method used to calculate gradients"
}
profile_3d_ic{
    init_A = 1.0
        .type = float
        .help = "Initial Ikeda-Carpenter A"
    min_A = 1e-1
        .type = float
        .help = "Min A for optimization"
    max_A = 20.0
        .type = float
        .help = "Max A for optimization"
    init_B = 0.1
        .type = float
        .help = "Initial Ikeda-Carpenter B"
    min_B = 1e-2
        .type = float
        .help = "Min B for optimization"
    max_B = 2.0
        .type = float
        .help = "Max B for optimization"
    init_R = 0.05
        .type = float
        .help = "Initial fast/slow neutron ratio (0 to 1)"
    min_R = 0.0
        .type = float
        .help = "Min R for optimization"
    max_R = 0.5
        .type = float
        .help = "Max R for optimization"
    min_SigX = 0.1
        .type = float
        .help = "Min SigX for optimization"
    max_SigX = 10.0
        .type = float
        .help = "Max SigX for optimization"
    min_SigY = 0.1
        .type = float
        .help = "Min SigY for optimization"
    max_SigY = 10.0
        .type = float
        .help = "Max SigY for optimization"
    init_SigP = 0.0
        .type = float
        .help = "Initial spatial correlation (-1 to 1)"
    min_SigP = -0.9
        .type = float
        .help = "Min SigP for optimization"
    max_SigP = 0.9
        .type = float
        .help = "Max SigP for optimization"
    hat_width = 5.0
        .type = float
        .help = "Half-width of rectangular convolution kernel in ToF bins"
    kconv = 120.0
        .type = float
        .help = "Gaussian convolution kernel decay constant"
    n_restarts = 1000
        .type = int(value_min=0)
        .help = "If fit fails, number of additional attempts with perturbed params"
    optimize_convolution_params = True
        .type = bool
        .help = "If True, hat_width and kconv are included in the optimization"
}

mp{
    nproc = auto
        .type = int(value_min=1)
        .help = "Number of processors to use during parallelized steps."
        "If set to auto DIALS will choose automatically."
}
bbox_tof_padding = 2
    .type = int
    .help = "Additional ToF frames added to calculated bounding boxes"
bbox_xy_padding = 1
    .type = int
    .help = "Additional pixels added to calculated bounding boxes"
partiality_n_sigma_xy = 2.5
    .type = float (value_min=0.1)
    .help = "Assumed number of sigma the bounding box half-width spans in x/y."
            "Used to estimate the Gaussian sigma for partiality calculation."
partiality_n_sigma_z = 2.5
    .type = float (value_min=0.1)
    .help = "Assumed number of sigma the bounding box half-width spans in z (ToF)."
            "Used to estimate the Gaussian sigma for partiality calculation."
keep_shoeboxes = False
    .type = bool
    .help = "Retain shoeboxes in output reflection table"
"""
)

"""
Usage:
$ dials.tof_integrate.py refined.expt refined.refl
$ dials.tof_integrate refined.expt refined.refl corrections.incident_run=vanadium_run.nxs corrections.empty_run=empty_run.nxs corrections.lorentz=True bbox_tof_padding=15 bbox_xy_padding=5 mask=ellipse method=profile_3d_gutmann mp.nproc=16 background_model=linear3d
"""

phil_scope.adopt_scope(integrator_phil_scope())


def get_corrections_data(
    experiments: ExperimentList, params: libtbx.phil.scope_extract
) -> Dict:
    """
    Extracts all the corrections in params related to incident
    and absorption corrections
    """

    corrections = {}

    experiment_cls = experiments[0].imageset.get_format_class()
    incident_proton_charge = None
    empty_proton_charge = None

    if applying_incident_and_empty_runs(params):
        incident_fmt_class = experiment_cls.get_instance(
            params.corrections.incident_run
        )
        incident_data = experiment_cls(params.corrections.incident_run).get_imageset(
            params.corrections.incident_run
        )
        corrections["incident_data"] = incident_data

        empty_data = experiment_cls(params.corrections.empty_run).get_imageset(
            params.corrections.empty_run
        )
        corrections["empty_data"] = empty_data
        empty_fmt_class = experiment_cls.get_instance(params.corrections.empty_run)
        incident_proton_charge = incident_fmt_class.get_proton_charge()
        empty_proton_charge = empty_fmt_class.get_proton_charge()

        corrections["incident_proton_charge"] = incident_proton_charge
        corrections["empty_proton_charge"] = empty_proton_charge

    if applying_spherical_absorption_correction(params):
        corrections["absorption_params"] = TOFAbsorptionParams(
            params.corrections.absorption.target_spectrum.sample_radius,
            params.corrections.absorption.target_spectrum.scattering_x_section,
            params.corrections.absorption.target_spectrum.absorption_x_section,
            params.corrections.absorption.target_spectrum.sample_number_density,
            params.corrections.absorption.incident_spectrum.sample_radius,
            params.corrections.absorption.incident_spectrum.scattering_x_section,
            params.corrections.absorption.incident_spectrum.absorption_x_section,
            params.corrections.absorption.incident_spectrum.sample_number_density,
        )

    return corrections


def calculate_shoebox_masks(
    experiment: Experiment,
    reflections: flex.reflection_table,
    params: libtbx.phil.scope_extract,
) -> flex.reflection_table:
    nproc = params.mp.nproc
    method = params.mask
    if method == "seed_skewness":
        logger.info("    Calculating seed skewness foreground/background mask")
        tof_calculate_seed_skewness_shoebox_mask(
            reflections, experiment, 1e-7, 10, nproc
        )
    else:
        logger.info("    Calculating ellipse foreground/background mask")
        scale = params.ellipse_mask.scale
        tof_calculate_ellipse_shoebox_mask(reflections, experiment, nproc, scale)

    return reflections


def integrate_reflection_table_for_experiment(
    expt: Experiment,
    expt_reflections: flex.reflection_table,
    expt_data: ImageSequence,
    params: libtbx.phil.scope_extract,
    **kwargs: Dict,
) -> flex.reflection_table:
    apply_lorentz = params.corrections.lorentz
    profile_1d_params = None
    profile_3d_gutmann_params = None
    profile_3d_ic_params = None
    incident_params = None
    absorption_params = None

    logger.info(f"    Integrating using {params.method}")

    show_profile_failures = logger.getEffectiveLevel() == logging.DEBUG
    if params.method == "profile_1d":
        alpha = params.profile_1d.init_alpha
        beta = params.profile_1d.init_beta
        A = 1.0
        min_alpha = params.profile_1d.min_alpha
        max_alpha = params.profile_1d.max_alpha
        min_beta = params.profile_1d.min_beta
        max_beta = params.profile_1d.max_beta
        n_restarts = params.profile_1d.n_restarts
        profile_1d_params = TOFProfile1DParams(
            A,
            alpha,
            min_alpha,
            max_alpha,
            beta,
            min_beta,
            max_beta,
            n_restarts,
            True,
            show_profile_failures,
        )
    elif params.method == "profile_3d_gutmann":
        alpha = params.profile_3d_gutmann.init_alpha
        beta = params.profile_3d_gutmann.init_beta
        min_alpha = params.profile_3d_gutmann.min_alpha
        max_alpha = params.profile_3d_gutmann.max_alpha
        min_beta = params.profile_3d_gutmann.min_beta
        max_beta = params.profile_3d_gutmann.max_beta
        n_restarts = params.profile_3d_gutmann.n_restarts
        use_central_diff = (
            params.profile_3d_gutmann.gradient_method == "central_difference"
        )
        profile_3d_gutmann_params = TOFProfile3DGutmannParams(
            alpha,
            min_alpha,
            max_alpha,
            beta,
            min_beta,
            max_beta,
            n_restarts,
            True,
            use_central_diff,
            show_profile_failures,
        )
    elif params.method == "profile_3d_ic":
        p = params.profile_3d_ic
        profile_3d_ic_params = TOFProfile3DICParams(
            p.init_A,
            p.min_A,
            p.max_A,
            p.init_B,
            p.min_B,
            p.max_B,
            p.init_R,
            p.min_R,
            p.max_R,
            p.min_SigX,
            p.max_SigX,
            p.min_SigY,
            p.max_SigY,
            p.init_SigP,
            p.min_SigP,
            p.max_SigP,
            p.hat_width,
            p.kconv,
            p.n_restarts,
            True,
            p.optimize_convolution_params,
            show_profile_failures,
        )

    if apply_lorentz:
        logger.info("    Adding Lorentz correction")

    if "incident_data" in kwargs:
        incident_params = TOFIncidentSpectrumParams(
            kwargs["incident_data"],
            kwargs["empty_data"],
            kwargs["expt_proton_charge"],
            kwargs["incident_proton_charge"],
            kwargs["empty_proton_charge"],
        )
        logger.info("    Adding incident spectrum correction")

    if "absorption_params" in kwargs:
        logger.info("    Adding absorption correction")
        absorption_params = kwargs["absorption_params"]

    integrate_reflection_table(
        expt_reflections,
        expt,
        expt_data,
        incident_params,
        absorption_params,
        apply_lorentz,
        params.mp.nproc,
        profile_1d_params,
        profile_3d_gutmann_params,
        profile_3d_ic_params,
    )

    return expt_reflections


def remove_overlapping_reflections(
    reflections: flex.reflection_table,
) -> flex.reflection_table:
    overlaps = reflections.find_overlaps()
    overlap_sel = flex.bool(len(reflections), False)
    for item in overlaps.edges():
        i0 = overlaps.source(item)
        i1 = overlaps.target(item)
        overlap_sel[i0] = True
        overlap_sel[i1] = True
    logger.info("Rejecting %i overlapping bounding boxes", overlap_sel.count(True))
    reflections = reflections.select(~overlap_sel)
    return reflections


def overlapping_foreground_reflections(
    reflections: flex.reflection_table, experiment: Experiment
) -> flex.bool:

    overlaps_filter = OverlapsFilter(reflections, experiment)
    overlaps_filter.create_referenced_mask(overlaps_filter.code_fgd, "foreground")
    """
    for j in range(10):
        for i in overlaps_filter.masks["foreground"][j]:
            val = overlaps_filter.masks["foreground"][j][i]
            if len(val) > 1:
                print(
                    f"TEST overlapping val({i}:{val}) {[(reflections['bbox'][k], reflections['panel'][k]) for k in val]}"
                )
    """

    return overlaps_filter.filter_overlaps_using_referenced_mask("foreground")


def _gaussian_fraction(center: float, lo: float, hi: float, sigma: float) -> float:
    """Fraction of N(center, sigma^2) integral lying in [lo, hi]."""
    s2 = sigma * sqrt(2)
    return 0.5 * (erf((hi - center) / s2) - erf((lo - center) / s2))


def compute_partiality(
    bbox: Sequence,
    image_size: Sequence,
    centroid: Sequence,
    n_sigma_xy: float = 2.5,
    n_sigma_z: float = 2.5,
) -> float:
    """
    Estimate partiality by modelling intensity as a 3D separable Gaussian and
    computing the fraction of the total integral that lies within the observed
    (panel/frame-clipped) region.

    sigma in each dimension is estimated from the bounding box half-width:
      sigma = half_width / n_sigma
    so n_sigma controls how many standard deviations the bbox edge is from the
    centroid when the reflection is fully visible.
    """
    cx0 = max(image_size[0], bbox[0])
    cx1 = min(image_size[1], bbox[1])
    cy0 = max(image_size[2], bbox[2])
    cy1 = min(image_size[3], bbox[3])
    cz0 = max(image_size[4], bbox[4])
    cz1 = min(image_size[5], bbox[5])

    if cx0 >= cx1 or cy0 >= cy1 or cz0 >= cz1:
        return 0.0

    sigma_x = (bbox[1] - bbox[0]) / (2.0 * n_sigma_xy)
    sigma_y = (bbox[3] - bbox[2]) / (2.0 * n_sigma_xy)
    sigma_z = (bbox[5] - bbox[4]) / (2.0 * n_sigma_z)

    if sigma_x <= 0.0 or sigma_y <= 0.0 or sigma_z <= 0.0:
        return 0.0

    cx, cy, cz = centroid

    # Integral over the clipped (observed) region
    p_clip = (
        _gaussian_fraction(cx, cx0, cx1, sigma_x)
        * _gaussian_fraction(cy, cy0, cy1, sigma_y)
        * _gaussian_fraction(cz, cz0, cz1, sigma_z)
    )

    # Integral over the full bbox (normaliser — accounts for Gaussian tails
    # beyond the bbox so a fully-visible reflection returns partiality 1.0)
    p_full = (
        _gaussian_fraction(cx, bbox[0], bbox[1], sigma_x)
        * _gaussian_fraction(cy, bbox[2], bbox[3], sigma_y)
        * _gaussian_fraction(cz, bbox[4], bbox[5], sigma_z)
    )

    if p_full < 1e-12:
        return 0.0

    return p_clip / p_full


def update_bounding_box(
    bbox: Tuple,
    centroid: Tuple,
    new_centroid: Tuple,
    padding: Tuple,  # x,y,z
    image_size: Tuple,
    n_sigma_xy: float = 2.5,
    n_sigma_z: float = 2.5,
) -> Tuple[Tuple, float]:
    """
    Calculates a new bounding box originally at centroid at position
    new_centroid with padding.
    Returns the bounding box and its partiality.
    """

    diff_centroid = (
        new_centroid[0] - centroid[0],
        new_centroid[1] - centroid[1],
        new_centroid[2] - centroid[2],
    )

    updated_bbox = list(deepcopy(bbox))

    updated_bbox[0] += diff_centroid[0] - padding[0]
    updated_bbox[1] += diff_centroid[0] + padding[0]
    updated_bbox[2] += diff_centroid[1] - padding[1]
    updated_bbox[3] += diff_centroid[1] + padding[1]
    updated_bbox[4] += diff_centroid[2] - padding[2]
    updated_bbox[5] += diff_centroid[2] + padding[2]

    partiality = compute_partiality(
        updated_bbox, image_size, new_centroid, n_sigma_xy, n_sigma_z
    )

    updated_bbox[0] = max(floor(updated_bbox[0]), image_size[0])
    updated_bbox[1] = min(ceil(updated_bbox[1]), image_size[1])
    updated_bbox[2] = max(floor(updated_bbox[2]), image_size[2])
    updated_bbox[3] = min(ceil(updated_bbox[3]), image_size[3])
    updated_bbox[4] = max(floor(updated_bbox[4]), image_size[4])
    updated_bbox[5] = min(ceil(updated_bbox[5]), image_size[5])

    return tuple(updated_bbox), partiality


def get_predicted_observed_reflections(
    params: libtbx.phil.scope_extract,
    experiments: ExperimentList,
    reflections: flex.reflection_table,
) -> flex.reflection_table:
    """
    Returns predicted reflections matched to reflections
    """

    # Calculate dmin
    min_s0_idx = min(
        range(len(reflections["wavelength"])), key=reflections["wavelength"].__getitem__
    )
    min_s0 = reflections["s0"][min_s0_idx]
    dmin = None
    for experiment in experiments:
        expt_dmin = experiment.detector.get_max_resolution(min_s0)
        if dmin is None or expt_dmin < dmin:
            dmin = expt_dmin

    logger.info(f"Calculated dmin from observed reflections: {round(dmin, 3)}")

    predicted_reflections = None
    miller_indices = reflections["miller_index"]

    # Get predicted reflections for observed miller indices
    for idx, experiment in enumerate(experiments):
        predictor = TOFReflectionPredictor(experiment, dmin)
        if predicted_reflections is None:
            predicted_reflections = predictor.indices_for_ub(miller_indices)
            predicted_reflections["id"] = cctbx.array_family.flex.int(
                len(predicted_reflections), idx
            )
            predicted_reflections["imageset_id"] = cctbx.array_family.flex.int(
                len(predicted_reflections), idx
            )
        else:
            r = predictor.indices_for_ub(miller_indices)
            r["id"] = cctbx.array_family.flex.int(len(r), idx)
            r["imageset_id"] = cctbx.array_family.flex.int(len(r), idx)
            predicted_reflections.extend(r)

    # Update params
    predicted_reflections["s0"] = predicted_reflections["s0_cal"]
    predicted_reflections.calculate_entering_flags(experiments)

    for i in range(len(experiments)):
        predicted_reflections.experiment_identifiers()[i] = experiments[i].identifier

    # Updates flags to set which reflections to use in generating reference profiles
    matched, reflections, _ = predicted_reflections.match_with_reference(reflections)

    predicted_reflections = predicted_reflections.select(matched)
    if "idx" in reflections:
        predicted_reflections["idx"] = reflections["idx"]

    predicted_reflections["xyzobs.px.value"] = reflections["xyzobs.px.value"]

    # Calculate predicted bounding boxes and partiality
    tof_padding = params.bbox_tof_padding
    xy_padding = params.bbox_xy_padding
    n_sigma_xy = params.partiality_n_sigma_xy
    n_sigma_z = params.partiality_n_sigma_z
    image_size = experiments[0].detector[0].get_image_size()
    tof_size = len(experiments[0].scan.get_property("time_of_flight"))
    bboxes = flex.int6(len(predicted_reflections))
    partiality = flex.double(len(predicted_reflections))
    for i in range(len(predicted_reflections)):
        bboxes[i], partiality[i] = update_bounding_box(
            reflections["bbox"][i],
            reflections["xyzobs.px.value"][i],
            predicted_reflections["xyzcal.px"][i],
            (int(xy_padding), int(xy_padding), int(tof_padding)),
            (0, image_size[0], 0, image_size[1], 0, tof_size),
            n_sigma_xy=n_sigma_xy,
            n_sigma_z=n_sigma_z,
        )
    predicted_reflections["bbox"] = bboxes
    predicted_reflections["partiality"] = partiality

    predicted_reflections.compute_d(experiments)
    predicted_reflections["shoebox"] = flex.shoebox(
        predicted_reflections["panel"],
        predicted_reflections["bbox"],
        allocate=False,
        flatten=False,
    )

    predicted_reflections.map_centroids_to_reciprocal_space(
        experiments, calculated=True
    )
    return predicted_reflections


def get_predicted_calculated_reflections(
    params: libtbx.phil.scope_extract,
    experiments: ExperimentList,
    reflections: flex.reflection_table,
) -> flex.reflection_table:
    """
    Returns predicted reflections out to params.calculated.dmin
    """

    dmin = params.calculated.dmin
    assert dmin is not None, (
        "Integrating calculated reflections but calculated.dmin has not been set"
    )

    # Get predicted reflections
    predicted_reflections = None
    for idx, experiment in enumerate(experiments):
        predictor = TOFReflectionPredictor(experiment, dmin)
        if predicted_reflections is None:
            predicted_reflections = predictor.for_ub(experiment.crystal.get_A())
            predicted_reflections["id"] = cctbx.array_family.flex.int(
                len(predicted_reflections), idx
            )
            predicted_reflections["imageset_id"] = cctbx.array_family.flex.int(
                len(predicted_reflections), idx
            )
        else:
            r = predictor.for_ub(experiment.crystal.get_A())
            r["id"] = cctbx.array_family.flex.int(len(r), idx)
            r["imageset_id"] = cctbx.array_family.flex.int(len(r), idx)
            predicted_reflections.extend(r)
    predicted_reflections["s0"] = predicted_reflections["s0_cal"]
    predicted_reflections.calculate_entering_flags(experiments)

    for i in range(len(experiments)):
        predicted_reflections.experiment_identifiers()[i] = experiments[i].identifier

    logger.info(f"Predicted {len(predicted_reflections)} using dmin {dmin}")

    used_in_ref = reflections.get_flags(reflections.flags.used_in_refinement)
    model_reflections = reflections.select(used_in_ref)
    logger.info(
        f"Using {len(model_reflections)} observed reflections to calculate bounding boxes"
    )

    # Get the closest observed reflection for each calculated reflection
    # Use the observed bounding box as a basis for the calculated reflection

    tof_padding = params.bbox_tof_padding
    xy_padding = params.bbox_xy_padding
    n_sigma_xy = params.partiality_n_sigma_xy
    n_sigma_z = params.partiality_n_sigma_z
    image_size = experiments[0].detector[0].get_image_size()
    tof_size = len(experiments[0].scan.get_property("time_of_flight"))
    predicted_reflections["bbox"] = flex.int6(len(predicted_reflections))
    predicted_reflections["partiality"] = flex.double(len(predicted_reflections))

    # If no observed reflections on a given panel, use the average bbox size
    x0, x1, y0, y1, z0, z1 = reflections["bbox"].parts()
    num_reflections = len(reflections)
    avg_bbox_size = (
        (sum(x1 - x0) / num_reflections) * 0.5,
        (sum(y1 - y0) / num_reflections) * 0.5,
        (sum(z1 - z0) / num_reflections) * 0.5,
    )

    for panel_idx in range(len(experiments[0].detector)):
        for expt_idx in range(len(experiments)):
            p_sel = (predicted_reflections["id"] == expt_idx) & (
                predicted_reflections["panel"] == panel_idx
            )
            expt_p_reflections = predicted_reflections.select(p_sel)
            bboxes = flex.int6(len(expt_p_reflections))
            partiality = flex.double(len(expt_p_reflections))

            m_sel = (model_reflections["id"] == expt_idx) & (
                model_reflections["panel"] == panel_idx
            )
            expt_m_reflections = model_reflections.select(m_sel)

            for i in range(len(expt_p_reflections)):
                if len(expt_m_reflections) == 0:
                    c = expt_p_reflections["xyzcal.px"][i]
                    bbox = (
                        c[0] - avg_bbox_size[0],
                        c[0] + avg_bbox_size[0],
                        c[1] - avg_bbox_size[1],
                        c[1] + avg_bbox_size[1],
                        c[2] - avg_bbox_size[2],
                        c[2] + avg_bbox_size[2],
                    )

                    bboxes[i], partiality[i] = update_bounding_box(
                        bbox,
                        expt_p_reflections["xyzcal.px"][i],
                        expt_p_reflections["xyzcal.px"][i],
                        (int(xy_padding), int(xy_padding), int(tof_padding)),
                        (0, image_size[0], 0, image_size[1], 0, tof_size),
                        n_sigma_xy=n_sigma_xy,
                        n_sigma_z=n_sigma_z,
                    )

                else:
                    distances = (
                        expt_m_reflections["xyzobs.px.value"]
                        - expt_p_reflections["xyzcal.px"][i]
                    ).norms()
                    min_distance_idx = np.argmin(flumpy.to_numpy(distances))

                    bboxes[i], partiality[i] = update_bounding_box(
                        expt_m_reflections["bbox"][min_distance_idx],
                        expt_m_reflections["xyzobs.px.value"][min_distance_idx],
                        expt_p_reflections["xyzcal.px"][i],
                        (int(xy_padding), int(xy_padding), int(tof_padding)),
                        (0, image_size[0], 0, image_size[1], 0, tof_size),
                        n_sigma_xy=n_sigma_xy,
                        n_sigma_z=n_sigma_z,
                    )
            predicted_reflections["bbox"].set_selected(p_sel, bboxes)
            predicted_reflections["partiality"].set_selected(p_sel, partiality)

    _, _, _, _, z1, z2 = predicted_reflections["bbox"].parts()
    sel = z2 > z1
    predicted_reflections = predicted_reflections.select(sel)
    _, _, y1, y2, _, _ = predicted_reflections["bbox"].parts()
    sel = y2 > y1
    predicted_reflections = predicted_reflections.select(sel)
    x1, x2, _, _, _, _ = predicted_reflections["bbox"].parts()
    sel = x2 > x1
    predicted_reflections = predicted_reflections.select(sel)

    predicted_reflections.compute_d(experiments)
    predicted_reflections["shoebox"] = flex.shoebox(
        predicted_reflections["panel"],
        predicted_reflections["bbox"],
        allocate=False,
        flatten=False,
    )

    predicted_reflections.map_centroids_to_reciprocal_space(
        experiments, calculated=True
    )
    return predicted_reflections


def run():
    """
    Input setup
    """

    phil = phil_scope.fetch()

    usage = "usage: dials.tof_integrate.py refined.expt refined.refl"
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

    integrated_reflections = run_integrate(params, experiments, reflections)
    integrated_reflections.as_msgpack_file(params.output.reflections)
    experiments.as_file(params.output.experiments)


def applying_spherical_absorption_correction(params: libtbx.phil.scope_extract) -> bool:
    """
    Returns True if all params are present for absorption correction.
    Raises error if only some are present
    """

    all_params_present = True
    some_params_present = False
    for i in dir(params.corrections.absorption.target_spectrum):
        if i.startswith("__"):
            continue
        if getattr(params.corrections.absorption.target_spectrum, i) is not None:
            some_params_present = True
        else:
            all_params_present = False
    if some_params_present and not all_params_present:
        raise ValueError(
            "Trying to apply spherical absorption correction but some corrections are None."
        )
    return all_params_present


def applying_incident_and_empty_runs(params: libtbx.phil.scope_extract) -> bool:
    """
    Returns True if all params are present for incident correction.
    Raises error if only some are present
    """

    if params.corrections.incident_run is not None:
        assert params.corrections.empty_run is not None, (
            "Incident run given without empty run."
        )
        return True
    elif params.corrections.empty_run is not None:
        assert params.corrections.incident_run is not None, (
            "Empty run given without incident run."
        )
        return True
    return False


def run_integrate(
    params: libtbx.phil.scope_extract,
    experiments: ExperimentList,
    reflections: flex.reflection_table,
) -> flex.reflection_table:
    if params.mp.nproc is libtbx.Auto:
        params.mp.nproc = multiprocessing.cpu_count()

    reflections, _ = process_reference(reflections)

    """
    Predict reflections using experiment crystal
    """

    if params.integration_type == "observed":
        predicted_reflections = get_predicted_observed_reflections(
            params=params, experiments=experiments, reflections=reflections
        )
    elif params.integration_type == "calculated":
        predicted_reflections = get_predicted_calculated_reflections(
            params=params, experiments=experiments, reflections=reflections
        )

    corrections_data = get_corrections_data(experiments=experiments, params=params)

    experiment_cls = experiments[0].imageset.get_format_class()
    for idx, expt in enumerate(experiments):
        logger.info(f"Integrating experiment {idx}")

        expt_data = expt.imageset

        if "incident_proton_charge" in corrections_data:
            expt_proton_charge = experiment_cls.get_instance(
                expt.imageset.paths()[0], **expt.imageset.data().get_params()
            ).get_proton_charge()
            corrections_data["expt_proton_charge"] = expt_proton_charge

        sel_expt = predicted_reflections["id"] == idx
        expt_reflections = predicted_reflections.select(sel_expt)

        logger.info("    Extracting shoeboxes")
        tof_extract_shoeboxes_to_reflection_table(
            expt_reflections, expt, expt_data, False
        )

        expt_reflections = calculate_shoebox_masks(expt, expt_reflections, params)
        expt_reflections.is_overloaded(experiments)
        expt_reflections.contains_invalid_pixels()

        # Filter reflections with a high fraction of masked foreground
        valid_foreground_threshold = 1.0  # DIALS default
        sboxs = expt_reflections["shoebox"]
        nvalfg = sboxs.count_mask_values(MaskCode.Valid | MaskCode.Foreground)
        nforeg = sboxs.count_mask_values(MaskCode.Foreground)
        fraction_valid = nvalfg.as_double() / nforeg.as_double()
        selection = fraction_valid < valid_foreground_threshold
        expt_reflections.set_flags(selection, expt_reflections.flags.dont_integrate)

        expt_reflections["num_pixels.valid"] = sboxs.count_mask_values(MaskCode.Valid)
        expt_reflections["num_pixels.background"] = sboxs.count_mask_values(
            MaskCode.Valid | MaskCode.Background
        )
        expt_reflections["num_pixels.background_used"] = sboxs.count_mask_values(
            MaskCode.Valid | MaskCode.Background | MaskCode.BackgroundUsed
        )
        expt_reflections["num_pixels.foreground"] = nvalfg

        logger.info(f"    Calculating background using {params.background_model}")
        params.integration.background.simple.model.algorithm = params.background_model
        background_algorithm = SimpleBackgroundExt(
            params=params, experiments=experiments
        )
        success = background_algorithm.compute_background(expt_reflections)
        expt_reflections.set_flags(
            ~success, expt_reflections.flags.failed_during_background_modelling
        )

        overlap_sel = overlapping_foreground_reflections(
            reflections=expt_reflections, experiment=expt
        )
        expt_reflections.set_flags(~overlap_sel, expt_reflections.flags.dont_integrate)

        logger.info(f"    Removed {overlap_sel.count(False)} due to foreground overlap")

        expt_reflections = integrate_reflection_table_for_experiment(
            expt,
            expt_reflections,
            expt_data,
            params,
            **corrections_data,
        )

        predicted_reflections.set_selected(sel_expt, expt_reflections)

    integration_report = IntegrationReport(experiments, predicted_reflections)
    logger.info("")
    logger.info(integration_report.as_str(prefix=" "))

    if not params.keep_shoeboxes:
        del predicted_reflections["shoebox"]
    return predicted_reflections


if __name__ == "__main__":
    run()
