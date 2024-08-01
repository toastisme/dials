# LIBTBX_SET_DISPATCHER_NAME dials.tof_integrate
from __future__ import annotations

import logging
import multiprocessing

import numpy as np

import cctbx.array_family.flex
import libtbx

import dials.util.log
from dials.algorithms.integration.fit.tof_line_profile import (
    compute_line_profile_intensity,
)
from dials.algorithms.integration.report import IntegrationReport
from dials.algorithms.profile_model.gaussian_rs import Model as GaussianRSProfileModel
from dials.algorithms.profile_model.gaussian_rs.calculator import (
    ComputeEsdBeamDivergence,
)
from dials.algorithms.shoebox import MaskCode
from dials.array_family import flex
from dials.command_line.integrate import process_reference
from dials.extensions.simple_background_ext import SimpleBackgroundExt
from dials.util.options import ArgumentParser, reflections_and_experiments_from_files
from dials.util.phil import parse
from dials.util.version import dials_version
from dials_tof_scaling_ext import (
    TOFCorrectionsData,
    tof_calculate_shoebox_foreground,
    tof_extract_shoeboxes_to_reflection_table,
)

logger = logging.getLogger("dials.command_line.simple_integrate")

phil_scope = parse(
    """
input{
    incident_run = None
        .type = str
        .help = "Path to incident run to normalize intensities (e.g. Vanadium)."
    empty_run = None
        .type = str
        .help = "Path to empty run to correct empty counts."
}
output {
experiments = 'integrated.expt'
    .type = str
    .help = "The experiments output filename"
reflections = 'integrated.refl'
    .type = str
    .help = "The integrated output filename"
output_hkl = False
    .type = bool
    .help = "Output the integrated intensities as a SHELX hkl file"
hkl =  'integrated.hkl'
    .type = str
    .help = "The hkl output filename"
phil = 'dials.simple_integrate.phil'
    .type = str
    .help = "The output phil file"
log = 'tof_integrate.log'
    .type = str
    .help = "The log filename"
}
method{
line_profile_fitting = False
    .type = bool
    .help = "Use integration by profile fitting using a Gaussian"
    "convoluted with back-to-back exponential functions"
}
corrections{
    lorentz = False
        .type = bool
        .help = "Apply the Lorentz correction to target spectrum."
}
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
mp{
    nproc = Auto
        .type = int(value_min=1)
        .help = "Number of processors to use during parallelized steps."
        "If set to Auto DIALS will choose automatically."
}
sigma_b = 0.01
    .type = float
    .help = "Used to calculate xy bounding box of predicted reflections"
sigma_m = 10
    .type = float
    .help = "Used to calculate z bounding box of predicted reflections"
foreground_radius=0.5
    .type = float
    .help = "Foreground mask radius in inverse angtroms"
keep_shoeboxes = False
    .type = bool
    .help = "Retain shoeboxes in output reflection table"
"""
)

"""
Kabsch 2010 refers to
Kabsch W., Integration, scaling, space-group assignment and
post-refinment, Acta Crystallographica Section D, 2010, D66, 133-144
Usage:
$ dials.tof_integrate.py refined.expt refined.refl
"""


def output_reflections_as_hkl(reflections, filename):
    def get_corrected_intensity_and_variance(reflections, idx):
        intensity = reflections["intensity.sum.value"][idx]
        variance = reflections["intensity.sum.variance"][idx]
        return intensity, variance

    def get_line_profile_intensity_and_variance(reflections, idx):
        intensity = reflections["intensity.prf.value"][idx]
        variance = reflections["intensity.prf.variance"][idx]
        return intensity, variance

    def valid_intensity(intensity, variance):
        from math import isinf, isnan

        if isnan(intensity) or isinf(intensity):
            return False
        if isnan(variance) or isinf(variance):
            return False
        return intensity > 0 and variance > 0

    with open(filename, "w") as g:
        for i in range(len(reflections)):
            h, k, l = reflections["miller_index"][i]
            batch_number = 1
            intensity, variance = get_corrected_intensity_and_variance(reflections, i)
            if not valid_intensity(intensity, variance):
                continue
            intensity = round(intensity, 2)
            sigma = round(np.sqrt(variance), 2)
            wavelength = round(reflections["wavelength_cal"][i], 4)
            g.write(
                ""
                + "{:4d}{:4d}{:4d}{:8.1f}{:8.2f}{:4d}{:8.4f}\n".format(
                    int(h),
                    int(k),
                    int(l),
                    float(intensity),
                    float(sigma),
                    int(batch_number),
                    float(wavelength),
                )
            )
            # g.write(f"{int(h)} {int(k)} {int(l)} {float(intensity)} {float(sigma)} {int(batch_number)} {float(wavelength)}\n")
        g.write(
            ""
            + "{:4d}{:4d}{:4d}{:8.1f}{:9.2f}{:4d}{:8.4f}\n".format(
                int(0), int(0), int(0), float(0.00), float(0.00), int(0), float(0.0000)
            )
        )
        # g.write(f"{int(0)} {int(0)} {int(0)} {float(0.00)} {float(0.00)} {int(0)} {float(0.0000)}")

    if "intensity.prf.value" not in reflections:
        return
    with open("prf_" + filename, "w") as g:
        for i in range(len(reflections)):
            h, k, l = reflections["miller_index"][i]
            batch_number = 1
            intensity, variance = get_line_profile_intensity_and_variance(
                reflections, i
            )
            if not valid_intensity(intensity, variance):
                continue
            intensity = round(intensity, 2)
            sigma = round(np.sqrt(variance), 2)
            wavelength = round(reflections["wavelength_cal"][i], 4)
            g.write(
                ""
                + "{:4d}{:4d}{:4d}{:8.1f}{:8.2f}{:4d}{:8.4f}\n".format(
                    int(h),
                    int(k),
                    int(l),
                    float(intensity),
                    float(sigma),
                    int(batch_number),
                    float(wavelength),
                )
            )
        g.write(
            ""
            + "{:4d}{:4d}{:4d}{:8.1f}{:9.2f}{:4d}{:8.4f}\n".format(
                int(0), int(0), int(0), float(0.00), float(0.00), int(0), float(0.0000)
            )
        )


def split_reflections(reflections, n, by_panel=False):
    if by_panel:
        for i in range(max(reflections["panel"]) + 1):
            sel = reflections["panel"] == i
            yield reflections.select(sel)
    else:
        d, r = divmod(len(reflections), n)
        for i in range(n):
            si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
            yield reflections[si : si + (d + 1 if i < r else d)]


def join_reflections(list_of_reflections):
    reflections = list_of_reflections[0]
    for i in range(1, len(list_of_reflections)):
        reflections.extend(list_of_reflections[i])
    return reflections


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
    if params.output.output_hkl:
        output_reflections_as_hkl(integrated_reflections, params.output.hkl)


def applying_spherical_absorption_correction(params):
    all_params_present = True
    some_params_present = False
    for i in dir(params.target_spectrum):
        if i.startswith("__"):
            continue
        if getattr(params.target_spectrum, i) is not None:
            some_params_present = True
        else:
            all_params_present = False
    if some_params_present and not all_params_present:
        raise ValueError(
            "Trying to apply spherical absorption correction but some corrections are None."
        )
    return all_params_present


def applying_incident_and_empty_runs(params):
    if params.input.incident_run is not None:
        assert (
            params.input.empty_run is not None
        ), "Incident run given without empty run."
        return True
    elif params.input.empty_run is not None:
        assert (
            params.input.incident_run is not None
        ), "Empty run given without incident run."
        return True
    return False


def run_integrate(params, experiments, reflections):
    nproc = params.mp.nproc
    if nproc is libtbx.Auto:
        nproc = multiprocessing.cpu_count()

    reflections, _ = process_reference(reflections)

    """
    Predict reflections using experiment crystal
    """

    min_s0_idx = min(
        range(len(reflections["wavelength"])), key=reflections["wavelength"].__getitem__
    )
    min_s0 = reflections["s0"][min_s0_idx]
    dmin = None
    for experiment in experiments:
        expt_dmin = experiment.detector.get_max_resolution(min_s0)
        if dmin is None or expt_dmin < dmin:
            dmin = expt_dmin

    predicted_reflections = None
    for idx, experiment in enumerate(experiments):

        if predicted_reflections is None:
            predicted_reflections = flex.reflection_table.from_predictions(
                experiment, padding=1.0, dmin=dmin
            )
            predicted_reflections["id"] = cctbx.array_family.flex.int(
                len(predicted_reflections), idx
            )
            predicted_reflections["imageset_id"] = cctbx.array_family.flex.int(
                len(predicted_reflections), idx
            )
        else:
            r = flex.reflection_table.from_predictions(
                experiment, padding=1.0, dmin=dmin
            )
            r["id"] = cctbx.array_family.flex.int(len(r), idx)
            r["imageset_id"] = cctbx.array_family.flex.int(len(r), idx)
            predicted_reflections.extend(r)
    predicted_reflections["s0"] = predicted_reflections["s0_cal"]
    predicted_reflections.calculate_entering_flags(experiments)

    for i in range(len(experiments)):
        predicted_reflections.experiment_identifiers()[i] = experiments[i].identifier

    # Updates flags to set which reflections to use in generating reference profiles
    matched, reflections, unmatched = predicted_reflections.match_with_reference(
        reflections
    )
    # sel = predicted_reflections.get_flags(predicted_reflections.flags.reference_spot)
    predicted_reflections = predicted_reflections.select(matched)
    if "idx" in reflections:
        predicted_reflections["idx"] = reflections["idx"]

    """
    Create profile model and add it to experiment.
    This is used to predict reflection properties.
    """

    # Filter reflections to use to create the model
    used_in_ref = reflections.get_flags(reflections.flags.used_in_refinement)
    model_reflections = reflections.select(used_in_ref)

    sigma_b = ComputeEsdBeamDivergence(
        experiment.detector, model_reflections, centroid_definition="s1"
    ).sigma()

    # sigma_m in 3.1 of Kabsch 2010
    sigma_m = params.sigma_m
    # sigma_b = 0.001
    # The Gaussian model given in 2.3 of Kabsch 2010
    for idx, experiment in enumerate(experiments):
        experiments[idx].profile = GaussianRSProfileModel(
            params=params, n_sigma=2.5, sigma_b=sigma_b, sigma_m=sigma_m
        )

    """
    Compute properties for predicted reflections using profile model,
    accessed via experiment.profile_model. These reflection_table
    methods are largely just wrappers for profile_model.compute_bbox etc.
    Note: I do not think all these properties are needed for integration,
    but are all present in the current dials.integrate output.
    """

    predicted_reflections.compute_bbox(experiments)
    x1, x2, y1, y2, t1, t2 = predicted_reflections["bbox"].parts()
    predicted_reflections = predicted_reflections.select(
        (t1 > 0)
        & (t2 < max([e.scan.get_image_range()[1] for e in experiments]))
        & (x1 > 0)
        & (y1 > 0)
    )

    predicted_reflections.compute_d(experiments)
    # predicted_reflections.compute_partiality(experiments)

    # Shoeboxes
    print("Getting shoebox data")
    predicted_reflections["shoebox"] = flex.shoebox(
        predicted_reflections["panel"],
        predicted_reflections["bbox"],
        allocate=False,
        flatten=False,
    )

    experiment_cls = experiments[0].imageset.get_format_class()
    predicted_reflections.map_centroids_to_reciprocal_space(
        experiments, calculated=True
    )

    if applying_incident_and_empty_runs(params):
        logger.info("Subtracting empty run from target and Vanadium runs")
        incident_fmt_class = experiment_cls.get_instance(params.input.incident_run)
        empty_fmt_class = experiment_cls.get_instance(params.input.empty_run)

        incident_data = experiment_cls(params.input.incident_run).get_imageset(
            params.input.incident_run
        )
        empty_data = experiment_cls(params.input.empty_run).get_imageset(
            params.input.empty_run
        )
        incident_proton_charge = incident_fmt_class.get_proton_charge()
        empty_proton_charge = empty_fmt_class.get_proton_charge()

        for expt in experiments:
            expt_data = expt.imageset
            expt_proton_charge = experiment_cls.get_instance(
                expt.imageset.paths()[0], **expt.imageset.data().get_params()
            ).get_proton_charge()

            if applying_spherical_absorption_correction(params):
                logger.info(
                    "Applying spherical absorption correction to target and Vanadium runs"
                )
                logger.info("Normalising target run with Vanadium run")
                if params.corrections.lorentz:
                    logger.info("Applying Lorentz correction to target run")
                corrections_data = TOFCorrectionsData(
                    expt_proton_charge,
                    incident_proton_charge,
                    empty_proton_charge,
                    params.target_spectrum.sample_radius,
                    params.target_spectrum.scattering_x_section,
                    params.target_spectrum.absorption_x_section,
                    params.target_spectrum.sample_number_density,
                    params.incident_spectrum.sample_radius,
                    params.incident_spectrum.scattering_x_section,
                    params.incident_spectrum.absorption_x_section,
                    params.incident_spectrum.sample_number_density,
                )

                tof_extract_shoeboxes_to_reflection_table(
                    predicted_reflections,
                    expt,
                    expt_data,
                    incident_data,
                    empty_data,
                    corrections_data,
                    params.corrections.lorentz,
                )
            else:
                logger.info("Normalising target run with Vanadium run")
                if params.corrections.lorentz:
                    logger.info("Applying Lorentz correction to target run")
                tof_extract_shoeboxes_to_reflection_table(
                    predicted_reflections,
                    expt,
                    expt_data,
                    incident_data,
                    empty_data,
                    expt_proton_charge,
                    incident_proton_charge,
                    empty_proton_charge,
                    params.corrections.lorentz,
                )
    else:
        if params.corrections.lorentz:
            logger.info("Applying Lorentz correction to target run")
        for expt in experiments:
            expt_data = expt.imageset
            tof_extract_shoeboxes_to_reflection_table(
                predicted_reflections,
                expt,
                expt_data,
                params.corrections.lorentz,
            )

    # tof_calculate_shoebox_mask(predicted_reflections, expt)
    tof_calculate_shoebox_foreground(
        predicted_reflections, expt, params.foreground_radius
    )
    predicted_reflections.is_overloaded(experiments)
    predicted_reflections.contains_invalid_pixels()
    predicted_reflections["partiality"] = flex.double(len(predicted_reflections), 1.0)

    # Background calculated explicitly to expose underlying algorithm
    background_algorithm = SimpleBackgroundExt(params=None, experiments=experiments)
    success = background_algorithm.compute_background(predicted_reflections)
    predicted_reflections.set_flags(
        ~success, predicted_reflections.flags.failed_during_background_modelling
    )

    # Centroids calculated explicitly to expose underlying algorithm
    # centroid_algorithm = SimpleCentroidExt(params=None, experiments=experiments)
    # centroid_algorithm.compute_centroid(predicted_reflections)

    print("Calculating summed intensities")
    predicted_reflections.compute_summed_intensity()

    if params.method.line_profile_fitting:
        print("Calculating line profile fitted intensities")
        predicted_reflections = compute_line_profile_intensity(predicted_reflections)

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

    integration_report = IntegrationReport(experiments, predicted_reflections)
    logger.info("")
    logger.info(integration_report.as_str(prefix=" "))

    if not params.keep_shoeboxes:
        del predicted_reflections["shoebox"]
    return predicted_reflections


if __name__ == "__main__":
    run()
