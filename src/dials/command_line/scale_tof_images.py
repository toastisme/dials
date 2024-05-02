# LIBTBX_SET_DISPATCHER_NAME dev.dials.normalize_tof_images
from __future__ import annotations

import logging
import multiprocessing
from os.path import splitext
from typing import Tuple

import numpy as np
from scipy.signal import savgol_filter

from dxtbx import flumpy
from dxtbx.format import Format
from dxtbx.model import Beam, Detector, Experiment
from libtbx import Auto, phil

import dials.util.log
from dials.util.options import ArgumentParser
from dials.util.phil import parse
from dials.util.version import dials_version
from dials_scaling_ext import (
    tof_lorentz_correction,
    tof_spherical_absorption_correction,
)

logger = logging.getLogger("dials.command_line.normalize_tof_images")

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
corrections{
    lorentz = True
        .type = bool
        .help = "Apply the Lorentz correction to target spectrum."
    spherical_absorption = True
        .type = bool
        .help = "Apply a spherical absorption correction."
    incident_and_empty = True
        .type = bool
        .help = "Divide the target by the incident (e.g. Vanadium) run"q
                "and subtract the empty run."
    normalize_by_bin_width = False
        .type = bool
        .help = "Multiply ToF bin widths by their ToF width."
    smoothing_window_length = 51
        .type = int
        .help = "The length of the filter window used in the savgol_filter"
                "for smoothing incident and empty runs."
    smoothing_polyorder = 3
        .type = int
        .help = "The order of polynormial used in the savgol_filter for "
                "smoothing incident and empty run."
    shrink_factor = 7
        .type = int
        .help = "Factor used to shrink incident and empty spectra to speed up"
                "computation time."
}
incident_spectrum{
    sample_number_density = 0.0722
        .type = float
        .help = "Sample number density for incident run."
                "Default is Vanadium used at SXD"
    sample_radius = 0.3
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
output {
    image_file_suffix = '_scaled'
        .type = str
        .help = "Suffix of the corrected image file."
    phil = 'dials.normalize_tof_images.phil'
        .type = str
        .help = "The output phil file."
    log = 'dials.normalize_tof_images.log'
        .type = str
        .help = "The log filename."
}
"""
)


def run_interpolate(y: np.array, window_length: int, polyorder: int) -> np.array:
    return savgol_filter(y, window_length, polyorder)


def smooth_spectra(
    spectra_arr: np.array, window_length: int, polyorder: int, nproc: int
) -> np.array:

    """
    Smooth spectra array along ToF to reduce noise.
    Assumes spectra_arr has shape (1, spectra_num, ToF).
    window_length and polyorder are params for savgol_filter.
    See  https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
    """

    assert (
        spectra_arr.ndim == 3
    ), "spectra_arr assumed to have shape (1, spectra_num, ToF)"

    pool = multiprocessing.Pool(nproc)

    processes = [
        pool.apply_async(run_interpolate, args=(y, window_length, polyorder))
        for y in spectra_arr[0]
    ]
    result = np.array([p.get() for p in processes])

    return result.reshape(1, result.shape[0], result.shape[1])


def shrink_spectra(spectra_arr: np.array, shrink_factor: int) -> np.array:

    """
    Reduce spectra size by shrink_factor.
    Assumes spectra_arr has shape (1, spectra_num, ToF).
    Main purpose of this is to speed up computation.
    """

    assert (
        spectra_arr.ndim == 3
    ), "spectra_arr assumed to have shape (1, spectra_num, ToF)"

    # Slice off end if it does not neatly divide
    remainder = spectra_arr.shape[-1] % shrink_factor
    if remainder > 0:
        spectra_arr = spectra_arr[:, :, :-remainder]

    return (
        spectra_arr.reshape(
            spectra_arr.shape[0], spectra_arr.shape[1], -1, shrink_factor
        ).sum(axis=3),
        remainder,
    )


def expand_spectra(
    spectra_arr: np.array, expand_factor: int, remainder: int, divide: bool = False
) -> np.array:

    """
    Expand spectra by expand_factor, then pad by remainder.
    Remainder is padded using the final value for each spectra.
    """

    assert (
        spectra_arr.ndim == 3
    ), "spectra_arr assumed to have shape (1, spectra_num, ToF)"

    if divide:
        spectra_arr = spectra_arr / expand_factor
    spectra_arr = np.repeat(spectra_arr, expand_factor, axis=2)

    if remainder > 0:
        # Just extend final value to make up remainder
        final_col = spectra_arr[:, :, -1]
        final_col = final_col.reshape(1, final_col.shape[1], 1)
        spectra_arr = spectra_arr[:, :, :-1]
        final_col = np.repeat(final_col, remainder + 1, axis=2)
        spectra_arr = np.append(spectra_arr, final_col, axis=2)

    return spectra_arr


def process_incident_and_empty_spectra(
    params: phil.scope_extract,
    experiment: Experiment,
    incident_instance: Format,
    empty_instance: Format,
    shrink_factor: int,
) -> Tuple(np.array, np.array):

    """
    Smooths both the incident and empty runs.
    Corrects incident run for empty run and absorption.
    """

    incident_spectra = incident_instance.get_raw_spectra(
        normalize_by_proton_charge=True
    )

    empty_spectra = empty_instance.get_raw_spectra(normalize_by_proton_charge=True)

    assert incident_spectra.shape == empty_spectra.shape

    # Preprocess to remove noise / decrease computation time
    logger.info("Smoothing empty spectra")
    empty_spectra, _ = shrink_spectra(empty_spectra, shrink_factor)

    empty_spectra = smooth_spectra(
        empty_spectra,
        params.corrections.smoothing_window_length,
        params.corrections.smoothing_polyorder,
        params.mp.nproc,
    )

    logger.info("Smoothing incident spectra")
    incident_spectra, remainder = shrink_spectra(incident_spectra, shrink_factor)

    incident_spectra = smooth_spectra(
        incident_spectra,
        params.corrections.smoothing_window_length,
        params.corrections.smoothing_polyorder,
        params.mp.nproc,
    )

    # Subtract empty spectra to correct for empty values
    incident_spectra = incident_spectra - empty_spectra

    empty_spectra = expand_spectra(empty_spectra, shrink_factor, remainder, divide=True)
    incident_spectra = expand_spectra(
        incident_spectra, shrink_factor, remainder, divide=False
    )

    # Normalise with absorption correction
    if params.corrections.spherical_absorption:
        logger.info("Calculating incident absorption correction")
        incident_spectra = get_incident_absorption_correction(
            params, incident_instance, experiment.detector, experiment.beam
        )

    return incident_spectra, empty_spectra


def get_incident_absorption_correction(
    params: phil.scope_extract,
    incident_spectrum_instance: Format,
    detector: Detector,
    beam: Beam,
) -> np.array:

    sample_number_density = params.incident_spectrum.sample_number_density
    radius = params.incident_spectrum.sample_radius
    scattering_x_section = params.incident_spectrum.scattering_x_section
    absorption_x_section = params.incident_spectrum.absorption_x_section
    linear_absorption_c = absorption_x_section * sample_number_density
    linear_scattering_c = scattering_x_section * sample_number_density

    spectra = incident_spectrum_instance.get_raw_spectra(
        normalize_by_proton_charge=True
    )
    two_theta = np.array(
        incident_spectrum_instance.get_raw_spectra_two_theta(detector, beam)
    )
    # TODO correct wavelengths for each pixel location
    wavelengths = np.array(incident_spectrum_instance.get_wavelength_channels_in_ang())

    muR_arr = (linear_scattering_c + (linear_absorption_c / 1.8) * wavelengths) * radius
    two_theta_deg_arr = two_theta * 180 / np.pi
    two_theta_idx_arr = (two_theta_deg_arr / 10.0).astype(int)

    tof_spherical_absorption_correction(
        flumpy.from_numpy(spectra[0]),
        flumpy.from_numpy(muR_arr),
        flumpy.from_numpy(two_theta),
        flumpy.from_numpy(two_theta_idx_arr),
    )

    spectra[np.isinf(spectra)] = 0
    spectra[np.isnan(spectra)] = 0

    return spectra


def update_image_path(expt: Experiment, new_image_path: str) -> Experiment:
    """
    Change image_path of expt to new_image_path
    """
    reader = expt.imageset.reader()
    reader.set_path(new_image_path)
    return expt


def get_corrected_image_name(expt: Experiment, file_suffix: str) -> str:
    reader = expt.imageset.reader()
    current_filename, ext = splitext(reader.paths()[0])
    return current_filename + file_suffix + ext


def apply_lorentz_correction(
    expt_instance: Format,
    experiment: Experiment,
    spectra: np.array,
    two_theta_spectra: np.array,
) -> np.array:

    """
    Returns spectra with the Lorentz correction applied.
    (sin^2(theta)/lambda^4)
    """

    L0 = experiment.beam.get_sample_to_moderator_distance() * 10**-3
    L1_spectra = np.array(expt_instance.get_raw_spectra_L1(experiment.detector))
    tof = np.array(expt_instance.get_tof_in_seconds())
    two_theta_spectra_sq = np.square(np.sin(two_theta_spectra * 0.5))

    tof_lorentz_correction(
        flumpy.from_numpy(spectra[0]),
        float(L0),
        flumpy.from_numpy(L1_spectra),
        flumpy.from_numpy(tof),
        flumpy.from_numpy(two_theta_spectra_sq),
    )

    return spectra


def correct_spectra_for_bin_width(expt_instance: Format, spectra: np.array) -> np.array:

    """
    Divides each value in spectra by its ToF bin width to correct for
    intensities being artificially increased for larger bin widths.
    """

    spectra = spectra / expt_instance.get_bin_width_correction()
    return spectra


def sanity_check_params(params: phil.scope_extract) -> None:

    # Params of incident_and_empty
    if params.corrections.incident_and_empty:

        assert params.input.incident_run is not None, (
            "Trying to correct for incident run but "
            "input.incident_run has not been specified"
        )
        assert params.input.empty_run is not None, (
            "Trying to correct for empty run but "
            "input.empty_run has not been specified"
        )

    # Params for absorption correction
    if params.corrections.spherical_absorption:

        assert params.target_spectrum.sample_number_density is not None, (
            "Trying to correct target for absorption but "
            "target_spectrum.sample_number_density has not been set"
        )
        assert params.target_spectrum.sample_radius is not None, (
            "Trying to correct target for absorption but "
            "target_spectrum.sample_radius has not been set"
        )
        assert params.target_spectrum.scattering_x_section is not None, (
            "Trying to correct target for absorption but "
            "target_spectrum.scattering_x_section has not been set"
        )
        assert params.target_spectrum.absorption_x_section is not None, (
            "Trying to correct target for absorption but "
            "target_spectrum.absorption_x_section has not been set"
        )

        if params.corrections.incident_and_empty:
            assert params.incident_spectrum.sample_number_density is not None, (
                "Trying to correct incident for absorption but "
                "incident_spectrum.sample_number_density has not been set"
            )
            assert params.incident_spectrum.sample_radius is not None, (
                "Trying to correct incident for absorption but "
                "incident_spectrum.sample_radius has not been set"
            )
            assert params.incident_spectrum.scattering_x_section is not None, (
                "Trying to correct incident for absorption but "
                "incident_spectrum.scattering_x_section has not been set"
            )
            assert params.incident_spectrum.absorption_x_section is not None, (
                "Trying to correct incident for absorption but "
                "incident_spectrum.absorption_x_section has not been set"
            )


def run() -> None:

    """
    Input setup
    """

    phil = phil_scope.fetch()

    usage = "usage: dev.dials.scale_tof_images imported.expt incident_run=vanadium_run.nxs empty_run=empty_run.nxs"
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

    sanity_check_params(params)

    if params.mp.nproc is Auto:
        params.mp.nproc = multiprocessing.cpu_count()
        logger.info(f"Using {params.mp.nproc} processors.")

    """
    Load the files
    """

    experiments = params.input.experiments[0].data
    experiment = experiments[0]
    experiment_cls = experiment.imageset.get_format_class()
    expt_instance = experiment_cls.get_instance(
        experiment.imageset.paths()[0],
        **experiment.imageset.data().get_params(),
    )

    """
    Correct experiment spectra
    """

    spectra = expt_instance.get_raw_spectra(normalize_by_proton_charge=True)

    if params.corrections.incident_and_empty:

        incident_instance = experiment_cls(params.input.incident_run)
        empty_instance = experiment_cls(params.input.empty_run)

        incident_spectra, empty_spectra = process_incident_and_empty_spectra(
            params,
            experiment,
            incident_instance,
            empty_instance,
            params.corrections.shrink_factor,
        )

        # Correct for empty counts
        spectra = spectra - empty_spectra

        # Normalize intensities
        spectra = spectra / incident_spectra
        spectra[np.isinf(spectra)] = 0
        spectra[np.isnan(spectra)] = 0

    if params.corrections.spherical_absorption:

        logger.info("Calculating target absorption correction")
        sample_number_density = params.target_spectrum.sample_number_density
        radius = params.target_spectrum.sample_radius
        scattering_x_section = params.target_spectrum.scattering_x_section
        absorption_x_section = params.target_spectrum.absorption_x_section
        linear_absorption_c = absorption_x_section * sample_number_density
        linear_scattering_c = scattering_x_section * sample_number_density

        two_theta = np.array(
            expt_instance.get_raw_spectra_two_theta(
                experiment.detector, experiment.beam
            )
        )
        # TODO correct wavelengths for each pixel location
        wavelengths = np.array(expt_instance.get_wavelength_channels_in_ang())
        muR_arr = (
            linear_scattering_c + (linear_absorption_c / 1.8) * wavelengths
        ) * radius
        two_theta_deg_arr = two_theta * 180 / np.pi
        two_theta_idx_arr = (two_theta_deg_arr / 10.0).astype(int)

        tof_spherical_absorption_correction(
            flumpy.from_numpy(spectra[0]),
            flumpy.from_numpy(muR_arr),
            flumpy.from_numpy(two_theta),
            flumpy.from_numpy(two_theta_idx_arr),
        )

        spectra[np.isinf(spectra)] = 0
        spectra[np.isnan(spectra)] = 0

    if params.corrections.normalize_by_bin_width:

        logger.info("Correcting for ToF bin width")

        spectra = correct_spectra_for_bin_width(expt_instance, spectra)

    if params.corrections.lorentz:

        logger.info("Applying Lorentz correction")

        two_theta = np.array(
            expt_instance.get_raw_spectra_two_theta(
                experiment.detector, experiment.beam
            )
        )
        spectra = apply_lorentz_correction(
            expt_instance, experiment, spectra, two_theta
        )

    output_filename = get_corrected_image_name(
        experiments[0], params.output.image_file_suffix
    )
    expt_instance.save_spectra(spectra, output_filename)
    logger.info(f"Scaled image saved as {output_filename}")
    experiments[0] = update_image_path(experiments[0], output_filename)
    experiments.as_file(params.output.experiments)
    logger.info(f"Experiment updated to point at {output_filename}")


if __name__ == "__main__":
    run()
