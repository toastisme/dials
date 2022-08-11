# LIBTBX_SET_DISPATCHER_NAME dev.dials.normalize_tof_images
from __future__ import annotations

import logging
import multiprocessing
from os.path import splitext

import numpy as np
from scipy import interpolate

from dxtbx.format.FormatISISSXD import FormatISISSXD as fc
from dxtbx.model import Experiment

import dials.util.log
from dials.algorithms.scaling.tof_absorption_correction import SphericalAbsorption
from dials.util.options import ArgumentParser
from dials.util.phil import parse
from dials.util.version import dials_version

logger = logging.getLogger("dials.command_line.normalize_tof_images")

phil_scope = parse(
    """
input{
vanadium_run = None
    .type = str
    .help = "Vanaduim run to normalize intensities"
empty_run = None
    .type = str
    .help = "Empty run to correct empty counts"
}
output {
experiments = 'imported.expt'
    .type = str
    .help = "The experiments output filename"
image_file_suffix = '_corrected'
    .type = str
    .help = "Suffix of the corrected image file"
phil = 'dials.normalize_tof_images.phil'
    .type = str
    .help = "The output phil file"
log = 'dials.normalize_tof_images.log'
    .type = str
    .help = "The log filename"
}
"""
)


def run_interpolate(x, y, smooth_param):
    tck = interpolate.splrep(x, y, k=3, s=smooth_param)
    return interpolate.splev(x, tck, der=0)


def smooth_spectra(spectra_arr, smooth_param):
    nproc = 14
    pool = multiprocessing.Pool(nproc)
    x = range(len(spectra_arr[0, 0]))

    processes = [
        pool.apply_async(run_interpolate, args=(x, y, smooth_param))
        for y in spectra_arr[0]
    ]
    result = [p.get() for p in processes]

    for idx, _ in enumerate(spectra_arr[0]):
        spectra_arr[0, idx] = result[idx]
    return spectra_arr


def shrink_spectra(spectra_arr, shrink_factor):

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


def expand_spectra(spectra_arr, expand_factor, remainder):

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


def calculate_absorption_correction(
    fmt_instance,
    radius,
    sample_number_density,
    scattering_x_section,
    absorption_x_section,
    spectra=None,
):

    if spectra is None:
        spectra = fmt_instance.get_raw_spectra(normalize_by_proton_charge=True)

    spherical_absorption = SphericalAbsorption(
        radius=radius,
        sample_number_density=sample_number_density,
        scattering_x_section=scattering_x_section,
        absorption_x_section=absorption_x_section,
    )

    two_theta = np.array(fmt_instance.get_raw_spectra_two_theta())
    wavelengths = np.array(fmt_instance.get_wavelength_channels_in_ang())

    return spherical_absorption.get_absorption_correction_vec(
        spectra_arr=spectra, wavelength_arr=wavelengths, two_theta_arr=two_theta
    )


def process_correction_spectra(vanadium_instance, empty_instance, shrink_factor):

    vanadium_spectra = vanadium_instance.get_raw_spectra(
        normalize_by_proton_charge=True
    )
    empty_spectra = empty_instance.get_raw_spectra(normalize_by_proton_charge=True)

    assert vanadium_spectra.shape == empty_spectra.shape

    # Preprocess to remove noise / decrease computation time
    logger.info("Smoothing empty spectra")
    empty_spectra, _ = shrink_spectra(empty_spectra, shrink_factor)
    empty_spectra = smooth_spectra(empty_spectra, smooth_param=110000)
    logger.info("Smoothing vanadium spectra")
    vanadium_spectra, remainder = shrink_spectra(vanadium_spectra, shrink_factor)
    vanadium_spectra = smooth_spectra(vanadium_spectra, smooth_param=110000)

    # Subtract empty spectra to correct for empty values
    vanadium_spectra = vanadium_spectra - empty_spectra

    empty_spectra = expand_spectra(empty_spectra, shrink_factor, remainder)
    vanadium_spectra = expand_spectra(vanadium_spectra, shrink_factor, remainder)

    # Normalise with absorption correction
    absorption_correction = get_vanadium_absorption_correction(vanadium_instance)
    vanadium_spectra = vanadium_spectra / absorption_correction

    return vanadium_spectra, empty_spectra


def get_vanadium_absorption_correction(vanadium_instance):
    vanadium_sample_number_density = 0.0722
    vanadium_radius = 0.3
    scattering_x_section = 5.158
    absorption_x_section = 4.883
    logger.info("Calculating vanadium absorption correction")
    absorption_correction = calculate_absorption_correction(
        fmt_instance=vanadium_instance,
        radius=vanadium_radius,
        sample_number_density=vanadium_sample_number_density,
        scattering_x_section=scattering_x_section,
        absorption_x_section=absorption_x_section,
    )
    return absorption_correction


def update_image_path(expt: Experiment, new_image_path: str) -> Experiment:
    reader = expt.imageset.reader()
    reader.set_path(new_image_path)
    return expt


def get_corrected_image_name(expt: Experiment, file_suffix: str) -> str:
    reader = expt.imageset.reader()
    current_filename, ext = splitext(reader.paths()[0])
    return current_filename + file_suffix + ext


def run():

    """
    Input setup
    """

    phil = phil_scope.fetch()

    usage = "usage: dev.dials.normalize_tof_images imported.expt vanadium_run=vanadium_run.nxs empty_run=empty_run.nxs"
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
    Load the files
    """

    experiments = params.input.experiments[0].data
    expt_instance = (
        experiments[0]
        .imageset.get_format_class()
        .get_instance(
            experiments[0].imageset.paths()[0],
            **experiments[0].imageset.data().get_params(),
        )
    )
    vanadium_instance = fc(params.input.vanadium_run)
    empty_instance = fc(params.input.empty_run)

    vanadium_spectra, empty_spectra = process_correction_spectra(
        vanadium_instance, empty_instance, shrink_factor=7
    )

    """
    Correct experiment spectra
    """

    spectra = expt_instance.get_raw_spectra(normalize_by_proton_charge=True)

    # Correct for empty counts
    spectra = spectra - empty_spectra

    # Normalize intensities
    spectra = spectra / vanadium_spectra
    spectra[np.isinf(spectra)] = 0
    spectra[np.isnan(spectra)] = 0

    # Apply absorption correction
    nacl_sample_number_density = 0.0223
    nacl_radius = 0.3
    scattering_x_section = 10.040
    absorption_x_section = 17.015

    logger.info("Calculating target absorption correction")
    absorption_correction = calculate_absorption_correction(
        fmt_instance=expt_instance,
        radius=nacl_radius,
        sample_number_density=nacl_sample_number_density,
        scattering_x_section=scattering_x_section,
        absorption_x_section=absorption_x_section,
    )
    spectra = spectra / absorption_correction
    spectra[np.isinf(spectra)] = 0
    spectra[np.isnan(spectra)] = 0

    output_filename = get_corrected_image_name(
        experiments[0], params.output.image_file_suffix
    )
    expt_instance.save_spectra(spectra, output_filename)
    logger.info(f"Corrected image saved as {output_filename}")
    experiments[0] = update_image_path(experiments[0], output_filename)
    experiments.as_file(params.output.experiments)
    logger.info(f"Experiment updated to point at {output_filename}")


if __name__ == "__main__":
    run()
