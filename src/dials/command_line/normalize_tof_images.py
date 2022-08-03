# LIBTBX_SET_DISPATCHER_NAME dev.dials.normalize_tof_images
from __future__ import annotations

import logging

from scipy import interpolate

from dxtbx.model.experiment_list import ExperimentListFactory

import dials.util.log
from dials.algorithms.scaling.tof_absorption_correction import SphericalAbsorption
from dials.util.options import ArgumentParser
from dials.util.phil import parse
from dials.util.version import dials_version

logger = logging.getLogger("dials.command_line.normalize_tof_images")

phil_scope = parse(
    """
input{
vanadium_run
    .type = str
    .help = "Vanaduim run to normalize intensities"
empty_run
    .type = str
    .help "Empty run to correct empty counts"
}
sample_number_density
    .type float
    .help "Numberof atoms or formula units per A^3"
output {
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


def get_smooth_spectra(spectra_arr, smooth_param=100):
    for idx, spectra in enumerate(spectra_arr[0]):
        tck = interpolate(range(len(spectra), k=3, s=smooth_param))
        spectra_arr[0, idx] = interpolate.splev(range(len(spectra), tck, der=0))
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
        spectra = fmt_instance.get_raw_spectra()

    spherical_absorption = SphericalAbsorption(
        fmt_instance=fmt_instance,
        radius=radius,
        sample_number_density=sample_number_density,
        scattering_x_section=scattering_x_section,
        absorption_x_section=absorption_x_section,
    )

    two_theta = fmt_instance.get_two_theta()
    wavelengths = fmt_instance.get_wavelengths()

    return spherical_absorption.get_absorption_correction(
        spectra_arr=spectra, wavelength_arr=wavelengths, two_theta_arr=two_theta
    )


def get_corrected_vanadium_spectra(vanadium_instance, empty_spectra):
    spectra = vanadium_instance.get_raw_spectra()
    spectra = get_smooth_spectra(spectra)
    spectra -= empty_spectra
    vanadium_sample_number_density = 0.0722
    vanadium_radius = 0.003
    scattering_x_section = 5.158
    absorption_x_section = 4.883
    absorption_correction = calculate_absorption_correction(
        fmt_instance=vanadium_instance,
        radius=vanadium_radius,
        sample_number_density=vanadium_sample_number_density,
        scattering_x_section=scattering_x_section,
        absorption_x_section=absorption_x_section,
    )
    return spectra / absorption_correction


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
    Load the files
    """

    experiments = ExperimentListFactory.from_filenames(params.input.experiments)
    vanadium_run = ExperimentListFactory.from_filenames(params.input.vanadium_run)
    empty_run = ExperimentListFactory.from_filenames(params.input.empty_run)

    vanadium_instance = vanadium_run.imageset.get_format_class().get_instance(
        vanadium_run.imageset.paths()[0], **vanadium_run.imageset.data().get_params()
    )
    empty_instance = empty_run.imageset.get_format_class().get_instance(
        empty_run.imageset.paths()[0], **empty_run.imageset.data().get_params()
    )

    empty_spectra = get_smooth_spectra(empty_instance.get_raw_spectra())

    vanadium_spectra = get_corrected_vanadium_spectra(vanadium_instance, empty_spectra)

    """
    Correct experiment spectra
    """

    for expt in experiments:

        # Get access to format class
        expt_instance = expt.imageset.get_format_class().get_instance(
            expt.imageset.paths()[0], **expt.imageset.data().get_params()
        )
        spectra = expt_instance.get_raw_spectra()

        # Correct for empty counts
        spectra -= empty_spectra

        # Normalize intensities
        spectra /= vanadium_spectra

        # Apply absorption correction
        absorption_correction = calculate_absorption_correction(expt_instance)
        spectra /= absorption_correction

        expt_instance.save_spectra(spectra, params.output.image_file_suffix)


if __name__ == "__main__":
    run()
