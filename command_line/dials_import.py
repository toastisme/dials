# LIBTBX_SET_DISPATCHER_NAME dials.import

import logging
import pickle
from collections import namedtuple

import dxtbx.model.compare as compare
from dxtbx.imageset import ImageGrid, RotImageSequence, TOFImageSequence
from dxtbx.model.experiment_list import (
    Experiment,
    ExperimentList,
    ExperimentListFactory,
)
from libtbx.phil import parse

from dials.util import Sorry, show_mail_handle_errors
from dials.util.multi_dataset_handling import generate_experiment_identifiers
from dials.util.options import flatten_experiments

logger = logging.getLogger("dials.command_line.import")


def _pickle_load(fh):
    return pickle.load(fh, encoding="bytes")


help_message = """

This program is used to import image data files into a format that can be used
within dials. The program looks at the metadata for each image along with the
filenames to determine the relationship between sets of images. Once all the
images have been analysed, a experiments object is written to file which specifies
the relationship between files. For example if two sets of images which belong
to two rotation scans have been given, two image sequences will be saved. Images to
be processed are specified as command line arguments. Sometimes, there is a
maximum number of arguments that can be given on the command line and the number
of files may exceed this. In this case image filenames can be input on stdin
as shown in the examples below. Alternatively a template can be specified using
the template= parameter where the consecutive digits representing the image
numbers in the filenames are replaced with '#' characters.

The geometry can be set manually, either by using the reference_geometry=
parameter to specify an experiment list .expt file containing
the reference geometry, by using the mosflm_beam_centre= parameter to set
the Mosflm beam centre, or by specifying each variable to be overridden
using various geometry parameters.

Examples::

  dials.import /data/directory-containing-images/

  dials.import image_*.cbf

  dials.import image_1_*.cbf image_2_*.cbf

  dials.import directory/with/images

  dials.import template=image_1_####.cbf

  dials.import directory=directory/with/images

  find . -name "image_*.cbf" | dials.import

  dials.import << EOF
  image_1.cbf
  image_2.cbf
  EOF
"""


# Create the phil parameters
phil_scope = parse(
    """

  output {

    experiments = imported.expt
      .type = str
      .help = "The output JSON or pickle file"

    log = 'dials.import.log'
      .type = str
      .help = "The log filename"

    compact = False
      .type = bool
      .help = "For JSON output use compact representation"

  }

  identifier_type = *uuid timestamp None
    .type = choice
    .help = "Type of unique identifier to generate."

  input {

    ignore_unhandled = False
      .type = bool
      .help = "Don't throw exception if some args are unhandled"

    template = None
      .type = str
      .help = "The image sequence template"
      .multiple = True

    directory = None
      .type = str
      .help = "A directory with images"
      .multiple = True

    reference_geometry = None
      .type = path
      .help = "Experimental geometry from this models.expt "
              "will override the geometry from the "
              "image headers."

    check_reference_geometry = True
      .type = bool
      .expert_level = 2
      .help = "If True, assert the reference geometry is similar to"
              "the image geometry"

    use_beam_reference = True
      .type = bool
      .expert_level = 2
      .help = "If True, the beam from reference_geometry will override "
              "the beam from the image headers."

    use_gonio_reference = True
      .type = bool
      .expert_level = 2
      .help = "If True, the goniometer from reference_geometry will override "
              "the goniometer from the image headers."

    use_detector_reference = True
      .type = bool
      .expert_level = 2
      .help = "If True, the detector from reference_geometry will override "
              "the detector from the image headers."

    allow_multiple_sequences = True
      .type = bool
      .help = "If False, raise an error if multiple sequences are found"

    as_grid_scan = False
      .type = bool
      .help = "Import as grid scan"

    grid_size = None
      .type = ints(size=2)
      .help = "If importing as a grid scan set the size"
  }

  include scope dials.util.options.format_phil_scope

  include scope dials.util.options.geometry_phil_scope

  lookup {
    mask = None
      .type = str
      .help = "Apply a mask to the imported data"

    gain = None
      .type = str
      .help = "Apply a gain to the imported data"

    pedestal = None
      .type = str
      .help = "Apply a pedestal to the imported data"

    dx = None
      .type = str
      .help = "Apply an x geometry offset"
              "If both dx and dy are set then"
              "OffsetParallaxCorrectedPxMmStrategy will be used"

    dy = None
      .type = str
      .help = "Apply an y geometry offset"
              "If both dx and dy are set then"
              "OffsetParallaxCorrectedPxMmStrategy will be used"
  }
""",
    process_includes=True,
)


def _extract_or_read_imagesets(params):
    """
    Return a list of ImageSets, importing them via alternative means if necessary.

    The "Alternative Means" means via params.input.template or .directory,
    if the images to import haven't been specified directly.

    Args:
        params: The phil.scope_extract from dials.import

    Returns: A list of ImageSet objects
    """

    # Get the experiments
    experiments = flatten_experiments(params.input.experiments)

    # Check we have some filenames
    if len(experiments) == 0:

        # FIXME Should probably make this smarter since it requires editing here
        # and in dials.import phil scope
        try:
            format_kwargs = {
                "dynamic_shadowing": params.format.dynamic_shadowing,
                "multi_panel": params.format.multi_panel,
            }
        except AttributeError:
            format_kwargs = None

        # Check if a template has been set and print help if not, otherwise try to
        # import the images based on the template input
        if len(params.input.template) > 0:
            experiments = ExperimentListFactory.from_templates(
                params.input.template,
                image_range=params.geometry.scan.image_range,
                format_kwargs=format_kwargs,
            )
            if len(experiments) == 0:
                raise Sorry(
                    "No experiments found matching template %s"
                    % params.input.experiments
                )
        elif len(params.input.directory) > 0:
            experiments = ExperimentListFactory.from_filenames(
                params.input.directory, format_kwargs=format_kwargs
            )
            if len(experiments) == 0:
                raise Sorry(
                    "No experiments found in directories %s" % params.input.directory
                )
        else:
            raise Sorry("No experiments found")

    # TODO (Nick):  This looks redundant as the experiments are immediately discarded.
    #               verify this, and remove if it is.
    if params.identifier_type:
        generate_experiment_identifiers(experiments, params.identifier_type)

    # Get a list of all imagesets
    imageset_list = experiments.imagesets()

    # Return the experiments
    return imageset_list


class ReferenceGeometryUpdater:
    """
    A class to replace beam + detector with a reference
    """

    def __init__(self, params):
        """
        Load the reference geometry
        """
        self.params = params
        self.reference = self.load_reference_geometry(params)

    def __call__(self, imageset):
        """
        Replace with the reference geometry
        """
        if self.params.input.check_reference_geometry:
            # Check static detector items are the same
            assert self.reference.detector.is_similar_to(
                imageset.get_detector(), static_only=True
            ), "Reference detector model does not match input detector model"

        # Set beam and detector
        if self.params.input.use_beam_reference:
            imageset.set_beam(self.reference.beam)
        if self.params.input.use_detector_reference:
            imageset.set_detector(self.reference.detector)
        if self.params.input.use_gonio_reference:
            imageset.set_goniometer(self.reference.goniometer)
        return imageset

    def load_reference_geometry(self, params):
        """
        Load a reference geometry file
        """
        # Load reference geometry
        reference_detector = None
        reference_beam = None
        if params.input.reference_geometry is not None:
            from dxtbx.serialize import load

            experiments = None
            experiments = load.experiment_list(
                params.input.reference_geometry, check_format=False
            )
            assert experiments, "Could not import reference geometry"
            assert len(experiments.detectors()) >= 1
            assert len(experiments.beams()) >= 1
            if len(experiments.detectors()) > 1:
                raise Sorry(
                    "The reference geometry file contains %d detector definitions, but only a single definition is allowed."
                    % len(experiments.detectors())
                )
            if len(experiments.beams()) > 1:
                raise Sorry(
                    "The reference geometry file contains %d beam definitions, but only a single definition is allowed."
                    % len(experiments.beams())
                )
            reference_detector = experiments.detectors()[0]
            reference_beam = experiments.beams()[0]
            reference_goniometer = experiments.goniometers()[0]
        Reference = namedtuple("Reference", ["detector", "beam", "goniometer"])
        return Reference(
            detector=reference_detector,
            beam=reference_beam,
            goniometer=reference_goniometer,
        )


class ManualGeometryUpdater:
    """
    A class to update the geometry manually
    """

    def __init__(self, params):
        """
        Save the params
        """
        self.params = params

    def __call__(self, imageset):
        """
        Override the parameters
        """
        from copy import deepcopy

        from dxtbx.imageset import ImageSetFactory, RotImageSequence
        from dxtbx.model import (
            BeamFactory,
            DetectorFactory,
            GoniometerFactory,
            SequenceFactory,
        )

        if self.params.geometry.convert_sequences_to_stills:
            imageset = ImageSetFactory.imageset_from_anyset(imageset)
            for j in imageset.indices():
                imageset.set_sequence(None, j)
                imageset.set_goniometer(None, j)
        if not isinstance(imageset, RotImageSequence):
            if self.params.geometry.convert_stills_to_sequences:
                imageset = self.convert_stills_to_sequence(imageset)
        if isinstance(imageset, RotImageSequence):
            beam = BeamFactory.from_phil(self.params.geometry, imageset.get_beam())
            detector = DetectorFactory.from_phil(
                self.params.geometry, imageset.get_detector(), beam
            )
            goniometer = GoniometerFactory.from_phil(
                self.params.geometry, imageset.get_goniometer()
            )
            sequence = SequenceFactory.from_phil(
                self.params.geometry, deepcopy(imageset.get_sequence())
            )
            i0, i1 = sequence.get_array_range()
            j0, j1 = imageset.get_sequence().get_array_range()
            if i0 < j0 or i1 > j1:
                imageset = self.extrapolate_imageset(
                    imageset=imageset,
                    beam=beam,
                    detector=detector,
                    goniometer=goniometer,
                    sequence=sequence,
                )
            else:
                imageset.set_beam(beam)
                imageset.set_detector(detector)
                imageset.set_goniometer(goniometer)
                imageset.set_sequence(sequence)
        else:
            for i in range(len(imageset)):
                beam = BeamFactory.from_phil(self.params.geometry, imageset.get_beam(i))
                detector = DetectorFactory.from_phil(
                    self.params.geometry, imageset.get_detector(i), beam
                )
                goniometer = GoniometerFactory.from_phil(
                    self.params.geometry, imageset.get_goniometer(i)
                )
                sequence = SequenceFactory.from_phil(
                    self.params.geometry, imageset.get_sequence(i)
                )
                imageset.set_beam(beam, i)
                imageset.set_detector(detector, i)
                imageset.set_goniometer(goniometer, i)
                imageset.set_sequence(sequence, i)
        return imageset

    def extrapolate_imageset(
        self, imageset=None, beam=None, detector=None, goniometer=None, sequence=None
    ):
        from dxtbx.imageset import ImageSetFactory

        first, last = sequence.get_image_range()
        sequence = ImageSetFactory.make_sequence(
            template=imageset.get_template(),
            indices=list(range(first, last + 1)),
            format_class=imageset.get_format_class(),
            beam=beam,
            detector=detector,
            goniometer=goniometer,
            sequence=sequence,
            format_kwargs=imageset.params(),
        )
        return sequence

    def convert_stills_to_sequence(self, imageset):
        from dxtbx.model import Scan

        assert self.params.geometry.scan.oscillation is not None
        beam = imageset.get_beam(index=0)
        detector = imageset.get_detector(index=0)
        goniometer = imageset.get_goniometer(index=0)
        for i in range(1, len(imageset)):
            b_i = imageset.get_beam(i)
            d_i = imageset.get_detector(i)
            g_i = imageset.get_goniometer(i)
            assert (beam is None and b_i is None) or beam.is_similar_to(
                imageset.get_beam(index=i),
                wavelength_tolerance=self.params.input.tolerance.beam.wavelength,
                direction_tolerance=self.params.input.tolerance.beam.direction,
                polarization_normal_tolerance=self.params.input.tolerance.beam.polarization_normal,
                polarization_fraction_tolerance=self.params.input.tolerance.beam.polarization_fraction,
            )
            assert (detector is None and d_i is None) or detector.is_similar_to(
                imageset.get_detector(index=i),
                fast_axis_tolerance=self.params.input.tolerance.detector.fast_axis,
                slow_axis_tolerance=self.params.input.tolerance.detector.slow_axis,
                origin_tolerance=self.params.input.tolerance.detector.origin,
            )
            assert (goniometer is None and g_i is None) or goniometer.is_similar_to(
                imageset.get_goniometer(index=i),
                rotation_axis_tolerance=self.params.input.tolerance.goniometer.rotation_axis,
                fixed_rotation_tolerance=self.params.input.tolerance.goniometer.fixed_rotation,
                setting_rotation_tolerance=self.params.input.tolerance.goniometer.setting_rotation,
            )
        oscillation = self.params.geometry.scan.oscillation
        from dxtbx.imageset import ImageSetFactory
        from dxtbx.sequence_filenames import template_regex_from_list

        template, indices = template_regex_from_list(imageset.paths())
        image_range = (min(indices), max(indices))
        assert (image_range[1] + 1 - image_range[0]) == len(indices)
        scan = Scan(image_range=image_range, oscillation=oscillation)
        if template is None:
            paths = [imageset.get_path(i) for i in range(len(imageset))]
            assert len(set(paths)) == 1
            template = paths[0]
        new_sequence = ImageSetFactory.make_sequence(
            template=template,
            indices=indices,
            format_class=imageset.reader().get_format_class(),
            beam=beam,
            detector=detector,
            goniometer=goniometer,
            sequence=scan,
        )
        return new_sequence


class MetaDataUpdater:
    """
    A class to manage updating the experiments metadata
    """

    def __init__(self, params):
        """
        Init the class
        """
        from dials.util.options import geometry_phil_scope

        self.params = params

        # Create the geometry updater
        self.update_geometry = []
        update_order = []

        # First add reference geometry is present
        if self.params.input.reference_geometry is not None:
            self.update_geometry.append(ReferenceGeometryUpdater(self.params))
            update_order.append("Reference geometry")

        # Then add manual geometry
        working_phil = geometry_phil_scope.format(self.params)
        diff_phil = geometry_phil_scope.fetch_diff(source=working_phil)
        if diff_phil.as_str() != "":
            self.update_geometry.append(ManualGeometryUpdater(self.params))
            update_order.append("Manual geometry")

        if len(update_order) > 0:
            logger.info("")
            logger.info("Applying input geometry in the following order:")
            for i, item in enumerate(update_order, start=1):
                logger.info("  %d. %s", i, item)
            logger.info("")

    def __call__(self, imageset_list):
        """
        Transform the metadata
        """
        # Import the lookup data
        lookup = self.import_lookup_data(self.params)

        # Convert all to ImageGrid
        if self.params.input.as_grid_scan:
            imageset_list = self.convert_to_grid_scan(imageset_list, self.params)

        # Create the experiments
        experiments = ExperimentList()

        # Loop through imagesets
        for imageset in imageset_list:

            # Set the external lookups
            imageset = self.update_lookup(imageset, lookup)

            # Update the geometry
            for updater in self.update_geometry:
                imageset = updater(imageset)

            # Check beam and detector are present
            if imageset.get_beam() is None or imageset.get_detector() is None:
                raise Sorry(
                    """
          Imageset contains no beam or detector model. This means you will be
          unable to process your data.

          Possible causes of this error are:
             - A problem reading the images with one of the dxtbx format classes
             - A lack of header information in the file itself.

          You can override this by specifying the metadata as geometry parameters
        """
                )

            # Check if dx and dy are set
            if [
                imageset.external_lookup.dx.filename,
                imageset.external_lookup.dy.filename,
            ].count(None) == 0:
                imageset.update_detector_px_mm_data()
            elif [
                imageset.external_lookup.dx.filename,
                imageset.external_lookup.dy.filename,
            ].count(None) == 1:
                raise Sorry(
                    """
          Only 1 offset map is set. Need to set both dx and d
        """
                )

            # Append to new imageset list
            if type(imageset) in [RotImageSequence, TOFImageSequence]:
                if (
                    isinstance(imageset, RotImageSequence)
                    and imageset.get_sequence().is_still()
                ):
                    # make lots of experiments all pointing at one
                    # image set

                    # check if user has overridden the input - if yes, recall
                    # that these are in people numbers (1...) and are inclusive
                    if self.params.geometry.scan.image_range:
                        user_start, user_end = self.params.geometry.scan.image_range
                        offset = imageset.get_sequence().get_array_range()[0]
                        start, end = user_start - 1, user_end
                    else:
                        start, end = imageset.get_sequence().get_array_range()
                        offset = 0

                    for j in range(start, end):
                        subset = imageset[j - offset : j - offset + 1]
                        experiments.append(
                            Experiment(
                                imageset=imageset,
                                beam=imageset.get_beam(),
                                detector=imageset.get_detector(),
                                goniometer=imageset.get_goniometer(),
                                sequence=subset.get_sequence(),
                                crystal=None,
                            )
                        )
                else:
                    # have just one experiment
                    experiments.append(
                        Experiment(
                            imageset=imageset,
                            beam=imageset.get_beam(),
                            detector=imageset.get_detector(),
                            goniometer=imageset.get_goniometer(),
                            sequence=imageset.get_sequence(),
                            crystal=None,
                        )
                    )
            else:
                for i in range(len(imageset)):
                    experiments.append(
                        Experiment(
                            imageset=imageset[i : i + 1],
                            beam=imageset.get_beam(i),
                            detector=imageset.get_detector(i),
                            goniometer=imageset.get_goniometer(i),
                            sequence=imageset.get_sequence(i),
                            crystal=None,
                        )
                    )

        if self.params.identifier_type:
            generate_experiment_identifiers(experiments, self.params.identifier_type)
        # Return the experiments
        return experiments

    def update_lookup(self, imageset, lookup):
        from dxtbx.format.image import ImageBool, ImageDouble

        if lookup.size is not None:
            d = imageset.get_detector()
            assert len(lookup.size) == len(d), "Incompatible size"
            for s, p in zip(lookup.size, d):
                assert s == p.get_image_size()[::-1], "Incompatible size"
            if lookup.mask.filename is not None:
                imageset.external_lookup.mask.filename = lookup.mask.filename
                imageset.external_lookup.mask.data = ImageBool(lookup.mask.data)
            if lookup.gain.filename is not None:
                imageset.external_lookup.gain.filename = lookup.gain.filename
                imageset.external_lookup.gain.data = ImageDouble(lookup.gain.data)
            if lookup.dark.filename is not None:
                imageset.external_lookup.pedestal.filename = lookup.dark.filename
                imageset.external_lookup.pedestal.data = ImageDouble(lookup.dark.data)
            if lookup.dx.filename is not None:
                imageset.external_lookup.dx.filename = lookup.dx.filename
                imageset.external_lookup.dx.data = ImageDouble(lookup.dx.data)
            if lookup.dy.filename is not None:
                imageset.external_lookup.dy.filename = lookup.dy.filename
                imageset.external_lookup.dy.data = ImageDouble(lookup.dy.data)
        return imageset

    def import_lookup_data(self, params):
        """
        Get the lookup data
        """
        # Check the lookup inputs
        mask_filename = None
        gain_filename = None
        dark_filename = None
        dx_filename = None
        dy_filename = None
        mask = None
        gain = None
        dark = None
        dx = None
        dy = None
        lookup_size = None
        if params.lookup.mask is not None:
            mask_filename = params.lookup.mask
            with open(mask_filename, "rb") as fh:
                mask = _pickle_load(fh)
            if not isinstance(mask, tuple):
                mask = (mask,)
            lookup_size = [m.all() for m in mask]
        if params.lookup.gain is not None:
            gain_filename = params.lookup.gain
            with open(gain_filename, "rb") as fh:
                gain = _pickle_load(fh)
            if not isinstance(gain, tuple):
                gain = (gain,)
            if lookup_size is None:
                lookup_size = [g.all() for g in gain]
            else:
                assert len(gain) == len(lookup_size), "Incompatible size"
                for s, g in zip(lookup_size, gain):
                    assert s == g.all(), "Incompatible size"
        if params.lookup.pedestal is not None:
            dark_filename = params.lookup.pedestal
            with open(dark_filename, "rb") as fh:
                dark = _pickle_load(fh)
            if not isinstance(dark, tuple):
                dark = (dark,)
            if lookup_size is None:
                lookup_size = [d.all() for d in dark]
            else:
                assert len(dark) == len(lookup_size), "Incompatible size"
                for s, d in zip(lookup_size, dark):
                    assert s == d.all(), "Incompatible size"
        if params.lookup.dx is not None:
            dx_filename = params.lookup.dx
            with open(dx_filename, "rb") as fh:
                dx = _pickle_load(fh)
            if not isinstance(dx, tuple):
                dx = (dx,)
            if lookup_size is None:
                lookup_size = [d.all() for d in dx]
            else:
                assert len(dx) == len(lookup_size), "Incompatible size"
                for s, d in zip(lookup_size, dx):
                    assert s == d.all(), "Incompatible size"
        if params.lookup.dy is not None:
            dy_filename = params.lookup.dy
            with open(dx_filename, "rb") as fh:
                dy = _pickle_load(fh)
            if not isinstance(dy, tuple):
                dy = (dy,)
            if lookup_size is None:
                lookup_size = [d.all() for d in dy]
            else:
                assert len(dy) == len(lookup_size), "Incompatible size"
                for s, d in zip(lookup_size, dy):
                    assert s == d.all(), "Incompatible size"
        Lookup = namedtuple("Lookup", ["size", "mask", "gain", "dark", "dx", "dy"])
        Item = namedtuple("Item", ["data", "filename"])
        return Lookup(
            size=lookup_size,
            mask=Item(data=mask, filename=mask_filename),
            gain=Item(data=gain, filename=gain_filename),
            dark=Item(data=dark, filename=dark_filename),
            dx=Item(data=dx, filename=dx_filename),
            dy=Item(data=dy, filename=dy_filename),
        )

    def convert_to_grid_scan(self, imageset_list, params):
        """
        Convert the imagesets to grid scans
        """
        if params.input.grid_size is None:
            raise Sorry("The input.grid_size parameter is required")
        result = []
        for imageset in imageset_list:
            result.append(
                ImageGrid.from_imageset(imageset.as_imageset(), params.input.grid_size)
            )
        return result


class ImageImporter:
    """Class to parse the command line options."""

    def __init__(self, phil=phil_scope):
        """Set the expected options."""
        from dials.util.options import OptionParser

        # Create the option parser
        usage = "dials.import [options] /path/to/image/files"
        self.parser = OptionParser(
            usage=usage,
            sort_options=True,
            phil=phil,
            read_experiments_from_images=True,
            epilog=help_message,
        )

    def import_image(self, args=None):
        """Parse the options."""

        # Parse the command line arguments in two passes to set up logging early
        params, options = self.parser.parse_args(
            args=args, show_diff_phil=False, quick_parse=True
        )

        # Configure logging, if this is the main process
        if __name__ == "__main__":
            from dials.util import log

            log.config(verbosity=options.verbose, logfile=params.output.log)

        from dials.util.version import dials_version

        logger.info(dials_version())

        # Parse the command line arguments completely
        if params.input.ignore_unhandled:
            params, options, unhandled = self.parser.parse_args(
                args=args, show_diff_phil=False, return_unhandled=True
            )
            # Remove any False values from unhandled (eliminate empty strings)
            unhandled = [x for x in unhandled if x]
        else:
            params, options = self.parser.parse_args(args=args, show_diff_phil=False)
            unhandled = None

        # Log the diff phil
        diff_phil = self.parser.diff_phil.as_str()
        if diff_phil:
            logger.info("The following parameters have been modified:\n")
            logger.info(diff_phil)

        # Print a warning if something unhandled
        if unhandled:
            msg = "Unable to handle the following arguments:\n"
            msg += "\n".join(["  %s" % a for a in unhandled])
            msg += "\n"
            logger.warning(msg)

        # Print help if no input
        if len(params.input.experiments) == 0 and not (
            params.input.template or params.input.directory
        ):
            self.parser.print_help()
            return

        # Re-extract the imagesets to rebuild experiments from
        imagesets = _extract_or_read_imagesets(params)

        metadata_updater = MetaDataUpdater(params)
        experiments = metadata_updater(imagesets)

        # Compute some numbers
        num_sweeps = 0
        num_still_sequences = 0
        num_stills = 0
        num_images = 0

        # importing a lot of experiments all pointing at one imageset should
        # work gracefully
        counted_imagesets = []

        for e in experiments:
            if e.imageset in counted_imagesets:
                continue
            if type(e.imageset) in [RotImageSequence, TOFImageSequence]:
                if (
                    isinstance(e.imageset, RotImageSequence)
                    and e.imageset.get_sequence().is_still()
                ):
                    num_still_sequences += 1
                else:
                    num_sweeps += 1
            else:
                num_stills += 1
            num_images += len(e.imageset)
            counted_imagesets.append(e.imageset)

        format_list = {str(e.imageset.get_format_class()) for e in experiments}

        # Print out some bulk info
        logger.info("-" * 80)
        for f in format_list:
            logger.info("  format: %s", f)
        logger.info("  num images: %d", num_images)
        logger.info("  sequences:")
        logger.info("    still:    %d", num_still_sequences)
        logger.info("    sweep:    %d", num_sweeps)
        logger.info("  num stills: %d", num_stills)

        # Print out info for all experiments
        for experiment in experiments:

            # Print some experiment info - override the output of image range
            # if appropriate
            image_range = params.geometry.scan.image_range
            if isinstance(experiment.imageset, RotImageSequence):
                imageset_type = "sequence"
            else:
                imageset_type = "stills"

            logger.debug("-" * 80)
            logger.debug("  format: %s", str(experiment.imageset.get_format_class()))
            logger.debug("  imageset type: %s", imageset_type)
            if image_range is None:
                logger.debug("  num images:    %d", len(experiment.imageset))
            else:
                logger.debug("  num images:    %d", image_range[1] - image_range[0] + 1)

            logger.debug("")
            logger.debug(experiment.imageset.get_beam())
            logger.debug(experiment.imageset.get_goniometer())
            logger.debug(experiment.imageset.get_detector())
            logger.debug(experiment.imageset.get_sequence())

        # Only allow a single sequence
        if params.input.allow_multiple_sequences is False:
            self.assert_single_sequence(experiments, params)

        # Write the experiments to file
        self.write_experiments(experiments, params)

    def write_experiments(self, experiments, params):
        """
        Output the experiments to file.
        """
        if params.output.experiments:
            logger.info("-" * 80)
            logger.info("Writing experiments to %s", params.output.experiments)
            experiments.as_file(
                params.output.experiments, compact=params.output.compact
            )

    def assert_single_sequence(self, experiments, params):
        """
        Print an error message if more than 1 sequence
        """
        sequences = [
            e.imageset for e in experiments if isinstance(e.imageset, RotImageSequence)
        ]

        if len(sequences) > 1:

            # Print some info about multiple sequences
            self.diagnose_multiple_sequences(sequences, params)

            # Raise exception
            raise Sorry(
                """
        More than 1 sequence was found. Two things may be happening here:

        1. There really is more than 1 sequence. If you expected this to be the
           case, set the parameter allow_multiple_sequences=True. If you don't
           expect this, then check the input to dials.import.

        2. There may be something wrong with your image headers (for example,
           the rotation ranges of each image may not match up). You should
           investigate what went wrong, but you can force dials.import to treat
           your images as a single sequence by using the template=image_####.cbf
           parameter (see help).
      """
            )

    def diagnose_multiple_sequences(self, sequences, params):
        """
        Print a diff between sequences.
        """
        logger.info("")
        for i in range(1, len(sequences)):
            logger.info("=" * 80)
            logger.info("Diff between sequence %d and %d", i - 1, i)
            logger.info("")
            self.print_sequence_diff(sequences[i - 1], sequences[i], params)
        logger.info("=" * 80)
        logger.info("")

    @staticmethod
    def print_sequence_diff(sequence1, sequence2, params):
        """
        Print a diff between sequences.
        """
        logger.info(
            compare.sequence_diff(
                sequence1, sequence2, tolerance=params.input.tolerance
            )
        )


@show_mail_handle_errors()
def run(args=None):
    importer = ImageImporter()
    importer.import_image(args)


if __name__ == "__main__":
    run()
