import os
import sys

from PIL import Image

import iotbx.phil
from dxtbx.model.detector_helpers import project_2d

from dials.algorithms.image.threshold import DispersionThresholdDebug
from dials.array_family import flex
from dials.util import Sorry, show_mail_handle_errors
from dials.util.image_viewer.slip_viewer.tile_generation import (
    get_flex_image,
    get_flex_image_multipanel,
)
from dials.util.options import OptionParser, flatten_experiments

help_message = """

Export raw diffraction image files as bitmap images, optionally exporting
images from intermediate spot-finding steps (local mean and variance maps,
or sigma_b, sigma_s or threshold-filtered images). Appearance of the images
can be altered via the brightness and colour_scheme parameters, and optionally
binning of pixels can be used to reduce image sizes.

Examples::

  dials.export_bitmaps image.cbf

  dials.export_bitmaps models.expt

  dials.export_bitmaps image.cbf display=variance colour_scheme=inverse_greyscale
"""

phil_scope = iotbx.phil.parse(
    """
binning = 1
  .type = int(value_min=1)
brightness = 100
  .type = float(value_min=0.0)
colour_scheme = *greyscale rainbow heatmap inverse_greyscale
  .type = choice
projection = lab *image
  .type = choice
padding = 4
  .type = int(value_min=0)
imageset_index = None
  .type = int
  .multiple = True
  .help = "The index/indices from an imageset to export. The first image of "
          "the set is 1."
  .expert_level=2
display = *image mean variance dispersion sigma_b \
          sigma_s threshold global_threshold
  .type = choice
nsigma_b = 6
  .type = float(value_min=0)
nsigma_s = 3
  .type = float(value_min=0)
global_threshold = 0
  .type = float(value_min=0)
kernel_size = 3,3
  .type = ints(size=2, value_min=1)
min_local = 2
  .type = int
gain = 1
  .type = float(value_min=0)
saturation = 0
  .type = int
show_mask = False
  .type = bool
png {
  compress_level = 1
    .type = int(value_min=0, value_max=9)
    .help = "ZLIB compression level, a number between 0 and 9: 1 gives best "
            "speed, 9 gives best compression, 0 gives no compression at all."
}
jpeg {
  quality = 75
    .type = int(value_min=1, value_max=95)
    .help = "The image quality, on a scale from 1 (worst) to 95 (best)"
}

include scope dials.util.options.format_phil_scope

output {
  prefix = "image"
    .type = str
  directory = None
    .type = path
  file = None
    .type = str
    .help = "Full name of the output file. Overrides 'prefix' and the default "
            "file extension. Only makes sense if a single file is written."
  format = jpeg *png tiff
    .type = choice
}""",
    process_includes=True,
)

colour_schemes = {"greyscale": 0, "rainbow": 1, "heatmap": 2, "inverse_greyscale": 3}


@show_mail_handle_errors()
def run(args=None):
    usage = "dials.export_bitmaps [options] models.expt | image.cbf"

    parser = OptionParser(
        usage=usage,
        phil=phil_scope,
        read_experiments=True,
        read_experiments_from_images=True,
        check_format=True,
        epilog=help_message,
    )

    params, options = parser.parse_args(args, show_diff_phil=True)

    experiments = flatten_experiments(params.input.experiments)
    if len(experiments) == 0:
        parser.print_help()
        exit(0)

    imagesets = experiments.imagesets()

    for imageset in imagesets:
        imageset_as_bitmaps(imageset, params)


def imageset_as_bitmaps(imageset, params):
    brightness = params.brightness / 100
    vendortype = "made up"
    # check that binning is a power of 2
    binning = params.binning
    if not (binning > 0 and ((binning & (binning - 1)) == 0)):
        raise Sorry("binning must be a power of 2")
    output_dir = params.output.directory
    if output_dir is None:
        output_dir = "."
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_files = []

    detector = imageset.get_detector()

    # Furnish detector with 2D projection axes
    detector.projected_2d = project_2d(detector)
    detector.projection = params.projection

    panel = detector[0]
    scan = imageset.get_sequence()
    # XXX is this inclusive or exclusive?
    saturation = panel.get_trusted_range()[1]
    if params.saturation:
        saturation = params.saturation
    if scan is not None and scan.get_oscillation()[1] > 0 and not params.imageset_index:
        start, end = scan.get_image_range()
    else:
        start, end = 1, len(imageset)
    # If the user specified an image range index, only export those
    image_range = [
        i
        for i in range(start, end + 1)
        if not params.imageset_index or i in params.imageset_index
    ]
    if params.output.file and len(image_range) != 1:
        sys.exit("output.file can only be specified if a single image is exported")
    for i_image in image_range:
        image = imageset.get_raw_data(i_image - start)

        mask = imageset.get_mask(i_image - start)
        if mask is None:
            mask = [p.get_trusted_range_mask(im) for im, p in zip(image, detector)]

        if params.show_mask:
            for rd, m in zip(image, mask):
                rd.set_selected(~m, -2)

        image = image_filter(
            image,
            mask,
            display=params.display,
            gain_value=params.gain,
            nsigma_b=params.nsigma_b,
            nsigma_s=params.nsigma_s,
            global_threshold=params.global_threshold,
            min_local=params.min_local,
            kernel_size=params.kernel_size,
        )

        show_untrusted = params.show_mask
        if len(detector) > 1:
            # FIXME This doesn't work properly, as flex_image.size2() is incorrect
            # also binning doesn't work
            flex_image = get_flex_image_multipanel(
                brightness=brightness,
                detector=detector,
                image_data=image,
                binning=binning,
                beam=imageset.get_beam(),
                show_untrusted=show_untrusted,
            )
        else:
            flex_image = get_flex_image(
                brightness=brightness,
                data=image[0],
                binning=binning,
                saturation=saturation,
                vendortype=vendortype,
                show_untrusted=show_untrusted,
            )

        flex_image.setWindow(0, 0, 1)
        flex_image.adjust(color_scheme=colour_schemes.get(params.colour_scheme))

        # now export as a bitmap
        flex_image.prep_string()

        # XXX is size//binning safe here?
        pil_img = Image.frombytes(
            "RGB", (flex_image.ex_size2(), flex_image.ex_size1()), flex_image.as_bytes()
        )
        if params.output.file:
            path = os.path.join(output_dir, params.output.file)
        else:
            path = os.path.join(
                output_dir,
                "{prefix}{image:0{padding}}.{format}".format(
                    image=i_image,
                    prefix=params.output.prefix,
                    padding=params.padding,
                    format=params.output.format,
                ),
            )

        print(f"Exporting {path}")
        output_files.append(path)
        with open(path, "wb") as tmp_stream:
            pil_img.save(
                tmp_stream,
                format=params.output.format,
                compress_level=params.png.compress_level,
                quality=params.jpeg.quality,
            )

    return output_files


def image_filter(
    raw_data,
    mask,
    display,
    gain_value,
    nsigma_b,
    nsigma_s,
    global_threshold,
    min_local,
    kernel_size,
):

    if display == "image":
        return raw_data

    assert gain_value > 0
    gain_map = [flex.double(rd.accessor(), gain_value) for rd in raw_data]

    kabsch_debug_list = [
        DispersionThresholdDebug(
            data.as_double(),
            mask[i_panel],
            gain_map[i_panel],
            kernel_size,
            nsigma_b,
            nsigma_s,
            global_threshold,
            min_local,
        )
        for i_panel, data in enumerate(raw_data)
    ]

    if display == "mean":
        display_data = [kabsch.mean() for kabsch in kabsch_debug_list]
    elif display == "variance":
        display_data = [kabsch.variance() for kabsch in kabsch_debug_list]
    elif display == "dispersion":
        display_data = [kabsch.index_of_dispersion() for kabsch in kabsch_debug_list]
    elif display == "sigma_b":
        cv = [kabsch.index_of_dispersion() for kabsch in kabsch_debug_list]
        display_data = (kabsch.cv_mask() for kabsch in kabsch_debug_list)
        display_data = [_mask.as_1d().as_double() for _mask in display_data]
        for i, _mask in enumerate(display_data):
            _mask.reshape(cv[i].accessor())
    elif display == "sigma_s":
        cv = [kabsch.index_of_dispersion() for kabsch in kabsch_debug_list]
        display_data = (kabsch.value_mask() for kabsch in kabsch_debug_list)
        display_data = [_mask.as_1d().as_double() for _mask in display_data]
        for i, _mask in enumerate(display_data):
            _mask.reshape(cv[i].accessor())
    elif display == "global_threshold":
        cv = [kabsch.index_of_dispersion() for kabsch in kabsch_debug_list]
        display_data = (kabsch.global_mask() for kabsch in kabsch_debug_list)
        display_data = [_mask.as_1d().as_double() for _mask in display_data]
        for i, _mask in enumerate(display_data):
            _mask.reshape(cv[i].accessor())
    elif display == "threshold":
        cv = [kabsch.index_of_dispersion() for kabsch in kabsch_debug_list]
        display_data = (kabsch.final_mask() for kabsch in kabsch_debug_list)
        display_data = [_mask.as_1d().as_double() for _mask in display_data]
        for i, _mask in enumerate(display_data):
            _mask.reshape(cv[i].accessor())

    return display_data


if __name__ == "__main__":
    run()
