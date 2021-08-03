"""
Print a table of the orientation for every image of a dataset. The
orientation is expressed as a zone axis (a direction referenced to the direct
lattice) [uvw] giving the beam direction with respect to the crystal lattice.
Take into account any scan-varying models.

Usage: dials.frame_orientations refined.expt
"""


import sys

import matplotlib

from scitbx import matrix

import dials.util
from dials.util import tabulate
from dials.util.options import OptionParser, flatten_experiments

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Script:
    """A class for running the script."""

    def __init__(self):
        """Initialise the script."""
        from libtbx.phil import parse

        # The phil scope
        phil_scope = parse(
            """
scale = unit *max_cell ewald_sphere_radius
    .type = choice
    .help = "Choose the scale for the direction vector in orthogonal"
            "coordinates prior to transformation into fractional"
            "coordinates [uvw]"

plot_filename = None
    .type = str
    .help = "Filename for a plot of angle between neighbouring frames"
            "(set to None for no plot)"
""",
            process_includes=True,
        )

        usage = "dials.frame_orientations refined.expt refined.refl"

        # Create the parser
        self.parser = OptionParser(
            usage=usage,
            phil=phil_scope,
            read_experiments=True,
            check_format=False,
            epilog=__doc__,
        )

    def run(self, args=None):
        """Execute the script."""

        # Parse the command line
        self.params, _ = self.parser.parse_args(args, show_diff_phil=True)

        if not self.params.input.experiments:
            self.parser.print_help()
            sys.exit()

        # Try to load the models
        experiments = flatten_experiments(self.params.input.experiments)
        nexp = len(experiments)
        if nexp == 0:
            self.parser.print_help()
            sys.exit("No Experiments found in the input")

        # Set up a plot if requested
        if self.params.plot_filename:
            plt.figure()

        header = [
            "Image",
            "Beam direction (xyz)",
            "Zone axis [uvw]",
            "Angles between beam\nand axes a, b, c (deg)",
            "Angle from\nprevious image (deg)",
        ]
        for iexp, exp in enumerate(experiments):
            print(f"For Experiment id = {iexp}")
            print(exp.beam)
            print(exp.crystal)
            print(exp.sequence)

            if self.params.scale == "ewald_sphere_radius":
                scale = 1.0 / exp.beam.get_wavelength()
            elif self.params.scale == "max_cell":
                uc = exp.crystal.get_unit_cell()
                scale = max(uc.parameters()[0:3])
            else:
                scale = 1.0
            print(
                "Beam direction scaled by {} = {:.3f} to "
                "calculate zone axis\n".format(self.params.scale, scale)
            )

            dat = extract_experiment_data(exp, scale)
            images = dat["images"]
            directions = dat["directions"]
            zone_axes = dat["zone_axes"]
            real_space_axes = dat["real_space_axes"]

            # calculate the angle between the beam and each crystal axis
            axis_angles = []
            for d, rsa in zip(directions, real_space_axes):
                angles = [d.angle(a, deg=True) for a in rsa]
                axis_angles.append("{:.2f} {:.2f} {:.2f}".format(*angles))

            # calculate the orientation offset between each image
            offset = [
                e1.angle(e2, deg=True) for e1, e2 in zip(zone_axes[:-1], zone_axes[1:])
            ]
            str_off = ["---"] + [f"{e:.8f}" for e in offset]

            rows = []
            for i, d, z, a, o in zip(
                images,
                directions,
                zone_axes,
                axis_angles,
                str_off,
            ):
                row = [
                    str(i),
                    "{:.8f} {:.8f} {:.8f}".format(*d.elems),
                    "{:.8f} {:.8f} {:.8f}".format(*z.elems),
                    a,
                    o,
                ]
                rows.append(row)

            # Print the table
            print(tabulate(rows, header))

            # Add to the plot, if requested
            if self.params.plot_filename:
                plt.scatter(images[1:], offset, s=1)

        # Finish and save plot, if requested
        if self.params.plot_filename:
            plt.xlabel("Image number")
            plt.ylabel(r"Angle from previous image $\left(^\circ\right)$")
            plt.title(r"Angle between neighbouring images")
            print(f"Saving plot to {self.params.plot_filename}")
            plt.savefig(self.params.plot_filename)

        print()


def extract_experiment_data(exp, scale=1):
    """Extract lists of the image number, beam direction and zone axis from an
    experiment"""
    crystal = exp.crystal
    beam = exp.beam
    scan = exp.sequence
    gonio = exp.goniometer

    image_range = scan.get_image_range()
    images = list(range(image_range[0], image_range[1] + 1))

    if beam.num_scan_points > 0:
        # There is one more scan point than the number of images. For simplicity,
        # omit the final scan point, to leave a list of the beam directions at
        # the _beginning_ of each image. Likewise for gonio and crystal, below.
        directions = []
        for i in range(beam.num_scan_points - 1):
            s0 = matrix.col(beam.get_s0_at_scan_point(i))
            directions.append(s0.normalize())
    else:
        directions = [matrix.col(beam.get_unit_s0()) for _ in images]

    if gonio.num_scan_points > 0:
        S_mats = [
            matrix.sqr(gonio.get_setting_rotation_at_scan_point(i))
            for i in range(gonio.num_scan_points - 1)
        ]
    else:
        S_mats = [matrix.sqr(gonio.get_setting_rotation()) for _ in images]

    F_mats = [matrix.sqr(gonio.get_fixed_rotation()) for _ in images]
    array_range = scan.get_array_range()
    R_mats = []
    axis = matrix.col(gonio.get_rotation_axis_datum())
    for i in range(*array_range):
        phi = scan.get_angle_from_array_index(i, deg=True)
        R = matrix.sqr(axis.axis_and_angle_as_r3_rotation_matrix(phi, deg=True))
        R_mats.append(R)

    if crystal.num_scan_points > 0:
        UB_mats = [
            matrix.sqr(crystal.get_A_at_scan_point(i))
            for i in range(crystal.num_scan_points - 1)
        ]
    else:
        UB_mats = [matrix.sqr(crystal.get_A()) for _ in images]

    assert len(directions) == len(S_mats) == len(F_mats) == len(R_mats) == len(UB_mats)

    # Construct full setting matrix for each image
    SRFUB = (S * R * F * UB for S, R, F, UB in zip(S_mats, R_mats, F_mats, UB_mats))

    # SFRUB is the orthogonalisation matrix for the reciprocal space laboratory
    # frame. We want the real space fractionalisation matrix, which is its
    # transpose (https://dials.github.io/documentation/conventions.html)
    frac_mats = [m.transpose() for m in SRFUB]
    zone_axes = [frac * (d * scale) for frac, d in zip(frac_mats, directions)]

    # Now get the real space orthogonalisation matrix to calculate the real
    # space cell vectors at each image
    orthog_mats = (frac.inverse() for frac in frac_mats)
    h = matrix.col((1, 0, 0))
    k = matrix.col((0, 1, 0))
    l = matrix.col((0, 0, 1))
    real_space_axes = [(o * h, o * k, o * l) for o in orthog_mats]
    return {
        "images": images,
        "directions": directions,
        "zone_axes": zone_axes,
        "real_space_axes": real_space_axes,
    }


@dials.util.show_mail_handle_errors()
def run(args=None):
    script = Script()
    script.run(args)


if __name__ == "__main__":
    run()
