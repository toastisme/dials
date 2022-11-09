# LIBTBX_SET_DISPATCHER_NAME dev.dials.simple_tof_integrate
from __future__ import annotations

import logging
import multiprocessing
from math import ceil, floor
from typing import Tuple

import numpy as np
from numpy.linalg import det
from scipy.optimize import curve_fit
from scipy.special import erfc

import cctbx.array_family.flex
from dxtbx import flumpy
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
experiments = 'integrated.expt'
    .type = str
    .help = "The experiments output filename"
reflections = 'integrated.refl'
    .type = str
    .help = "The integrated output filename"
output_hkl = True
    .type = bool
    .help = "Output the integrated intensities as a SHELX hkl file"
hkl =  'integrated.hkl'
    .type = str
    .help = "The hkl output filename"
phil = 'dials.simple_integrate.phil'
    .type = str
    .help = "The output phil file"
log = 'simple_tof_integrate.log'
    .type = str
    .help = "The log filename"
}
method{
profile_fitting = False
    .type = bool
    .help = "Use integration by profile fitting"
}
"""
)

"""
Kabsch 2010 refers to
Kabsch W., Integration, scaling, space-group assignment and
post-refinment, Acta Crystallographica Section D, 2010, D66, 133-144
Usage:
$ dev.dials.simple_tof_integrate.py refined.expt refined.refl
"""


class GutmannProfile:
    def __init__(self, reflection, alpha, beta):
        self.centroid = reflection["xyzcal.px"]
        s = reflection["shoebox"]
        s.flatten()
        self.intensities = flumpy.to_numpy(s.values())
        self.coords = list(s.coords())
        self.dx = self.get_dx(self.centroid, self.coords)
        self.dy = self.get_dy(self.centroid, self.coords)
        self.dt = self.get_dt(self.centroid, self.coords)
        self.alpha = alpha
        self.beta = beta
        self.H = self.init_H(self.coords)
        self.cov = None
        self.params = None

    def init_H(self, coords):
        x = [i[0] for i in coords]
        y = [i[1] for i in coords]
        t = [i[2] for i in coords]
        H1 = 1 / (max(x) - min(x))
        H4 = 1 / (max(y) - min(y))
        H6 = 1 / (max(t) - min(t))
        H3 = 0.01
        H5 = 0.01
        H2 = 0.01
        return np.array((H1, H2, H3, H4, H5, H6))

    def get_dx(self, centroid, coords):
        x = [i[0] for i in coords]
        return np.array([i - centroid[0] for i in x])

    def get_dy(self, centroid, coords):
        y = [i[1] for i in coords]
        return np.array([i - centroid[1] for i in y])

    def get_dt(self, centroid, coords):
        t = [i[2] for i in coords]
        return np.array([i - centroid[2] for i in t])

    def plot_sum(self):
        import matplotlib.pyplot as plt

        predicted = self.func((self.dx, self.dy, self.dt), *(self.params))
        sum_predicted = []
        sum_intensities = []

        tof = []
        for idx, i in enumerate(self.coords):
            new_tof = i[2]
            if len(tof) == 0 or abs(new_tof - tof[-1]) > 1e-7:
                sum_predicted.append(predicted[idx])
                sum_intensities.append(self.intensities[idx])
                tof.append(new_tof)
            else:
                sum_predicted[-1] += predicted[idx]
                sum_intensities[-1] += self.intensities[idx]

        plt.figure(figsize=(12, 10))
        plt.plot(tof, sum_intensities, label="Observed")
        plt.plot(tof, sum_predicted, label="Fit")
        plt.legend(fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel("ToF", fontsize=20)
        plt.ylabel("Intensity", fontsize=20)
        plt.show()

    def func(self, coords, H1, H2, H3, H4, H5, H6, alpha, beta):
        H_mat = np.array(((H1, H2, H3), (H2, H4, H5), (H3, H5, H6)))
        dx, dy, dt = coords
        a = alpha
        b = beta
        N = (a * b) / (2 * (a + b))
        N_g = np.sqrt(det(H_mat)) / (2 * np.pi) ** (3 / 2.0)
        N_g = 1
        u = 0.5 * a * (a + 2 * H6 * dt + 2 * H3 * dx + 2 * H5 * dy)
        v = 0.5 * b * (b - 2 * H6 * dt - 2 * H3 * dx - 2 * H5 * dy)
        y = (a + H6 * dt + H3 * dx + H5 * dy) / (np.sqrt(2 * H6))
        w = (b - H6 * dt - H3 * dx - H5 * dy) / (np.sqrt(2 * H6))
        f1 = N * N_g * np.sqrt(np.pi / (2 * H6))
        f2 = np.exp(
            -0.5 * H1 * np.square(dx)
            - H2 * dx * dy
            - 0.5 * H4 * np.square(dy)
            + (
                (
                    np.square(H3) * np.square(dx)
                    + 2 * H3 * H5 * dx * dy
                    + np.square(H5) * np.square(dy)
                )
                / (2 * H6)
            )
        )
        f3 = np.exp(u) * erfc(y) + np.exp(v) * erfc(w)
        return f1 * f2 * f3

    def fit(self, H):
        intensities = self.intensities
        params, cov = curve_fit(
            f=self.func,
            xdata=(self.dx, self.dy, self.dt),
            ydata=intensities,
            p0=(*H, self.alpha, self.beta),
            maxfev=10000000,
        )
        self.params = params
        self.cov = cov


class ReferenceProfileGrid:
    def __init__(
        self,
        image_range: Tuple[3],
        grid_size: Tuple[3],
        subdivision: Tuple[3],
        num_panels: int,
    ):

        self.grid_size = grid_size
        self.profiles = {i: {} for i in range(num_panels)}
        self.subdivision = subdivision
        self.step_size = tuple([image_range[i] / grid_size[i] for i in range(3)])

    def get_subdivided_data(self, reflection, subdivision):
        sbox = reflection["shoebox"]

        data = flumpy.to_numpy(sbox.data)
        mask = flumpy.to_numpy(sbox.mask)

        data = self.subdivide_array(data, subdivision)
        mask = self.subdivide_array(mask, subdivision)
        return data, mask

    def add_reflection_data(self, reflections, weight_func):

        for r in range(len(reflections)):
            reflection = reflections[r]

            data, mask = self.get_subdivided_data(reflection, self.subdivision)
            if not np.sum(data) > 0:
                continue

            pxyz = reflection["xyzcal.px"]

            nearest_grid_idxs = self.get_nearest_grid_idxs(pxyz)
            weights = self.get_weights_for_idxs(nearest_grid_idxs, pxyz, weight_func)

            panel = reflection["panel"]
            for i, idx in enumerate(nearest_grid_idxs):
                if idx in self.profiles[panel]:
                    self.profiles[panel][idx] += data * weights[i] / np.sum(data)
                else:
                    self.profiles[panel][idx] = data * weights[i] / np.sum(data)

    def profile_fit(self, reflections):
        for r in range(len(reflections)):
            reflection = reflections[r]
            data, mask = self.get_subdivided_data(reflection, self.subdivision)

    def get_weights_for_idxs(self, idxs, pxyz, weight_func):
        weights = []
        for idx in idxs:
            weights.append(self.get_weight_for_idx(idx, pxyz, weight_func))
        return weights

    def get_weight_for_idx(self, idx, pxyz, weight_func):
        idx_coords = self.get_idx_panel_coords(idx)
        x = (idx_coords[0] - pxyz[0]) / self.step_size[0]
        y = (idx_coords[1] - pxyz[1]) / self.step_size[1]
        z = (idx_coords[2] - pxyz[2]) / self.step_size[2]
        distance = x * x + y * y + z * z
        return weight_func(distance)

    def get_idx_panel_coords(self, idx: Tuple[3]) -> Tuple[3]:
        return (
            idx[0] * self.step_size[0],
            idx[1] * self.step_size[1],
            idx[2] * self.step_size[2],
        )

    def get_nearest_grid_idxs(self, pxyz: Tuple[3]):
        x, y, z = pxyz
        idxs = []
        for i in (floor, ceil):
            for j in (floor, ceil):
                for k in (floor, ceil):
                    idxs.append(
                        (
                            int(i(x) / self.step_size[0]),
                            int(j(y) / self.step_size[1]),
                            int(k(z) / self.step_size[2]),
                        )
                    )
        return idxs

    def subdivide_array(self, arr, subdivision: Tuple):
        assert len(arr.shape) == len(subdivision)
        for i in range(len(subdivision)):
            arr = np.repeat(arr, [subdivision[i]], axis=i)
        arr = arr / np.product(subdivision)
        return arr


class ReferenceProfile:
    def init(self, idx: int, size: Tuple[3]):
        self.idx = idx


def print_data(reflections, panel):
    r = reflections.select(reflections["panel"] == panel)
    s = ""
    for i in range(len(r)):
        s += (
            f'{r[i]["tof"]*10**6} {r[i]["wavelength"]} {r[i]["xyzobs.px.value"]} {r[i]["intensity.sum.value"]} {np.sqrt(r[i]["intensity.sum.variance"])} {r[i]["miller_index"]}'
            + "\n"
        )
    print(s)


def output_reflections_as_hkl(reflections, filename):
    def get_corrected_intensity_and_sigma(reflections, idx):
        intensity = reflections["intensity.sum.value"][idx]
        variance = reflections["intensity.sum.variance"][idx]
        return intensity, np.sqrt(variance)

    def valid_intensity(intensity):
        from math import isinf, isnan

        if isnan(intensity) or isinf(intensity):
            return False
        return intensity > 0

    with open(filename, "w") as g:
        for i in range(len(reflections)):
            h, k, l = reflections["miller_index"][i]
            batch_number = 1
            intensity, sigma = get_corrected_intensity_and_sigma(reflections, i)
            if not valid_intensity(intensity):
                continue
            intensity = round(intensity, 2)
            sigma = round(sigma, 2)
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


def output_expt_as_ins(expt, filename):
    def LATT_SYMM(s, space_group, decimal=False):
        Z = space_group.conventional_centring_type_symbol()
        Z_dict = {
            "P": 1,
            "I": 2,
            "R": 3,
            "F": 4,
            "A": 5,
            "B": 6,
            "C": 7,
        }
        try:
            LATT_N = Z_dict[Z]
        except Exception:
            raise RuntimeError("Error: Lattice type not supported by SHELX.")
        # N must be made negative if the structure is non-centrosymmetric.
        if space_group.is_centric():
            if not space_group.is_origin_centric():
                raise RuntimeError(
                    "Error: "
                    + " SHELX manual: If the structure is centrosymmetric, the"
                    + " origin MUST lie on a center of symmetry."
                )
        else:
            LATT_N = -LATT_N
        print("LATT", LATT_N, file=s)
        # The operator x,y,z is always assumed, so MUST NOT be input.
        for i in range(1, space_group.n_smx()):
            print(
                "SYMM",
                space_group(i).as_xyz(
                    decimal=decimal, t_first=False, symbol_letters="XYZ", separator=","
                ),
                file=s,
            )

    uc = expt.crystal.get_recalculated_unit_cell() or expt.crystal.get_unit_cell()
    uc_sd = (
        expt.crystal.get_recalculated_cell_parameter_sd()
        or expt.crystal.get_cell_parameter_sd()
    )
    sg = expt.crystal.get_space_group()
    wl = 0.7

    with open(filename, "w") as f:
        f.write(
            f"TITL {sg.type().number()} in {sg.type().lookup_symbol().replace(' ','')}\n"
        )
        f.write(
            "CELL {:8.5f} {:8.4f} {:8.4f} {:8.4f} {:8.3f} {:8.3f} {:8.3f}\n".format(
                wl, *uc.parameters()
            )
        )
        if uc_sd:
            f.write(
                "ZERR {:8.3f} {:8.4f} {:8.4f} {:8.4f} {:8.3f} {:8.3f} {:8.3f}\n".format(
                    sg.order_z(), *uc_sd
                )
            )
        f.write("ZERR 4.0\n")
        LATT_SYMM(f, sg)
        f.write("SFAC Na Cl\n")
        f.write("UNIT 2 2\n")

        f.write(
            "MERG 0\nL.S. 10\nFMAP -2\nPLAN 10\nREM BASF 1 1 1 1 1 1 1 1 1 1\nREM BASF 1 1 1 1 1 1 1 1 1 1 1\nREM BASF 1 1 1 1 1 1 1 1 1 1 1\nREM BASF 1 1 1 1 1 1 1 1 1 1 1\nREM BASF 1 1 1 1 1 1 1 1 1 1 1\nREM BASF 1 1 1 1 1 1 1 1 1 1 1\nWGHT 0.1\nFVAR 1.0\n"
        )
        f.write("HKLF 2\nEND")


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

    usage = "usage: dev.dials.simple_tof_integrate.py refined.expt refined.refl"
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
    experiments.as_file(params.output.experiments)
    if params.output.output_hkl:
        output_reflections_as_hkl(integrated_reflections, params.output.hkl)


def run_simple_integrate(params, experiments, reflections):
    nproc = 11
    pool = multiprocessing.Pool(nproc)

    experiment = experiments[0]

    reflections, _ = process_reference(reflections)

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
    sel = predicted_reflections.get_flags(predicted_reflections.flags.reference_spot)
    predicted_reflections = predicted_reflections.select(sel)

    """
    Create profile model and add it to erperiment.
    This is used to predict reflection properties.
    """

    # Filter reflections to use to create the model
    used_in_ref = reflections.get_flags(reflections.flags.used_in_refinement)
    model_reflections = reflections.select(used_in_ref)

    # sigma_m in 3.1 of Kabsch 2010
    sigma_m = 0.01
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
    x1, x2, y1, y2, t1, t2 = predicted_reflections["bbox"].parts()
    predicted_reflections = predicted_reflections.select(
        t2 < experiment.sequence.get_image_range()[1]
    )
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

    predicted_reflections.experiment_identifiers()[0] = experiment.identifier

    """
    Load modeller that will calculate reference profiles and
    do the actual profile fitting integration.
    """

    sel = predicted_reflections.get_flags(predicted_reflections.flags.reference_spot)
    reference_reflections = predicted_reflections.select(sel)

    if params.method.profile_fitting is False:
        integration_report = IntegrationReport(experiments, predicted_reflections)
        logger.info("")
        logger.info(integration_report.as_str(prefix=" "))
        return predicted_reflections

    fit_method = 1  # reciprocal space fitter (called explicitly below)
    grid_method = 2  # regular grid
    grid_size = 5  # Downsampling grid size described in 3.3 of Kabsch 2010
    # Get the number of scan points
    num_scan_points = 72
    n_sigma = 4.5  # multiplier to expand bounding boxes
    fitting_threshold = 0.02
    goniometer = Goniometer()
    reference_profile_modeller = GaussianRSProfileModeller(
        experiment.beam,
        experiment.detector,
        goniometer,
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

    processes = [
        pool.apply_async(reference_profile_modeller.model_tof_return, args=(r,))
        for r in split_reflections(reference_reflections, nproc, by_panel=True)
    ]
    result = [p.get() for p in processes]
    for i in result:
        reference_profile_modeller.accumulate(i)

    reference_profile_modeller.normalize_profiles()

    profile_model_report = ProfileModelReport(
        experiments, [reference_profile_modeller], model_reflections
    )
    logger.info("")
    logger.info(profile_model_report.as_str(prefix=" "))

    """
    Carry out the integration by fitting to reference profiles in 1D.
    (Calculates intensity using 3.4 of Kabsch 2010)
    """

    sel = predicted_reflections.get_flags(predicted_reflections.flags.dont_integrate)
    sel = ~sel
    predicted_reflections = predicted_reflections.select(sel)

    # Avoid trying to model predicted reflections far from observed
    pz = reflections["xyzobs.px.value"].parts()[2]
    pred_pz = predicted_reflections["xyzcal.px"].parts()[2]
    sel = pred_pz > min(pz - 10) and pred_pz < max(pz + 10)
    predicted_reflections = predicted_reflections.select(sel)

    processes = [
        pool.apply_async(
            reference_profile_modeller.fit_reciprocal_space_tof_return, args=(r,)
        )
        for r in split_reflections(predicted_reflections, nproc, by_panel=True)
    ]
    result = [p.get() for p in processes]
    predicted_reflections = result[0]
    for i in range(1, len(result)):
        predicted_reflections.extend(result[i])
    integration_report = IntegrationReport(experiments, predicted_reflections)
    logger.info("")
    logger.info(integration_report.as_str(prefix=" "))

    """
    Remove shoeboxes
    """

    del predicted_reflections["shoebox"]
    predicted_reflections.experiment_identifiers()[0] = experiment.identifier
    return predicted_reflections


if __name__ == "__main__":
    run()
