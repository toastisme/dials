from __future__ import annotations

import json
import logging
from bisect import bisect_left
from copy import deepcopy
from enum import Enum
from typing import Dict, Tuple

import gemmi
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.chebyshev import chebval
from scipy.optimize import least_squares

from dxtbx import flumpy
from dxtbx.serialize import load
from scitbx.matrix import col

from dials.array_family import flex
from dials.array_family.flex import reflection_table

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger("dials.command_line.tof_wavelength_normalization")


class IntegrationType(Enum):
    summation = 0
    profile = 1


class ReflectionGroup:

    """
    A set of reflections related by symmetry that should have the same
    intensities.
    """

    def __init__(
        self,
        reflections: reflection_table,
        init_f_lambda: np.ndarray,
        init_b_image: np.ndarray,
        init_s_image: np.ndarray,
        normalized_wavelength_bins: Tuple[float, ...],
        integration_type: IntegrationType = IntegrationType.summation,
        regularization_factor: float = 2,
    ):

        self.reflections = reflections
        self.wl_bin_idxs = self.get_wl_bin_idxs(normalized_wavelength_bins)
        self.f_lambda = init_f_lambda
        self.s_image = init_s_image
        self.b_image = init_b_image
        self.integration_type = integration_type
        self.raw_intensities = self.get_raw_intensities()
        self.raw_intensities_sigma = self.get_raw_intensities_sigma()
        self.regularization_factor = regularization_factor

    def get_wl_bin_idxs(self, normalized_wavelength_bins: Tuple[float]) -> Tuple[int]:

        """
        Which wavelength bin each reflection falls into
        """

        bin_idxs = []
        for i in range(len(self.reflections)):
            idx = TOFWavelengthNormalizer.get_wavelength_bin_idx(
                self.reflections[i]["normalized_wavelength_cal"],
                normalized_wavelength_bins,
            )
            bin_idxs.append(idx)
        return tuple(bin_idxs)

    def get_raw_intensities(self) -> flex.double:
        if self.integration_type == IntegrationType.profile:
            return flumpy.to_numpy(self.reflections["intensity.prf.value"])
        else:
            return flumpy.to_numpy(self.reflections["intensity.sum.value"])

    def get_raw_intensities_sigma(self) -> flex.double:
        if self.integration_type == IntegrationType.profile:
            return flumpy.to_numpy(
                flex.sqrt(self.reflections["intensity.prf.variance"])
            )
        else:
            return flumpy.to_numpy(
                flex.sqrt(self.reflections["intensity.sum.variance"])
            )

    def get_f_image(self, refl_idx: int) -> float:

        """
        (8) of section 5.5
        """

        image_idx = self.reflections["imageset_id"][refl_idx]
        if image_idx == 0:
            return 1.0

        wl = self.reflections["normalized_wavelength_cal"][refl_idx]
        f_l = self.reflections["lorentz_factor"][refl_idx]
        s_i = self.s_image[image_idx]
        b_i = self.b_image[image_idx]
        return s_i * np.exp(-2 * b_i * (f_l / np.square(wl)))

    def get_corrected_intensities(self, return_corrections: bool = False) -> np.ndarray:
        intensities = []
        corrections = []
        for i in range(len(self.raw_intensities)):
            correction = self.reflections["lorentz_factor"][i]
            correction *= self.f_lambda[self.wl_bin_idxs[i]]
            correction *= self.get_f_image(i)
            intensity = self.raw_intensities[i] * correction
            intensities.append(intensity)
            corrections.append(correction)
        if return_corrections:
            return np.array(intensities), np.array(corrections)
        return np.array(intensities)

    def get_corrected_intensities_sigma(self) -> np.ndarray:
        sigma_vals = []
        for i in range(len(self.raw_intensities_sigma)):
            sigma_val = (
                self.reflections["lorentz_factor"][i] * self.raw_intensities_sigma[i]
            )
            sigma_val *= self.f_lambda[self.wl_bin_idxs[i]]
            sigma_val *= self.get_f_image(i)
            sigma_vals.append(sigma_val)
        return np.array(sigma_vals)

    def get_corrected_avg_intensity(self) -> float:
        corrected_intensities = self.get_corrected_intensities()
        return sum(corrected_intensities) / len(corrected_intensities)

    def get_residual(self) -> float:

        """
        (2) of section 3.
        """

        val = 0
        intensities = self.get_corrected_intensities()
        sigmas = self.get_corrected_intensities_sigma()
        avg_intensity = self.get_corrected_avg_intensity()
        avg_raw_intensity = self.get_raw_avg_intensity()

        for idx, i in enumerate(intensities):
            if sigmas[idx] > 0 and i > 0:
                val += np.square((i - avg_intensity) / (sigmas[idx] + 1e-3))

        constraint = np.square(avg_intensity - avg_raw_intensity)

        return val + (constraint * self.regularization_factor)

    def get_raw_avg_intensity(self) -> float:
        return sum(self.raw_intensities) / len(self.raw_intensities)

    def get_raw_residual(self) -> float:
        val = 0
        intensities = self.get_raw_intensities()
        sigmas = self.get_raw_intensities_sigma()
        avg_intensity = self.get_raw_avg_intensity()
        for idx, i in enumerate(intensities):
            val += np.square((i - avg_intensity) / (sigmas[idx]))
        return val


class TOFWavelengthNormalizer:

    """
    Wavelength normalisation for time-of-flight data

    Params are taken from
    Azrt S., et al. (1999), LSCALE - the new normalization, scaling and
    absorption correction program in the Daresbury Laue software suite
    J. Appl. Cryst. (1999). 32, 554-562

    Corresponding variables to sections in the paper:
    f_lorentz: 5.1
    s_image: 5.5
    b_image: 5.5
    f_lambda: 5.7
    """

    def __init__(
        self,
        experiments_path: str,
        reflections_path: str,
        lambda_polynomial_degree: int = 10,
        num_lambda_bins: int = 200,
        s_image: np.ndarray | None = None,
        b_image: np.ndarray | None = None,
        f_lambda_coeffs: np.ndarray | None = None,
        integration_type: IntegrationType = IntegrationType.summation,
        min_partiality: float = 0.99,
        min_i_sigma: float = 1.0,
        min_i: float = 1.0,
        ref_wavelength=None,
        intensity_scaling_factor: int = 1000,
        regularization_factor: float = 2,
    ) -> None:

        logger.info("Setting up scaler..")

        # Data
        self.integration_type = integration_type
        self.intensity_scaling_factor = intensity_scaling_factor
        self.regularization_factor = regularization_factor
        self.experiments_file_path = experiments_path
        self.reflections_file_path = reflections_path
        self.experiments = load.experiment_list(experiments_path)
        self.raw_reflections = reflection_table.from_msgpack_file(reflections_path)
        self.reflections = reflection_table.from_msgpack_file(reflections_path)
        logger.info(
            f"Loaded {len(self.reflections)} reflections from {len(self.experiments)} experiment(s)"
        )
        self.reflections = self.get_filtered_reflections(
            min_partiality=min_partiality,
            min_i_sigma=min_i_sigma,
            min_i=min_i,
        )
        self.multiply_reflection_intensities_by_scaling_factor()

        wl_range = self.get_wavelength_range()

        logger.info(f"Wavelength range from all reflections: {wl_range}")
        logger.info(
            f"Wavelength normalisation will be generated for {num_lambda_bins} bins across this range"
        )
        logger.info(f"Scaling using intensities from {integration_type}")

        self.normalized_wavelength_bins = (
            TOFWavelengthNormalizer.get_normalized_wavelength_bins(
                wl_range, num_bins=num_lambda_bins
            )
        )

        # Optimization params
        self.num_optimization_iterations = 0
        self.residual_history = []
        self.save_filename = None
        self.f_lorentz = None  # Size of len(self.reflections)
        self.s_image = None
        self.b_image = None
        self.ref_wavelength = None
        self.f_lambda_coeffs = None
        self.setup_optimization_params(
            f_lambda_coeffs=f_lambda_coeffs,
            s_image=s_image,
            b_image=b_image,
            ref_wavelength=ref_wavelength,
            lambda_polynomial_degree=lambda_polynomial_degree,
            wl_range=wl_range,
        )

        self.reflection_groups = self.get_reflection_groups()

        logger.info("Finished setup")

    def setup_optimization_params(
        self,
        f_lambda_coeffs: np.ndarray,
        s_image: np.ndarray,
        b_image: np.ndarray,
        ref_wavelength: float,
        lambda_polynomial_degree: int,
        wl_range: Tuple[float, ...],
    ):

        if s_image is not None:
            logger.info("Starting from previous s_image values")
            self.s_image = s_image  # Size of len(imageset_ids) + 1
        else:
            self.s_image = np.ones(max(self.reflections["imageset_id"]) + 1)

        if b_image is not None:
            logger.info("Starting from previous b_image values")
            self.b_image = b_image  # Size of len(imageset_ids) + 1
        else:
            self.b_image = np.ones(max(self.reflections["imageset_id"]) + 1)

        if ref_wavelength is not None:
            logger.info(f"Reference wavelength set as {ref_wavelength} (A)")
            self.ref_wavelength = ref_wavelength
            self.normalized_ref_wavelength = (
                TOFWavelengthNormalizer.normalize_wavelength(
                    ref_wavelength, wl_range[0], wl_range[1]
                )
            )
        else:
            ref_wl = round((wl_range[0] + wl_range[1]) * 0.5, 3)
            logger.info(
                f"Reference wavelength set by default as half the range ({ref_wl} (A))"
            )
            self.ref_wavelength = ref_wl
            self.normalized_ref_wavelength = (
                TOFWavelengthNormalizer.normalize_wavelength(
                    ref_wl, wl_range[0], wl_range[1]
                )
            )
            self.ref_wavelength_bin_idx = (
                TOFWavelengthNormalizer.get_wavelength_bin_idx(
                    self.normalized_ref_wavelength, self.normalized_wavelength_bins
                )
            )
        if f_lambda_coeffs is not None:
            logger.info("Starting from previous f_lambda_coeffs values")
            self.f_lambda_coeffs = (
                f_lambda_coeffs  # Size of len(lambda_polynomial_degree) + 1
            )
        else:
            self.f_lambda_coeffs = np.ones(lambda_polynomial_degree + 1)
        self.f_lambda = self.get_f_lambda()

        logger.info(f"Using a lambda polynomial degree of {lambda_polynomial_degree}")
        self.lambda_polynomial_degree = lambda_polynomial_degree

    def optimize(
        self,
        save_filename: str = "checkpoint",
        diff_step: float = 0.001,
        max_nfev: int = 32,
    ) -> None:

        self.save_filename = save_filename
        params = np.concatenate((self.f_lambda_coeffs, self.s_image, self.b_image))
        result = least_squares(
            self.residual, params, diff_step=diff_step, max_nfev=max_nfev
        )

        self.residual(result.x)

    def get_f_lambda(self) -> np.ndarray:
        g_lambda = chebval(self.normalized_wavelength_bins, self.f_lambda_coeffs)
        ref_g_lambda = chebval(
            [self.normalized_wavelength_bins[self.ref_wavelength_bin_idx]],
            self.f_lambda_coeffs,
        )
        f_lambda = np.exp(g_lambda) / (np.exp(ref_g_lambda) + 1e-7)
        return f_lambda

    def update_reflection_groups_refine_params(
        self, f_lambda: np.ndarray, s_image: np.ndarray, b_image: np.ndarray
    ) -> None:
        for i in self.reflection_groups:
            self.reflection_groups[i].f_lambda = f_lambda
            self.reflection_groups[i].s_image = s_image
            self.reflection_groups[i].b_image = b_image

    def residual(self, params: np.ndarray) -> None:

        num_f_lambda = len(self.f_lambda_coeffs)
        num_s_image = len(self.s_image)

        self.f_lambda_coeffs = params[:num_f_lambda]
        self.s_image = params[num_f_lambda : num_f_lambda + num_s_image]
        self.b_image = params[num_f_lambda + num_s_image :]

        self.f_lambda = self.get_f_lambda()

        logger.debug("New params: ")
        logger.debug(f"f_lambda: {self.f_lambda}")
        logger.debug(f"s_image: {self.s_image}")
        logger.debug(f"b_image: {self.b_image}")

        self.update_reflection_groups_refine_params(
            self.f_lambda, self.s_image, self.b_image
        )

        if self.save_filename is not None:
            if self.num_optimization_iterations == 0:
                filename = f"{self.save_filename}_params.json"
                self.save_optimization_params(filename)

        residuals = []
        for i in self.reflection_groups:
            residuals.append(self.reflection_groups[i].get_residual())

        avg_residual = sum(residuals) / len(residuals)
        logger.info(f"{self.num_optimization_iterations} residual: {avg_residual}")
        self.residual_history.append(avg_residual)
        self.num_optimization_iterations += 1
        return np.array(residuals)

    def save_optimization_params(self, filename: str) -> None:
        logger.info(f"Saving optimization params to {filename}")
        with open(filename, "w") as f:
            json.dump(
                {
                    "f_lambda_coeffs": self.f_lambda_coeffs.tolist(),
                    "s_image": self.s_image.tolist(),
                    "b_image": self.b_image.tolist(),
                },
                f,
            )

    def load_optimization_params(self, filename: str) -> None:
        logger.info(f"Loading optimization params from {filename}")
        with open(filename, "r") as f:
            params = json.load(f)
            self.f_lambda_coeffs = np.array(params["f_lambda_coeffs"])
            self.s_image = np.array(params["s_image"])
            self.b_image = np.array(params["b_image"])
            self.f_lambda = self.get_f_lambda()
            self.update_reflection_groups_refine_params(
                self.f_lambda, self.s_image, self.b_image
            )

    def get_lorentz_factors(self) -> flex.double:

        """
        Computes sin^2(theta) for all xyzcal.px positions of self.reflections
        """

        lorentz_factors = flex.double(len(self.reflections))
        # Assume all experiments share the same detector model
        detector = self.experiments[0].detector
        # Assume all experiments share the same detector model
        unit_s0 = col(self.experiments[0].beam.get_unit_s0())

        for idx, panel in enumerate(detector):
            sel = self.reflections["panel"] == idx
            panel_refl = self.reflections.select(sel)
            panel_lorentz_factor = flex.double(len(panel_refl))
            x, y, _ = panel_refl["xyzcal.px"].parts()
            panel_xy = flex.vec2_double(x, y)
            for i in range(len(panel_refl)):
                theta = panel.get_two_theta_at_pixel(unit_s0, panel_xy[i]) * 0.5
                panel_lorentz_factor[i] = np.square(np.sin(theta))
            lorentz_factors.set_selected(sel, panel_lorentz_factor)
        return lorentz_factors

    def get_filtered_reflections(
        self,
        min_partiality: float = 0.99,
        min_i_sigma: float = 1.0,
        min_i: float = 1.0,
    ) -> reflection_table:

        sel = self.reflections["partiality"] > min_partiality
        reflections = self.reflections.select(
            self.reflections["partiality"] > min_partiality
        )
        logger.info(
            f"Removed {sel.count(False)} reflections with partiality < {min_partiality}"
        )

        if self.integration_type == IntegrationType.profile:
            sel = ~reflections.get_flags(
                reflections.flags.failed_during_profile_fitting
            )
            reflections = reflections.select(
                ~reflections.get_flags(reflections.flags.failed_during_profile_fitting)
            )
            logger.info(
                f"Removed {sel.count(False)} reflections that failed during profile fitting"
            )

            sel = reflections["intensity.prf.value"] > min_i

            reflections = reflections.select(sel)
            logger.info(
                f"Removed {sel.count(False)} reflections with intensity < {min_i}"
            )
            i_sigma = reflections["intensity.sum.value"] / flex.sqrt(
                reflections["intensity.sum.variance"]
            )
            sel = i_sigma > min_i_sigma
            reflections = reflections.select(sel)
            logger.info(
                f"Removed {sel.count(False)} reflections with i/sigma < {min_i_sigma}"
            )
        else:
            sel = reflections["intensity.sum.value"] > min_i
            reflections = reflections.select(sel)
            logger.info(
                f"Removed {sel.count(False)} reflections with intensity < {min_i}"
            )
            i_sigma = reflections["intensity.sum.value"] / flex.sqrt(
                reflections["intensity.sum.variance"]
            )
            sel = i_sigma > min_i_sigma
            reflections = reflections.select(sel)
            logger.info(
                f"Removed {sel.count(False)} reflections with i/sigma < {min_i_sigma}"
            )
        logger.info(f"Number of reflections after filtering: {len(reflections)}")
        return reflections

    def multiply_reflection_intensities_by_scaling_factor(self) -> None:

        if self.intensity_scaling_factor != 1:
            logger.info(f"Scaling intensities by {self.intensity_scaling_factor}")
        self.reflections["intensity.sum.value"] = (
            self.reflections["intensity.sum.value"] * self.intensity_scaling_factor
        )
        self.reflections["intensity.sum.variance"] = (
            self.reflections["intensity.sum.variance"]
            * self.intensity_scaling_factor**2
        )
        logger.info(
            f"Summation intensity range ({round(min(self.reflections['intensity.sum.value']),3)} - {round(max(self.reflections['intensity.sum.value']),3)})"
        )
        logger.info(
            f"Summation variance range ({round(min(self.reflections['intensity.sum.variance']),3)} - {round(max(self.reflections['intensity.sum.variance']),3)})"
        )

        if self.integration_type == IntegrationType.profile:
            self.reflections["intensity.prf.value"] = (
                self.reflections["intensity.prf.value"] * self.intensity_scaling_factor
            )
            self.reflections["intensity.prf.variance"] = (
                self.reflections["intensity.prf.variance"]
                * self.intensity_scaling_factor**2
            )
            logger.info(
                f"Profile intensity range ({round(min(self.reflections['intensity.prf.value']),3)} - {round(max(self.reflections['intensity.prf.value']),3)})"
            )
            logger.info(
                f"Profile variance range ({round(min(self.reflections['intensity.prf.variance']),3)} - {round(max(self.reflections['intensity.prf.variance']),3)})"
            )

    def get_scaled_reflections(self) -> reflection_table:
        def get_f_image(reflection, s_image, b_image):
            image_idx = reflection["imageset_id"]
            if image_idx == 0:
                return 1.0
            wl = reflection["normalized_wavelength_cal"]
            f_l = reflection["lorentz_factor"]
            s_i = s_image[image_idx]
            b_i = b_image[image_idx]
            return s_i * np.exp(-2 * b_i * (f_l / np.square(wl)))

        scaled_reflections = deepcopy(self.reflections)
        scaled_intensity = flex.double(len(scaled_reflections))
        scaled_variance = flex.double(len(scaled_reflections))
        scaled_prf_intensity = flex.double(len(scaled_reflections))
        scaled_prf_variance = flex.double(len(scaled_reflections))

        wavelengths = scaled_reflections["normalized_wavelength_cal"]

        for i in range(len(self.reflections)):
            prf_intensity = None
            prf_variance = None
            intensity = self.reflections["intensity.sum.value"][i]
            variance = self.reflections["intensity.sum.variance"][i]
            if self.integration_type == IntegrationType.profile:
                prf_intensity = self.reflections["intensity.prf.value"][i]
                prf_variance = self.reflections["intensity.prf.variance"][i]

            wl_idx = TOFWavelengthNormalizer.get_wavelength_bin_idx(
                wavelengths[i], self.normalized_wavelength_bins
            )
            f_i = get_f_image(scaled_reflections[i], self.s_image, self.b_image)
            f_l = scaled_reflections["lorentz_factor"][i]
            scaled_intensity[i] = f_l * self.f_lambda[wl_idx] * f_i * intensity
            scaled_variance[i] = np.square(f_l * self.f_lambda[wl_idx] * f_i) * variance
            if self.integration_type == IntegrationType.profile:
                scaled_prf_intensity[i] = (
                    f_l * self.f_lambda[wl_idx] * f_i * prf_intensity
                )
                scaled_prf_variance[i] = (
                    np.square(f_l * self.f_lambda[wl_idx] * f_i) * prf_variance
                )

        if self.integration_type == IntegrationType.profile:
            scaled_reflections["intensity.prf.value"] = scaled_prf_intensity
            scaled_reflections["intensity.prf.variance"] = scaled_prf_variance

        scaled_reflections["wavelength_cal"] = flex.double(
            len(scaled_reflections), self.ref_wavelength
        )
        scaled_reflections["intensity.sum.value"] = scaled_intensity
        scaled_reflections["intensity.sum.variance"] = scaled_variance
        return scaled_reflections

    def get_equivalent_intensity_sets(self) -> Tuple[Tuple[float, ...], ...]:

        """
        Returns indices in self.reflections grouped by equivalence by symmetry
        """

        def get_equivalent_hkls(hkl, space_group):
            # Generate all symmetry-equivalent hkls
            equiv_hkls = []
            for op in space_group.operations():
                sym_hkl = tuple(op.apply_to_hkl(hkl))
                equiv_hkls.append(sym_hkl)
            return equiv_hkls

        # Initialize space group
        # Assume one crystal shared by all experiments
        space_group_name = (
            self.experiments[0]
            .crystal.get_space_group()
            .type()
            .universal_hermann_mauguin_symbol()
        )
        space_group = gemmi.SpaceGroup(space_group_name)
        equiv_hkl_map = {}

        # Build map of equivalent hkls
        for i, hkl in enumerate(self.reflections["miller_index"]):

            # Generate all symmetry-equivalent HKLs
            equiv_hkls = get_equivalent_hkls(hkl, space_group)
            for sym_hkl in equiv_hkls:
                if sym_hkl not in equiv_hkl_map:
                    equiv_hkl_map[sym_hkl] = []
                equiv_hkl_map[sym_hkl].append(i)

        # Group reflections into equivalence groups
        equivalence_groups = []
        seen_reflections = set()

        for i, hkl in enumerate(self.reflections["miller_index"]):

            if hkl not in equiv_hkl_map:
                logger.debug(f"{hkl} not found in equiv_hkl_map")
                continue

            if i in seen_reflections:
                continue

            # Collect indices of equivalent reflections
            group = []
            for idx in equiv_hkl_map[hkl]:
                if idx not in seen_reflections:
                    group.append(idx)
                    seen_reflections.add(idx)

            if len(group) > 1:
                equivalence_groups.append(group)

        return tuple(equivalence_groups)

    def get_wavelength_range(self) -> Tuple[float, float]:
        wls = self.reflections["wavelength_cal"]
        return (round(min(wls), 3), round(max(wls), 3))

    def add_normalized_wavelengths_to_reflections(self) -> None:
        wls = self.reflections["wavelength_cal"]
        wl_range = self.get_wavelength_range()
        normalized_wls = flex.double(len(wls))
        for i in range(len(wls)):
            normalized_wls[i] = TOFWavelengthNormalizer.normalize_wavelength(
                wls[i], wl_range[0], wl_range[1]
            )
        self.reflections["normalized_wavelength_cal"] = normalized_wls

    def add_lorentz_factors_to_reflections(self) -> None:
        logger.info("Adding lorentz factors to reflections")
        lorentz_factors = self.get_lorentz_factors()
        self.reflections["lorentz_factor"] = lorentz_factors

    def get_reflection_groups(self) -> Dict[Tuple, ReflectionGroup]:

        logger.info("Splitting reflections by symmetry..")

        idx_groups = self.get_equivalent_intensity_sets()
        if "lorentz_factor" not in self.reflections:
            self.add_lorentz_factors_to_reflections()
        if "normalized_wavelength_cal" not in self.reflections:
            self.add_normalized_wavelengths_to_reflections()

        logger.info(
            f"Identified {len(idx_groups)} groups of reflections equivalent by symmetry"
        )
        reflection_groups = {}
        for idx_group in idx_groups:
            flex_idx_group = flex.size_t(idx_group)
            sel_refls = self.reflections.select(flex_idx_group)
            refl_group = ReflectionGroup(
                reflections=sel_refls,
                init_f_lambda=self.f_lambda,
                init_s_image=self.s_image,
                init_b_image=self.b_image,
                normalized_wavelength_bins=self.normalized_wavelength_bins,
                integration_type=self.integration_type,
                regularization_factor=self.regularization_factor,
            )

            reflection_groups[tuple(idx_group)] = refl_group
        return reflection_groups

    def get_mean_intensities_per_wavelength_bin(
        self, raw_intensities: bool = False
    ) -> np.ndarray:
        mean_intensities = np.zeros(len(self.normalized_wavelength_bins))
        mean_intensities_count = np.zeros(len(self.normalized_wavelength_bins))

        for i in self.reflection_groups:
            if raw_intensities:
                corrected_intensities = self.reflection_groups[i].get_raw_intensities()
            else:
                corrected_intensities = self.reflection_groups[
                    i
                ].get_corrected_intensities()
            bin_idxs = self.reflection_groups[i].wl_bin_idxs
            for j in range(len(corrected_intensities)):
                mean_intensities[bin_idxs[j]] += corrected_intensities[j]
                mean_intensities_count[bin_idxs[j]] += 1

        mean_intensities = np.divide(
            mean_intensities, mean_intensities_count, where=mean_intensities_count > 0
        )
        return mean_intensities

    def get_errors_per_wavelength_bin(
        self, raw_intensities: bool = False
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        mean_intensities = self.get_mean_intensities_per_wavelength_bin(
            raw_intensities=raw_intensities
        )
        errors_y = []
        errors_x = []

        for i in self.reflection_groups:
            if raw_intensities:
                intensities = self.reflection_groups[i].get_raw_intensities()
            else:
                intensities = self.reflection_groups[i].get_corrected_intensities()
            bin_idxs = self.reflection_groups[i].wl_bin_idxs
            for j in range(len(intensities)):
                errors_x.append(self.normalized_wavelength_bins[bin_idxs[j]])
                i_mean = mean_intensities[bin_idxs[j]]
                errors_y.append((intensities[j] - i_mean) / i_mean)

        return tuple(errors_x), tuple(errors_y)

    def plot_errors_per_wavelength_bin(self, raw_intensities: bool = False) -> None:
        errors_x, errors_y = self.get_errors_per_wavelength_bin(
            raw_intensities=raw_intensities
        )
        plt.plot(errors_x, errors_y, "o", markersize=2)
        plt.plot(
            self.normalized_wavelength_bins,
            [0 for i in range(len(self.normalized_wavelength_bins))],
            color="black",
        )
        plt.xlabel("Normalized wavelength (AU)")
        plt.ylabel(r"$(I_{scaled} - I_{mean})/I_{mean}$")
        plt.show()

    def plot_wavelength_normalization_curve(
        self,
        scale_factor: float = 1,
        xlim: Tuple[float, float] | None = None,
        ylim: Tuple[float, float] | None = None,
        raw_intensities: bool = False,
    ) -> None:

        mean_intensities = self.get_mean_intensities_per_wavelength_bin(
            raw_intensities=raw_intensities
        )

        plt.scatter(
            self.normalized_wavelength_bins,
            mean_intensities * scale_factor,
            label="Normalized Mean Intensity",
        )
        plt.plot(
            self.normalized_wavelength_bins,
            self.f_lambda,
            label="Chebyshev Fit",
            color="red",
        )
        plt.xlabel("Normalized Wavelength (AU)")
        plt.ylabel("Mean Intensity (AU)")
        if xlim is not None:
            plt.xlim(xlim[0], xlim[1])
        if ylim is not None:
            plt.ylim(ylim[0], ylim[1])
        plt.legend()
        plt.grid()
        plt.show()

    def plot_residual_history(self):
        plt.plot(list(range(len(self.residual_history))), self.residual_history)
        plt.xlabel("Num Iterations")
        plt.ylabel("Residual (AU)")
        plt.show()

    @staticmethod
    def normalize_wavelength(
        wavelength: float, min_wavelength: float, max_wavelength: float
    ) -> float:
        return (2 * wavelength - max_wavelength - min_wavelength) / (
            max_wavelength - min_wavelength
        )

    @staticmethod
    def get_wavelength_bin_idx(wavelength: float, wavelength_bins: Tuple) -> int:
        idx = bisect_left(wavelength_bins, wavelength) - 1
        return max(0, idx)

    @staticmethod
    def get_wavelength_bins(
        wavelength_range: Tuple[float, float], num_bins: int
    ) -> Tuple[float, ...]:

        delta_wavelength = (wavelength_range[1] - wavelength_range[0]) / num_bins
        wavelength_bins = []
        for i in range(num_bins):
            wavelength_bins.append(wavelength_range[0] + (delta_wavelength * i))

        return wavelength_bins

    @staticmethod
    def get_normalized_wavelength_bins(
        wavelength_range: Tuple[float, float], num_bins: int
    ) -> Tuple[float, ...]:

        delta_wavelength = (wavelength_range[1] - wavelength_range[0]) / num_bins
        wavelength_bins = []
        for i in range(num_bins):
            wl_bin = wavelength_range[0] + (delta_wavelength * i)
            normalized_wl_bin = TOFWavelengthNormalizer.normalize_wavelength(
                wl_bin, wavelength_range[0], wavelength_range[1]
            )
            wavelength_bins.append(normalized_wl_bin)

        return wavelength_bins
