from __future__ import annotations

from multiprocessing import Pool

import numpy as np
from scipy import integrate
from scipy.optimize import least_squares
from scipy.special import erfc

import cctbx.array_family.flex
from dxtbx import flumpy

from dials.algorithms.shoebox import MaskCode
from dials.array_family import flex


class BackToBackExponential:
    """
    https://www.nature.com/articles/srep36628.pdf
    """

    def __init__(self, tof, intensities, A, alpha, beta, sigma, T):
        # Clean the input data
        self.tof = np.nan_to_num(tof, nan=0, posinf=None, neginf=None)
        self.intensities = np.nan_to_num(intensities, nan=0, posinf=None, neginf=None)
        self.params = (A, alpha, beta, sigma, T)
        self.cov = None

    def func(self, tof, A, alpha, beta, sigma, T):
        dT = tof - T
        sigma2 = np.square(sigma)
        sigma_sqrt = np.sqrt(2 * sigma2)

        u = alpha * 0.5 * (alpha * sigma2 + 2 * dT)
        v = beta * 0.5 * (beta * sigma2 - 2 * dT)
        y = (alpha * sigma2 + dT) / sigma_sqrt
        z = (beta * sigma2 - dT) / sigma_sqrt

        N = (alpha * beta) / (2 * (alpha + beta))

        # Handle numerical stability for exponentials
        exp_u = np.exp(np.clip(u, -700, 700))
        exp_v = np.exp(np.clip(v, -700, 700))

        # Handle erfc to avoid domain errors
        erfc_y = erfc(np.clip(y, -10, 10))
        erfc_z = erfc(np.clip(z, -10, 10))

        result = A * N * (exp_u * erfc_y + exp_v * erfc_z)

        regularization = 1e-10
        return np.where(np.isfinite(result), result, regularization)

    def fit(self):
        try:

            def residuals(params):
                A, alpha, beta, sigma, T = params
                return self.intensities - self.func(self.tof, A, alpha, beta, sigma, T)

            res = least_squares(
                residuals,
                x0=self.params,
                bounds=(
                    (1, 0, 0, 1, min(self.tof)),
                    (1000000, 1, 100000, 10000000, max(self.tof)),
                ),
            )
            self.params = res.x
            self.cov = None
        except Exception:

            self.params = None
            self.cov = None
            raise ValueError

        except RuntimeWarning:
            pass

    def result(self):
        return self.func(self.tof, *(self.params))

    def calc_intensity(self):
        predicted = self.result()
        return integrate.simpson(predicted, x=self.tof)


def compute_line_profile_data_for_shoebox(
    shoebox, alpha=1.0, beta=0.2, sigma=1.0, integration_method="summation"
):

    bg_code = MaskCode.Valid | MaskCode.Background | MaskCode.BackgroundUsed

    data = flumpy.to_numpy(shoebox.data).ravel()
    background = flumpy.to_numpy(shoebox.background).ravel()
    mask = flumpy.to_numpy(shoebox.mask).ravel()
    coords = flumpy.to_numpy(shoebox.coords())

    bg_mask = (mask & bg_code) == bg_code

    foreground_mask = (mask & MaskCode.Foreground) == MaskCode.Foreground
    valid_mask = (mask & MaskCode.Valid) == MaskCode.Valid
    not_overlapped_mask = (mask & MaskCode.Overlapped) == 0
    intensity_mask = foreground_mask & valid_mask & not_overlapped_mask
    n_background = np.sum(np.bitwise_and(~intensity_mask, bg_mask))
    n_signal = np.sum(intensity_mask)

    # Remove background and project onto ToF axis
    background = background[intensity_mask]
    avg_background = sum(background) / len(background)
    intensity = data - avg_background
    background_sum = np.sum(background)
    summation_intensity = float(np.sum(intensity))
    tof = coords[:, 2]

    summed_values = {}
    summed_background_values = {}

    for j in np.unique(tof):
        indices = np.where(tof == j)
        summed_values[j] = np.sum(intensity[indices])
        summed_background_values[j] = avg_background * len(indices)

    projected_intensity = np.array(list(summed_values.values()))
    projected_background = np.array(list(summed_background_values.values()))
    tof = np.array(list(summed_values.keys()))
    if n_background > 0:
        m_n = n_signal / n_background
    else:
        m_n = 0.0
    summation_std = np.sqrt(
        abs(summation_intensity) + abs(background_sum) * (1.0 + m_n)
    )

    if integration_method == "profile1d":
        try:
            T = tof[np.argmax(projected_intensity)]
            l = BackToBackExponential(
                tof=tof,
                intensities=projected_intensity,
                A=max(5, max(projected_intensity)),
                alpha=alpha,
                beta=beta,
                sigma=sigma,
                T=T,
            )
            l.fit()
            line_profile = l.result()
            fit_intensity = integrate.simpson(line_profile, x=tof)
            fit_std = np.sqrt(abs(fit_intensity) + abs(background_sum) * (1.0 + m_n))

            return (
                tof,
                projected_intensity,
                projected_background,
                line_profile,
                fit_intensity,
                fit_std,
                summation_intensity,
                summation_std,
            )
        except ValueError as e:
            print("fit error", e)
            return (
                tof,
                projected_intensity,
                projected_background,
                [],
                -1,
                -1,
                summation_intensity,
                summation_std,
            )

    return (
        tof,
        projected_intensity,
        projected_background,
        [],
        -1,
        -1,
        summation_intensity,
        summation_std,
    )


def process_reflection(args):
    i, data, background, coords, mask, alpha, beta, sigma = args
    intensity = data.ravel()
    coords = flumpy.to_numpy(coords)

    bg_code = MaskCode.Valid | MaskCode.Background | MaskCode.BackgroundUsed
    bg_mask = (mask & bg_code) == bg_code

    foreground_mask = (mask & MaskCode.Foreground) == MaskCode.Foreground
    valid_mask = (mask & MaskCode.Valid) == MaskCode.Valid
    not_overlapped_mask = (mask & MaskCode.Overlapped) == 0
    intensity_mask = foreground_mask & valid_mask & not_overlapped_mask
    n_background = np.sum(np.bitwise_and(~intensity_mask, bg_mask))
    n_signal = np.sum(intensity_mask)

    # Remove background and project onto ToF axis
    background = background[intensity_mask]
    background_sum = np.sum(background)
    tof = coords[:, 2]

    summed_values = {}

    # Remove background and project onto ToF axis
    for j in np.unique(tof):
        indices = np.where(tof == j)
        summed_values[j] = np.sum(intensity[indices])

    projected_intensity = np.array(list(summed_values.values()))
    tof = np.array(list(summed_values.keys()))

    fit_intensity = None
    try:
        T = tof[np.argmax(projected_intensity)]
        l = BackToBackExponential(
            tof=tof,
            intensities=projected_intensity,
            A=max(5, max(projected_intensity)),
            alpha=alpha,
            beta=beta,
            sigma=sigma,
            T=T,
        )
        l.fit()
        fit_intensity = l.calc_intensity()
    except RuntimeWarning:
        return i, -1, -1
    except ValueError:
        return i, -1, -1

    if n_background > 0:
        m_n = n_signal / n_background
    else:
        m_n = 0.0

    fit_variance = abs(fit_intensity) + abs(background_sum) * (1.0 + m_n)
    return i, fit_intensity, fit_variance


def compute_line_profile_intensity(reflections, nproc=8):
    alpha = 1.0
    beta = 0.2
    sigma = 1.0

    fit_intensities = cctbx.array_family.flex.double(len(reflections))
    fit_variances = cctbx.array_family.flex.double(len(reflections))

    args = [
        (
            i,
            flumpy.to_numpy(
                reflections[i]["shoebox"].data - reflections["background.mean"][i]
            ),
            flumpy.to_numpy(reflections[i]["shoebox"].background),
            flumpy.to_numpy(reflections[i]["shoebox"].coords()),
            flumpy.to_numpy(reflections[i]["shoebox"].mask),
            alpha,
            beta,
            sigma,
        )
        for i in range(len(reflections))
    ]

    with Pool(processes=nproc) as pool:
        results = pool.map(process_reflection, args)

    for i, fit_intensity, fit_variance in results:
        fit_intensities[i] = fit_intensity
        fit_variances[i] = fit_variance

    reflections["intensity.prf.value"] = fit_intensities
    reflections["intensity.prf.variance"] = fit_variances
    reflections.set_flags(
        reflections["intensity.prf.value"] < 0,
        reflections.flags.failed_during_profile_fitting,
    )

    i_sig_sum = reflections["intensity.sum.value"] / flex.sqrt(
        reflections["intensity.sum.variance"]
    )
    i_sig_prf = reflections["intensity.prf.value"] / flex.sqrt(
        reflections["intensity.prf.variance"]
    )

    reflections.set_flags(
        i_sig_sum > i_sig_prf,
        reflections.flags.failed_during_profile_fitting,
    )

    reflections.set_flags(
        (reflections["intensity.prf.value"] > 0) & (i_sig_prf > i_sig_sum),
        reflections.flags.integrated_prf,
    )
    return reflections
