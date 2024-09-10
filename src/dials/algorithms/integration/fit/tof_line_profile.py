from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.optimize import least_squares
from scipy.special import erfc

import cctbx.array_family.flex
from dxtbx import flumpy

from dials.algorithms.shoebox import MaskCode


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
            # Use least_squares for robustness
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
            # Covariance might not be available in least_squares
            self.cov = None
        except Exception as e:
            print(f"An error occurred during fitting: {e}")
            self.params = None
            self.cov = None

    def result(self):
        return self.func(self.tof, *(self.params))

    def calc_intensity(self):
        predicted = self.result()
        return integrate.simpson(predicted, x=self.tof)


def compute_line_profile_data_for_reflection(
    reflection_table, A=200.0, alpha=0.4, beta=0.4, sigma=8.0
):

    assert len(reflection_table) == 1

    bg_code = MaskCode.Valid | MaskCode.Background | MaskCode.BackgroundUsed

    shoebox = reflection_table["shoebox"][0]
    data = flumpy.to_numpy(shoebox.data).ravel()
    background = flumpy.to_numpy(shoebox.background).ravel()
    mask = flumpy.to_numpy(shoebox.mask).ravel()
    coords = flumpy.to_numpy(shoebox.coords())
    m = mask & MaskCode.Foreground == MaskCode.Foreground
    bg_m = mask & bg_code == bg_code
    n_background = np.sum(np.bitwise_and(~m, bg_m))

    m = np.bitwise_and(m, mask & MaskCode.Valid == MaskCode.Valid)
    m = np.bitwise_and(m, mask & MaskCode.Overlapped == 0)

    n_signal = np.sum(m)

    background = background[m]
    intensity = data[m] - background
    background_sum = np.sum(background)
    summation_intensity = float(np.sum(intensity))
    coords = coords[m]
    tof = coords[:, 2]

    summed_values = {}
    summed_background_values = {}

    for j in np.unique(tof):
        indices = np.where(tof == j)
        summed_values[j] = np.sum(intensity[indices])
        summed_background_values[j] = np.sum(background[indices])

    # Remove background and project onto ToF axis
    projected_intensity = np.array(list(summed_values.values()))
    projected_background = np.array(list(summed_background_values.values()))
    tof = np.array(list(summed_values.keys()))

    try:
        T = tof[np.argmax(projected_intensity)]
        l = BackToBackExponential(
            tof=tof,
            intensities=projected_intensity,
            A=max(projected_intensity),
            alpha=alpha,
            beta=beta,
            sigma=sigma,
            T=T,
        )
        l.fit()
        line_profile = l.result()
        fit_intensity = integrate.simpson(line_profile, x=tof)
    except ValueError as e:
        print("fit error", e)
        return [], [], [], [], -1, -1, -1, -1

    if n_background > 0:
        m_n = n_signal / n_background
    else:
        m_n = 0.0
    fit_std = np.sqrt(abs(fit_intensity) + abs(background_sum) * (1.0 + m_n))
    summation_std = np.sqrt(
        abs(summation_intensity) + abs(background_sum) * (1.0 + m_n)
    )

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


def compute_line_profile_intensity(reflections):

    # A = 200.0
    alpha = 0.4
    beta = 0.4
    sigma = 8.0

    bg_code = MaskCode.Valid | MaskCode.Background | MaskCode.BackgroundUsed

    fit_intensities = cctbx.array_family.flex.double(len(reflections))
    fit_variances = cctbx.array_family.flex.double(len(reflections))

    for i in range(len(reflections)):
        shoebox = reflections[i]["shoebox"]
        data = flumpy.to_numpy(shoebox.data).ravel()
        background = flumpy.to_numpy(shoebox.background).ravel()
        mask = flumpy.to_numpy(shoebox.mask).ravel()
        coords = flumpy.to_numpy(shoebox.coords())
        m = mask & MaskCode.Foreground == MaskCode.Foreground
        bg_m = mask & bg_code == bg_code
        n_background = np.sum(np.bitwise_and(~m, bg_m))

        m = np.bitwise_and(m, mask & MaskCode.Valid == MaskCode.Valid)
        m = np.bitwise_and(m, mask & MaskCode.Overlapped == 0)

        n_signal = np.sum(m)

        background = background[m]
        intensity = data[m] - background
        background_sum = np.sum(background)
        coords = coords[m]
        tof = coords[:, 2]

        summed_values = {}

        for j in np.unique(tof):
            indices = np.where(tof == j)
            summed_values[j] = np.sum(intensity[indices])

        # Remove background and project onto ToF axis
        projected_intensity = np.array(list(summed_values.values()))
        tof = np.array(list(summed_values.keys()))

        fit_intensity = None
        try:
            T = tof[np.argmax(projected_intensity)]
            l = BackToBackExponential(
                tof=tof,
                intensities=projected_intensity,
                A=max(projected_intensity),
                alpha=alpha,
                beta=beta,
                sigma=sigma,
                T=T,
            )
            l.fit()
            fit_intensity = l.calc_intensity()
            fit_intensities[i] = fit_intensity
        except ValueError:
            fit_intensities[i] = -1
            fit_variances[i] = -1
            continue

        if n_background > 0:
            m_n = n_signal / n_background
        else:
            m_n = 0.0
        fit_variance = abs(fit_intensity) + abs(background_sum) * (1.0 + m_n)
        fit_variances[i] = fit_variance

    reflections["intensity.prf.value"] = fit_intensities
    reflections["intensity.prf.variance"] = fit_variances
    reflections.set_flags(
        reflections["intensity.prf.value"] < 0,
        reflections.flags.failed_during_profile_fitting,
    )
    reflections.set_flags(
        reflections["intensity.prf.value"] > 0,
        reflections.flags.integrated_prf,
    )
    return reflections
