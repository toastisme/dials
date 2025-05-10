from __future__ import annotations

from multiprocessing import Pool

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erfc

from dxtbx import flumpy

from dials.algorithms.shoebox import MaskCode
from dials.array_family import flex


class GutmannProfile:
    """
    https://www.sciencedirect.com/science/article/abs/pii/S0168900216312906
    """

    def __init__(self, coords, intensities, alpha, beta):

        self.coords = coords.T

        self.raw_intensities = intensities
        self.intensities = np.nan_to_num(intensities, nan=0, posinf=None, neginf=None)
        self.intensities /= np.sum(self.intensities)

        # Initialize H using inverse covariance
        cov = np.cov(self.coords)
        cov += np.eye(3) * 1e-6  # regularize to ensure SPD
        L = np.linalg.cholesky(np.linalg.inv(cov))

        # Extract 6 independent lower-triangle elements of L
        l11, l21, l31 = L[0, 0], L[1, 0], L[2, 0]
        l22, l32 = L[1, 1], L[2, 1]
        l33 = L[2, 2]

        self.params = np.array([l11, l21, l31, l22, l32, l33, alpha, beta])

    def build_H_from_L(self, params):
        l11, l21, l31, l22, l32, l33 = params[:6]
        L = np.array([[l11, 0.0, 0.0], [l21, l22, 0.0], [l31, l32, l33]])
        H = L @ L.T
        return H

    def func(self, coords, H, alpha, beta):
        H1, H2, H3 = H[0]
        _, H4, H5 = H[1]
        _, _, H6 = H[2]

        dx, dy, dt = coords
        a = alpha
        b = beta

        N = (a * b) / (2 * (a + b))
        N_g = np.sqrt(np.linalg.det(H)) / (2 * np.pi) ** (3 / 2.0)
        f1 = N * N_g * np.sqrt(np.pi / (2 * H6))

        u = 0.5 * a * (a + 2 * H6 * dt + 2 * H3 * dx + 2 * H5 * dy)
        v = 0.5 * b * (b - 2 * H6 * dt - 2 * H3 * dx - 2 * H5 * dy)

        y = (a + H6 * dt + H3 * dx + H5 * dy) / np.sqrt(2 * H6)
        w = (b - H6 * dt - H3 * dx - H5 * dy) / np.sqrt(2 * H6)

        # Clip u, v to avoid exp overflow
        u = np.clip(u, -700, 700)
        v = np.clip(v, -700, 700)

        # Clip erfc inputs to avoid log(0) or nan in erfc
        EPS = 1e-300
        erfc_y = np.clip(erfc(y), EPS, None)
        erfc_w = np.clip(erfc(w), EPS, None)

        f2 = np.exp(
            -0.5 * H1 * dx**2
            - H2 * dx * dy
            - 0.5 * H4 * dy**2
            + (H3**2 * dx**2 + 2 * H3 * H5 * dx * dy + H5**2 * dy**2) / (2 * H6)
        )

        f3 = np.exp(u) * erfc_y + np.exp(v) * erfc_w

        result = f1 * f2 * f3
        return result

    def fit(self):
        try:

            def residuals(params):
                try:
                    H = self.build_H_from_L(params)
                    alpha, beta = params[6:]
                    r = self.intensities - self.func(self.coords, H, alpha, beta)
                    if not np.all(np.isfinite(r)):
                        r = np.nan_to_num(r, nan=1e6, posinf=1e6, neginf=-1e6)
                    if np.any(np.abs(r) > 1e10):
                        r = np.clip(r, -1e10, 1e10)
                    return r

                except np.linalg.LinAlgError:
                    # If H is not SPD for any reason (numerical), penalize
                    return 1e6 * np.ones_like(self.intensities)

            bounds_lower = [1e-4, -1e-3, -1e-3, 1e-4, -1e-3, -1e-3, 0, 0]
            bounds_upper = [1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 100, 100]

            res = least_squares(
                residuals,
                x0=self.params,
                loss="soft_l1",
                f_scale=1.0,
                x_scale="jac",
                bounds=(bounds_lower, bounds_upper),
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
        H = self.build_H_from_L(self.params)
        alpha, beta = self.params[6:]
        return self.func(self.coords, H, alpha, beta)

    def calc_intensity(self):
        predicted = self.result()

        scale_factor = np.sum(self.raw_intensities)
        return sum(predicted) * scale_factor


def process_reflection(args):
    i, data, background, coords, mask, alpha, beta = args
    intensity = data.ravel()

    bg_code = MaskCode.Valid | MaskCode.Background | MaskCode.BackgroundUsed
    bg_mask = (mask & bg_code) == bg_code

    foreground_mask = (mask & MaskCode.Foreground) == MaskCode.Foreground
    valid_mask = (mask & MaskCode.Valid) == MaskCode.Valid
    not_overlapped_mask = (mask & MaskCode.Overlapped) == 0
    intensity_mask = foreground_mask & valid_mask & not_overlapped_mask
    n_background = np.sum(np.bitwise_and(~intensity_mask, bg_mask))
    n_signal = np.sum(intensity_mask)

    background = background[intensity_mask]
    background_sum = np.sum(background)

    fit_intensity = None
    try:
        profile = GutmannProfile(
            coords=coords,
            intensities=intensity,
            alpha=alpha,
            beta=beta,
        )
        profile.fit()
        fit_intensity = profile.calc_intensity()
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


def compute_gutmann_profile_intensity(
    reflections, nproc=8, integration_method="observed"
):
    alpha = 1.0
    beta = 0.2

    fit_intensities = flex.double(len(reflections))
    fit_variances = flex.double(len(reflections))

    args = []
    for i in range(len(reflections)):

        # Get coords relative to centroid
        coords = flumpy.to_numpy(reflections[i]["shoebox"].coords())
        if integration_method == "observed":
            coords -= reflections[i]["xyzobs.px.value"]
        elif integration_method == "calculated":
            coords -= reflections[i]["xyzcal.px"]
        else:
            raise ValueError(f"Unknown integration method {integration_method}")

        args.append(
            (
                i,
                flumpy.to_numpy(
                    reflections[i]["shoebox"].data - reflections["background.mean"][i]
                ),
                flumpy.to_numpy(reflections[i]["shoebox"].background),
                coords,
                flumpy.to_numpy(reflections[i]["shoebox"].mask),
                alpha,
                beta,
            )
        )

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
