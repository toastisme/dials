from __future__ import annotations

import numpy as np


class SphericalAbsorption:

    # Taken from
    # https://github.com/mantidproject/mantid/blob/main/Framework/Crystal/inc/MantidCrystal/AnvredCorrection.h

    pc = np.array(
        (
            (
                -6.4910e-07,
                -6.8938e-07,
                -7.8149e-07,
                8.1682e-08,
                1.8008e-06,
                3.3916e-06,
                4.5095e-06,
                4.7970e-06,
                4.4934e-06,
                3.6700e-06,
                2.5881e-06,
                1.5007e-06,
                3.7669e-07,
                -7.9487e-07,
                -1.7935e-06,
                -2.5563e-06,
                -3.1113e-06,
                -3.3993e-06,
                -3.5091e-06,
            ),
            (
                1.0839e-05,
                1.1582e-05,
                1.1004e-05,
                -2.2848e-05,
                -8.1974e-05,
                -1.3268e-04,
                -1.6486e-04,
                -1.6839e-04,
                -1.5242e-04,
                -1.1949e-04,
                -7.8682e-05,
                -3.7973e-05,
                2.9117e-06,
                4.4823e-05,
                8.0464e-05,
                1.0769e-04,
                1.2753e-04,
                1.3800e-04,
                1.4190e-04,
            ),
            (
                8.7140e-05,
                9.0870e-05,
                1.6706e-04,
                6.9008e-04,
                1.4781e-03,
                2.0818e-03,
                2.3973e-03,
                2.3209e-03,
                1.9935e-03,
                1.4508e-03,
                8.1903e-04,
                1.9608e-04,
                -4.1128e-04,
                -1.0205e-03,
                -1.5374e-03,
                -1.9329e-03,
                -2.2212e-03,
                -2.3760e-03,
                -2.4324e-03,
            ),
            (
                -2.9549e-03,
                -3.1360e-03,
                -4.2431e-03,
                -8.1103e-03,
                -1.2989e-02,
                -1.6012e-02,
                -1.6815e-02,
                -1.4962e-02,
                -1.1563e-02,
                -6.8581e-03,
                -1.7302e-03,
                3.2400e-03,
                7.9409e-03,
                1.2528e-02,
                1.6414e-02,
                1.9394e-02,
                2.1568e-02,
                2.2758e-02,
                2.3182e-02,
            ),
            (
                1.7934e-02,
                1.9304e-02,
                2.4706e-02,
                3.6759e-02,
                4.8351e-02,
                5.1049e-02,
                4.5368e-02,
                3.0864e-02,
                1.2086e-02,
                -1.0254e-02,
                -3.2992e-02,
                -5.4495e-02,
                -7.4205e-02,
                -9.2818e-02,
                -1.0855e-01,
                -1.2068e-01,
                -1.2954e-01,
                -1.3451e-01,
                -1.3623e-01,
            ),
            (
                6.2799e-02,
                6.3892e-02,
                6.4943e-02,
                6.4881e-02,
                7.2169e-02,
                9.5669e-02,
                1.3082e-01,
                1.7694e-01,
                2.2559e-01,
                2.7655e-01,
                3.2483e-01,
                3.6888e-01,
                4.0783e-01,
                4.4330e-01,
                4.7317e-01,
                4.9631e-01,
                5.1334e-01,
                5.2318e-01,
                5.2651e-01,
            ),
            (
                -1.4949e00,
                -1.4952e00,
                -1.4925e00,
                -1.4889e00,
                -1.4867e00,
                -1.4897e00,
                -1.4948e00,
                -1.5025e00,
                -1.5084e00,
                -1.5142e00,
                -1.5176e00,
                -1.5191e00,
                -1.5187e00,
                -1.5180e00,
                -1.5169e00,
                -1.5153e00,
                -1.5138e00,
                -1.5125e00,
                -1.5120e00,
            ),
            (
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
                0.0000e00,
            ),
        )
    )

    def __init__(
        self, radius, sample_number_density, scattering_x_section, absorption_x_section
    ):

        self.radius = radius
        self.sample_number_density = sample_number_density
        self.linear_absorption_c = absorption_x_section * sample_number_density
        self.linear_scattering_c = scattering_x_section * sample_number_density

    def get_absorption_correction(self, spectra_arr, wavelength_arr, two_theta_arr):

        corrections = np.zeros(spectra_arr.shape)

        for i, spectra in enumerate(spectra_arr):
            for j, bin in enumerate(spectra):
                wavelength = wavelength_arr[j]
                two_theta = two_theta_arr[i]
                muR = (
                    self.linear_scattering_c
                    + (self.linear_absorption_c / 1.8) * wavelength
                ) * self.radius

                assert muR <= 8.0
                two_theta_deg = two_theta * (180 / np.pi)
                assert two_theta_deg > 0 and two_theta_deg < 180

                theta_idx = int(two_theta_deg)

                ln_transmission_1 = 0
                ln_transmission_2 = 0

                ncoef = self.pc.shape[0] / self.pc.shape[1]
                for n in range(ncoef):
                    ln_transmission_1 *= muR + self.pc[n][theta_idx]
                    ln_transmission_2 *= muR + self.pc[n][theta_idx + 1]

                transmission_1 = np.exp(ln_transmission_1)
                transmission_2 = np.exp(ln_transmission_2)

                sin_theta_1 = np.square(np.sin(theta_idx * 5.0 * (np.pi / 180)))
                sin_theta_2 = np.square(np.sin((theta_idx + 1) * 5.0 * (np.pi / 180)))

                L1 = (transmission_1 - transmission_2) / (sin_theta_1 - sin_theta_2)
                L0 = transmission_1 - L1 * sin_theta_1

                absorption_correction = 1 / (L0 + L1 * np.square(np.sin(two_theta)))
                corrections[i, j] = absorption_correction

        return corrections
