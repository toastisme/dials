
#ifndef DIALS_ALGORITHMS_SCALING_TOF_SCALING_CORRECTIONS_H
#define DIALS_ALGORITHMS_SCALING_TOF_SCALING_CORRECTIONS_H

#include <dials/array_family/scitbx_shared_and_versa.h>
#include <scitbx/constants.h>
#include <cmath>

namespace dials { namespace algorithms {

  using scitbx::deg_as_rad;
  using scitbx::constants::m_n;
  using scitbx::constants::Planck;

  // Taken from
  // https://github.com/mantidproject/mantid/blob/main/Framework/Crystal/inc/MantidCrystal/AnvredCorrection.h
  const double pc[8][19] = {{-6.4910e-07,
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
                             -3.5091e-06},
                            {1.0839e-05,
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
                             1.4190e-04},
                            {8.7140e-05,
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
                             -2.4324e-03},
                            {-2.9549e-03,
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
                             2.3182e-02},
                            {1.7934e-02,
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
                             -1.3623e-01},
                            {6.2799e-02,
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
                             5.2651e-01},
                            {-1.4949e+00,
                             -1.4952e+00,
                             -1.4925e+00,
                             -1.4889e+00,
                             -1.4867e+00,
                             -1.4897e+00,
                             -1.4948e+00,
                             -1.5025e+00,
                             -1.5084e+00,
                             -1.5142e+00,
                             -1.5176e+00,
                             -1.5191e+00,
                             -1.5187e+00,
                             -1.5180e+00,
                             -1.5169e+00,
                             -1.5153e+00,
                             -1.5138e+00,
                             -1.5125e+00,
                             -1.5120e+00},
                            {0.0000e+00,
                             0.0000e+00,
                             0.0000e+00,
                             0.0000e+00,
                             0.0000e+00,
                             0.0000e+00,
                             0.0000e+00,
                             0.0000e+00,
                             0.0000e+00,
                             0.0000e+00,
                             0.0000e+00,
                             0.0000e+00,
                             0.0000e+00,
                             0.0000e+00,
                             0.0000e+00,
                             0.0000e+00,
                             0.0000e+00,
                             0.0000e+00,
                             0.0000e+00}};

  void tof_lorentz_correction(scitbx::af::versa<double, af::flex_grid<> > spectra,
                              double L0,
                              scitbx::af::shared<double> L1,
                              scitbx::af::shared<double> tof,
                              scitbx::af::shared<double> two_theta_spectra_sq) {
    DIALS_ASSERT(spectra.accessor().all()[0] == L1.size());
    DIALS_ASSERT(spectra.accessor().all()[0] == two_theta_spectra_sq.size());
    DIALS_ASSERT(spectra.accessor().all()[1] == tof.size());

    for (std::size_t i = 0; i < spectra.accessor().all()[0]; ++i) {
      for (std::size_t j = 0; j < spectra.accessor().all()[1]; ++j) {
        double wl = ((Planck * tof[j]) / (m_n * (L0 + L1[i]))) * std::pow(10, 10);
        spectra(i, j) *= two_theta_spectra_sq[i] / std::pow(wl, 4);
      }
    }
  }

  void tof_spherical_absorption_correction(
    scitbx::af::versa<double, af::flex_grid<> > spectra,
    scitbx::af::shared<double> muR_arr,
    scitbx::af::shared<double> two_thetas,
    scitbx::af::shared<long> two_theta_idxs) {
    DIALS_ASSERT(spectra.accessor().all()[0] == two_thetas.size());
    DIALS_ASSERT(spectra.accessor().all()[0] == two_theta_idxs.size());
    DIALS_ASSERT(spectra.accessor().all()[1] == muR_arr.size());

    const double pc_size = sizeof(pc) / sizeof(pc[0]);

    for (std::size_t i = 0; i < two_thetas.size(); ++i) {
      const int theta_idx = two_theta_idxs[i];
      const double theta = two_thetas[i] * .5;
      for (std::size_t j = 0; j < muR_arr.size(); ++j) {
        const double muR = muR_arr[j];
        double ln_t1 = 0;
        double ln_t2 = 0;
        for (std::size_t k = 0; k < pc_size; ++k) {
          ln_t1 = ln_t1 * muR + pc[k][theta_idx];
          ln_t2 = ln_t2 * muR + pc[k][theta_idx + 1];
        }
        const double t1 = exp(ln_t1);
        const double t2 = exp(ln_t2);
        const double sin_theta_1 = pow(sin(deg_as_rad(theta_idx * 5.0)), 2);
        const double sin_theta_2 = pow(sin(deg_as_rad((theta_idx + 1) * 5.0)), 2);
        const double l1 = (t1 - t2) / (sin_theta_1 - sin_theta_2);
        const double l0 = t1 - l1 * sin_theta_1;
        const double correction = 1 / (l0 + l1 * pow(sin(theta), 2));
        spectra(i, j) /= correction;
      }
    }
  }
}}  // namespace dials::algorithms

#endif /* DIALS_ALGORITHMS_SCALING_TOF_SCALING_CORRECTIONS_H */