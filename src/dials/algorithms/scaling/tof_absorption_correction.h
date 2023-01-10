#ifndef DIALS_SCALING_TOF_ABSORPTION_CORRECTION_H
#define DIALS_SCALING_TOF_ABSORPTION_CORRECTION_H

#include <dials/array_family/scitbx_shared_and_versa.h>
#include <scitbx/constants.h>

namespace dials_scaling {
using namespace boost::python;
using namespace scitbx::constants;

class TOFSphericalAbsorption {
public:
  virtual ~TOFSphericalAbsorption() {}

  void absorption_correction_for_pixel_array() {}
  scitbx::af::shared<double> absorption_correction_for_pixel(
    const double &muR,
    const int &theta_idx,
    const scitbx::af::shared<double> &two_theta) {
    double ln_transmission_1 = 0.0;
    double ln_transmission_2 = 0.0;

    for (std::size_t i = 0; i < pc.size(); ++i) {
      ln_transmission_1 = ln_transmission_1 * muR + pc[i][theta_idx];
      ln_transmission_2 = ln_transmission_2 * muR + pc[i][theta_idx + 1];
    }

    double transmission_1 = std::exp(ln_transmission_1);
    double transmission_2 = std::exp(ln_transmission_2);

    double sin_theta_1 = std::pow(std::sin(theta_idx * 5.0 * pi_180), 2);
    double sin_theta_2 = std::pow(std::sin((theta_idx + 1) * 5.0 * pi_180), 2);

    double l1 = (transmission_1 - transmission_2) / (sin_theta_1 - sin_theta_2);
    double l0 = transmission_1 - l1 * sin_theta_1;

    return 1 / (l0 + l1 * std::pow(std::sin(two_theta * .5), 2));
  }

private:
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
}
}  // namespace dials_scaling

#endif  // DIALS_SCALING_TOF_ABSORPTION_CORRECTION_H
