
#ifndef DIALS_ALGORITHMS_SCALING_TOF_SCALING_CORRECTIONS_H
#define DIALS_ALGORITHMS_SCALING_TOF_SCALING_CORRECTIONS_H

#include <dials/array_family/scitbx_shared_and_versa.h>
#include <dxtbx/format/image.h>
#include <scitbx/constants.h>
#include <scitbx/vec2.h>
#include <scitbx/vec3.h>
#include <dxtbx/imageset.h>
#include <dxtbx/model/detector.h>
#include <dxtbx/model/beam.h>
#include <dxtbx/model/scan.h>
#include <dials/algorithms/integration/processor.h>

#include <cmath>

namespace dials { namespace algorithms {

  using dxtbx::ImageSequence;
  using dxtbx::model::Beam;
  using dxtbx::model::Detector;
  using dxtbx::model::Scan;
  using format::Image;
  using format::ImageTile;
  using scitbx::deg_as_rad;
  using scitbx::vec2;
  using scitbx::vec3;
  using scitbx::constants::m_n;
  using scitbx::constants::pi;
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

  double tof_pixel_spherical_absorption_correction(double pixel_data,
                                                   double muR,
                                                   double two_theta,
                                                   int two_theta_idx) {
    double ln_t1 = 0;
    double ln_t2 = 0;
    for (std::size_t k = 0; k < pc_size; ++k) {
      ln_t1 = ln_t1 * muR + pc[k][two_theta_idx];
      ln_t2 = ln_t2 * muR + pc[k][two_theta_idx + 1];
    }
    const double t1 = exp(ln_t1);
    const double t2 = exp(ln_t2);
    const double sin_theta_1 = pow(sin(deg_as_rad(two_theta_idx * 5.0)), 2);
    const double sin_theta_2 = pow(sin(deg_as_rad((two_theta_idx + 1) * 5.0)), 2);
    const double l1 = (t1 - t2) / (sin_theta_1 - sin_theta_2);
    const double l0 = t1 - l1 * sin_theta_1;
    const double correction = 1 / (l0 + l1 * pow(sin(two_theta * .5), 2));
    return correction;
  }

  void tof_extract_shoeboxes_to_reflection_table(af::reflection_table &reflection_table,
                                                 Experiment &experiment,
                                                 ImageSequence &vandium_data,
                                                 ImageSequence &empty_data,
                                                 double sample_radius,
                                                 double scattering_x_section,
                                                 double absorption_x_section,
                                                 double linear_absorption_c,
                                                 double linear_scattering_c) {
    Detector detector = experiment.get_detector();
    Scan scan = experiment.get_scan();

    PolychromaticBeam beam = experiment.get_beam();
    vec3<double> unit_s0 = beam.get_unit_s0();
    double sample_to_source_distance = beam.get_sample_to_source_distance();

    scitbx::af::shared<double> img_tof = scan.get_property("time_of_flight");
    img_tof *= std::pow(10, -6);  //(s)

    ImageSequence data = experiment.get_imageset();
    int n_panels = detector.size();
    int num_images = data.size();
    DIALS_ASSERT(num_images == img_tof.size());

    ShoeboxProcessor shoebox_processor(
      reflection_table, n_panels, 0, num_images, false);

    for (std::size_t img_num = 0; img_num < num_images; ++img_num) {
      // Image for each panel
      Image<int> imgs = data.get_corrected_data(img_num);
      Image<int> v_imgs = vanadium_data.get_corrected_data(img_num);
      Image<int> e_imgs = empty_data.get_corrected_data(img_num);

      DIALS_ASSERT(imgs.size() == v_imgs.size() == e_imgs.size());

      Image<double> corrected_imgs;
      double tof = img_tof[j];

      for (std::size_t panel_num = 0; panel_num < imgs.size(); ++panel_num) {
        scitbx::vec2<int> img_size = imgs[panel_num].accessor().all();
        DIALS_ASSERT(img_size == v_imgs[panel_num].accessor().all());
        DIALS_ASSERT(img_size == e_imgs[panel_num].accessor().all());

        scitbx::af::vera<double, scitbx::af::c_grid<2>(img_size[0], img_size[1])>
          corrected_img_data;

        for (std::size_t i = 0; i < img_size[0]; ++i) {
          for (std::size_t j = 0; j < img_size[1]; ++j) {
            // Get data for pixel
            int pixel_data = imgs[panel_num](i, j);
            int vanadium_pixel_data = v_imgs[panel_num](i, j);
            int empty_pixel_data = e_imgs[panel_num](i, j);

            // Subtract empty and normalise by vanadium
            // If vanadium pixel is less than 0, set pixel data to 0
            pixel_data -= empty_pixel_data;
            vanadium_pixel_data -= empty_pixel_data;
            if (vanadium_pixel_data <= 0) {
              corrected_img_data(i, j) = 0.0;
              continue;
            }
            pixel_data /= vanadium_pixel_data;

            // Spherical absorption correction
            double two_theta = detector[panel_num].get_two_theta_at_pixel(
              unit_s0, scitbx::vec2<double>(i, j));
            double two_theta_deg = two_theta * (180 / pi);
            int two_theta_idx = static_cast<int>(two_theta_deg / 10);

            scitbx::vec3<double> s1 =
              detector[panel_num].get_pixel_lab_coord(scitbx::vec2<double>(i, j));
            double distance = s1.length() + sample_to_source_distance;
            distance *= std::pow(10, -3);  // (m)

            double wl = ((Planck * tof) / (m_n * (distance))) * std::pow(10, 10);
            double muR =
              (linear_scattering_c + (linear_absorption_c / 1.8) * wl) * sample_radius;
            double absorption_correction = tof_pixel_spherical_absorption_correction(
              pixel_data, muR, two_theta, two_theta_idx);
            if (absorption_correction <= 0) {
              corrected_img_data(i, j) = 0.0;
              continue;
            }

            pixel_data /= absorption_correction;

            // Lorentz correction
            double sin_two_theta_sq = std::pow(sin(two_theta * .5), 2);
            double lorentz_correction = sin_two_theta_sq / std::pow(wl, 4);
            pixel_data *= lorentz_correction;
            corrected_img_data(i, j) = pixel_data;
          }
        }
        corrected_img.push_back(ImageTile<double>(corrected_img_data));
      }
      shoebox_processor.next_data_only(corrected_img);
    }
  }
}}  // namespace dials::algorithms

#endif /* DIALS_ALGORITHMS_SCALING_TOF_SCALING_CORRECTIONS_H */