
#ifndef DIALS_ALGORITHMS_SCALING_TOF_SCALING_CORRECTIONS_H
#define DIALS_ALGORITHMS_SCALING_TOF_SCALING_CORRECTIONS_H

#include <dxtbx/model/detector.h>
#include <dxtbx/model/beam.h>
#include <dxtbx/model/scan.h>
#include <dxtbx/model/goniometer.h>
#include <dials/array_family/scitbx_shared_and_versa.h>
#include <dxtbx/format/image.h>
#include <scitbx/constants.h>
#include <scitbx/vec2.h>
#include <scitbx/vec3.h>
#include <dxtbx/imageset.h>
#include <dxtbx/array_family/flex_table.h>
#include <dxtbx/array_family/flex_table_suite.h>
#include <dials/algorithms/integration/processor.h>
#include <dials/model/data/shoebox.h>
#include <dials/algorithms/shoebox/mask_code.h>

#include <cmath>
#include <iostream>
namespace dials { namespace algorithms {

  using dials::model::Background;
  using dials::model::Foreground;
  using dxtbx::ImageSequence;
  using dxtbx::af::flex_table;
  using dxtbx::model::Detector;
  using dxtbx::model::Experiment;
  using dxtbx::model::Goniometer;
  using dxtbx::model::PolychromaticBeam;
  using dxtbx::model::Scan;
  using dxtbx::model::scan_property_types;
  using scitbx::deg_as_rad;
  using scitbx::mat3;
  using scitbx::vec2;
  using scitbx::vec3;
  using scitbx::constants::m_n;
  using scitbx::constants::pi;
  using scitbx::constants::Planck;

  double tof_pixel_spherical_absorption_correction(double pixel_data,
                                                   double muR,
                                                   double two_theta,
                                                   int two_theta_idx) {
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
    double ln_t1 = 0;
    double ln_t2 = 0;
    for (std::size_t k = 0; k < 8; ++k) {
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

  struct TOFCorrectionsData {
    double sample_proton_charge;
    double incident_proton_charge;
    double empty_proton_charge;
    double sample_radius;
    double sample_scattering_x_section;
    double sample_absorption_x_section;
    double sample_number_density;
    double incident_radius;
    double incident_scattering_x_section;
    double incident_absorption_x_section;
    double incident_number_density;

    TOFCorrectionsData(double sample_proton_charge,
                       double incident_proton_charge,
                       double empty_proton_charge,
                       double sample_radius,
                       double sample_scattering_x_section,
                       double sample_absorption_x_section,
                       double sample_number_density,
                       double incident_radius,
                       double incident_scattering_x_section,
                       double incident_absorption_x_section,
                       double incident_number_density)
        : sample_proton_charge(sample_proton_charge),
          incident_proton_charge(incident_proton_charge),
          empty_proton_charge(empty_proton_charge),
          sample_radius(sample_radius),
          sample_scattering_x_section(sample_scattering_x_section),
          sample_absorption_x_section(sample_absorption_x_section),
          sample_number_density(sample_number_density),
          incident_radius(incident_radius),
          incident_scattering_x_section(incident_scattering_x_section),
          incident_absorption_x_section(incident_absorption_x_section),
          incident_number_density(incident_number_density) {}
  };

  void tof_extract_shoeboxes_to_reflection_table(
    af::reflection_table &reflection_table,
    Experiment &experiment,
    ImageSequence &data,
    ImageSequence &incident_data,
    ImageSequence &empty_data,
    TOFCorrectionsData &corrections_data,
    bool apply_lorentz_correction,
    bool apply_spherical_absorption_correction) {
    double sample_linear_scattering_c = corrections_data.sample_number_density
                                        * corrections_data.sample_scattering_x_section;
    double incident_linear_scattering_c =
      corrections_data.incident_number_density
      * corrections_data.incident_scattering_x_section;
    double sample_linear_absorption_c = corrections_data.sample_number_density
                                        * corrections_data.sample_absorption_x_section;
    double incident_linear_absorption_c =
      corrections_data.incident_number_density
      * corrections_data.incident_absorption_x_section;

    Detector detector = *experiment.get_detector();
    Scan scan = *experiment.get_scan();

    std::shared_ptr<dxtbx::model::BeamBase> beam_ptr = experiment.get_beam();
    std::shared_ptr<PolychromaticBeam> beam =
      std::dynamic_pointer_cast<PolychromaticBeam>(beam_ptr);
    DIALS_ASSERT(beam != nullptr);

    vec3<double> unit_s0 = beam->get_unit_s0();
    double sample_to_source_distance = beam->get_sample_to_source_distance();

    scitbx::af::shared<double> img_tof = scan.get_property<double>("time_of_flight");

    int n_panels = detector.size();
    int num_images = data.size();
    vec2<std::size_t> image_size = detector[0].get_image_size();
    DIALS_ASSERT(num_images == img_tof.size());

    boost::python::dict d;
    af::reflection_table i_reflection_table =
      dxtbx::af::flex_table_suite::deepcopy(reflection_table, d);
    af::reflection_table e_reflection_table =
      dxtbx::af::flex_table_suite::deepcopy(reflection_table, d);

    ShoeboxProcessor shoebox_processor(
      reflection_table, n_panels, 0, num_images, false);

    ShoeboxProcessor incident_shoebox_processor(
      i_reflection_table, n_panels, 0, num_images, false);

    ShoeboxProcessor empty_shoebox_processor(
      e_reflection_table, n_panels, 0, num_images, false);

    for (std::size_t img_num = 0; img_num < num_images; ++img_num) {
      dxtbx::format::Image<double> img = data.get_corrected_data(img_num);
      dxtbx::format::Image<bool> mask = data.get_mask(img_num);

      af::shared<scitbx::af::versa<double, scitbx::af::c_grid<2> > > output_data(
        n_panels);
      af::shared<scitbx::af::versa<bool, scitbx::af::c_grid<2> > > output_mask(
        n_panels);

      for (std::size_t i = 0; i < output_data.size(); ++i) {
        output_data[i] = img.tile(i).data();
        output_mask[i] = mask.tile(i).data();
      }
      shoebox_processor.next_data_only(
        model::Image<double>(output_data.const_ref(), output_mask.const_ref()));
    }

    for (std::size_t img_num = 0; img_num < num_images; ++img_num) {
      dxtbx::format::Image<double> img = incident_data.get_corrected_data(img_num);
      dxtbx::format::Image<bool> mask = incident_data.get_mask(img_num);

      af::shared<scitbx::af::versa<double, scitbx::af::c_grid<2> > > output_data(
        n_panels);
      af::shared<scitbx::af::versa<bool, scitbx::af::c_grid<2> > > output_mask(
        n_panels);

      for (std::size_t i = 0; i < output_data.size(); ++i) {
        output_data[i] = img.tile(i).data();
        output_mask[i] = mask.tile(i).data();
      }
      incident_shoebox_processor.next_data_only(
        model::Image<double>(output_data.const_ref(), output_mask.const_ref()));
    }

    for (std::size_t img_num = 0; img_num < num_images; ++img_num) {
      dxtbx::format::Image<double> img = empty_data.get_corrected_data(img_num);
      dxtbx::format::Image<bool> mask = empty_data.get_mask(img_num);

      af::shared<scitbx::af::versa<double, scitbx::af::c_grid<2> > > output_data(
        n_panels);
      af::shared<scitbx::af::versa<bool, scitbx::af::c_grid<2> > > output_mask(
        n_panels);

      for (std::size_t i = 0; i < output_data.size(); ++i) {
        output_data[i] = img.tile(i).data();
        output_mask[i] = mask.tile(i).data();
      }
      empty_shoebox_processor.next_data_only(
        model::Image<double>(output_data.const_ref(), output_mask.const_ref()));
    }

    af::shared<Shoebox<> > shoeboxes = reflection_table["shoebox"];
    af::shared<Shoebox<> > e_shoeboxes = i_reflection_table["shoebox"];
    af::shared<Shoebox<> > i_shoeboxes = e_reflection_table["shoebox"];
    af::const_ref<int6> bboxes = reflection_table["bbox"];

    for (std::size_t i = 0; i < reflection_table.size(); ++i) {
      Shoebox<> shoebox = shoeboxes[i];
      Shoebox<> i_shoebox = e_shoeboxes[i];
      Shoebox<> e_shoebox = i_shoeboxes[i];
      int6 bbox = bboxes[i];
      int panel = shoebox.panel;

      for (std::size_t z = 0; z < shoebox.zsize(); ++z) {
        int frame_z = bbox[4] + z;
        double tof = img_tof[frame_z] * std::pow(10, -6);  // (s)

        for (std::size_t x = 0; x < shoebox.xsize(); ++x) {
          int panel_x = bbox[0] + x;
          if (panel_x > image_size[0] || panel_x < 0) {
            continue;
          }
          for (std::size_t y = 0; y < shoebox.ysize(); ++y) {
            int panel_y = bbox[2] + y;
            if (panel_y > image_size[1] || panel_y < 0) {
              continue;
            }

            double incident_pixel_data = i_shoebox.data(z, y, x);
            double empty_pixel_data = e_shoebox.data(z, y, x);
            double pixel_data = shoebox.data(z, y, x);
            /*
            if (i==0){
                    std::cout<<"TEST pixel data " << pixel_data  << " empty data " <<
            empty_pixel_data << " incident data " << incident_pixel_data << std::endl;
            }
            */

            pixel_data /= corrections_data.sample_proton_charge;
            incident_pixel_data /= corrections_data.incident_proton_charge;
            empty_pixel_data /= corrections_data.empty_proton_charge;
            /*
            if (i==0){
                    std::cout<<"TEST after proton charge pixel data " << pixel_data  <<
            " empty data " << empty_pixel_data << " incident data " <<
            incident_pixel_data << std::endl;
            }
            */

            // Subtract empty from incident and sample
            pixel_data -= empty_pixel_data;
            incident_pixel_data -= empty_pixel_data;
            /*
            if (i==0){
                    std::cout<<"TEST after empty subtract pixel data " << pixel_data  <<
            " empty data " << empty_pixel_data << " incident data " <<
            incident_pixel_data << std::endl;
            }
            */

            double two_theta = detector[panel].get_two_theta_at_pixel(
              unit_s0, scitbx::vec2<double>(panel_x, panel_y));
            double two_theta_deg = two_theta * (180 / pi);
            int two_theta_idx = static_cast<int>(two_theta_deg / 10);

            scitbx::vec3<double> s1 = detector[panel].get_pixel_lab_coord(
              scitbx::vec2<double>(panel_x, panel_y));
            double distance = s1.length() + sample_to_source_distance;
            distance *= std::pow(10, -3);  // (m)
            double wl = ((Planck * tof) / (m_n * (distance))) * std::pow(10, 10);

            // Spherical absorption correction
            if (apply_spherical_absorption_correction) {
              double sample_muR =
                (sample_linear_scattering_c + (sample_linear_absorption_c / 1.8) * wl)
                * corrections_data.sample_radius;
              double sample_absorption_correction =
                tof_pixel_spherical_absorption_correction(
                  pixel_data, sample_muR, two_theta, two_theta_idx);

              /*
              if (i==0){
                      std::cout<<"TEST sample absorption correction: " <<
              sample_absorption_correction << std::endl;
              }
              */
              if (sample_absorption_correction < 1e-5) {
                shoebox.data(z, y, x) = 0;
                continue;
              }

              double incident_muR = (incident_linear_scattering_c
                                     + (incident_linear_absorption_c / 1.8) * wl)
                                    * corrections_data.incident_radius;
              double incident_absorption_correction =
                tof_pixel_spherical_absorption_correction(
                  pixel_data, incident_muR, two_theta, two_theta_idx);

              /*
              if (i==0){
                      std::cout<<"TEST incident absorption correction: " <<
              incident_absorption_correction << std::endl;
              }
              */
              if (incident_absorption_correction < 1e-5) {
                shoebox.data(z, y, x) = 0;
                continue;
              }

              incident_pixel_data /= incident_absorption_correction;
              /*
              if (i==0){
                      std::cout<<"TEST incident after correction: " <<
              incident_pixel_data << std::endl;
              }
              */
              if (incident_pixel_data < 1e-5) {
                shoebox.data(z, y, x) = 0;
                continue;
              }

              pixel_data /= incident_pixel_data;
              /*
              if (i==0){
                      std::cout<<"TEST pixel data after normalisation: " << pixel_data
              << std::endl;
              }
              */
              pixel_data /= sample_absorption_correction;
              /*
              if (i==0){
                      std::cout<<"TEST pixel data after correction: " << pixel_data <<
              std::endl;
              }
              */
            }

            else {
              if (incident_pixel_data < 1e-5) {
                shoebox.data(z, y, x) = 0;
                continue;
              }

              pixel_data /= incident_pixel_data;
            }

            // Lorentz correction
            if (apply_lorentz_correction) {
              double sin_two_theta_sq = std::pow(sin(two_theta * .5), 2);
              double lorentz_correction = sin_two_theta_sq / std::pow(wl, 4);
              // std::cout<<"TEST lorentz two_theta " << two_theta << " sin_two_theta_sq
              // " << sin_two_theta_sq << " correction " << lorentz_correction << "
              // pixel data " << pixel_data << " empty " << empty_pixel_data <<
              // std::endl;
              pixel_data *= lorentz_correction;
              // if (i==0){
              //   std::cout<<"TEST lorentz  " << lorentz_correction << " " <<
              //   pixel_data << std::endl;
              // }
            }

            shoebox.data(z, y, x) = double(pixel_data);
          }
        }
      }
    }
  }

  void tof_calculate_shoebox_foreground(af::reflection_table &reflection_table,
                                        Experiment &experiment,
                                        double foreground_radius) {
    af::shared<Shoebox<> > shoeboxes = reflection_table["shoebox"];
    Scan scan = *experiment.get_scan();
    Detector detector = *experiment.get_detector();
    Goniometer goniometer = *experiment.get_goniometer();
    mat3<double> setting_rotation = goniometer.get_setting_rotation();

    std::shared_ptr<dxtbx::model::BeamBase> beam_ptr = experiment.get_beam();
    std::shared_ptr<PolychromaticBeam> beam =
      std::dynamic_pointer_cast<PolychromaticBeam>(beam_ptr);
    DIALS_ASSERT(beam != nullptr);
    vec3<double> unit_s0 = beam->get_unit_s0();
    double sample_to_source_distance = beam->get_sample_to_source_distance();

    scitbx::af::shared<double> img_tof = scan.get_property<double>("time_of_flight");
    af::const_ref<int6> bboxes = reflection_table["bbox"];
    scitbx::af::shared<vec3<double> > rlps = reflection_table["rlp"];

    for (std::size_t i = 0; i < reflection_table.size(); ++i) {
      Shoebox<> shoebox = shoeboxes[i];
      af::ref<int, af::c_grid<3> > mask = shoebox.mask.ref();
      int panel = shoebox.panel;
      int6 bbox = bboxes[i];
      vec3<double> rlp = rlps[i];

      for (std::size_t z = 0; z < shoebox.zsize(); ++z) {
        int frame_z = bbox[4] + z;
        double tof = img_tof[frame_z] * std::pow(10, -6);  // (s)

        for (std::size_t y = 0; y < shoebox.ysize(); ++y) {
          int panel_y = bbox[2] + y;

          for (std::size_t x = 0; x < shoebox.xsize(); ++x) {
            int panel_x = bbox[0] + x;
            vec3<double> s1 =
              detector[panel].get_pixel_lab_coord(vec2<double>(panel_x, panel_y));
            double pixel_distance = s1.length() + sample_to_source_distance;
            pixel_distance *= std::pow(10, -3);  // (m)
            double wl =
              ((Planck * tof) / (m_n * (pixel_distance))) * std::pow(10, 10);  // (A)
            vec3<double> s0 = unit_s0 / wl;
            s1 = s1 / s1.length() * (1 / wl);
            vec3<double> S = s1 - s0;
            S = setting_rotation.inverse() * S;
            double distance = (S - rlp).length();

            int mask_value = (distance <= foreground_radius) ? Foreground : Background;
            mask(z, y, x) |= mask_value;
          }
        }
      }
    }
  }

}}  // namespace dials::algorithms

#endif /* DIALS_ALGORITHMS_SCALING_TOF_SCALING_CORRECTIONS_H */