#ifndef DIALS_ALGORITHMS_INTEGRATION_TOF_INTEGRATION_H
#define DIALS_ALGORITHMS_INTEGRATION_TOF_INTEGRATION_H

#include <dxtbx/imageset.h>
#include <dxtbx/format/image.h>
#include <dxtbx/array_family/flex_table.h>
#include <dials/model/data/shoebox.h>
#include <dxtbx/model/detector.h>
#include <dxtbx/model/beam.h>
#include <dxtbx/model/scan.h>
#include <dxtbx/model/goniometer.h>
#include <dials/array_family/scitbx_shared_and_versa.h>
#include <dials/algorithms/integration/processor.h>
#include <scitbx/vec2.h>
#include <scitbx/vec3.h>
#include <scitbx/constants.h>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <iostream>

#define GEMMI_WRITE_IMPLEMENTATION
#include <gemmi/mtz.hpp>
#include <gemmi/unitcell.hpp>
#include <gemmi/symmetry.hpp>

namespace dials { namespace algorithms {

  using dials::algorithms::Shoebox;
  using dials::algorithms::ShoeboxProcessor;
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
  using scitbx::af::int6;
  using scitbx::constants::m_n;
  using scitbx::constants::pi;
  using scitbx::constants::Planck;

  void get_asu_reflections(af::shared<cctbx::miller::index<int> > indices,
                           af::shared<cctbx::miller::index<int> > asu_predicted_indices,
                           af::shared<double> wavelengths,
                           af::shared<double> asu_predicted_wavelengths,
                           af::shared<bool> asu_reflection,
                           cctbx::sgtbx::space_group space_group

  ) {
    /*
     * Updates asu_reflections with true for each asu reflection
     */

    DIALS_ASSERT(indices.size() == asu_reflection.size());
    DIALS_ASSERT(indices.size() == wavelengths.size());
    DIALS_ASSERT(asu_predicted_indices.size() == asu_predicted_wavelengths.size());

    const char* hall_symbol = space_group.type().hall_symbol().c_str();
    const gemmi::SpaceGroup* gemmi_sg_ptr =
      gemmi::find_spacegroup_by_ops(gemmi::symops_from_hall(hall_symbol));
    if (gemmi_sg_ptr == nullptr) {
      throw DIALS_ERROR("Space group not found: " + *hall_symbol);
    }

    gemmi::UnmergedHklMover hkl_mover(gemmi_sg_ptr);
    af::shared<cctbx::miller::index<> > merged_hkls(indices.size());

    for (int i_refl = 0; i_refl < indices.size(); ++i_refl) {
      cctbx::miller::index<> miller_index = indices[i_refl];
      std::array<int, 3> hkl = {miller_index[0], miller_index[1], miller_index[2]};
      int isym = hkl_mover.move_to_asu(hkl);
      merged_hkls[i_refl] = cctbx::miller::index<>(hkl[0], hkl[1], hkl[3]);
    }

    for (std::size_t i = 0; i < asu_predicted_indices.size(); ++i) {
      cctbx::miller::index<> p_hkl = asu_predicted_indices[i];
      int closest_match = -1;
      double min_wl_diff = -1;
      for (std::size_t j = 0; j < merged_hkls.size(); ++j) {
        if (p_hkl == merged_hkls[j]) {
          double wl_diff = std::abs(wavelengths[j] - asu_predicted_wavelengths[i]);
          if (min_wl_diff < 0 || wl_diff < min_wl_diff) {
            closest_match = j;
          }
        }
        if (closest_match >= 0) {
          asu_reflection[closest_match] = true;
        }
      }
    }
  }

  void tof_calculate_shoebox_foreground(af::reflection_table& reflection_table,
                                        Experiment& experiment,
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
            // std::cout << "Distance " << distance << " mask value " << mask_value
            //           << " fr " << foreground_radius << std::endl;
            mask(z, y, x) |= mask_value;
          }
        }
      }
    }
  }

  // Function to compute the mean, eigenvectors, and axes lengths of the ellipsoid
  void compute_ellipsoid(std::vector<Eigen::Vector3d>& points,
                         Eigen::Vector3d& mean,
                         Eigen::Matrix3d& eigenvectors,
                         Eigen::Vector3d& axes_lengths) {
    // covariance matrix of the points
    Eigen::MatrixXd pointsMatrix(points.size(), 3);
    for (size_t i = 0; i < points.size(); ++i) {
      pointsMatrix.row(i) = points[i];
    }
    Eigen::RowVector3d mean_row = pointsMatrix.colwise().mean();
    mean = mean_row.transpose();
    Eigen::MatrixXd centered = pointsMatrix.rowwise() - mean_row;
    Eigen::MatrixXd covMatrix =
      (centered.adjoint() * centered) / double(pointsMatrix.rows() - 1);

    // eigenvalues and eigenvectors of the covariance matrix
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(covMatrix);
    if (eigensolver.info() != Eigen::Success) {
      throw DIALS_ERROR("Eigen decomposition failed");
    }
    Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
    eigenvectors = eigensolver.eigenvectors();
    axes_lengths = eigenvalues.cwiseSqrt();
  }

  bool point_inside_ellipsoid(const Eigen::Vector3d& point,
                              const Eigen::Vector3d& mean,
                              const Eigen::Matrix3d& eigenvectors,
                              const Eigen::Vector3d& axes_lengths) {
    Eigen::Vector3d centered_point = point - mean;
    Eigen::Vector3d transformed_point = eigenvectors.transpose() * centered_point;
    Eigen::Vector3d normalized_point = transformed_point.cwiseQuotient(axes_lengths);
    double distance_squared = normalized_point.squaredNorm();
    return distance_squared <= 1.0;
  }

  std::vector<Eigen::Vector3d> get_shoebox_rlps(const Shoebox<>& shoebox,
                                                Detector& detector,
                                                int& panel,
                                                int6& bbox,
                                                scitbx::af::shared<double>& img_tof,
                                                vec3<double>& unit_s0,
                                                double& sample_to_source_distance,
                                                mat3<double>& setting_rotation) {
    std::vector<Eigen::Vector3d> points;
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
          points.emplace_back(Eigen::Vector3d(S[0], S[1], S[2]));
        }
      }
    }
    return points;
  }

  void tof_calculate_shoebox_mask(af::reflection_table& reflection_table,
                                  Experiment& experiment) {
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
      std::vector<Eigen::Vector3d> shoebox_rlps =
        get_shoebox_rlps(shoebox,
                         detector,
                         panel,
                         bbox,
                         img_tof,
                         unit_s0,
                         sample_to_source_distance,
                         setting_rotation);
      Eigen::Vector3d mean;
      Eigen::Matrix3d eigenvectors;
      Eigen::Vector3d axes_lengths;
      compute_ellipsoid(shoebox_rlps, mean, eigenvectors, axes_lengths);
      int count = 0;
      for (std::size_t z = 0; z < shoebox.zsize(); ++z) {
        for (std::size_t y = 0; y < shoebox.ysize(); ++y) {
          for (std::size_t x = 0; x < shoebox.xsize(); ++x) {
            int mask_value = point_inside_ellipsoid(
                               shoebox_rlps[count], mean, eigenvectors, axes_lengths)
                               ? Foreground
                               : Background;
            mask(z, y, x) |= mask_value;
            count++;
          }
        }
      }
    }
  }
}}  // namespace dials::algorithms

#endif /* DIALS_ALGORITHMS_INTEGRATION_TOF_INTEGRATION_H */
