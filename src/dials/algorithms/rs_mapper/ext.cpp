#include <boost/python.hpp>
#include <scitbx/array_family/shared.h>
#include <scitbx/array_family/flex_types.h>
#include <scitbx/constants.h>
#include <cctype>
#include <dxtbx/model/panel.h>
#include <dxtbx/imageset.h>
#include <iostream>
#include <dials/error.h>

namespace recviewer { namespace ext {
  using namespace scitbx;
  using scitbx::constants::m_n;
  using scitbx::constants::pi;
  using scitbx::constants::Planck;

  typedef scitbx::af::flex<vec3<double> >::type flex_vec3_double;
  typedef scitbx::af::flex<vec2<double> >::type flex_vec2_double;

  static af::shared<vec2<double> > get_target_pixels(dxtbx::model::Panel panel,
                                                     vec3<double> s0,
                                                     int nfast,
                                                     int nslow,
                                                     double maxres) {
    af::shared<vec2<double> > ret;
    vec2<double> xy;

    for (size_t y = 0; y < nslow; y++) {
      for (size_t x = 0; x < nfast; x++) {
        xy[0] = x;
        xy[1] = y;

        // get_resolution_at_pixel() no longer returns INF, so this is safe
        // Expects coord given in terms of (fast, slow), which is column, row...
        if (panel.get_resolution_at_pixel(s0, xy) > maxres) {
          ret.push_back(xy);
        }
      }
    }
    return ret;
  }

  static void fill_voxels_tof(scitbx::af::shared<std::size_t> image_idxs,
                              dxtbx::ImageSequence imageset,
                              double max_resolution,
                              af::flex_double &grid,
                              af::flex_int &counts) {
    DIALS_ASSERT(imageset.get_detector() != NULL);
    DIALS_ASSERT(imageset.get_scan() != NULL);
    DIALS_ASSERT(imageset.get_beam() != NULL);
    DIALS_ASSERT(imageset.get_goniometer() != NULL);
    dxtbx::model::Detector detector = *imageset.get_detector();
    dxtbx::model::Scan scan = *imageset.get_scan();
    dxtbx::model::Goniometer goniometer = *imageset.get_goniometer();
    std::shared_ptr<dxtbx::model::BeamBase> beam_ptr = imageset.get_beam();
    std::shared_ptr<dxtbx::model::PolychromaticBeam> beam =
      std::dynamic_pointer_cast<dxtbx::model::PolychromaticBeam>(beam_ptr);
    DIALS_ASSERT(beam != nullptr);

    DIALS_ASSERT(scan.contains("time_of_flight"));
    scitbx::af::shared<double> tof_bins = scan.get_property<double>("time_of_flight");

    vec3<double> unit_s0 = beam->get_unit_s0();
    double sample_to_source_distance = beam->get_sample_to_source_distance();

    af::shared<vec3<double> > rlp;
    vec2<double> xy;
    af::tiny<long unsigned int, 2> image_size;
    vec3<double> s1;
    vec3<double> s0;
    vec3<double> S;
    double tof;
    double pixel_size;
    double wavelength;
    double pixel_distance;
    dxtbx::Image<int> img_data;
    dxtbx::model::Panel panel;

    // Grid variables
    int npoints = grid.accessor().all()[0];
    double rec_range = 1 / max_resolution;
    double step = 2 * rec_range / npoints;
    int grid_x;
    int grid_y;
    int grid_z;

    DIALS_ASSERT(imageset.size() == tof_bins.size());

    for (size_t img_idx = 0; img_idx < image_idxs.size(); img_idx++) {
      DIALS_ASSERT(img_idx >= 0 && img_idx < tof_bins.size());
      tof = tof_bins[img_idx] * std::pow(10, -6);  // (s);
      img_data = imageset.get_raw_data(img_idx).as_int();
      for (size_t panel_idx = 0; panel_idx < detector.size(); panel_idx++) {
        panel = detector[panel_idx];
        pixel_size = panel.get_pixel_size()[0];
        image_size = panel.get_image_size();

        for (size_t y = 0; y < image_size[1]; y++) {
          for (size_t x = 0; x < image_size[0]; x++) {
            xy[0] = x;
            xy[1] = y;

            s1 = panel.get_lab_coord(xy * pixel_size);
            pixel_distance =
              (s1.length() + sample_to_source_distance) * std::pow(10, -3);  // (m);
            wavelength =
              ((Planck * tof) / (m_n * pixel_distance)) * std::pow(10, 10);  // (A)
            s0 = unit_s0 / wavelength;

            if (panel.get_resolution_at_pixel(s0, xy) > max_resolution) {
              // Get rlp
              s1 = s1 / s1.length() * (1 / wavelength);
              S = s1 - s0;
              scitbx::mat3<double> setting_rotation = goniometer.get_setting_rotation();
              S = setting_rotation.inverse() * S;

              // Get corresponding point on reciprocal space grid
              grid_x = S[0] / step + npoints / 2 + 0.5;
              grid_y = S[1] / step + npoints / 2 + 0.5;
              grid_z = S[2] / step + npoints / 2 + 0.5;

              // Add image intensity at that point
              if (grid_x >= npoints || grid_y >= npoints || grid_z >= npoints
                  || grid_x < 0 || grid_y < 0 || grid_z < 0)
                continue;
              grid(grid_x, grid_y, grid_z) += img_data.tile(panel_idx).data()(y, x);
              counts(grid_x, grid_y, grid_z)++;
            }
          }
        }
      }
    }
  }

  static void fill_voxels(const af::flex_int &image,
                          af::flex_double &grid,
                          af::flex_int &counts,
                          const flex_vec3_double &rotated_S,
                          const flex_vec2_double &xy,
                          const double rec_range) {
    int npoints = grid.accessor().all()[0];
    double step = 2 * rec_range / npoints;

    for (int i = 0, ilim = xy.size(); i < ilim; i++) {
      int ind_x = rotated_S[i][0] / step + npoints / 2 + 0.5;
      int ind_y = rotated_S[i][1] / step + npoints / 2 + 0.5;
      int ind_z = rotated_S[i][2] / step + npoints / 2 + 0.5;
      int x = xy[i][0];
      int y = xy[i][1];

      if (ind_x >= npoints || ind_y >= npoints || ind_z >= npoints || ind_x < 0
          || ind_y < 0 || ind_z < 0)
        continue;
      grid(ind_x, ind_y, ind_z) += image(y, x);
      counts(ind_x, ind_y, ind_z)++;
    }
  }

  static void normalize_voxels(af::flex_double &grid, af::flex_int &counts) {
    for (int i = 0, ilim = grid.size(); i < ilim; i++) {
      if (counts[i] != 0) {
        grid[i] /= counts[i];
      }
    }
  }

  void init_module() {
    using namespace boost::python;
    def("get_target_pixels", get_target_pixels);
    def("fill_voxels", fill_voxels);
    def("fill_voxels_tof", fill_voxels_tof);
    def("normalize_voxels", normalize_voxels);
  }

}}  // namespace recviewer::ext

BOOST_PYTHON_MODULE(recviewer_ext) {
  recviewer::ext::init_module();
}
