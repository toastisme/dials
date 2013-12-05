/*
 * mask_foreground.h
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#ifndef DIALS_ALGORITHMS_SHOEBOX_MASK_FOREGROUND_H
#define DIALS_ALGORITHMS_SHOEBOX_MASK_FOREGROUND_H

#include <scitbx/vec2.h>
#include <scitbx/vec3.h>
#include <dxtbx/model/beam.h>
#include <dxtbx/model/goniometer.h>
#include <dxtbx/model/detector.h>
#include <dxtbx/model/scan.h>
#include <dials/model/data/reflection.h>
#include <dials/algorithms/reflection_basis/coordinate_system.h>
#include <dials/algorithms/reflection_basis/beam_vector_map.h>
#include <dials/algorithms/shoebox/mask_code.h>
#include <dials/error.h>

namespace dials { namespace algorithms { namespace shoebox {

  using scitbx::vec2;
  using scitbx::vec3;
  using scitbx::af::int6;
  using dxtbx::model::Beam;
  using dxtbx::model::Detector;
  using dxtbx::model::Goniometer;
  using dxtbx::model::Scan;
  using dials::model::Reflection;
  using dials::algorithms::reflection_basis::CoordinateSystem;
  using dials::algorithms::reflection_basis::transform::beam_vector_map;

  /**
   * A class to mask foreground/background pixels
   */
  class MaskForeground {
  public:

    /**
     * Initialise the stuff needed to create the mask.
     * @param beam The beam model
     * @param detector The detector model
     * @param gonio The goniometer model
     * @param scan The scan model
     * @param delta_b nsigma * sigma_divergence
     * @param delta_m nsigma * mosaicity
     */
    MaskForeground(const Beam &beam, const Detector &detector,
                   const Goniometer &gonio, const Scan &scan,
                   const af::const_ref<double> &delta_b,
                   const af::const_ref<double> &delta_m)
      : s1_map_(beam_vector_map(detector, beam, false)),
        m2_(gonio.get_rotation_axis()),
        s0_(beam.get_s0()),
        phi0_(scan.get_oscillation()[0]),
        dphi_(scan.get_oscillation()[1]),
        index0_(scan.get_array_range()[0]) {
      DIALS_ASSERT(delta_b.size() == delta_m.size());
      DIALS_ASSERT(delta_b.size() == scan.get_num_images());
      DIALS_ASSERT(delta_b.all_gt(0.0));
      DIALS_ASSERT(delta_m.all_gt(0.0));
      delta_b_r_.resize(delta_b.size());
      delta_m_r_.resize(delta_m.size());
      for (std::size_t i = 0; i < delta_b.size(); ++i) {
        delta_b_r_[i] = 1.0 / delta_b[i];
        delta_m_r_[i] = 1.0 / delta_m[i];
      }
    }

    /**
     * Initialise the stuff needed to create the mask.
     * @param beam The beam model
     * @param detector The detector model
     * @param gonio The goniometer model
     * @param scan The scan model
     * @param delta_b nsigma * sigma_divergence
     * @param delta_m nsigma * mosaicity
     */
    MaskForeground(const Beam &beam, const Detector &detector,
                   const Goniometer &gonio, const Scan &scan,
                   double delta_b, double delta_m)
      : s1_map_(beam_vector_map(detector, beam, false)),
        m2_(gonio.get_rotation_axis()),
        s0_(beam.get_s0()),
        phi0_(scan.get_oscillation()[0]),
        dphi_(scan.get_oscillation()[1]),
        index0_(scan.get_array_range()[0]) {
      DIALS_ASSERT(delta_b > 0.0);
      DIALS_ASSERT(delta_m > 0.0);
      delta_b_r_.resize(scan.get_num_images());
      delta_m_r_.resize(scan.get_num_images());
      for (std::size_t i = 0; i < scan.get_num_images(); ++i) {
        delta_b_r_[i] = 1.0 / delta_b;
        delta_m_r_[i] = 1.0 / delta_m;
      }
    }

    /**
     * Set all the foreground/background pixels in the reflection mask.
     * @param reflection The reflection to mask
     */
    void operator()(Reflection &reflection) const {

      // Ensure the reflection is valid
      if (reflection.is_valid()) {

        // Get some bits from the reflection
        af::ref< int, af::c_grid<3> > mask = reflection.get_shoebox_mask().ref();
        vec3<double> s1 = reflection.get_beam_vector();
        double phi = reflection.get_rotation_angle();
        int6 bbox = reflection.get_bounding_box();
        int x0 = bbox[0], x1 = bbox[1];
        int y0 = bbox[2], y1 = bbox[3];
        int z0 = bbox[4], z1 = bbox[5];
        int xsize = x1 - x0;
        int ysize = y1 - y0;
        int zsize = z1 - z0;

        int z = (int)floor(reflection.get_frame_number());
        DIALS_ASSERT(z >= 0 && z < delta_b_r_.size());
        double delta_b_r = delta_b_r_[z];
        //double delta_m_r = delta_m_r_[z];

        // Check the size of the mask
        DIALS_ASSERT(mask.accessor()[0] == zsize);
        DIALS_ASSERT(mask.accessor()[1] == ysize);
        DIALS_ASSERT(mask.accessor()[2] == xsize);

        // Create the coordinate system and generators
        CoordinateSystem cs(m2_, s0_, s1, phi);

        // Get the size of the image
        std::size_t width = s1_map_.accessor()[1];
        std::size_t height = s1_map_.accessor()[0];

        // Loop through all the pixels in the shoebox, transform the point
        // to the reciprocal space coordinate system and check that it is
        // within the ellipse defined by:
        // (c1 / delta_b)^2 + (c2 / delta_b)^2 <= 1
        // Mark those points within as Foreground and those without as
        // Background.
        for (int j = 0; j < ysize; ++j) {
          for (int i = 0; i < xsize; ++i) {
            if (x0 + i >= 0 && y0 + j >= 0 &&
                x0 + i < width && y0 + j < height) {
              vec2<double> gxy = cs.from_beam_vector(s1_map_(y0 + j, x0 + i));
              double gxa2 = (gxy[0] * delta_b_r)*(gxy[0] * delta_b_r);
              double gyb2 = (gxy[1] * delta_b_r)*(gxy[1] * delta_b_r);
              int mask_value = (gxa2 + gyb2 <= 1.0) ? Foreground : Background;
              for (std::size_t k = 0; k < zsize; ++k) {
                mask(k, j, i) |= mask_value;
              }
            }
          }
        }
      }
    }

    /**
     * Mask all the foreground/background pixels for all the reflections
     * @param reflections The reflection list
     */
    void operator()(af::ref<Reflection> reflections) const {
      for (std::size_t i = 0; i < reflections.size(); ++i) {
        this->operator()(reflections[i]);
      }
    }

  private:
    af::versa< vec3<double>, af::c_grid<2> > s1_map_;
    vec3<double> m2_;
    vec3<double> s0_;
    double phi0_;
    double dphi_;
    int index0_;
    af::shared<double> delta_b_r_;
    af::shared<double> delta_m_r_;
  };

}}} // namespace dials::algorithms::shoebox

#endif /* DIALS_ALGORITHMS_SHOEBOX_MASK_FOREGROUND_H */
