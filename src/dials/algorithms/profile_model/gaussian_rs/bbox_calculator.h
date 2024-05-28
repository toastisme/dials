/*
 * bbox_calculator.h
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#ifndef DIALS_ALGORITHMS_PROFILE_MODEL_GAUSSIAN_RS_BBOX_CALCULATOR_H
#define DIALS_ALGORITHMS_PROFILE_MODEL_GAUSSIAN_RS_BBOX_CALCULATOR_H

#include <cmath>
#include <boost/shared_ptr.hpp>
#include <scitbx/constants.h>
#include <scitbx/vec3.h>
#include <scitbx/array_family/tiny_types.h>
#include <scitbx/array_family/ref_reductions.h>
#include <dxtbx/model/beam.h>
#include <dxtbx/model/detector.h>
#include <dxtbx/model/goniometer.h>
#include <dxtbx/model/sequence.h>
#include <dials/array_family/scitbx_shared_and_versa.h>
#include <dials/algorithms/profile_model/gaussian_rs/coordinate_system.h>

namespace dials {
  namespace algorithms {
    namespace profile_model {
      namespace gaussian_rs {

  // Use a load of stuff from other namespaces
  using dxtbx::model::Detector;
  using dxtbx::model::Goniometer;
  using dxtbx::model::MonoBeam;
  using dxtbx::model::PolyBeam;
  using dxtbx::model::Scan;
  using dxtbx::model::TOFSequence;
  using scitbx::vec2;
  using scitbx::vec3;
  using scitbx::af::double2;
  using scitbx::af::double3;
  using scitbx::af::double4;
  using scitbx::af::int6;
  using scitbx::af::max;
  using scitbx::af::min;
  using std::ceil;
  using std::floor;

  /**
   * Interface for bounding box calculator.
   */
  class BBoxCalculatorIface {
  public:
    virtual ~BBoxCalculatorIface() {}

    virtual int6 single(vec3<double> s1, double frame, std::size_t panel) const = 0;

    virtual af::shared<int6> array(const af::const_ref<vec3<double> > &s1,
                                   const af::const_ref<double> &frame,
                                   const af::const_ref<std::size_t> &panel) const = 0;
  };

  /** Calculate the bounding box for each reflection */
  class BBoxCalculator3D : public BBoxCalculatorIface {
  public:
    /**
     * Initialise the bounding box calculation.
     * @param beam The beam parameters
     * @param detector The detector parameters
     * @param goniometer The goniometer parameters
     * @param delta_divergence The xds delta_divergence parameter
     * @param delta_mosaicity The xds delta_mosaicity parameter
     */
    BBoxCalculator3D(const boost::python::object &beam,
                     const Detector &detector,
                     const Goniometer &gonio,
                     const boost::python::object &scan,
                     double delta_divergence,
                     double delta_mosaicity)
        : s0_(boost::python::extract<vec3<double> >(beam.attr("get_s0"))),
          m2_(gonio.get_rotation_axis()),
          detector_(detector),
          scan_(boost::python::extract<Scan>(scan)),
          delta_divergence_(1, delta_divergence),
          delta_mosaicity_(1, delta_mosaicity) {
      DIALS_ASSERT(delta_divergence > 0.0);
      DIALS_ASSERT(delta_mosaicity > 0.0);
    }

    /**
     * Initialise the bounding box calculation.
     * @param beam The beam parameters
     * @param detector The detector parameters
     * @param goniometer The goniometer parameters
     * @param delta_divergence The xds delta_divergence parameter
     * @param delta_mosaicity The xds delta_mosaicity parameter
     */
    BBoxCalculator3D(const boost::python::object &beam,
                     const Detector &detector,
                     const Goniometer &gonio,
                     const boost::python::object &scan,
                     const af::const_ref<double> &delta_divergence,
                     const af::const_ref<double> &delta_mosaicity)
        : s0_(boost::python::extract<vec3<double> >(beam.attr("get_s0"))),
          m2_(gonio.get_rotation_axis()),
          detector_(detector),
          scan_(boost::python::extract<Scan>(scan)),
          delta_divergence_(delta_divergence.begin(), delta_divergence.end()),
          delta_mosaicity_(delta_mosaicity.begin(), delta_mosaicity.end()) {
      DIALS_ASSERT(delta_divergence.all_gt(0.0));
      DIALS_ASSERT(delta_mosaicity.all_gt(0.0));
      DIALS_ASSERT(delta_divergence_.size() == delta_mosaicity_.size());
      DIALS_ASSERT(delta_divergence_.size()
                   == boost::python::extract<int>(scan.attr("get_num_images")));
      DIALS_ASSERT(delta_divergence_.size() > 0);
    }

    /**
     * Calculate the bbox on the detector image volume for the reflection.
     *
     * The roi is calculated using the parameters delta_divergence and
     * delta_mosaicity. The reflection mask comprises all pixels where:
     *  |e1| <= delta_d, |e2| <= delta_d, |e3| <= delta_m
     *
     * We transform the coordinates of the box
     *   (-delta_d, -delta_d, 0)
     *   (+delta_d, -delta_d, 0)
     *   (-delta_d, +delta_d, 0)
     *   (+delta_d, +delta_d, 0)
     *
     * to the detector image volume and return the minimum and maximum values
     * for the x, y, z image volume coordinates.
     *
     * @param s1 The diffracted beam vector
     * @param frame The predicted frame number
     * @returns A 6 element array: (minx, maxx, miny, maxy, minz, maxz)
     */
    virtual int6 single(vec3<double> s1, double frame, std::size_t panel) const {
      // Ensure our values are ok
      DIALS_ASSERT(s1.length_sq() > 0);

      // Get the rotation angle
      double phi = scan_.get_angle_from_array_index(frame);

      // Create the coordinate system for the reflection
      CoordinateSystem xcs(m2_, s0_, s1, phi);

      // Get the divergence and mosaicity for this point
      double delta_d = 0.0;
      double delta_m = 0.0;
      if (delta_divergence_.size() == 1) {
        delta_d = delta_divergence_[0];
        delta_m = delta_mosaicity_[0];
      } else {
        int frame0 = scan_.get_array_range()[0];
        int index = (int)std::floor(frame) - frame0;
        if (index < 0) {
          delta_d = delta_divergence_.front();
          delta_m = delta_mosaicity_.front();
        } else if (index >= delta_divergence_.size()) {
          delta_d = delta_divergence_.back();
          delta_m = delta_mosaicity_.back();
        } else {
          delta_d = delta_divergence_[index];
          delta_m = delta_mosaicity_[index];
        }
      }

      // Calculate the beam vectors at the following xds coordinates:
      //   (-delta_d, -delta_d, 0)
      //   (+delta_d, -delta_d, 0)
      //   (-delta_d, +delta_d, 0)
      //   (+delta_d, +delta_d, 0)
      double point = delta_d;
      double3 sdash1 = xcs.to_beam_vector(double2(-point, -point));
      double3 sdash2 = xcs.to_beam_vector(double2(+point, -point));
      double3 sdash3 = xcs.to_beam_vector(double2(-point, +point));
      double3 sdash4 = xcs.to_beam_vector(double2(+point, +point));

      // Get the detector coordinates (px) at the ray intersections
      double2 xy1 = detector_[panel].get_ray_intersection_px(sdash1);
      double2 xy2 = detector_[panel].get_ray_intersection_px(sdash2);
      double2 xy3 = detector_[panel].get_ray_intersection_px(sdash3);
      double2 xy4 = detector_[panel].get_ray_intersection_px(sdash4);

      /// Calculate the rotation angles at the following XDS
      // e3 coordinates: -delta_m, +delta_m
      double phi1 = xcs.to_rotation_angle_fast(-delta_m);
      double phi2 = xcs.to_rotation_angle_fast(+delta_m);

      // Get the array indices at the rotation angles
      double z1 = scan_.get_array_index_from_angle(phi1);
      double z2 = scan_.get_array_index_from_angle(phi2);

      // Return the roi in the following form:
      // (minx, maxx, miny, maxy, minz, maxz)
      // Min's are rounded down to the nearest integer, Max's are rounded up
      double4 x(xy1[0], xy2[0], xy3[0], xy4[0]);
      double4 y(xy1[1], xy2[1], xy3[1], xy4[1]);
      double2 z(z1, z2);
      int6 bbox((int)floor(min(x)),
                (int)ceil(max(x)),
                (int)floor(min(y)),
                (int)ceil(max(y)),
                (int)floor(min(z)),
                (int)ceil(max(z)));

      vec2<int> array_range = scan_.get_array_range();
      DIALS_ASSERT(bbox[4] <= frame && frame < bbox[5]);
      bbox[4] = std::max(bbox[4], array_range[0]);
      bbox[4] = std::min(bbox[4], array_range[1] - 1);
      bbox[5] = std::min(bbox[5], array_range[1]);
      bbox[5] = std::max(bbox[5], array_range[0] + 1);
      DIALS_ASSERT(bbox[1] > bbox[0]);
      DIALS_ASSERT(bbox[3] > bbox[2]);
      DIALS_ASSERT(bbox[5] > bbox[4]);
      return bbox;
    }

    /**
     * Calculate the rois for an array of reflections given by the array of
     * diffracted beam vectors and rotation angles.
     * @param s1 The array of diffracted beam vectors
     * @param frame The array of frame numbers.
     */
    virtual af::shared<int6> array(const af::const_ref<vec3<double> > &s1,
                                   const af::const_ref<double> &frame,
                                   const af::const_ref<std::size_t> &panel) const {
      DIALS_ASSERT(s1.size() == frame.size());
      DIALS_ASSERT(s1.size() == panel.size());
      af::shared<int6> result(s1.size(), af::init_functor_null<int6>());
      for (std::size_t i = 0; i < s1.size(); ++i) {
        result[i] = single(s1[i], frame[i], panel[i]);
      }
      return result;
    }

  private:
    vec3<double> s0_;
    vec3<double> m2_;
    Detector detector_;
    Scan scan_;
    af::shared<double> delta_divergence_;
    af::shared<double> delta_mosaicity_;
  };

  /** Calculate the bounding box for each reflection */
  class BBoxCalculator2D : public BBoxCalculatorIface {
  public:
    /**
     * Initialise the bounding box calculation.
     * @param beam The beam parameters
     * @param detector The detector parameters
     * @param delta_divergence The xds delta_divergence parameter
     * @param delta_mosaicity The xds delta_mosaicity parameter
     */
    BBoxCalculator2D(const boost::python::object &beam,
                     const Detector &detector,
                     double delta_divergence,
                     double delta_mosaicity)
        : s0_(boost::python::extract<vec3<double> >(beam.attr("get_s0"))),
          detector_(detector),
          delta_divergence_(delta_divergence) {
      DIALS_ASSERT(delta_divergence > 0.0);
      DIALS_ASSERT(delta_mosaicity >= 0.0);
    }

    /**
     * Calculate the bbox on the detector image volume for the reflection.
     *
     * The roi is calculated using the parameters delta_divergence and
     * delta_mosaicity. The reflection mask comprises all pixels where:
     *  |e1| <= delta_d, |e2| <= delta_d, |e3| <= delta_m
     *
     * We transform the coordinates of the box
     *   (-delta_d, -delta_d, 0)
     *   (+delta_d, -delta_d, 0)
     *   (-delta_d, +delta_d, 0)
     *   (+delta_d, +delta_d, 0)
     *
     * to the detector image volume and return the minimum and maximum values
     * for the x, y, z image volume coordinates.
     *
     * @param s1 The diffracted beam vector
     * @param frame The predicted frame number
     * @returns A 6 element array: (minx, maxx, miny, maxy, minz, maxz)
     */
    virtual int6 single(vec3<double> s1, double frame, std::size_t panel) const {
      // Ensure our values are ok
      DIALS_ASSERT(s1.length_sq() > 0);

      // Create the coordinate system for the reflection
      CoordinateSystem2d xcs(s0_, s1);

      // Get the divergence and mosaicity for this point
      double delta_d = delta_divergence_;

      // Calculate the beam vectors at the following xds coordinates:
      //   (-delta_d, -delta_d, 0)
      //   (+delta_d, -delta_d, 0)
      //   (-delta_d, +delta_d, 0)
      //   (+delta_d, +delta_d, 0)
      double point = delta_d;
      double3 sdash1 = xcs.to_beam_vector(double2(-point, -point));
      double3 sdash2 = xcs.to_beam_vector(double2(+point, -point));
      double3 sdash3 = xcs.to_beam_vector(double2(-point, +point));
      double3 sdash4 = xcs.to_beam_vector(double2(+point, +point));

      // Get the detector coordinates (px) at the ray intersections
      double2 xy1 = detector_[panel].get_ray_intersection_px(sdash1);
      double2 xy2 = detector_[panel].get_ray_intersection_px(sdash2);
      double2 xy3 = detector_[panel].get_ray_intersection_px(sdash3);
      double2 xy4 = detector_[panel].get_ray_intersection_px(sdash4);

      // Return the roi in the following form:
      // (minx, maxx, miny, maxy, minz, maxz)
      // Min's are rounded down to the nearest integer, Max's are rounded up
      double4 x(xy1[0], xy2[0], xy3[0], xy4[0]);
      double4 y(xy1[1], xy2[1], xy3[1], xy4[1]);
      int6 bbox((int)floor(min(x)),
                (int)ceil(max(x)),
                (int)floor(min(y)),
                (int)ceil(max(y)),
                (int)floor(frame),
                (int)floor(frame) + 1);
      DIALS_ASSERT(bbox[1] > bbox[0]);
      DIALS_ASSERT(bbox[3] > bbox[2]);
      DIALS_ASSERT(bbox[5] > bbox[4]);
      return bbox;
    }

    /**
     * Calculate the rois for an array of reflections given by the array of
     * diffracted beam vectors and rotation angles.
     * @param s1 The array of diffracted beam vectors
     * @param phi The array of rotation angles.
     */
    virtual af::shared<int6> array(const af::const_ref<vec3<double> > &s1,
                                   const af::const_ref<double> &frame,
                                   const af::const_ref<std::size_t> &panel) const {
      DIALS_ASSERT(s1.size() == frame.size());
      DIALS_ASSERT(s1.size() == panel.size());
      af::shared<int6> result(s1.size(), af::init_functor_null<int6>());
      for (std::size_t i = 0; i < s1.size(); ++i) {
        result[i] = single(s1[i], frame[i], panel[i]);
      }
      return result;
    }

  private:
    vec3<double> s0_;
    Detector detector_;
    double delta_divergence_;
  };

  class BBoxCalculatorTOF {
  public:
    /**
     * Initialise the bounding box calculation.
     * @param beam The beam parameters
     * @param detector The detector parameters
     * @param delta_divergence The xds delta_divergence parameter
     * @param delta_mosaicity The xds delta_mosaicity parameter
     */
    BBoxCalculatorTOF(const PolyBeam &beam,
                      const Detector &detector,
                      const TOFSequence &sequence,
                      double delta_divergence,
                      double delta_mosaicity)
        : detector_(detector),
          sequence_(sequence),
          beam_(beam),
          delta_divergence_(delta_divergence),
          delta_mosaicity_(delta_mosaicity) {
      DIALS_ASSERT(delta_divergence > 0.0);
      DIALS_ASSERT(delta_mosaicity >= 0.0);
    }

    /**
     * Calculate the bbox on the detector image volume for the reflection.
     *
     * The roi is calculated using the parameters delta_divergence and
     * delta_mosaicity. The reflection mask comprises all pixels where:
     *  |e1| <= delta_d, |e2| <= delta_d, |e3| <= delta_m
     *
     * We transform the coordinates of the box
     *   (-delta_d, -delta_d, 0)
     *   (+delta_d, -delta_d, 0)
     *   (-delta_d, +delta_d, 0)
     *   (+delta_d, +delta_d, 0)
     *
     * to the detector image volume and return the minimum and maximum values
     * for the x, y, z image volume coordinates.
     *
     * @param s1 The diffracted beam vector
     * @param frame The predicted frame number
     * @returns A 6 element array: (minx, maxx, miny, maxy, minz, maxz)
     */
    virtual int6 single(vec3<double> s0,
                        vec3<double> s1,
                        double frame,
                        double L1,
                        std::size_t panel) const {
      // Ensure our values are ok
      DIALS_ASSERT(s1.length_sq() > 0);

      // Create the coordinate system for the reflection
      CoordinateSystemTOF xcs(s0, s1, L1);

      // Get the divergence and mosaicity for this point
      double delta_d = delta_divergence_;
      double delta_m = delta_mosaicity_;

      // Calculate the beam vectors at the following xds coordinates:
      //   (-delta_d, -delta_d, 0)
      //   (+delta_d, -delta_d, 0)
      //   (-delta_d, +delta_d, 0)
      //   (+delta_d, +delta_d, 0)
      double point = delta_d;
      double3 sdash1 = xcs.to_beam_vector(double2(-point, -point));
      double3 sdash2 = xcs.to_beam_vector(double2(+point, -point));
      double3 sdash3 = xcs.to_beam_vector(double2(-point, +point));
      double3 sdash4 = xcs.to_beam_vector(double2(+point, +point));

      // Get the detector coordinates (px) at the ray intersections
      double2 xy1 = detector_[panel].get_ray_intersection_px(sdash1);
      double2 xy2 = detector_[panel].get_ray_intersection_px(sdash2);
      double2 xy3 = detector_[panel].get_ray_intersection_px(sdash3);
      double2 xy4 = detector_[panel].get_ray_intersection_px(sdash4);

      // Return the roi in the following form:
      // (minx, maxx, miny, maxy, minz, maxz)
      // Min's are rounded down to the nearest integer, Max's are rounded up
      double4 x(xy1[0], xy2[0], xy3[0], xy4[0]);
      double4 y(xy1[1], xy2[1], xy3[1], xy4[1]);

      int x0 = (int)floor(min(x));
      int x1 = (int)ceil(max(x));
      int y0 = (int)floor(min(y));
      int y1 = (int)ceil(max(y));

      /// Calculate the rotation angles at the following XDS
      // e3 coordinates: -delta_m, +delta_m
      /*
      double wavelength1 = xcs.to_wavelength(-delta_m, s1);
      double wavelength2 = xcs.to_wavelength(+delta_m, s1);
      double tof1 = beam_.get_tof_from_wavelength(wavelength1, L1);
      double tof2 = beam_.get_tof_from_wavelength(wavelength2, L1);

      // Get the array indices at the rotation angles
      double z1 = sequence_.get_frame_from_tof(tof1);
      double z2 = sequence_.get_frame_from_tof(tof2);
      double2 z(z1, z2);
      */
      double z0 = frame - delta_m;
      if (z0 < 0) {
        z0 = 0;
      }
      double max_z = sequence_.get_array_range()[1];
      double z1 = frame + delta_m;
      if (z1 > max_z) {
        z1 = max_z;
      }
      if (z0 > z1) {
        z0 = -1;
        z1 = 0;
      }

      int6 bbox(x0, x1, y0, y1, z0, z1);
      DIALS_ASSERT(bbox[1] > bbox[0]);
      DIALS_ASSERT(bbox[3] > bbox[2]);
      DIALS_ASSERT(bbox[5] > bbox[4]);
      return bbox;
    }

    /**
     * Calculate the rois for an array of reflections given by the array of
     * diffracted beam vectors and rotation angles.
     * @param s1 The array of diffracted beam vectors
     * @param phi The array of rotation angles.
     */
    virtual af::shared<int6> array(const af::const_ref<vec3<double> > &s0,
                                   const af::const_ref<vec3<double> > &s1,
                                   const af::const_ref<double> &frame,
                                   const af::const_ref<double> &L1,
                                   const af::const_ref<std::size_t> &panel) const {
      DIALS_ASSERT(s1.size() == frame.size());
      DIALS_ASSERT(s1.size() == panel.size());
      DIALS_ASSERT(s0.size() == frame.size());
      DIALS_ASSERT(s0.size() == panel.size());
      af::shared<int6> result(s1.size(), af::init_functor_null<int6>());
      for (std::size_t i = 0; i < s1.size(); ++i) {
        result[i] = single(s0[i], s1[i], frame[i], L1[i], panel[i]);
      }
      return result;
    }

  private:
    Detector detector_;
    TOFSequence sequence_;
    PolyBeam beam_;
    double delta_divergence_;
    double delta_mosaicity_;
  };
  /**
   * Class to help compute bbox for multiple experiments.
   */
  class BBoxMultiCalculator {
  public:
    /**
     * Add a bbox calculator to the list.
     */
    void push_back(boost::shared_ptr<BBoxCalculatorIface> obj) {
      compute_.push_back(obj);
    }

    /**
     * Get the number of calculators.
     */
    std::size_t size() const {
      return compute_.size();
    }

    /**
     * Calculate the rois for an array of reflections given by the array of
     * diffracted beam vectors and rotation angles.
     * @param id The experiment id
     * @param s1 The array of diffracted beam vectors
     * @param phi The array of rotation angles.
     * @param panel The panel number
     */
    af::shared<int6> operator()(const af::const_ref<std::size_t> &id,
                                const af::const_ref<vec3<double> > &s1,
                                const af::const_ref<double> &phi,
                                const af::const_ref<std::size_t> &panel) const {
      DIALS_ASSERT(s1.size() == id.size());
      DIALS_ASSERT(s1.size() == phi.size());
      DIALS_ASSERT(s1.size() == panel.size());
      af::shared<int6> result(s1.size(), af::init_functor_null<int6>());
      for (std::size_t i = 0; i < s1.size(); ++i) {
        DIALS_ASSERT(id[i] < size());
        result[i] = compute_[id[i]]->single(s1[i], phi[i], panel[i]);
      }
      return result;
    }

  private:
    std::vector<boost::shared_ptr<BBoxCalculatorIface> > compute_;
  };

}}}}  // namespace dials::algorithms::profile_model::gaussian_rs

#endif  // DIALS_ALGORITHMS_PROFILE_MODEL_GAUSSIAN_RS_BBOX_CALCULATOR_H
