/*
 * gaussian_rs_ext.cc
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <boost_adaptbx/std_pair_conversion.h>
#include <dials/algorithms/profile_model/gaussian_rs/bbox_calculator.h>
#include <dials/algorithms/profile_model/gaussian_rs/partiality_calculator.h>
#include <dials/algorithms/profile_model/gaussian_rs/mask_calculator.h>
#include <dials/algorithms/profile_model/gaussian_rs/ideal_profile.h>
#include <dials/algorithms/profile_model/gaussian_rs/coordinate_system.h>
#include <dials/algorithms/profile_model/gaussian_rs/modeller.h>
#include <dials/algorithms/profile_model/modeller/boost_python/empirical_profile_modeller_wrapper.h>

namespace dials {
  namespace algorithms {
    namespace profile_model {
      namespace gaussian_rs {
  namespace boost_python {

    using namespace boost::python;

    /**
     * Helper function to calculate zeta for an array of s1.
     */
    af::shared<double> zeta_factor_array(vec3<double> m2,
                                         vec3<double> s0,
                                         const af::const_ref<vec3<double> >& s1) {
      af::shared<double> result(s1.size(), 0);
      for (std::size_t i = 0; i < s1.size(); ++i) {
        result[i] = zeta_factor(m2, s0, s1[i]);
      }
      return result;
    }

    /**
     * Helper function to calculate zeta for an array of s1 with experiment
     * indices
     */
    af::shared<double> zeta_factor_array_multi(const af::const_ref<vec3<double> >& m2,
                                               const af::const_ref<vec3<double> >& s0,
                                               const af::const_ref<vec3<double> >& s1,
                                               const af::const_ref<int>& index) {
      DIALS_ASSERT(m2.size() == s0.size());
      DIALS_ASSERT(s1.size() == index.size());
      af::shared<double> result(s1.size());
      for (std::size_t i = 0; i < index.size(); ++i) {
        int j = index[i];
        DIALS_ASSERT(j >= 0);
        DIALS_ASSERT(j < m2.size());
        result[i] = zeta_factor(m2[j], s0[j], s1[i]);
      }
      return result;
    }

    struct GaussianRSProfileModellerPickleSuite : boost::python::pickle_suite {
      static boost::python::tuple getinitargs(const GaussianRSProfileModeller& obj) {
        return boost::python::make_tuple(obj.beam(),
                                         obj.detector(),
                                         obj.goniometer(),
                                         obj.scan(),
                                         obj.sigma_b(),
                                         obj.sigma_m(),
                                         obj.n_sigma(),
                                         obj.grid_size(),
                                         obj.num_scan_points(),
                                         obj.threshold(),
                                         obj.grid_method(),
                                         obj.fit_method());
      }

      static boost::python::tuple getstate(const GaussianRSProfileModeller& obj) {
        typedef GaussianRSProfileModeller::data_type data_type;
        typedef GaussianRSProfileModeller::mask_type mask_type;
        boost::python::list data_list;
        boost::python::list mask_list;
        boost::python::list nref_list;
        boost::python::list panel_list;
        for (std::size_t i = 0; i < obj.size(); ++i) {
          nref_list.append(obj.n_reflections(i));
          panel_list.append(obj.panel(i));
          try {
            data_list.append(obj.data(i));
            mask_list.append(obj.mask(i));
          } catch (dials::error const&) {
            data_list.append(data_type());
            mask_list.append(mask_type());
          }
        }
        return boost::python::make_tuple(
          data_list, mask_list, nref_list, obj.finalized(), panel_list);
      }

      static void setstate(GaussianRSProfileModeller& obj, boost::python::tuple state) {
        typedef GaussianRSProfileModeller::data_type data_type;
        typedef GaussianRSProfileModeller::mask_type mask_type;
        DIALS_ASSERT(boost::python::len(state) == 5);
        boost::python::list data_list = extract<boost::python::list>(state[0]);
        boost::python::list mask_list = extract<boost::python::list>(state[1]);
        boost::python::list nref_list = extract<boost::python::list>(state[2]);
        bool finalized = extract<bool>(state[3]);
        boost::python::list panel_list = extract<boost::python::list>(state[4]);
        DIALS_ASSERT(boost::python::len(data_list) == boost::python::len(mask_list));
        DIALS_ASSERT(boost::python::len(data_list) == obj.size());
        DIALS_ASSERT(boost::python::len(nref_list) == obj.size());
        DIALS_ASSERT(boost::python::len(panel_list) == obj.size());
        for (std::size_t i = 0; i < obj.size(); ++i) {
          af::flex_double d = boost::python::extract<af::flex_double>(data_list[i]);
          af::flex_bool m = boost::python::extract<af::flex_bool>(mask_list[i]);
          DIALS_ASSERT(d.accessor().all().size() == 3);
          DIALS_ASSERT(m.accessor().all().size() == 3);
          obj.set_data(i, data_type(d.handle(), af::c_grid<3>(d.accessor())));
          obj.set_mask(i, mask_type(m.handle(), af::c_grid<3>(m.accessor())));
          obj.set_n_reflections(i, boost::python::extract<std::size_t>(nref_list[i]));
          obj.set_panel(i, boost::python::extract<int>(panel_list[i]));
        }
        obj.set_finalized(finalized);
      }
    };

    void export_modeller() {
      typedef class_<GaussianRSProfileModeller,  //> class_type;
                     bases<ProfileModellerIface> >
        class_type;

      /* class_type result("GaussianRSProfileModeller", no_init); */

      class_type result =
        dials::algorithms::boost_python::empirical_profile_modeller_wrapper<
          GaussianRSProfileModeller>("GaussianRSProfileModeller");
      result
        .def(init<const boost::python::object&,
                  const Detector&,
                  const Goniometer&,
                  const boost::python::object&,
                  double,
                  double,
                  double,
                  std::size_t,
                  std::size_t,
                  double,
                  int,
                  int>())
        .def("coord", &GaussianRSProfileModeller::coord)
        .def("coord_with_panel", &GaussianRSProfileModeller::coord_with_panel)
        .def("model", &GaussianRSProfileModeller::model, (arg("reflections")))
        .def("model_tof", &GaussianRSProfileModeller::model_tof, (arg("reflections")))
        .def("model_tof_return",
             &GaussianRSProfileModeller::model_tof_return,
             (arg("reflections")))
        .def("accumulate", &GaussianRSProfileModeller::accumulate, (arg("other")))
        .def("fit_reciprocal_space",
             &GaussianRSProfileModeller::fit_reciprocal_space,
             (arg("reflections")))
        .def("fit_reciprocal_space_tof",
             &GaussianRSProfileModeller::fit_reciprocal_space_tof,
             (arg("reflections")))
        .def("fit_reciprocal_space_tof_return",
             &GaussianRSProfileModeller::fit_reciprocal_space_tof_return,
             (arg("reflections")))
        .def("normalize_profiles", &GaussianRSProfileModeller::normalize_profiles)
        .def("nearest_profile", &GaussianRSProfileModeller::nearest_profile)
        .def("nearest_n_profiles", &GaussianRSProfileModeller::nearest_n_profiles)
        .def("profile_weight", &GaussianRSProfileModeller::profile_weight)
        .def("profile_coord", &GaussianRSProfileModeller::profile_coord)
        .def_pickle(GaussianRSProfileModellerPickleSuite());

      scope in_modeller = result;

      enum_<GaussianRSProfileModeller::GridMethod>("GridMethod")
        .value("single", GaussianRSProfileModeller::Single)
        .value("regular_grid", GaussianRSProfileModeller::RegularGrid)
        .value("circular_grid", GaussianRSProfileModeller::CircularGrid)
        .value("spherical_grid", GaussianRSProfileModeller::SphericalGrid);

      enum_<GaussianRSProfileModeller::FitMethod>("FitMethod")
        .value("reciprocal_space", GaussianRSProfileModeller::ReciprocalSpace)
        .value("reciprocal_space_tof", GaussianRSProfileModeller::ReciprocalSpace)
        .value("detector_space", GaussianRSProfileModeller::DetectorSpace);

      /* register_ptr_to_python< boost::shared_ptr<GaussianRSProfileModeller> >(); */
      /* implicitly_convertible< */
      /*   boost::shared_ptr<GaussianRSProfileModeller>, */
      /* boost::shared_ptr<ProfileModellerIface> >(); */
    }

    BOOST_PYTHON_MODULE(dials_algorithms_profile_model_gaussian_rs_ext) {
      export_modeller();

      class_<BBoxCalculatorIface, boost::noncopyable>("BBoxCalculatorIface", no_init)
        .def("__call__",
             &BBoxCalculatorIface::single,
             (arg("s1"), arg("frame"), arg("panel")))
        .def("__call__",
             &BBoxCalculatorIface::array,
             (arg("s1"), arg("frame"), arg("panel")));

      class_<BBoxCalculator3D, bases<BBoxCalculatorIface> >("BBoxCalculator3D", no_init)
        .def(init<const boost::python::object&,
                  const Detector&,
                  const Goniometer&,
                  const boost::python::object&,
                  double,
                  double>((arg("beam"),
                           arg("detector"),
                           arg("goniometer"),
                           arg("scan"),
                           arg("delta_divergence"),
                           arg("delta_mosaicity"))))
        .def(init<const boost::python::object&,
                  const Detector&,
                  const Goniometer&,
                  const boost::python::object&,
                  const af::const_ref<double>,
                  const af::const_ref<double> >((arg("beam"),
                                                 arg("detector"),
                                                 arg("goniometer"),
                                                 arg("scan"),
                                                 arg("delta_divergence"),
                                                 arg("delta_mosaicity"))));

      class_<BBoxCalculator2D, bases<BBoxCalculatorIface> >("BBoxCalculator2D", no_init)
        .def(init<const boost::python::object&, const Detector&, double, double>(
          (arg("beam"),
           arg("detector"),
           arg("delta_divergence"),
           arg("delta_mosaicity"))));

      class_<BBoxCalculatorTOF>("BBoxCalculatorTOF", no_init)
        .def(init<const PolyBeam&, const Detector&, const TOFSequence&, double, double>(
          (arg("beam"),
           arg("detector"),
           arg("sequence"),
           arg("delta_divergence"),
           arg("delta_mosaicity"))))
        .def("__call__",
             &BBoxCalculatorTOF::single,
             (arg("s0"), arg("s1"), arg("frame"), arg("L1"), arg("panel")))
        .def("__call__",
             &BBoxCalculatorTOF::array,
             (arg("s0"), arg("s1"), arg("frame"), arg("L1"), arg("panel")));

      class_<BBoxMultiCalculator>("BBoxMultiCalculator")
        .def("append", &BBoxMultiCalculator::push_back)
        .def("__len__", &BBoxMultiCalculator::size)
        .def("__call__", &BBoxMultiCalculator::operator());

      class_<MaskCalculatorIface, boost::noncopyable>("MaskCalculatorIface", no_init)
        .def("__call__",
             &MaskCalculatorIface::single,
             (arg("shoebox"), arg("s1"), arg("frame"), arg("panel")))
        .def("__call__",
             &MaskCalculatorIface::array,
             (arg("shoebox"), arg("s1"), arg("frame"), arg("panel")))
        .def("__call__",
             &MaskCalculatorIface::volume,
             (arg("volume"), arg("bbox"), arg("s1"), arg("frame"), arg("panel")))
        .def("__call__",
             &MaskCalculatorIface::single_with_s0,
             (arg("shoebox"), arg("s1"), arg("s0"), arg("frame"), arg("panel")))
        .def("__call__",
             &MaskCalculatorIface::array_with_s0,
             (arg("shoebox"), arg("s1"), arg("s0"), arg("frame"), arg("panel")))
        .def("__call__",
             &MaskCalculatorIface::volume_with_s0,
             (arg("volume"),
              arg("bbox"),
              arg("s1"),
              arg("s0"),
              arg("frame"),
              arg("panel")));

      class_<PartialityCalculatorIface, boost::noncopyable>("PartialityCalculatorIface",
                                                            no_init)
        .def("__call__",
             &PartialityCalculatorIface::single,
             (arg("s1"), arg("frame"), arg("bbox")))
        .def("__call__",
             &PartialityCalculatorIface::array,
             (arg("s1"), arg("frame"), arg("bbox")));

      class_<PartialityCalculator3D, bases<PartialityCalculatorIface> >(
        "PartialityCalculator3D", no_init)
        .def(
          init<const boost::python::object&,
               const Goniometer&,
               const boost::python::object&,
               double>((arg("beam"), arg("goniometer"), arg("scan"), arg("delta_m"))))
        .def(init<const boost::python::object&,
                  const Goniometer&,
                  const boost::python::object&,
                  const af::const_ref<double>&>(
          (arg("beam"), arg("goniometer"), arg("scan"), arg("delta_m"))));

      class_<PartialityCalculator2D, bases<PartialityCalculatorIface> >(
        "PartialityCalculator2D", no_init)
        .def(init<const boost::python::object&, double>((arg("beam"), arg("delta_m"))));

      class_<PartialityCalculatorTOF, bases<PartialityCalculatorIface> >(
        "PartialityCalculatorTOF", no_init)
        .def(init<const boost::python::object&, double>((arg("beam"), arg("delta_m"))));

      class_<PartialityMultiCalculator>("PartialityMultiCalculator")
        .def("append", &PartialityMultiCalculator::push_back)
        .def("__len__", &PartialityMultiCalculator::size)
        .def("__call__", &PartialityMultiCalculator::operator());

      class_<MaskCalculator3D, bases<MaskCalculatorIface> >("MaskCalculator3D", no_init)
        .def(init<const boost::python::object&,
                  const Detector&,
                  const Goniometer&,
                  const Scan&,
                  double,
                  double>((arg("beam"),
                           arg("detector"),
                           arg("goniometer"),
                           arg("scan"),
                           arg("delta_divergence"),
                           arg("delta_mosaicity"))))
        .def(init<const boost::python::object&,
                  const Detector&,
                  const Goniometer&,
                  const Scan&,
                  const af::const_ref<double>&,
                  const af::const_ref<double>&>((arg("beam"),
                                                 arg("detector"),
                                                 arg("goniometer"),
                                                 arg("scan"),
                                                 arg("delta_divergence"),
                                                 arg("delta_mosaicity"))));

      class_<MaskCalculator2D, bases<MaskCalculatorIface> >("MaskCalculator2D", no_init)
        .def(init<const boost::python::object&, const Detector&, double, double>(
          (arg("beam"),
           arg("detector"),
           arg("delta_divergence"),
           arg("delta_mosaicity"))));

      class_<MaskCalculatorTOF, bases<MaskCalculatorIface> >("MaskCalculatorTOF",
                                                             no_init)
        .def(init<const Detector&,
                  const boost::python::object&,
                  const boost::python::object&,
                  double,
                  double>((arg("detector"),
                           arg("scan"),
                           arg("beam"),
                           arg("delta_divergence"),
                           arg("delta_mosaicity"))));

      class_<MaskMultiCalculator>("MaskMultiCalculator")
        .def("append", &MaskMultiCalculator::push_back)
        .def("__len__", &MaskMultiCalculator::size)
        .def("__call__", &MaskMultiCalculator::operator());

      def("ideal_profile_float", &ideal_profile<float>);
      def("ideal_profile_double", &ideal_profile<double>);

      // Export zeta factor functions
      def("zeta_factor",
          (double (*)(vec3<double>, vec3<double>, vec3<double>)) & zeta_factor,
          (arg("m2"), arg("s0"), arg("s1")));
      def("zeta_factor",
          (double (*)(vec3<double>, vec3<double>)) & zeta_factor,
          (arg("m2"), arg("e1")));
      def("zeta_factor", &zeta_factor_array, (arg("m2"), arg("s0"), arg("s1")));
      def("zeta_factor",
          &zeta_factor_array_multi,
          (arg("m2"), arg("s0"), arg("s1"), arg("index")));

      // Export coordinate system 2d
      class_<CoordinateSystem2d>("CoordinateSystem2d", no_init)
        .def(init<vec3<double>, vec3<double> >((arg("s0"), arg("s1"))))
        .def("s0", &CoordinateSystem2d::s0)
        .def("s1", &CoordinateSystem2d::s1)
        .def("p_star", &CoordinateSystem2d::p_star)
        .def("e1_axis", &CoordinateSystem2d::e1_axis)
        .def("e2_axis", &CoordinateSystem2d::e2_axis)
        .def("from_beam_vector", &CoordinateSystem2d::from_beam_vector)
        .def("to_beam_vector", &CoordinateSystem2d::to_beam_vector);

      // Export coordinate system
      class_<CoordinateSystem>("CoordinateSystem", no_init)
        .def(init<vec3<double>, vec3<double>, vec3<double>, double>(
          (arg("m2"), arg("s0"), arg("s1"), arg("phi"))))
        .def("m2", &CoordinateSystem::m2)
        .def("s0", &CoordinateSystem::s0)
        .def("s1", &CoordinateSystem::s1)
        .def("phi", &CoordinateSystem::phi)
        .def("p_star", &CoordinateSystem::p_star)
        .def("e1_axis", &CoordinateSystem::e1_axis)
        .def("e2_axis", &CoordinateSystem::e2_axis)
        .def("e3_axis", &CoordinateSystem::e3_axis)
        .def("zeta", &CoordinateSystem::zeta)
        .def("lorentz_inv", &CoordinateSystem::lorentz_inv)
        .def("lorentz", &CoordinateSystem::lorentz)
        .def("path_length_increase", &CoordinateSystem::path_length_increase)
        .def("limits", &CoordinateSystem::limits)
        .def("from_beam_vector", &CoordinateSystem::from_beam_vector)
        .def("from_rotation_angle", &CoordinateSystem::from_rotation_angle)
        .def("from_rotation_angle_fast", &CoordinateSystem::from_rotation_angle_fast)
        .def("from_beam_vector_and_rotation_angle",
             &CoordinateSystem::from_beam_vector_and_rotation_angle)
        .def("to_beam_vector", &CoordinateSystem::to_beam_vector)
        .def("to_rotation_angle", &CoordinateSystem::to_rotation_angle)
        .def("to_rotation_angle_fast", &CoordinateSystem::to_rotation_angle_fast)
        .def("to_beam_vector_and_rotation_angle",
             &CoordinateSystem::to_beam_vector_and_rotation_angle);

      // Export coordinate system
      class_<CoordinateSystemTOF>("CoordinateSystemTOF", no_init)
        .def(
          init<vec3<double>, vec3<double>, double>((arg("s0"), arg("s1"), arg("L1"))))
        .def("s0", &CoordinateSystemTOF::s0)
        .def("s1", &CoordinateSystemTOF::s1)
        .def("L1", &CoordinateSystemTOF::L1)
        .def("p_star", &CoordinateSystemTOF::p_star)
        .def("e1_axis", &CoordinateSystemTOF::e1_axis)
        .def("e2_axis", &CoordinateSystemTOF::e2_axis)
        .def("e3_axis", &CoordinateSystemTOF::e3_axis)
        .def("from_beam_vector", &CoordinateSystemTOF::from_beam_vector)
        .def("to_beam_vector", &CoordinateSystemTOF::to_beam_vector)
        .def("to_wavelength", &CoordinateSystemTOF::to_wavelength)
        .def("from_wavelength", &CoordinateSystemTOF::from_wavelength);

      boost_adaptbx::std_pair_conversions::to_tuple<vec3<double>, double>();
    }

}}}}}  // namespace dials::algorithms::profile_model::gaussian_rs::boost_python
