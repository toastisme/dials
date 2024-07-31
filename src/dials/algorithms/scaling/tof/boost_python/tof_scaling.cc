#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <dials/algorithms/scaling/tof/tof_scaling.h>
#include <dials/algorithms/integration/tof/tof_integration.h>

namespace dials_scaling { namespace boost_python {

  using namespace boost::python;
  BOOST_PYTHON_MODULE(dials_tof_scaling_ext) {
    class_<TOFCorrectionsData>("TOFCorrectionsData", no_init)
      .def(init<double,
                double,
                double,
                double,
                double,
                double,
                double,
                double,
                double,
                double,
                double>());

    void (*extract_shoeboxes1)(dials::af::reflection_table &,
                               dxtbx::model::Experiment &,
                               dxtbx::ImageSequence &,
                               bool) = &tof_extract_shoeboxes_to_reflection_table;
    void (*extract_shoeboxes2)(dials::af::reflection_table &,
                               dxtbx::model::Experiment &,
                               dxtbx::ImageSequence &,
                               dxtbx::ImageSequence &,
                               dxtbx::ImageSequence &,
                               TOFCorrectionsData &,
                               bool) = &tof_extract_shoeboxes_to_reflection_table;
    void (*extract_shoeboxes3)(dials::af::reflection_table &,
                               dxtbx::model::Experiment &,
                               dxtbx::ImageSequence &,
                               dxtbx::ImageSequence &,
                               dxtbx::ImageSequence &,
                               double,
                               double,
                               double,
                               bool) = &tof_extract_shoeboxes_to_reflection_table;

    def("tof_extract_shoeboxes_to_reflection_table", extract_shoeboxes1);
    def("tof_extract_shoeboxes_to_reflection_table", extract_shoeboxes2);
    def("tof_extract_shoeboxes_to_reflection_table", extract_shoeboxes3);
    def("tof_calculate_shoebox_mask",
        &dials::algorithms::tof_calculate_shoebox_mask,
        (arg("reflection_table"), arg("experiment")));
    def("tof_calculate_shoebox_foreground",
        &dials::algorithms::tof_calculate_shoebox_foreground,
        (arg("reflection_table"), arg("experiment"), arg("foreground_radius")));
    def("get_asu_reflections",
        &dials::algorithms::get_asu_reflections,
        (arg("indices"),
         arg("predicted_indices"),
         arg("wavelengths"),
         arg("predicted_wavelengths"),
         arg("asu_reflection"),
         arg("space_group")));
  }

}}  // namespace dials_scaling::boost_python
