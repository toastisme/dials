
#define BOOST_PYTHON_MAX_ARITY 30
#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <dials/algorithms/integration/tof/tof_mask_calculator.h>
#include <dials/algorithms/integration/tof/tof_integration.h>
#include <dials/algorithms/integration/tof/tof_profile_1d.h>
#include <dials/algorithms/integration/tof/tof_profile_3d_gutmann.h>
#include <dials/algorithms/integration/tof/tof_profile_3d_ic.h>
#include <dials/algorithms/scaling/tof/tof_scaling.h>

namespace dials { namespace algorithms { namespace boost_python {

  using namespace boost::python;

  void integrate_reflection_table_wrapper(dials::af::reflection_table& reflection_table,
                                          dxtbx::model::Experiment& experiment,
                                          dxtbx::ImageSequence& data,
                                          object incident_params_obj,
                                          object absorption_params_obj,
                                          const bool& apply_lorentz,
                                          int n_threads,
                                          object profile_1d_params_obj,
                                          object profile_3d_gutmann_params_obj,
                                          object profile_3d_ic_params_obj) {
    boost::optional<TOFProfile1DParams> profile_1d_params;
    boost::optional<TOFProfile3DGutmannParams> profile_3d_gutmann_params;
    boost::optional<TOFProfile3DICParams> profile_3d_ic_params;

    if (!profile_1d_params_obj.is_none()) {
      profile_1d_params = extract<TOFProfile1DParams>(profile_1d_params_obj);
    }

    if (!profile_3d_gutmann_params_obj.is_none()) {
      profile_3d_gutmann_params =
        extract<TOFProfile3DGutmannParams>(profile_3d_gutmann_params_obj);
    }

    if (!profile_3d_ic_params_obj.is_none()) {
      profile_3d_ic_params = extract<TOFProfile3DICParams>(profile_3d_ic_params_obj);
    }

    if (absorption_params_obj.is_none() && incident_params_obj.is_none()) {
      integrate_reflection_table(reflection_table,
                                 experiment,
                                 data,
                                 apply_lorentz,
                                 n_threads,
                                 profile_1d_params,
                                 profile_3d_gutmann_params,
                                 profile_3d_ic_params);
      return;
    }

    if (incident_params_obj.is_none()) {
      // Absorption correction only (no incident spectrum normalisation)
      dials_scaling::TOFAbsorptionParams absorption_params =
        extract<dials_scaling::TOFAbsorptionParams>(absorption_params_obj);

      integrate_reflection_table(reflection_table,
                                 experiment,
                                 data,
                                 absorption_params,
                                 apply_lorentz,
                                 n_threads,
                                 profile_1d_params,
                                 profile_3d_gutmann_params,
                                 profile_3d_ic_params);
      return;
    }

    dials_scaling::TOFIncidentSpectrumParams incident_params =
      extract<dials_scaling::TOFIncidentSpectrumParams>(incident_params_obj);

    if (!absorption_params_obj.is_none()) {
      dials_scaling::TOFAbsorptionParams absorption_params =
        extract<dials_scaling::TOFAbsorptionParams>(absorption_params_obj);

      integrate_reflection_table(reflection_table,
                                 experiment,
                                 data,
                                 incident_params,
                                 absorption_params,
                                 apply_lorentz,
                                 n_threads,
                                 profile_1d_params,
                                 profile_3d_gutmann_params,
                                 profile_3d_ic_params);
    } else {
      integrate_reflection_table(reflection_table,
                                 experiment,
                                 data,
                                 incident_params,
                                 apply_lorentz,
                                 n_threads,
                                 profile_1d_params,
                                 profile_3d_gutmann_params,
                                 profile_3d_ic_params);
    }
  }

  boost::python::tuple fit_profile_3d_ic_wrapper(
    scitbx::af::versa<scitbx::vec3<double>, scitbx::af::c_grid<3>> coords,
    scitbx::af::versa<double, scitbx::af::c_grid<3>> intensities,
    scitbx::af::versa<double, scitbx::af::c_grid<3>> background_variances,
    TOFProfile3DICParams& profile_params,
    boost::python::object profile_3d_obj) {
    double I_prf = 0.0;
    boost::optional<scitbx::af::versa<double, scitbx::af::c_grid<3>>> profile_3d_out;
    if (!profile_3d_obj.is_none()) {
      profile_3d_out =
        extract<scitbx::af::versa<double, scitbx::af::c_grid<3>>>(profile_3d_obj);
    }
    const bool success = dials::algorithms::fit_profile_3d_ic(coords.const_ref(),
                                                              intensities,
                                                              background_variances,
                                                              profile_params,
                                                              I_prf,
                                                              profile_3d_out);
    return boost::python::make_tuple(success, I_prf);
  }

  BOOST_PYTHON_MODULE(dials_algorithms_tof_integration_ext) {
    class_<TOFProfile1DParams>("TOFProfile1DParams", no_init)
      .def(
        init<double, double, double, double, double, double, double, int, bool, bool>())
      .def_readwrite("A", &TOFProfile1DParams::A)
      .def_readwrite("alpha", &TOFProfile1DParams::alpha)
      .def_readwrite("alpha_min", &TOFProfile1DParams::alpha_min)
      .def_readwrite("alpha_max", &TOFProfile1DParams::alpha_max)
      .def_readwrite("beta", &TOFProfile1DParams::beta)
      .def_readwrite("beta_min", &TOFProfile1DParams::beta_min)
      .def_readwrite("beta_max", &TOFProfile1DParams::beta_max)
      .def_readwrite("n_restarts", &TOFProfile1DParams::n_restarts)
      .def_readwrite("optimize_profile", &TOFProfile1DParams::optimize_profile)
      .def_readwrite("show_profile_failures",
                     &TOFProfile1DParams::show_profile_failures);

    class_<TOFProfile3DGutmannParams>("TOFProfile3DGutmannParams", no_init)
      .def(
        init<double, double, double, double, double, double, int, bool, bool, bool>())
      .def_readwrite("alpha", &TOFProfile3DGutmannParams::alpha)
      .def_readwrite("alpha_min", &TOFProfile3DGutmannParams::alpha_min)
      .def_readwrite("alpha_max", &TOFProfile3DGutmannParams::alpha_max)
      .def_readwrite("beta", &TOFProfile3DGutmannParams::beta)
      .def_readwrite("beta_min", &TOFProfile3DGutmannParams::beta_min)
      .def_readwrite("beta_max", &TOFProfile3DGutmannParams::beta_max)
      .def_readwrite("n_restarts", &TOFProfile3DGutmannParams::n_restarts)
      .def_readwrite("optimize_profile", &TOFProfile3DGutmannParams::optimize_profile)
      .def_readwrite("use_central_diff", &TOFProfile3DGutmannParams::use_central_diff)
      .def_readwrite("show_profile_failures",
                     &TOFProfile3DGutmannParams::show_profile_failures);

    def("tof_calculate_ellipse_shoebox_mask",
        &tof_calculate_ellipse_shoebox_mask,
        (arg("reflection_table"),
         arg("experiment"),
         arg("n_threads") = 1,
         arg("scale") = 1));

    def("tof_calculate_seed_skewness_shoebox_mask",
        &tof_calculate_seed_skewness_shoebox_mask,
        (arg("reflection_table"),
         arg("experiment"),
         arg("d_skewness_threshold"),
         arg("min_iterations"),
         arg("n_threads") = 1));

    def("tof_calculate_bboxes_from_foreground_mask",
        &tof_calculate_bboxes_from_foreground_mask,
        (arg("reflection_table"),
         arg("xy_padding") = 2,
         arg("z_padding") = 2,
         arg("n_threads") = 1));

    def("integrate_reflection_table",
        &integrate_reflection_table_wrapper,
        (arg("reflection_table"),
         arg("experiment"),
         arg("data"),
         arg("incident_params"),
         arg("absorption_params"),
         arg("apply_lorentz_correction"),
         arg("n_threads"),
         arg("profile_1d_params") = object(),
         arg("profile_3d_gutmann_params") = object(),
         arg("profile_3d_ic_params") = object()));

    def("calculate_line_profile_for_reflection",
        static_cast<boost::python::tuple (*)(dials::af::reflection_table&,
                                             dxtbx::model::Experiment&,
                                             dxtbx::ImageSequence&,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             const bool&)>(
          &calculate_line_profile_for_reflection));

    def("calculate_line_profile_for_reflection",
        static_cast<boost::python::tuple (*)(dials::af::reflection_table&,
                                             dxtbx::model::Experiment&,
                                             dxtbx::ImageSequence&,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             const bool&,
                                             TOFProfile1DParams&)>(
          &calculate_line_profile_for_reflection));

    def("calculate_line_profile_for_reflection",
        static_cast<boost::python::tuple (*)(dials::af::reflection_table&,
                                             dxtbx::model::Experiment&,
                                             dxtbx::ImageSequence&,
                                             scitbx::af::shared<vec3<double>>,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             const bool&,
                                             TOFProfile3DGutmannParams&)>(
          &calculate_line_profile_for_reflection));

    def("calculate_line_profile_for_reflection",
        static_cast<boost::python::tuple (*)(dials::af::reflection_table&,
                                             dxtbx::model::Experiment&,
                                             dxtbx::ImageSequence&,
                                             scitbx::af::shared<vec3<double>>,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             const bool&,
                                             TOFProfile3DICParams&)>(
          &calculate_line_profile_for_reflection));

    def("calculate_line_profile_for_reflection",
        static_cast<boost::python::tuple (*)(
          dials::af::reflection_table&,
          dxtbx::model::Experiment&,
          dxtbx::ImageSequence&,
          const dials_scaling::TOFIncidentSpectrumParams&,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          const bool&)>(&calculate_line_profile_for_reflection));

    def("calculate_line_profile_for_reflection",
        static_cast<boost::python::tuple (*)(
          dials::af::reflection_table&,
          dxtbx::model::Experiment&,
          dxtbx::ImageSequence&,
          const dials_scaling::TOFIncidentSpectrumParams&,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          const bool&,
          TOFProfile1DParams&)>(&calculate_line_profile_for_reflection));

    def("calculate_line_profile_for_reflection",
        static_cast<boost::python::tuple (*)(
          dials::af::reflection_table&,
          dxtbx::model::Experiment&,
          dxtbx::ImageSequence&,
          const dials_scaling::TOFIncidentSpectrumParams&,
          scitbx::af::shared<vec3<double>>,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          const bool&,
          TOFProfile3DGutmannParams&)>(&calculate_line_profile_for_reflection));

    def("calculate_line_profile_for_reflection",
        static_cast<boost::python::tuple (*)(
          dials::af::reflection_table&,
          dxtbx::model::Experiment&,
          dxtbx::ImageSequence&,
          const dials_scaling::TOFIncidentSpectrumParams&,
          scitbx::af::shared<vec3<double>>,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          const bool&,
          TOFProfile3DICParams&)>(&calculate_line_profile_for_reflection));

    def("calculate_line_profile_for_reflection",
        static_cast<boost::python::tuple (*)(
          dials::af::reflection_table&,
          dxtbx::model::Experiment&,
          dxtbx::ImageSequence&,
          const dials_scaling::TOFIncidentSpectrumParams&,
          const dials_scaling::TOFAbsorptionParams&,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          const bool&)>(&calculate_line_profile_for_reflection));

    def("calculate_line_profile_for_reflection",
        static_cast<boost::python::tuple (*)(
          dials::af::reflection_table&,
          dxtbx::model::Experiment&,
          dxtbx::ImageSequence&,
          const dials_scaling::TOFIncidentSpectrumParams&,
          const dials_scaling::TOFAbsorptionParams&,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          const bool&,
          TOFProfile1DParams&)>(&calculate_line_profile_for_reflection));

    def("calculate_line_profile_for_reflection",
        static_cast<boost::python::tuple (*)(
          dials::af::reflection_table&,
          dxtbx::model::Experiment&,
          dxtbx::ImageSequence&,
          const dials_scaling::TOFIncidentSpectrumParams&,
          const dials_scaling::TOFAbsorptionParams&,
          scitbx::af::shared<vec3<double>>,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          const bool&,
          TOFProfile3DGutmannParams&)>(&calculate_line_profile_for_reflection));

    def("calculate_line_profile_for_reflection",
        static_cast<boost::python::tuple (*)(
          dials::af::reflection_table&,
          dxtbx::model::Experiment&,
          dxtbx::ImageSequence&,
          const dials_scaling::TOFIncidentSpectrumParams&,
          const dials_scaling::TOFAbsorptionParams&,
          scitbx::af::shared<vec3<double>>,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          scitbx::af::shared<double>,
          const bool&,
          TOFProfile3DICParams&)>(&calculate_line_profile_for_reflection));

    def("calculate_line_profile_for_reflection",
        static_cast<boost::python::tuple (*)(dials::af::reflection_table&,
                                             dxtbx::model::Experiment&,
                                             dxtbx::ImageSequence&,
                                             const dials_scaling::TOFAbsorptionParams&,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             const bool&)>(
          &calculate_line_profile_for_reflection));

    def("calculate_line_profile_for_reflection",
        static_cast<boost::python::tuple (*)(dials::af::reflection_table&,
                                             dxtbx::model::Experiment&,
                                             dxtbx::ImageSequence&,
                                             const dials_scaling::TOFAbsorptionParams&,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             const bool&,
                                             TOFProfile1DParams&)>(
          &calculate_line_profile_for_reflection));

    def("calculate_line_profile_for_reflection",
        static_cast<boost::python::tuple (*)(dials::af::reflection_table&,
                                             dxtbx::model::Experiment&,
                                             dxtbx::ImageSequence&,
                                             const dials_scaling::TOFAbsorptionParams&,
                                             scitbx::af::shared<vec3<double>>,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             const bool&,
                                             TOFProfile3DGutmannParams&)>(
          &calculate_line_profile_for_reflection));

    def("calculate_line_profile_for_reflection",
        static_cast<boost::python::tuple (*)(dials::af::reflection_table&,
                                             dxtbx::model::Experiment&,
                                             dxtbx::ImageSequence&,
                                             const dials_scaling::TOFAbsorptionParams&,
                                             scitbx::af::shared<vec3<double>>,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             scitbx::af::shared<double>,
                                             const bool&,
                                             TOFProfile3DICParams&)>(
          &calculate_line_profile_for_reflection));

    class_<TOFProfile3DICParams>("TOFProfile3DICParams", no_init)
      .def(init<double,
                double,
                double,  // A, A_min, A_max
                double,
                double,
                double,  // B, B_min, B_max
                double,
                double,
                double,  // R, R_min, R_max
                double,
                double,  // SigX_min, SigX_max
                double,
                double,  // SigY_min, SigY_max
                double,
                double,
                double,  // SigP, SigP_min, SigP_max
                double,
                double,  // HatWidth, KConv
                int,
                bool,
                bool,
                bool,
                bool,
                bool>())  // n_restarts, optimize_profile,
                          // optimize_convolution_params, optimize_moderator_params,
                          // use_analytic_jacobian, show_profile_failures
      .def_readwrite("A", &TOFProfile3DICParams::A)
      .def_readwrite("A_min", &TOFProfile3DICParams::A_min)
      .def_readwrite("A_max", &TOFProfile3DICParams::A_max)
      .def_readwrite("B", &TOFProfile3DICParams::B)
      .def_readwrite("B_min", &TOFProfile3DICParams::B_min)
      .def_readwrite("B_max", &TOFProfile3DICParams::B_max)
      .def_readwrite("R", &TOFProfile3DICParams::R)
      .def_readwrite("R_min", &TOFProfile3DICParams::R_min)
      .def_readwrite("R_max", &TOFProfile3DICParams::R_max)
      .def_readwrite("SigX_min", &TOFProfile3DICParams::SigX_min)
      .def_readwrite("SigX_max", &TOFProfile3DICParams::SigX_max)
      .def_readwrite("SigY_min", &TOFProfile3DICParams::SigY_min)
      .def_readwrite("SigY_max", &TOFProfile3DICParams::SigY_max)
      .def_readwrite("SigP", &TOFProfile3DICParams::SigP)
      .def_readwrite("SigP_min", &TOFProfile3DICParams::SigP_min)
      .def_readwrite("SigP_max", &TOFProfile3DICParams::SigP_max)
      .def_readwrite("HatWidth", &TOFProfile3DICParams::HatWidth)
      .def_readwrite("KConv", &TOFProfile3DICParams::KConv)
      .def_readwrite("n_restarts", &TOFProfile3DICParams::n_restarts)
      .def_readwrite("optimize_profile", &TOFProfile3DICParams::optimize_profile)
      .def_readwrite("optimize_convolution_params",
                     &TOFProfile3DICParams::optimize_convolution_params)
      .def_readwrite("optimize_moderator_params",
                     &TOFProfile3DICParams::optimize_moderator_params)
      .def_readwrite("use_analytic_jacobian",
                     &TOFProfile3DICParams::use_analytic_jacobian)
      .def_readwrite("show_profile_failures",
                     &TOFProfile3DICParams::show_profile_failures);

    def("fit_profile_3d_ic",
        &fit_profile_3d_ic_wrapper,
        (arg("coords"),
         arg("intensities"),
         arg("background_variances"),
         arg("profile_params"),
         arg("profile_3d_out") = object()));
  }

}}}  // namespace dials::algorithms::boost_python
