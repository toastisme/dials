#ifndef DIALS_ALGORITHMS_INTEGRATION_TOF_PROFILE_3D_GUTMANN_H
#define DIALS_ALGORITHMS_INTEGRATION_TOF_PROFILE_3D_GUTMANN_H

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include "tof_utils.h"
#include <scitbx/constants.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/NonLinearOptimization>
#include <dials/array_family/scitbx_shared_and_versa.h>

/*
3D profile fitting method based on
Gutmann, M.J, A 3D profile function suitable for integration of
neutron time-of-flight single crystal diffraction peaks,
Nucl. Instrum. Methods Phys. Res. A, 848, 170-173, (2016)
https://doi.org/10.1016/j.nima.2016.12.026
*/

namespace dials { namespace algorithms {

  /*
   * Holds params required for TOFProfile3DGutmann
   */

  struct TOFProfile3DGutmannParams {
    double alpha;
    double alpha_min;
    double alpha_max;
    double beta;
    double beta_min;
    double beta_max;
    int n_restarts;              // number of attempts when fitting
    bool optimize_profile;       // If false the profile is generated with input params
    bool use_central_diff;       // Use more expensive central differences for gradients
    bool show_profile_failures;  // Prints debugging information

    TOFProfile3DGutmannParams(double alpha,
                              double alpha_min,
                              double alpha_max,
                              double beta,
                              double beta_min,
                              double beta_max,
                              int n_restarts,
                              bool optimize_profile,
                              bool use_central_diff,
                              bool show_profile_failures)
        : alpha(alpha),
          alpha_min(alpha_min),
          alpha_max(alpha_max),
          beta(beta),
          beta_min(beta_min),
          beta_max(beta_max),
          n_restarts(n_restarts),
          optimize_profile(optimize_profile),
          use_central_diff(use_central_diff),
          show_profile_failures(show_profile_failures) {}
  };

  struct Profile3DGutmannFunctor {
    const scitbx::af::versa<vec3<double>, af::c_grid<3>> coords;  // (x,y,z)
    scitbx::af::shared<double> dt_widths;                         //(μs)
    const scitbx::af::versa<double, af::c_grid<3>> intensities;
    const scitbx::af::versa<double, af::c_grid<3>> background_variances;
    mutable Eigen::VectorXd last_params;
    mutable double cached_norm = 1.0;
    mutable double cached_A = 1.0;
    mutable bool cache_valid = false;
    std::array<double, 9> min_bounds;
    std::array<double, 9> max_bounds;
    int num_data_points, num_params;
    bool use_central_diff;

    Profile3DGutmannFunctor(
      const scitbx::af::versa<vec3<double>, af::c_grid<3>> coords_,
      const scitbx::af::versa<double, af::c_grid<3>> intensities_,
      const scitbx::af::versa<double, af::c_grid<3>> background_variances_,
      const std::array<double, 9>& minb,
      const std::array<double, 9>& maxb,
      bool use_central_diff_)
        : coords(coords_),
          intensities(intensities_),
          background_variances(background_variances_),
          use_central_diff(use_central_diff_) {
      min_bounds = minb;
      max_bounds = maxb;
      num_data_points = coords.size();
      num_params = 9;
      last_params =
        Eigen::VectorXd::Constant(9, std::numeric_limits<double>::quiet_NaN());

      dt_widths.resize(coords_.accessor()[2]);
      dt_widths[0] = 1.;
      for (std::size_t c_z = 1; c_z < coords.accessor()[2]; ++c_z) {
        dt_widths[c_z] = coords(0, 0, c_z)[2] - coords(0, 0, c_z - 1)[2];
      }
    }

    int values() const {
      return num_data_points;
    }

    int inputs() const {
      return num_params;
    }

    inline Eigen::VectorXd clamp_params(const Eigen::VectorXd& x) const {
      Eigen::VectorXd xc = x;
      for (int i = 0; i < x.size(); ++i)
        xc[i] = std::min(std::max(x[i], min_bounds[i]), max_bounds[i]);
      return xc;
    }

    Eigen::Matrix3d build_H_from_L(const Eigen::VectorXd& params) const {
      double l11 = std::exp(params[0]);
      double l21 = params[1];
      double l31 = params[2];
      double l22 = std::exp(params[3]);
      double l32 = params[4];
      double l33 = std::exp(params[5]);

      Eigen::Matrix3d L;
      L << l11, 0.0, 0.0, l21, l22, 0.0, l31, l32, l33;

      return L * L.transpose();
    }

    double func(const vec3<double>& c,
                const Eigen::Matrix3d& H,
                double alpha,
                double beta,
                double A_,
                double norm_factor,
                double dt_width,
                double T0 = 0.0) const {
      /**
       * func used to generate the actual profile
       * (Numbers) refer to equations in
       * https://doi.org/10.1016/j.nima.2016.12.026
       * T0 is the pulse onset offset: the model is evaluated at (dt - T0) so that
       * the model peak aligns with the data peak at dt=0.
       */

      double dx = c[0], dy = c[1], dt = c[2] - T0;

      double H1 = H(0, 0), H2 = H(0, 1), H3 = H(0, 2);
      double H4 = H(1, 1), H5 = H(1, 2), H6 = H(2, 2);

      double a = alpha;
      double b = beta;
      double PI = scitbx::constants::pi;

      double N = (a * b) / (2.0 * (a + b));  // (5)
      double detH = std::max(H.determinant(), 1e-12);
      double Ng = std::sqrt(detH) / std::pow(2.0 * PI, 1.5);  // (3)
      double f1 = N * Ng * std::sqrt(PI / (2.0 * H6));

      double u = 0.5 * a * (a + 2.0 * H6 * dt + 2.0 * H3 * dx + 2.0 * H5 * dy);
      double v = 0.5 * b * (b - 2.0 * H6 * dt - 2.0 * H3 * dx - 2.0 * H5 * dy);

      v = std::max(-700.0, std::min(v, 700.0));
      u = std::max(-700.0, std::min(u, 700.0));

      double y = (a + H6 * dt + H3 * dx + H5 * dy) / std::sqrt(2.0 * H6);
      double w = (b - H6 * dt - H3 * dx - H5 * dy) / std::sqrt(2.0 * H6);

      double f2 =
        std::exp(-0.5 * H1 * dx * dx - H2 * dx * dy - 0.5 * H4 * dy * dy
                 + (H3 * H3 * dx * dx + 2.0 * H3 * H5 * dx * dy + H5 * H5 * dy * dy)
                     / (2.0 * H6));

      auto F_tof = [&](double dt) {
        double u =
          0.5 * alpha * (alpha + 2.0 * H6 * dt + 2.0 * H3 * dx + 2.0 * H5 * dy);
        double v = 0.5 * beta * (beta - 2.0 * H6 * dt - 2.0 * H3 * dx - 2.0 * H5 * dy);

        double y = (alpha + H6 * dt + H3 * dx + H5 * dy) / std::sqrt(2.0 * H6);
        double w = (beta - H6 * dt - H3 * dx - H5 * dy) / std::sqrt(2.0 * H6);

        double term1 = exp_safe(u) * erfc_safe(y) / alpha;
        double term2 = exp_safe(v) * erfc_safe(w) / beta;

        return std::sqrt(PI / (2.0 * H6)) * (term1 + term2);
      };

      double dt_lo = dt - 0.5 * dt_width;
      double dt_hi = dt + 0.5 * dt_width;

      double f3 = F_tof(dt_hi) - F_tof(dt_lo);

      double result = A_ * (f1 * f2 * f3 / norm_factor);  // (6)
      if (!std::isfinite(result)) result = 0.0;
      return result;
    }

    double get_norm_factor(Eigen::Matrix3d H,
                           double alpha,
                           double beta,
                           double T0) const {
      /**
       * Numerical normalisation constant to ensure integration sums to 1
       */

      double sum = 0;
      for (std::size_t c_x = 0; c_x < coords.accessor()[0]; ++c_x) {
        for (std::size_t c_y = 0; c_y < coords.accessor()[1]; ++c_y) {
          for (std::size_t c_z = 0; c_z < coords.accessor()[2]; ++c_z) {
            sum +=
              func(coords(c_x, c_y, c_z), H, alpha, beta, 1.0, 1.0, dt_widths[c_z], T0);
          }
        }
      }
      return sum;
    }

    double calc_A(const Eigen::Matrix3d& H,
                  double alpha,
                  double beta,
                  double norm_factor,
                  double T0) const {
      /**
       * Obtain A directly with least squares
       * rather than optimize with other params.
       * Background variance is used as a weighting term
       */

      scitbx::af::shared<double> P(num_data_points);
      int count = 0;
      for (std::size_t c_x = 0; c_x < coords.accessor()[0]; ++c_x) {
        for (std::size_t c_y = 0; c_y < coords.accessor()[1]; ++c_y) {
          for (std::size_t c_z = 0; c_z < coords.accessor()[2]; ++c_z) {
            P[count] = func(coords(c_x, c_y, c_z),
                            H,
                            alpha,
                            beta,
                            1.0,
                            norm_factor,
                            dt_widths[c_z],
                            T0);
            count++;
          }
        }
      }

      double num = 0.0, den = 0.0;

      for (std::size_t i = 0; i < num_data_points; ++i) {
        double obs = intensities[i];
        double var = background_variances[i];
        if (!std::isfinite(var) || var <= 0.0) var = std::max(std::abs(obs), 1.0);
        double w = 1.0 / var;
        double p_i = P[i];
        num += w * obs * p_i;
        den += w * p_i * p_i;
      }

      double A = (den > 0.0) ? (num / den) : 0.0;
      return A;
    }

    int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const {
      // Get current params
      Eigen::VectorXd xc = clamp_params(x);
      Eigen::Matrix3d H = build_H_from_L(xc);
      double alpha = std::exp(xc[6]);
      double beta = std::exp(xc[7]);
      double T0 = xc[8];

      Eigen::VectorXd current_params(9);
      current_params << H(0, 0), H(0, 1), H(0, 2), H(1, 1), H(1, 2), H(2, 2), alpha,
        beta, T0;

      // Update A and norm factors if params have changed
      if (!cache_valid || (current_params - last_params).cwiseAbs().maxCoeff() > 1e-6) {
        cached_norm = get_norm_factor(H, alpha, beta, T0);
        if (cached_norm <= 0.0 || !std::isfinite(cached_norm)) cached_norm = 1.0;
        cached_A = calc_A(H, alpha, beta, cached_norm, T0);
        last_params = current_params;
        cache_valid = true;
      }
      double norm_factor = cached_norm;
      double A = cached_A;

      fvec.resize(num_data_points);

      // Calculate residuals
      int count = 0;
      for (std::size_t c_x = 0; c_x < coords.accessor()[0]; ++c_x) {
        for (std::size_t c_y = 0; c_y < coords.accessor()[1]; ++c_y) {
          for (std::size_t c_z = 0; c_z < coords.accessor()[2]; ++c_z) {
            double model = func(coords(c_x, c_y, c_z),
                                H,
                                alpha,
                                beta,
                                A,
                                norm_factor,
                                dt_widths[c_z],
                                T0);
            double obs = intensities(c_x, c_y, c_z);
            double var = background_variances(c_x, c_y, c_z);
            if (!std::isfinite(var) || var <= 0.0) {
              var = std::max(std::abs(obs), 1.0);
            }
            double sigma = std::sqrt(var);
            double diff = (obs - model) / sigma;

            fvec[count] = std::isfinite(diff) ? diff : 1e6;
            count++;
          }
        }
      }
      return 0;
    }

    int df(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const {
      /**
       * Finite difference using forward or central difference
       */

      J.resize(num_data_points, num_params);

      double eps = 1e-5;
      Eigen::VectorXd xc = clamp_params(x);  // init clamped params
      Eigen::VectorXd f0(num_data_points);
      operator()(xc, f0);

      for (int j = 0; j < num_params; ++j) {
        // Perturb param
        Eigen::VectorXd xp = xc;  // params + delta
        Eigen::VectorXd xm = xc;  // params - delta
        double delta = eps * std::abs(1.0 + std::abs(xc[j]));
        xp[j] += delta;
        xm[j] -= delta;

        Eigen::VectorXd xpc = clamp_params(xp);  // params + delta clamped
        Eigen::VectorXd xmc = clamp_params(xm);  // params - delta clamped

        // Handle parameters at boundaries
        bool at_upper_bound = std::abs(xpc[j] - xc[j]) < 1e-14;
        bool at_lower_bound = std::abs(xmc[j] - xc[j]) < 1e-14;

        if (use_central_diff && !at_upper_bound && !at_lower_bound) {
          // Central difference
          Eigen::VectorXd fp(num_data_points), fm(num_data_points);
          operator()(xpc, fp);
          operator()(xmc, fm);
          double step = xpc[j] - xmc[j];
          if (std::abs(step) < 1e-14) {
            J.col(j).setZero();
            continue;
          }
          J.col(j) = (fp - fm) / step;
        } else {
          // Forward difference
          Eigen::VectorXd fp(num_data_points);
          operator()(xpc, fp);
          double step = xpc[j] - xc[j];
          if (std::abs(step) < 1e-14) {
            J.col(j).setZero();
            continue;
          }
          J.col(j) = (fp - f0) / step;
        }
      }

      return 0;
    }
  };

  /*
   * Thin 5-parameter wrapper around Profile3DGutmannFunctor that fixes the
   * three spatial-ToF off-diagonal L elements to zero (params[2] and params[4])
   * and the x-y off-diagonal (params[1]).  Used for stage-1 (diagonal H) fitting
   * to give a stable warm-start before the full 9-parameter refinement.
   *
   * Parameter mapping:
   *   x5[0] -> params[0]  (log l11)
   *   x5[1] -> params[3]  (log l22)
   *   x5[2] -> params[5]  (log l33)
   *   x5[3] -> params[6]  (log alpha)
   *   x5[4] -> params[7]  (log beta)
   *   params[1] = params[2] = params[4] = 0
   */
  /*
   * 6-parameter diagonal wrapper (log_l11, log_l22, log_l33, log_α, log_β, T0).
   * Off-diagonal L elements are fixed to zero for the stage-1 warm-start fit.
   * T0 is included here because it is critical for aligning the model with the
   * data peak and must be roughly correct before full-H refinement.
   *
   *   x6[0] -> params[0]  (log l11)
   *   x6[1] -> params[3]  (log l22)
   *   x6[2] -> params[5]  (log l33)
   *   x6[3] -> params[6]  (log alpha)
   *   x6[4] -> params[7]  (log beta)
   *   x6[5] -> params[8]  (T0)
   *   params[1] = params[2] = params[4] = 0
   */
  struct DiagonalGutmannFunctor {
    Profile3DGutmannFunctor& base;
    int m_values, m_inputs;

    explicit DiagonalGutmannFunctor(Profile3DGutmannFunctor& b)
        : base(b), m_values(b.num_data_points), m_inputs(6) {}

    int values() const {
      return m_values;
    }
    int inputs() const {
      return m_inputs;
    }

    Eigen::VectorXd expand(const Eigen::VectorXd& x6) const {
      Eigen::VectorXd x9(9);
      x9 << x6[0], 0.0, 0.0, x6[1], 0.0, x6[2], x6[3], x6[4], x6[5];
      return x9;
    }

    int operator()(const Eigen::VectorXd& x6, Eigen::VectorXd& fvec) const {
      return base(expand(x6), fvec);
    }

    int df(const Eigen::VectorXd& x6, Eigen::MatrixXd& J) const {
      J.resize(m_values, 6);
      const double eps = 1e-5;
      Eigen::VectorXd f0(m_values);
      operator()(x6, f0);
      for (int j = 0; j < 6; ++j) {
        Eigen::VectorXd xp = x6;
        double delta = eps * (1.0 + std::abs(x6[j]));
        xp[j] += delta;
        Eigen::VectorXd fp(m_values);
        operator()(xp, fp);
        double step = xp[j] - x6[j];
        if (std::abs(step) < 1e-14)
          J.col(j).setZero();
        else
          J.col(j) = (fp - f0) / step;
      }
      return 0;
    }
  };

  class TOFProfile3DGutmann {
  public:
    // Coords in ToF
    const scitbx::af::versa<vec3<double>, af::c_grid<3>> coords;
    // Raw intensities
    const scitbx::af::versa<double, af::c_grid<3>> intensities;
    // Background variances used for weights during optimization
    const scitbx::af::versa<double, af::c_grid<3>> background_variances;
    // Intensities with no negative values
    scitbx::af::versa<double, af::c_grid<3>> y_norm;
    // Number of attempts at fitting
    int n_restarts;
    double intensity_max;
    boost::optional<Profile3DGutmannFunctor> functor;

    Eigen::VectorXd params;
    std::array<double, 9> min_bounds;
    std::array<double, 9> max_bounds;

    TOFProfile3DGutmann(
      scitbx::af::const_ref<vec3<double>, af::c_grid<3>> coords_,
      const scitbx::af::versa<double, af::c_grid<3>> intensities_,
      const scitbx::af::versa<double, af::c_grid<3>> background_variances_,
      double alpha,
      double beta,
      const std::array<double, 2> alpha_bounds,
      const std::array<double, 2> beta_bounds,
      int n_restarts_,
      bool use_central_diff)
        : coords(get_rel_coords(coords_, intensities_)),
          intensities(intensities_),
          background_variances(background_variances_),
          n_restarts(n_restarts_) {
      DIALS_ASSERT(coords.size() == intensities.size());

      const std::size_t n = intensities.size();
      // Ensure no negative values
      intensity_max = 1.0;
      if (!(n == 0)) {
        intensity_max = *std::max_element(intensities.begin(), intensities.end());
        if (intensity_max <= 0.0) intensity_max = 1.0;
      }

      y_norm.resize(intensities.accessor());

      for (std::size_t i = 0; i < n; ++i) {
        double v = intensities[i];
        if (!std::isfinite(v) || v < 0.0) v = 0.0;
        y_norm[i] = v;
      }

      // Compute weighted mean
      Eigen::Vector3d mean = Eigen::Vector3d::Zero();
      double weight_sum = 0.0;

      for (std::size_t c_x = 0; c_x < coords.accessor()[0]; ++c_x) {
        for (std::size_t c_y = 0; c_y < coords.accessor()[1]; ++c_y) {
          for (std::size_t c_z = 0; c_z < coords.accessor()[2]; ++c_z) {
            std::size_t idx =
              (c_x * coords.accessor()[1] + c_y) * coords.accessor()[2] + c_z;
            double w = y_norm[idx];
            vec3<double> coord = coords(c_x, c_y, c_z);
            mean += w * Eigen::Vector3d(coord[0], coord[1], coord[2]);
            weight_sum += w;
          }
        }
      }
      if (weight_sum <= 0.0) weight_sum = 1.0;
      mean /= weight_sum;

      // Compute weighted covariance
      Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
      for (std::size_t c_x = 0; c_x < coords.accessor()[0]; ++c_x) {
        for (std::size_t c_y = 0; c_y < coords.accessor()[1]; ++c_y) {
          for (std::size_t c_z = 0; c_z < coords.accessor()[2]; ++c_z) {
            std::size_t idx =
              (c_x * coords.accessor()[1] + c_y) * coords.accessor()[2] + c_z;
            double w = y_norm[idx];
            vec3<double> coord = coords(c_x, c_y, c_z);
            Eigen::Vector3d v(coord[0], coord[1], coord[2]);
            Eigen::Vector3d d = v - mean;
            C += w * (d * d.transpose());
          }
        }
      }
      C /= weight_sum;
      C += Eigen::Matrix3d::Identity() * 1e-6;

      // Block-diagonal H initialization: decouple spatial (x,y) from ToF (z).
      //
      // Use the total observed ToF variance for H6 directly rather than
      // subtracting an exp-tail contribution (1/α² + 1/β²). The subtraction is
      // theoretically motivated but blows up when init α/β are too small: with
      // init_alpha=1 and init_beta=0.1 the contribution is 101 frames², which
      // exceeds typical observed variances, driving H6 to 10000 (degenerate).
      // The optimizer can correct α/β once given a stable H6 starting point.
      Eigen::Matrix2d C_xy = C.block<2, 2>(0, 0);
      double tof_var = C(2, 2);
      double gauss_var_tof = std::max(tof_var, 1.0);

      Eigen::Matrix3d H_init = Eigen::Matrix3d::Zero();
      H_init.block<2, 2>(0, 0) = C_xy.inverse();
      H_init(2, 2) = 1.0 / gauss_var_tof;
      H_init += Eigen::Matrix3d::Identity() * 1e-8;

      Eigen::LLT<Eigen::Matrix3d> llt_H(H_init);
      Eigen::Matrix3d L_init = llt_H.matrixL();

      double l11 = std::max(L_init(0, 0), 1e-10);
      double l21 = L_init(1, 0);
      double l31 = L_init(2, 0);  // 0 for block-diagonal H_init
      double l22 = std::max(L_init(1, 1), 1e-10);
      double l32 = L_init(2, 1);  // 0 for block-diagonal H_init
      double l33 = std::max(L_init(2, 2), 1e-10);

      // Compute T0: the pulse onset offset that aligns the model peak with the data
      // peak. Without T0 the model peaks at approximately dt = beta/H6 = beta *
      // gauss_var_tof, which can be 10+ frames from the data peak. T0_init shifts the
      // onset so the model peak falls at dt=0 (the data peak, placed there by
      // get_rel_coords).
      double tof_min = std::numeric_limits<double>::max();
      double tof_max = std::numeric_limits<double>::lowest();
      for (std::size_t c_z = 0; c_z < coords.accessor()[2]; ++c_z) {
        double t = coords(0, 0, c_z)[2];
        tof_min = std::min(tof_min, t);
        tof_max = std::max(tof_max, t);
      }
      const double tof_range = tof_max - tof_min;
      const double T0_min = tof_min - 2.0 * tof_range;
      const double T0_max = -1e-6;
      const double T0_init = std::min(std::max(-beta * gauss_var_tof, T0_min), T0_max);

      // Params are stored in log space for smoother optimization; T0 is raw (can be
      // negative)
      params.resize(9);
      params << std::log(l11), l21, l31, std::log(l22), l32, std::log(l33),
        std::log(alpha), std::log(beta), T0_init;

      min_bounds = {std::log(1e-6),
                    -1e2,
                    -1e2,
                    std::log(1e-6),
                    -1e2,
                    std::log(1e-6),
                    std::log(alpha_bounds[0]),
                    std::log(beta_bounds[0]),
                    T0_min};
      max_bounds = {std::log(1e6),
                    1e2,
                    1e2,
                    std::log(1e6),
                    1e2,
                    std::log(1e6),
                    std::log(alpha_bounds[1]),
                    std::log(beta_bounds[1]),
                    T0_max};

      // Sanity check
      DIALS_ASSERT(alpha >= alpha_bounds[0] && alpha <= alpha_bounds[1]);
      DIALS_ASSERT(beta >= beta_bounds[0] && beta <= beta_bounds[1]);

      functor.emplace(
        coords, y_norm, background_variances, min_bounds, max_bounds, use_central_diff);
      // Initialise cached norm/A so result() works before the first fit() call
      {
        Eigen::Matrix3d H0 = functor->build_H_from_L(params);
        functor->cached_norm = functor->get_norm_factor(H0, alpha, beta, T0_init);
        if (functor->cached_norm <= 0.0 || !std::isfinite(functor->cached_norm))
          functor->cached_norm = 1.0;
        functor->cached_A =
          functor->calc_A(H0, alpha, beta, functor->cached_norm, T0_init);
        functor->cache_valid = false;  // will be refreshed by operator() on first call
      }
    }

    scitbx::af::versa<vec3<double>, af::c_grid<3>> get_rel_coords(
      scitbx::af::const_ref<vec3<double>, af::c_grid<3>> coords_,
      const scitbx::af::versa<double, af::c_grid<3>> intensities_) {
      /**
       * Calculates coords relative to the coordinate of the peak intensity
       */

      // Get max intensity
      double max_intensity = -1.;
      std::size_t coord_idx = 0;
      for (std::size_t i = 0; i < intensities_.size(); ++i) {
        double w = intensities_[i];
        if (!std::isfinite(w) || w <= 0.0) continue;
        if (w > max_intensity) {
          max_intensity = w;
          coord_idx = i;
        }
      }

      // Shift coordinates relative to peak
      vec3<double> peak = coords_[coord_idx];
      scitbx::af::versa<vec3<double>, af::c_grid<3>> rel_coords(coords_.accessor());
      for (std::size_t i = 0; i < coords_.size(); ++i) {
        rel_coords[i] = vec3<double>(
          coords_[i][0] - peak[0], coords_[i][1] - peak[1], coords_[i][2] - peak[2]);
      }

      return rel_coords;
    }

    // Alpha and beta are optimized in log-space; T0 is stored raw
    double get_alpha() const {
      return std::exp(params[6]);
    }
    double get_beta() const {
      return std::exp(params[7]);
    }
    double get_T0() const {
      return params[8];
    }

    scitbx::af::versa<double, af::c_grid<3>> result() const {
      /**
       * Returns the profile for all coords
       */

      Eigen::Matrix3d H = functor->build_H_from_L(params);
      double alpha = get_alpha();
      double beta = get_beta();
      double T0 = get_T0();

      double norm_factor = functor->cached_norm;
      double A = functor->cached_A;

      scitbx::af::versa<double, af::c_grid<3>> out(coords.accessor());
      for (std::size_t c_x = 0; c_x < coords.accessor()[0]; ++c_x) {
        for (std::size_t c_y = 0; c_y < coords.accessor()[1]; ++c_y) {
          for (std::size_t c_z = 0; c_z < coords.accessor()[2]; ++c_z) {
            double out_val = functor->func(coords(c_x, c_y, c_z),
                                           H,
                                           alpha,
                                           beta,
                                           A,
                                           norm_factor,
                                           functor->dt_widths[c_z],
                                           T0);
            out(c_x, c_y, c_z) = out_val;
          }
        }
      }
      return out;
    }

    double calc_intensity() const {
      /**
       * Intensity calculated as a sum as the functional form
       * is integrable analytically
       */

      scitbx::af::versa<double, af::c_grid<3>> r = result();

      const auto& acc = r.accessor();
      std::size_t nx = acc[0], ny = acc[1], nz = acc[2];

      double total = 0;
      for (std::size_t ix = 0; ix < nx; ++ix) {
        for (std::size_t iy = 0; iy < ny; ++iy) {
          for (std::size_t iz = 0; iz < nz; ++iz) {
            double v = r(ix, iy, iz);
            total += v;
          }
        }
      }
      return total;
    }

    bool fit(bool show_profile_failures,
             int maxfev = 200,
             double xtol = 1e-8,
             double ftol = 1e-8) {
      /*
       * Two-stage least-squares minimization.
       *
       * Stage 1 (diagonal H, 5 params): Fix the three off-diagonal L elements
       * to zero and optimize only the diagonal scales and alpha/beta.  This is a
       * well-conditioned sub-problem that reliably moves close to the true minimum
       * regardless of the starting point.
       *
       * Stage 2 (full H + T0, 9 params): Refine all parameters starting from the
       * stage-1 solution.  Because stage 1 already places us near the minimum,
       * stage 2 almost always converges in one attempt, making random restarts
       * rarely necessary.
       */

      typedef Eigen::LevenbergMarquardt<Profile3DGutmannFunctor, double> LM;
      typedef Eigen::LevenbergMarquardt<DiagonalGutmannFunctor, double> LM6;

      auto run_single_fit = [&](const Eigen::VectorXd& x_init,
                                double& final_error) -> bool {
        Eigen::VectorXd fvec1(functor->intensities.size());
        Eigen::VectorXd x = x_init;
        (*functor)(x, fvec1);
        final_error = fvec1.squaredNorm();
        LM lm(*functor);
        lm.parameters.maxfev = maxfev;
        lm.parameters.xtol = xtol;
        lm.parameters.ftol = ftol;

        int result = lm.minimize(x);
        if (result < 0) return false;

        x = functor->clamp_params(x);

        // Compute residual norm
        Eigen::VectorXd fvec(functor->intensities.size());
        (*functor)(x, fvec);
        final_error = fvec.squaredNorm();

        params = x;
        return true;
      };

      // Stage 1: diagonal H + T0 (6 params — log_l11, log_l22, log_l33, log_α, log_β,
      // T0)
      {
        DiagonalGutmannFunctor diag_functor(*functor);
        Eigen::VectorXd x6(6);
        x6 << params[0], params[3], params[5], params[6], params[7], params[8];

        LM6 lm6(diag_functor);
        lm6.parameters.maxfev = maxfev;
        lm6.parameters.xtol = xtol;
        lm6.parameters.ftol = ftol;
        lm6.minimize(x6);

        // Write back; off-diagonal elements (params[1], params[2], params[4]) stay
        // at 0 from block-diagonal init and are refined in stage 2.
        params[0] = x6[0];
        params[3] = x6[1];
        params[5] = x6[2];
        params[6] = x6[3];
        params[7] = x6[4];
        params[8] = x6[5];
        params = functor->clamp_params(params);
      }

      // Stage 2: full 9-parameter refinement from stage-1 warm start
      Eigen::VectorXd x0 = params;
      double fit_resid = std::numeric_limits<double>::infinity();
      bool success = run_single_fit(x0, fit_resid);
      double I_prf;

      if (success) {
        I_prf = this->calc_intensity();
        if (this->trust_result(fit_resid, I_prf, show_profile_failures)) {
          return true;
        }
      }

      // Stage 2 failed: perturb the stage-1 warm start and retry.
      // With the improved initialization this loop is rarely entered.
      std::mt19937 rng(std::random_device{}());
      std::uniform_real_distribution<double> unit_dist(0.0, 1.0);
      std::normal_distribution<double> norm_dist(0.0, 0.5);

      for (int i = 0; i < n_restarts; ++i) {
        Eigen::VectorXd x_try = x0;

        // H values should be correlated, so try scaling first
        if (i < n_restarts / 3) {
          // Small H scale perturbations
          for (int j = 0; j < 6; ++j) {
            double perturb = 0.8 + 0.4 * unit_dist(rng);
            x_try[j] += std::log(perturb);
          }
        } else if (i < 2 * n_restarts / 3) {
          // Larger H scale perturbations
          double scale = 0.5 + 1.5 * unit_dist(rng);
          for (int j = 0; j < 6; ++j)
            x_try[j] += std::log(scale);
        } else {
          // Random perturbations of H
          for (int j = 0; j < 6; ++j)
            x_try[j] = min_bounds[j] + unit_dist(rng) * (max_bounds[j] - min_bounds[j]);
        }

        // alpha, beta perturbations
        for (int j = 6; j < 8; ++j) {
          x_try[j] = x0[j] + norm_dist(rng);
        }
        // T0 perturbation (perturb by up to 20% of the T0 range)
        {
          double T0_range = max_bounds[8] - min_bounds[8];
          x_try[8] = x0[8] + norm_dist(rng) * 0.2 * T0_range;
        }

        x_try = functor->clamp_params(x_try);

        success = run_single_fit(x_try, fit_resid);
        if (!success) continue;
        I_prf = this->calc_intensity();
        if (this->trust_result(fit_resid, I_prf, show_profile_failures)) {
          return true;
        }
      }

      return false;
    }

    bool trust_result(double error, double I_prf, bool show_error = false) const {
      /**
       * Tests to check the fit is reasonable
       */

      // Check reasonable error
      if (!std::isfinite(error) || error <= 0.0) {
        if (show_error) {
          std::cerr << "profile_3d_gutmann fitting failure: invalid error value (error="
                    << error << ")\n";
        }
        return false;
      }

      // Check reasonable intensity
      if (I_prf < 1e-7) {
        if (show_error) {
          std::cerr
            << "profile_3d_gutmann fitting failure: profile intensity too small (I_prf="
            << I_prf << ")\n";
        }
        return false;
      }

      double alpha = get_alpha();
      double beta = get_beta();
      double T0 = get_T0();

      // Check positive H
      Eigen::Matrix3d H = functor->build_H_from_L(params);
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(H);
      double min_eig = eig.eigenvalues().minCoeff();
      if (min_eig <= 0.0) {
        if (show_error) {
          std::cerr
            << "profile_3d_gutmann fitting failure: non-positive Hessian eigenvalue "
            << "(min_eigenvalue=" << min_eig << ")\n";
        }
        return false;
      }

      // Check all values sensible
      const double exp_limit = 1e12;
      const double mean_limit = 1e12;
      const auto& acc = coords.accessor();
      const std::size_t nx = acc[0];
      const std::size_t ny = acc[1];
      const std::size_t nz = acc[2];

      double profile_peak = 0.0;
      double data_peak = 0.0;
      double sum_val = 0.0;
      double num = 0.0, denom_y = 0.0, denom_m = 0.0;

      double norm_factor = functor->cached_norm;
      double A = functor->cached_A;

      for (std::size_t ix = 0; ix < nx; ++ix) {
        for (std::size_t iy = 0; iy < ny; ++iy) {
          for (std::size_t iz = 0; iz < nz; ++iz) {
            double data_val = y_norm(ix, iy, iz);
            double val = functor->func(coords(ix, iy, iz),
                                       H,
                                       alpha,
                                       beta,
                                       A,
                                       norm_factor,
                                       functor->dt_widths[iz],
                                       T0);

            if (!std::isfinite(val)) {
              if (show_error) {
                std::cerr
                  << "profile_3d_gutmann fitting failure: non-finite model value "
                  << "(ix=" << ix << ", iy=" << iy << ", iz=" << iz << ", val=" << val
                  << ")\n";
              }
              return false;
            }

            if (val > exp_limit) {
              if (show_error) {
                std::cerr
                  << "profile_3d_gutmann fitting failure: model value exceeds limit "
                  << "(ix=" << ix << ", iy=" << iy << ", iz=" << iz << ", val=" << val
                  << ", limit=" << exp_limit << ")\n";
              }
              return false;
            }

            data_peak = std::max(data_peak, data_val);
            profile_peak = std::max(profile_peak, val);
            sum_val += val;
            num += data_val * val;
            denom_y += data_val * data_val;
            denom_m += val * val;
          }
        }
      }

      double mean_val = sum_val / (nx * ny * nz);
      if (mean_val > mean_limit) {
        if (show_error) {
          std::cerr
            << "profile_3d_gutmann fitting failure: mean profile value too large "
            << "(mean_val=" << mean_val << ", limit=" << mean_limit << ")\n";
        }
        return false;
      }

      // Cosine similarity between data and model
      double corr = num / std::sqrt(denom_y * denom_m + 1e-12);
      if (corr < 0.75) {
        if (show_error) {
          std::cerr << "profile_3d_gutmann fitting failure: low correlation with data "
                    << "(corr=" << corr << ")\n";
        }
        return false;
      }

      // Check peak height is within 50% of data peak
      double peak_diff = std::abs(profile_peak - data_peak);
      if (peak_diff > data_peak * 0.50) {
        if (show_error) {
          std::cerr << "profile_3d_gutmann fitting failure: peak height mismatch "
                    << "(profile_peak=" << profile_peak << ", data_peak=" << data_peak
                    << ", diff=" << peak_diff << ")\n";
        }
        return false;
      }

      return true;
    }
  };

  bool fit_profile_3d_gutmann(
    scitbx::af::const_ref<vec3<double>, af::c_grid<3>> coords,
    const scitbx::af::versa<double, af::c_grid<3>> intensities,
    const scitbx::af::versa<double, af::c_grid<3>> background_variances,
    TOFProfile3DGutmannParams& profile_params,
    double& I_prf_out,
    boost::optional<scitbx::af::versa<double, af::c_grid<3>>> profile_3d_out =
      boost::none,
    bool update_params = false) {
    /**
     * Wrapper for fitting a given reflection
     * If profile_3d_out is provided the profile is returned at every
     * position in coords
     */

    // Fit profile
    const std::array<double, 2> alpha_bounds = {profile_params.alpha_min,
                                                profile_params.alpha_max};
    const std::array<double, 2> beta_bounds = {profile_params.beta_min,
                                               profile_params.beta_max};

    TOFProfile3DGutmann profile(coords,
                                intensities,
                                background_variances,
                                profile_params.alpha,
                                profile_params.beta,
                                alpha_bounds,
                                beta_bounds,
                                profile_params.n_restarts,
                                profile_params.use_central_diff);

    bool profile_success = true;
    if (profile_params.optimize_profile) {
      profile_success = profile.fit(profile_params.show_profile_failures);
    }

    if (profile_success) {
      double I_prf = profile.calc_intensity();
      I_prf_out = I_prf;

      if (update_params) {
        profile_params.alpha = profile.get_alpha();
        profile_params.beta = profile.get_beta();
      }
      if (!profile_3d_out) {
        return profile_success;
      }

      scitbx::af::versa<double, af::c_grid<3>> pred = profile.result();
      scitbx::af::versa<double, af::c_grid<3>> profile_3d = *profile_3d_out;

      DIALS_ASSERT(pred.accessor().all_eq(profile_3d.accessor()));
      for (std::size_t x = 0; x < pred.accessor()[0]; ++x) {
        for (std::size_t y = 0; y < pred.accessor()[1]; ++y) {
          for (std::size_t z = 0; z < pred.accessor()[2]; ++z) {
            profile_3d(x, y, z) = pred(x, y, z);
          }
        }
      }
      return profile_success;
    }
    return false;
  }

}}  // namespace dials::algorithms

#endif  // DIALS_ALGORITHMS_INTEGRATION_TOF_PROFILE_3D_GUTMANN_H
