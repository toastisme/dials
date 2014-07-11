#
#  Copyright (C) (2013) STFC Rutherford Appleton Laboratory, UK.
#
#  Author: David Waterman.
#
#  This code is distributed under the BSD license, a copy of which is
#  included in the root directory of this package.
#

# Python imports
from __future__ import division

# cctbx imports
from scitbx import matrix

# dials imports
#from dials_refinement_helpers_ext import *
from dials.algorithms.refinement.parameterisation.prediction_parameters import \
    XYPhiPredictionParameterisation

from dials.array_family import flex

class VaryingCrystalPredictionParameterisation(XYPhiPredictionParameterisation):
  """Support crystal parameterisations that vary with time (via the proxy of
  "observed image number")"""

  def compose(self, reflections):
    """Compose scan-varying crystal parameterisations at the specified image
    number, for the specified experiment, for all reflections. Put the U, B and
    UB matrices in the reflection table, and cache the derivatives."""

    nref = len(reflections)
    # set columns if needed
    if not reflections.has_key('u_matrix'):
      reflections['u_matrix'] = flex.mat3_double(nref)
    if not reflections.has_key('b_matrix'):
      reflections['b_matrix'] = flex.mat3_double(nref)

    # set up arrays to store derivatives
    num_free_U_params = sum([e.num_free() for e in self._xl_orientation_parameterisations])
    num_free_B_params = sum([e.num_free() for e in self._xl_unit_cell_parameterisations])
    null = (0., 0., 0., 0., 0., 0., 0., 0., 0.)
    self._dU_dp = [flex.mat3_double(nref, null) for i in range(num_free_U_params)]
    self._dB_dp = [flex.mat3_double(nref, null) for i in range(num_free_B_params)]

    ori_offset = uc_offset = 0
    for iexp, exp in enumerate(self._experiments):

      # select the reflections of interest
      sel = reflections['id'] == iexp
      isel = sel.iselection()

      # get their frame numbers
      obs_image_numbers = (reflections['xyzobs.px.value'].parts()[2]).select(isel)

      # identify which crystal parameterisations to use for this experiment
      param_set = self._exp_to_param[iexp]
      xl_ori_param_id = param_set.xl_ori_param
      xl_uc_param_id = param_set.xl_uc_param
      xl_op = self._xl_orientation_parameterisations[param_set.xl_ori_param]
      xl_ucp = self._xl_unit_cell_parameterisations[param_set.xl_uc_param]

      # get state and derivatives for each reflection
      for i, frame in zip(isel, obs_image_numbers):

        # compose the models
        xl_op.compose(frame)
        xl_ucp.compose(frame)

        # set states
        row = {'u_matrix':xl_op.get_state().elems,
               'b_matrix':xl_ucp.get_state().elems}
        reflections[i] = row

        # set derivatives of the states
        for j, dU in enumerate(xl_op.get_ds_dp()):
          j2 = j + ori_offset
          self._dU_dp[j2][i] = dU
        for j, dB in enumerate(xl_ucp.get_ds_dp()):
          j2 = j + uc_offset
          self._dB_dp[j2][i] = dB

      ori_offset += xl_op.num_free()
      uc_offset += xl_ucp.num_free()

    # set the UB matrices for prediction
    reflections['ub_matrix'] = reflections['u_matrix'] * reflections['b_matrix']

    return

  def get_UB(self, obs_image_number, experiment_id):
    """Extract the setting matrix from the contained scan
    dependent crystal parameterisations at specified image number"""

    # called by refiner.run for setting the crystal scan points
    param_set = self._exp_to_param[experiment_id]
    xl_ori_param_id = param_set.xl_ori_param
    xl_uc_param_id = param_set.xl_uc_param
    xl_op = self._xl_orientation_parameterisations[param_set.xl_ori_param]
    xl_ucp = self._xl_unit_cell_parameterisations[param_set.xl_uc_param]

    xl_op.compose(obs_image_number)
    xl_ucp.compose(obs_image_number)

    UB = xl_op.get_state() * xl_ucp.get_state()
    return UB

  # overloaded for the scan-varying case
  def _get_U_B_for_experiment(self, crystal, reflections, isel):
    """helper function to return either a single U, B pair (for scan-static) or
    U, B arrays (scan-varying; overloaded in derived class) for a particular
    experiment."""

    # crystal ignored here (it is needed for the scan-static version only)
    U = reflections['u_matrix'].select(isel)
    B = reflections['b_matrix'].select(isel)
    return U, B

  def _get_gradients_core(self, reflections, D, s0, U, B, axis):
    """Calculate gradients of the prediction formula with respect to
    each of the parameters of the contained models, for reflection h
    that reflects at rotation angle phi with scattering vector s that
    intersects panel panel_id. That is, calculate dX/dp, dY/dp and
    dphi/dp"""

    # Spindle rotation matrices for every reflection
    #R = self._axis.axis_and_angle_as_r3_rotation_matrix(phi)
    #R = flex.mat3_double(len(reflections))
    # NB for now use flex.vec3_double.rotate_around_origin each time I need the
    # rotation matrix R.

    self._axis = axis
    self._s0 = s0

    # pv is the 'projection vector' for the ray along s1.
    self._D = D
    self._s1 = reflections['s1']
    self._pv = D * self._s1

    # also need quantities derived from pv, precalculated for efficiency
    u, v, w = self._pv.parts()
    self._w_inv = 1/w
    self._u_w_inv = u * self._w_inv
    self._v_w_inv = v * self._w_inv

    self._UB = U * B
    self._U = U
    self._B = B

    # r is the reciprocal lattice vector, in the lab frame
    self._h = reflections['miller_index'].as_vec3_double()
    self._phi_calc = reflections['xyzcal.mm'].parts()[2]
    self._r = (self._UB * self._h).rotate_around_origin(self._axis, self._phi_calc)

    # All of the derivatives of phi have a common denominator, given by
    # (e X r).s0, where e is the rotation axis. Calculate this once, here.
    self._e_X_r = self._axis.cross(self._r)
    self._e_r_s0 = (self._e_X_r).dot(self._s0)

    # Note that e_r_s0 -> 0 when the rotation axis, beam vector and
    # relp are coplanar. This occurs when a reflection just touches
    # the Ewald sphere.
    #
    # There is a relationship between e_r_s0 and zeta_factor.
    # Uncommenting the code below shows that
    # s0.(e X r) = zeta * |s X s0|

    #from dials.algorithms.reflection_basis import zeta_factor
    #from libtbx.test_utils import approx_equal
    #s = matrix.col(reflections['s1'][0])
    #z = zeta_factor(axis[0], s0[0], s)
    #ss0 = (s.cross(matrix.col(s0[0]))).length()
    #assert approx_equal(e_r_s0[0], z * ss0)

    # catch small values of e_r_s0
    e_r_s0_mag = flex.abs(self._e_r_s0)
    try:
      assert flex.min(e_r_s0_mag) > 1.e-6
    except AssertionError as e:
      imin = flex.min_index(e_r_s0_mag)
      print "(e X r).s0 too small:"
      print "for", (e_r_s0_mag <= 1.e-6).count(True), "reflections"
      print "out of", len(e_r_s0_mag), "total"
      print "such as", reflections['miller_index'][imin]
      print "with scattering vector", reflections['s1'][imin]
      print "where r =", self._r[imin]
      print "e =", self._axis[imin]
      print "s0 =", self._s0[imin]
      print ("this reflection forms angle with the equatorial plane "
             "normal:")
      vecn = matrix.col(self._s0[imin]).cross(matrix.col(self._axis[imin])).normalize()
      print matrix.col(reflections['s1'][imin]).accute_angle(vecn)
      raise e

    # Set up the lists of derivatives: a separate array over reflections for
    # each free parameter
    m = len(reflections)
    n = len(self) # number of free parameters
    dX_dp, dY_dp, dphi_dp = self._prepare_gradient_vectors(m, n)

    # determine experiment to indices mappings once, here
    experiment_to_idx = []
    for iexp, exp in enumerate(self._experiments):

      sel = reflections['id'] == iexp
      isel = sel.iselection()
      experiment_to_idx.append(isel)

    # reset a pointer to the parameter number
    self._iparam = 0

    ### Work through the parameterisations, calculating their contributions
    ### to derivatives d[pv]/dp and d[phi]/dp

    # loop over the detector parameterisations
    for dp in self._detector_parameterisations:

      # Determine (sub)set of reflections affected by this parameterisation
      isel = flex.size_t()
      for exp_id in dp.get_experiment_ids():
        isel.extend(experiment_to_idx[exp_id])

      # Access the detector model being parameterised
      detector = dp.get_model()

      # Get panel numbers of the affected reflections
      panel = reflections['panel'].select(isel)

      # loop through the panels in this detector
      for panel_id, _ in enumerate(exp.detector):

        # get the right subset of array indices to set for this panel
        sub_isel = isel.select(panel == panel_id)
        sub_pv = self._pv.select(sub_isel)
        sub_D = self._D.select(sub_isel)
        dpv_ddet_p = self._detector_derivatives(dp, sub_pv, sub_D, panel_id)

        # convert to dX/dp, dY/dp and assign the elements of the vectors
        # corresponding to this experiment and panel
        sub_w_inv = self._w_inv.select(sub_isel)
        sub_u_w_inv = self._u_w_inv.select(sub_isel)
        sub_v_w_inv = self._v_w_inv.select(sub_isel)
        dX_ddet_p, dY_ddet_p = self._calc_dX_dp_and_dY_dp_from_dpv_dp(
          sub_w_inv, sub_u_w_inv, sub_v_w_inv, dpv_ddet_p)
        iparam = self._iparam
        for dX, dY in zip(dX_ddet_p, dY_ddet_p):
          dX_dp[iparam].set_selected(sub_isel, dX)
          dY_dp[iparam].set_selected(sub_isel, dY)
          # increment the local parameter index pointer
          iparam += 1

      # increment the parameter index pointer to the last detector parameter
      self._iparam += dp.num_free()

    # loop over the beam parameterisations
    for bp in self._beam_parameterisations:

      # Determine (sub)set of reflections affected by this parameterisation
      isel = flex.size_t()
      for exp_id in bp.get_experiment_ids():
        isel.extend(experiment_to_idx[exp_id])

      # Get required data from those reflections
      r = self._r.select(isel)
      e_X_r = self._e_X_r.select(isel)
      e_r_s0 = self._e_r_s0.select(isel)
      D = self._D.select(isel)

      w_inv = self._w_inv.select(isel)
      u_w_inv = self._u_w_inv.select(isel)
      v_w_inv = self._v_w_inv.select(isel)

      dpv_dbeam_p, dphi_dbeam_p = self._beam_derivatives(bp, r, e_X_r, e_r_s0, D)

      # convert to dX/dp, dY/dp and assign the elements of the vectors
      # corresponding to this experiment
      dX_dbeam_p, dY_dbeam_p = self._calc_dX_dp_and_dY_dp_from_dpv_dp(
        w_inv, u_w_inv, v_w_inv, dpv_dbeam_p)
      for dX, dY, dphi in zip(dX_dbeam_p, dY_dbeam_p, dphi_dbeam_p):
        dphi_dp[self._iparam].set_selected(isel, dphi)
        dX_dp[self._iparam].set_selected(isel, dX)
        dY_dp[self._iparam].set_selected(isel, dY)
        # increment the parameter index pointer
        self._iparam += 1

    # loop over the crystal orientation parameterisations
    local_iparam = 0
    for xlop in self._xl_orientation_parameterisations:

      # Determine (sub)set of reflections affected by this parameterisation
      isel = flex.size_t()
      for exp_id in xlop.get_experiment_ids():
        isel.extend(experiment_to_idx[exp_id])

      # Get required data from those reflections
      axis = self._axis.select(isel)
      phi_calc = self._phi_calc.select(isel)
      h = self._h.select(isel)
      s1 = self._s1.select(isel)
      e_X_r = self._e_X_r.select(isel)
      e_r_s0 = self._e_r_s0.select(isel)
      B = self._B.select(isel)
      D = self._D.select(isel)

      w_inv = self._w_inv.select(isel)
      u_w_inv = self._u_w_inv.select(isel)
      v_w_inv = self._v_w_inv.select(isel)

      # get derivatives of the U matrix wrt the parameters
      dU_dxlo_p = [self._dU_dp[i].select(isel) for i in range(local_iparam,
        local_iparam + xlop.num_free())]
      dpv_dxlo_p, dphi_dxlo_p = self._xl_orientation_derivatives(
        dU_dxlo_p, axis, phi_calc, h, s1, e_X_r, e_r_s0, B, D)

      # convert to dX/dp, dY/dp and assign the elements of the vectors
      # corresponding to this experiment
      dX_dxlo_p, dY_dxlo_p = self._calc_dX_dp_and_dY_dp_from_dpv_dp(
        w_inv, u_w_inv, v_w_inv, dpv_dxlo_p)
      for dX, dY, dphi in zip(dX_dxlo_p, dY_dxlo_p, dphi_dxlo_p):
        dphi_dp[self._iparam].set_selected(isel, dphi)
        dX_dp[self._iparam].set_selected(isel, dX)
        dY_dp[self._iparam].set_selected(isel, dY)
        # increment the parameter index pointer
        self._iparam += 1
        local_iparam += 1

    # loop over the crystal unit cell parameterisations
    local_iparam = 0
    for xlucp in self._xl_unit_cell_parameterisations:

      # Determine (sub)set of reflections affected by this parameterisation
      isel = flex.size_t()
      for exp_id in xlop.get_experiment_ids():
        isel.extend(experiment_to_idx[exp_id])

      # Get required data from those reflections
      axis = self._axis.select(isel)
      phi_calc = self._phi_calc.select(isel)
      h = self._h.select(isel)
      s1 = self._s1.select(isel)
      e_X_r = self._e_X_r.select(isel)
      e_r_s0 = self._e_r_s0.select(isel)
      U = self._U.select(isel)
      D = self._D.select(isel)

      w_inv = self._w_inv.select(isel)
      u_w_inv = self._u_w_inv.select(isel)
      v_w_inv = self._v_w_inv.select(isel)

      dB_dxluc_p = [self._dB_dp[i].select(isel) for i in range(
        local_iparam, local_iparam + xlucp.num_free())]
      dpv_dxluc_p, dphi_dxluc_p =  self._xl_unit_cell_derivatives(
        dB_dxluc_p, axis, phi_calc, h, s1, e_X_r, e_r_s0, U, D)

      # convert to dX/dp, dY/dp and assign the elements of the vectors
      # corresponding to this experiment
      dX_dxluc_p, dY_dxluc_p = self._calc_dX_dp_and_dY_dp_from_dpv_dp(
        w_inv, u_w_inv, v_w_inv, dpv_dxluc_p)
      for dX, dY, dphi in zip(dX_dxluc_p, dY_dxluc_p, dphi_dxluc_p):
        dphi_dp[self._iparam].set_selected(isel, dphi)
        dX_dp[self._iparam].set_selected(isel, dX)
        dY_dp[self._iparam].set_selected(isel, dY)
        # increment the parameter index pointer
        self._iparam += 1
        local_iparam += 1

    return (dX_dp, dY_dp, dphi_dp)

  def _xl_orientation_derivatives(self, dU_dxlo_p, axis, phi_calc, h, s1, e_X_r, e_r_s0, B, D):
    """helper function to extend the derivatives lists by
    derivatives of the crystal orientation parameterisations"""

    # get derivatives of the U matrix wrt the parameters
    #dU_dxlo_p = xlop.get_ds_dp()

    dphi_dp = []
    dpv_dp = []

    # loop through the parameters
    for der_mat in dU_dxlo_p:

      # calculate the derivative of r for this parameter
      # FIXME COULD DO THIS BETTER WITH __rmul__?!
      tmp = der_mat * B * h
      dr = tmp.rotate_around_origin(axis, phi_calc)

      # calculate the derivative of phi for this parameter
      dphi = -1.0 * dr.dot(s1) / e_r_s0
      dphi_dp.append(dphi)

      # calculate the derivative of pv for this parameter
      dpv_dp.append(D * (dr + e_X_r * dphi))

    return dpv_dp, dphi_dp

  def _xl_unit_cell_derivatives(self, dB_dxluc_p, axis, phi_calc, h, s1, e_X_r, e_r_s0, U, D):
    """helper function to extend the derivatives lists by
    derivatives of the crystal unit cell parameterisations"""

    # get derivatives of the B matrix wrt the parameters
    #dB_dxluc_p = xlucp.get_ds_dp()

    dphi_dp = []
    dpv_dp = []

    # loop through the parameters
    for der_mat in dB_dxluc_p:

      # calculate the derivative of r for this parameter
      tmp = U * der_mat * h
      dr = tmp.rotate_around_origin(axis, phi_calc)

      # calculate the derivative of phi for this parameter
      dphi = -1.0 * dr.dot(s1) / e_r_s0
      dphi_dp.append(dphi)

      # calculate the derivative of pv for this parameter
      dpv_dp.append(D * (dr + e_X_r * dphi))

    return dpv_dp, dphi_dp

  def calculate_model_state_uncertainties(self, var_cov=None,
                                          obs_image_number=None,
                                          experiment_id=None):
    """
    Take the variance-covariance matrix of all free parameters calculated by
    the minimisation engine. For each parameterisation in the global model,
    extract the subset of this matrix for the associated block of parameters.
    Pass this on to the relevant model parameterisation to calculate its own
    uncertainty of state.

    This scan-varying version should first be called with var_cov set but
    obs_image_number=None and experiment_id=None. This calls the scan-static
    version to do the calculation for the scan-static parameterisations and
    also caches the subsets of var_cov relevant for the scan-varying
    parameterisations. Subsequent calls should provide obs_image_number and
    experiment_id to calculate for a particular crystal at a particular
    scan-point"""

    # first call, only a variance-covariance matrix is supplied
    if var_cov is not None:
      assert [obs_image_number, experiment_id].count(None) == 2
      super(VaryingCrystalPredictionParameterisation,
            self).calculate_model_state_uncertainties(var_cov)
      return

    # later calls, only an experiment and image number are supplied
    else:
      # identify the crystal parameterisations for this experiment
      param_set = self._exp_to_param[experiment_id]
      xl_ori_param_id = param_set.xl_ori_param
      xl_uc_param_id = param_set.xl_uc_param
      xl_op = self._xl_orientation_parameterisations[param_set.xl_ori_param]
      xl_ucp = self._xl_unit_cell_parameterisations[param_set.xl_uc_param]

      # compose at the requested image number
      xl_op.compose(obs_image_number)
      xl_ucp.compose(obs_image_number)

      # calculate using the cached varcov matrices. Take the first elt of the
      # list becase the crystal parameterisations are not multi-state
      U_cov = xl_op.calculate_state_uncertainties(var_cov=None)[0]
      B_cov = xl_ucp.calculate_state_uncertainties(var_cov=None)[0]

    return U_cov, B_cov

  def set_model_state_uncertainties(self, u_cov_list, b_cov_list,
                                          experiment_id=None):
    """Identify the parameterisation"""


    param_set = self._exp_to_param[experiment_id]
    xl_ori_param_id = param_set.xl_ori_param
    xl_uc_param_id = param_set.xl_uc_param
    xl_op = self._xl_orientation_parameterisations[param_set.xl_ori_param]
    xl_ucp = self._xl_unit_cell_parameterisations[param_set.xl_uc_param]

    xl_op.set_state_uncertainties(u_cov_list)
    xl_ucp.set_state_uncertainties(b_cov_list)

    return

class VaryingCrystalPredictionParameterisationFast(VaryingCrystalPredictionParameterisation):
  """Overloads compose to calculate UB model per frame rather than per
  reflection"""

  def compose(self, reflections):
    """Compose scan-varying crystal parameterisations at the specified image
    number, for the specified experiment, for each image. Put the U, B and
    UB matrices in the reflection table, and cache the derivatives."""

    nref = len(reflections)
    # set columns if needed
    if not reflections.has_key('u_matrix'):
      reflections['u_matrix'] = flex.mat3_double(nref)
    if not reflections.has_key('b_matrix'):
      reflections['b_matrix'] = flex.mat3_double(nref)

    # set up arrays to store derivatives
    num_free_U_params = sum([e.num_free() for e in self._xl_orientation_parameterisations])
    num_free_B_params = sum([e.num_free() for e in self._xl_unit_cell_parameterisations])
    null = (0., 0., 0., 0., 0., 0., 0., 0., 0.)
    self._dU_dp = [flex.mat3_double(nref, null) for i in range(num_free_U_params)]
    self._dB_dp = [flex.mat3_double(nref, null) for i in range(num_free_B_params)]

    ori_offset = uc_offset = 0

    for iexp, exp in enumerate(self._experiments):

      # select the reflections of interest
      sel = reflections['id'] == iexp
      isel = sel.iselection()

      # get their integer frame numbers
      frames = reflections['xyzobs.px.value'].parts()[2]
      obs_image_numbers = flex.floor((frames).select(isel)).iround()

      # identify which crystal parameterisations to use for this experiment
      param_set = self._exp_to_param[iexp]
      xl_ori_param_id = param_set.xl_ori_param
      xl_uc_param_id = param_set.xl_uc_param
      xl_op = self._xl_orientation_parameterisations[param_set.xl_ori_param]
      xl_ucp = self._xl_unit_cell_parameterisations[param_set.xl_uc_param]

      # get state and derivatives for each image
      for frame in xrange(flex.min(obs_image_numbers),
                          flex.max(obs_image_numbers) + 1):

        # compose the models
        xl_op.compose(frame)
        xl_ucp.compose(frame)

        # determine the subset of reflections this affects
        subsel = isel.select(obs_image_numbers == frame)

        # set states
        reflections['u_matrix'].set_selected(subsel, xl_op.get_state().elems)
        reflections['b_matrix'].set_selected(subsel, xl_ucp.get_state().elems)

        # set derivatives of the states
        for j, dU in enumerate(xl_op.get_ds_dp()):
          j2 = j + ori_offset
          self._dU_dp[j2].set_selected(subsel, dU)
        for j, dB in enumerate(xl_ucp.get_ds_dp()):
          j2 = j + uc_offset
          self._dB_dp[j2].set_selected(subsel, dB)

      ori_offset += xl_op.num_free()
      uc_offset += xl_ucp.num_free()

    # set the UB matrices for prediction
    reflections['ub_matrix'] = reflections['u_matrix'] * reflections['b_matrix']

    return
