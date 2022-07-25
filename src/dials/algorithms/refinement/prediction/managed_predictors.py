"""Managed reflection prediction for refinement.

* ScansRayPredictor adapts DIALS prediction for use in refinement, by keeping
  up to date with the current model geometry
* StillsRayPredictor predicts reflections without a goniometer, under
  the naive assumption that the relp is already in reflecting position
"""


from __future__ import annotations

from math import pi

from scipy.constants import Planck, m_n

from scitbx.array_family import flex

from dials.algorithms.spot_prediction import ScanStaticRayPredictor
from dials.algorithms.spot_prediction import ScanStaticReflectionPredictor as sc
from dials.algorithms.spot_prediction import ScanVaryingReflectionPredictor as sv
from dials.algorithms.spot_prediction import StillsReflectionPredictor as st
from dials.algorithms.spot_prediction import TOFReflectionPredictorPy

# constants
TWO_PI = 2.0 * pi


class ScansRayPredictor:
    """
    Predict for a relp based on the current states of models of the
    experimental geometry. This is a wrapper for DIALS' C++
    RayPredictor class, which does the real work. This class keeps track
    of the experimental geometry, and instantiates a RayPredictor when
    required.
    """

    def __init__(self, experiments, sequence_range=(0, 2.0 * pi)):
        """Construct by linking to instances of experimental model classes"""

        self._experiments = experiments
        self._sequence_range = sequence_range

    def __call__(self, hkl, experiment_id=0, UB=None):
        """
        Solve the prediction formula for the reflecting angle phi.

        If UB is given, override the contained crystal model. This is
        for use in refinement with time-varying crystal parameters
        """

        e = self._experiments[experiment_id]
        ray_predictor = ScanStaticRayPredictor(
            e.beam.get_s0(),
            e.goniometer.get_rotation_axis_datum(),
            e.goniometer.get_fixed_rotation(),
            e.goniometer.get_setting_rotation(),
            self._sequence_range,
        )

        UB_ = UB if UB else e.crystal.get_A()

        rays = ray_predictor(hkl, UB_)

        return rays


class ExperimentsPredictor:
    """
    Predict for relps based on the current states of models of the experimental
    geometry. This version manages multiple experiments, selecting the correct
    predictor in each case.
    """

    def __init__(self, experiments):
        """Construct by linking to instances of experimental model classes"""

        self._experiments = experiments

    def __call__(self, reflections):
        """Predict for all reflections at the current model geometry"""

        for iexp, e in enumerate(self._experiments):

            # select the reflections for this experiment only
            sel = reflections["id"] == iexp
            refs = reflections.select(sel)

            self._predict_one_experiment(e, refs)
            # write predictions back to overall reflections
            reflections.set_selected(sel, refs)

        reflections = self._post_prediction(reflections)

        return reflections

    def _predict_one_experiment(self, experiment, reflections):

        raise NotImplementedError()

    def _post_prediction(self, reflections):
        """Perform tasks on the whole reflection list after prediction before
        returning."""

        return reflections


class TOFExperimentsPredictor(ExperimentsPredictor):
    def _predict_one_experiment(self, experiment, reflections):

        min_s0_idx = min(
            range(len(reflections["wavelength"])),
            key=reflections["wavelength"].__getitem__,
        )
        min_s0 = reflections["s0"][min_s0_idx]
        dmin = experiment.detector.get_max_resolution(min_s0)
        predictor = TOFReflectionPredictorPy(experiment, dmin)
        UB = experiment.crystal.get_A()
        predictor.for_reflection_table(reflections, UB)

    def _post_prediction(self, reflections):

        if "tof_cal" not in reflections:
            reflections["tof_cal"] = flex.double(reflections.nrows())
        for i, expt in enumerate(self._experiments):
            if "imageset_id" in reflections:
                sel_expt = reflections["imageset_id"] == i
            else:
                sel_expt = reflections["id"] == i
            expt_reflections = reflections.select(sel_expt)
            L = expt.beam.get_sample_to_moderator_distance()
            expt_reflections = reflections.select(sel_expt)
            for idx, reflection in enumerate(expt_reflections):
                wavelength = expt_reflections[idx]["wavelength_cal"]
                expt_reflections[idx]["tof_cal"] = self._calc_tof_from_wavelength(
                    wavelength, L
                )

        return reflections

    def _calc_tof_from_wavelength(self, wavelength, L):
        return (wavelength * m_n * L) / Planck


class ScansExperimentsPredictor(ExperimentsPredictor):
    def _predict_one_experiment(self, experiment, reflections):

        # scan-varying
        if "ub_matrix" in reflections:
            predictor = sv(experiment)
            UB = reflections["ub_matrix"]
            s0 = reflections["s0_vector"]
            dmat = reflections["d_matrix"]
            Smat = reflections["S_matrix"]
            predictor.for_reflection_table(reflections, UB, s0, dmat, Smat)
        # scan static
        else:
            predictor = sc(experiment)
            UB = experiment.crystal.get_A()
            predictor.for_reflection_table(reflections, UB)

    def _post_prediction(self, reflections):

        if "xyzobs.mm.value" in reflections:
            reflections = self._match_full_turns(reflections)

        return reflections

    def _match_full_turns(self, reflections):
        """Modify the calculated phi values so that they match the full rotation
        from zero taken from the the observations, rather than being modulo 2*pi."""

        x_obs, y_obs, phi_obs = reflections["xyzobs.mm.value"].parts()
        x_calc, y_calc, phi_calc = reflections["xyzcal.mm"].parts()
        resid = phi_calc - (flex.fmod_positive(phi_obs, TWO_PI))
        # ensure this is the smaller of two possibilities
        resid = flex.fmod_positive((resid + pi), TWO_PI) - pi
        phi_calc = phi_obs + resid
        reflections["xyzcal.mm"] = flex.vec3_double(x_calc, y_calc, phi_calc)

        # Update xyzcal.px with the correct z_px values in keeping with above
        for iexp, e in enumerate(self._experiments):
            sel = reflections["id"] == iexp
            x_px, y_px, z_px = reflections["xyzcal.px"].select(sel).parts()
            scan = e.sequence
            if scan is not None:
                z_px = scan.get_array_index_from_angle(phi_calc.select(sel), deg=False)
            else:
                # must be a still image, z centroid not meaningful
                z_px = phi_calc.select(sel)
            xyzcal_px = flex.vec3_double(x_px, y_px, z_px)
            reflections["xyzcal.px"].set_selected(sel, xyzcal_px)

        return reflections


class StillsExperimentsPredictor(ExperimentsPredictor):

    spherical_relp_model = False

    def _predict_one_experiment(self, experiment, reflections):
        """
        if reflections.contains_valid_tof_data():
            self._predict_one_tof_experiment(experiment, reflections)
        else:
        """
        predictor = st(experiment, spherical_relp=self.spherical_relp_model)
        UB = experiment.crystal.get_A()
        predictor.for_reflection_table(reflections, UB)


class ExperimentsPredictorFactory:
    @staticmethod
    def from_experiments(experiments, force_stills=False, spherical_relp=False):

        if experiments.is_single_tof_experiment():
            return TOFExperimentsPredictor(experiments)

        # Determine whether or not to use a stills predictor
        if not force_stills:
            for exp in experiments:
                if exp.goniometer is None:
                    force_stills = True
                    break

        # Construct the predictor
        if force_stills:
            predictor = StillsExperimentsPredictor(experiments)
            predictor.spherical_relp_model = spherical_relp
        else:
            predictor = ScansExperimentsPredictor(experiments)

        return predictor
