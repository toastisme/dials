from __future__ import annotations

import dials_algorithms_spot_prediction_ext
from dials_algorithms_spot_prediction_ext import (
    IndexGenerator,
    NaveStillsReflectionPredictor,
    PixelLabeller,
    PixelToMillerIndex,
    ReekeIndexGenerator,
    RotationAngles,
    ScanStaticRayPredictor,
    ScanVaryingRayPredictor,
    SphericalRelpStillsReflectionPredictor,
    StillsDeltaPsiReflectionPredictor,
    StillsRayPredictor,
    TOFRayPredictor,
    TOFReflectionPredictor,
    ray_intersection,
)

__all__ = [
    "IndexGenerator",
    "NaveStillsReflectionPredictor",
    "PixelLabeller",
    "PixelToMillerIndex",
    "ray_intersection",
    "ReekeIndexGenerator",
    "RotationAngles",
    "ScanStaticRayPredictor",
    "ScanStaticReflectionPredictor",
    "ScanVaryingRayPredictor",
    "ScanVaryingReflectionPredictor",
    "SphericalRelpStillsReflectionPredictor",
    "StillsDeltaPsiReflectionPredictor",
    "StillsRayPredictor",
    "TOFRayPredictor",
    "StillsReflectionPredictor",
    "TOFReflectionPredictor",
]


import cctbx.array_family.flex
from scitbx.array_family import flex


def ScanStaticReflectionPredictor(experiment, dmin=None, margin=1, padding=0, **kwargs):
    """
    A constructor for the reflection predictor.

    :param experiment: The experiment to predict for
    :param dmin: The maximum resolution to predict to
    :param margin: The margin for prediction
    :return: The spot predictor
    """

    # Get dmin if it is not set
    if dmin is None:
        dmin = experiment.detector.get_max_resolution(experiment.beam.get_s0())

    # Only remove certain systematic absences
    space_group = experiment.crystal.get_space_group()
    space_group = space_group.build_derived_patterson_group()

    # Create the reflection predictor
    return dials_algorithms_spot_prediction_ext.ScanStaticReflectionPredictor(
        experiment.beam,
        experiment.detector,
        experiment.goniometer,
        experiment.sequence,
        experiment.crystal.get_unit_cell(),
        space_group.type(),
        dmin,
        margin,
        padding,
    )


def ScanVaryingReflectionPredictor(
    experiment, dmin=None, margin=1, padding=0, **kwargs
):
    """
    A constructor for the reflection predictor.

    :param experiment: The experiment to predict for
    :param dmin: The maximum resolution to predict to
    :param margin: The margin for prediction
    :return: The spot predictor
    """

    # Get dmin if it is not set
    if dmin is None:
        dmin = experiment.detector.get_max_resolution(experiment.beam.get_s0())

    # Only remove certain systematic absences
    space_group = experiment.crystal.get_space_group()
    space_group = space_group.build_derived_patterson_group()

    # Create the reflection predictor
    return dials_algorithms_spot_prediction_ext.ScanVaryingReflectionPredictor(
        experiment.beam,
        experiment.detector,
        experiment.goniometer,
        experiment.sequence,
        space_group.type(),
        dmin,
        margin,
        padding,
    )


class TOFReflectionPredictorPy:
    def __init__(self, experiment, dmin):
        self.experiment = experiment
        self.dmin = dmin
        self.predictor = TOFReflectionPredictor(
            experiment.beam,
            experiment.detector,
            experiment.goniometer,
            experiment.sequence,
            experiment.crystal.get_A(),
            experiment.crystal.get_unit_cell(),
            experiment.crystal.get_space_group().type(),
            dmin,
        )

    def post_prediction(self, reflections):

        if "tof_cal" not in reflections:
            reflections["tof_cal"] = flex.double(reflections.nrows())
        if "L1" not in reflections:
            reflections["L1"] = flex.double(reflections.nrows())

        tof_cal = flex.double(reflections.nrows())
        L1 = flex.double(reflections.nrows())

        panel_numbers = cctbx.array_family.flex.size_t(reflections["panel"])
        expt = self.experiment

        for i_panel in range(len(expt.detector)):
            sel = panel_numbers == i_panel
            expt_reflections = reflections.select(sel)
            x, y, _ = expt_reflections["xyzcal.mm"].parts()
            s1 = expt.detector[i_panel].get_lab_coord(
                cctbx.array_family.flex.vec2_double(x, y)
            )
            expt_L1 = s1.norms() * 10**-3
            expt_tof_cal = flex.double(expt_reflections.nrows())

            for idx in range(len(expt_reflections)):
                wavelength = expt_reflections[idx]["wavelength_cal"]
                tof = expt.beam.get_tof_from_wavelength(wavelength, expt_L1[idx])
                expt_tof_cal[idx] = tof
            tof_cal.set_selected(sel, expt_tof_cal)
            L1.set_selected(sel, expt_L1)

        reflections["tof_cal"] = tof_cal
        reflections["L1"] = L1

        # Filter out predicted reflections outside of experiment range
        wavelength_range = expt.beam.get_wavelength_range()
        sel = reflections["wavelength_cal"] >= wavelength_range[0]
        reflections = reflections.select(sel)
        sel = reflections["wavelength_cal"] <= wavelength_range[1]
        reflections = reflections.select(sel)

        return reflections

    def for_ub(self, ub):

        reflection_table = self.predictor.for_ub(ub)
        reflection_table = self.post_prediction(reflection_table)

        image_range = self.experiment.sequence.get_image_range()
        frames = list(range(image_range[0], image_range[1]))
        x, y, z = reflection_table["xyzcal.px"].parts()
        xyz = cctbx.array_family.flex.vec3_double(len(reflection_table))

        for i in range(len(reflection_table)):
            wavelength = reflection_table["wavelength_cal"][i]
            L1 = reflection_table["L1"][i]
            tof = self.experiment.beam.get_tof_from_wavelength(wavelength, L1)
            frame = self.experiment.sequence.get_frame_from_tof(tof)
            xyz[i] = (x[i], y[i], frame)
        x, y, z = xyz.parts()
        reflection_table["xyzcal.px"] = xyz
        sel = z > frames[0]
        return reflection_table.select(sel)

    def for_reflection_table(self, reflections, UB):
        return self.predictor.for_reflection_table(reflections, UB)


def StillsReflectionPredictor(experiment, dmin=None, spherical_relp=False, **kwargs):
    """
    A factory function for the reflection predictor.

    :param experiment: The experiment to predict for
    :param dmin: The maximum resolution to predict to
    :param spherical_relp: Whether to use the spherical relp prediction model
    :return: The spot predictor
    """

    # FIXME Selection of reflection predictor type is ugly. What is a better
    # way here? Should it be based entirely on the existence of certain types
    # of profile model within the experiment?

    # Get dmin if it is not set
    if dmin is None:
        dmin = experiment.detector.get_max_resolution(experiment.beam.get_s0())

    if spherical_relp:
        return SphericalRelpStillsReflectionPredictor(
            experiment.beam,
            experiment.detector,
            experiment.crystal.get_A(),
            experiment.crystal.get_unit_cell(),
            experiment.crystal.get_space_group().type(),
            dmin,
        )

    # Create the reflection predictor
    try:
        if (
            experiment.crystal.get_half_mosaicity_deg() is not None
            and experiment.crystal.get_domain_size_ang() is not None
        ):
            return NaveStillsReflectionPredictor(
                experiment.beam,
                experiment.detector,
                experiment.crystal.get_A(),
                experiment.crystal.get_unit_cell(),
                experiment.crystal.get_space_group().type(),
                dmin,
                experiment.crystal.get_half_mosaicity_deg(),
                experiment.crystal.get_domain_size_ang(),
            )
    except AttributeError:
        pass

    return StillsDeltaPsiReflectionPredictor(
        experiment.beam,
        experiment.detector,
        experiment.crystal.get_A(),
        experiment.crystal.get_unit_cell(),
        experiment.crystal.get_space_group().type(),
        dmin,
    )
