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

from scipy import interpolate

import cctbx.array_family.flex


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
            experiment.crystal.get_A(),
            experiment.crystal.get_unit_cell(),
            experiment.crystal.get_space_group().type(),
            dmin,
        )

    def for_ub(self, ub):
        reflection_table = self.predictor.for_ub(ub)
        wavelengths = list(self.experiment.sequence.get_wavelengths())
        image_range = self.experiment.sequence.get_image_range()
        frames = list(range(image_range[0], image_range[1]))
        spline_coefficients = interpolate.splrep(wavelengths, frames)
        x, y, z = reflection_table["xyzcal.px"].parts()
        xyz = cctbx.array_family.flex.vec3_double(len(reflection_table))
        for i in range(len(reflection_table)):
            wavelength = reflection_table["wavelength_cal"][i]
            frame = min(
                max(
                    frames[0], float(interpolate.splev(wavelength, spline_coefficients))
                ),
                frames[-1],
            )
            xyz[i] = (x[i], y[i], frame)

        reflection_table["xyzcal.px"] = xyz
        return reflection_table

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
