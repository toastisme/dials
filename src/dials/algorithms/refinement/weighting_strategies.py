"""Contains classes used to provide weighting schemes as strategies for
ReflectionManagers."""

from __future__ import annotations

from dials.algorithms.refinement import DialsRefineConfigError
from dials.array_family import flex


class StatisticalWeightingStrategy:
    """Defines a single method that provides a ReflectionManager with a strategy
    for calculating weights for refinement"""

    @staticmethod
    def calculate_weights(reflections):
        """set 'statistical weights', that is w(x) = 1/var(x)"""

        weights = (reflections["xyzobs.mm.variance"]).deep_copy()
        parts = weights.parts()
        for w in parts:
            sel = w > 0.0
            w.set_selected(sel, 1.0 / w.select(sel))
        reflections["xyzobs.mm.weights"] = flex.vec3_double(*parts)
        indexed = reflections.select(reflections.get_flags(reflections.flags.indexed))
        if any(indexed["xyzobs.mm.weights"].norms() == 0.0):
            raise DialsRefineConfigError(
                "Cannot set statistical weights as some indexed reflections have observed variances equal to zero"
            )

        return reflections


class StillsWeightingStrategy(StatisticalWeightingStrategy):
    """Defines a single method that provides a ReflectionManager with a strategy
    for calculating weights for refinement. This version uses statistical weights
    for X and Y and a fixed constant for the delta Psi part, defaulting to 1000000"""

    def __init__(self, delpsi_constant=1000000):
        self._delpsi_constant = delpsi_constant

    def calculate_weights(self, reflections):
        """Include weights for DeltaPsi"""

        # call parent class method to set X and Y weights
        reflections = super().calculate_weights(reflections)

        reflections["delpsical.weights"] = flex.double(
            len(reflections), self._delpsi_constant
        )

        return reflections


class ExternalDelPsiWeightingStrategy(StatisticalWeightingStrategy):
    """Defines a single method that provides a ReflectionManager with a strategy
    for calculating weights for stills refinement. This version uses statistical
    weights for X and Y and assume that the Delta Psi part is already provided in
    the reflection table"""

    def calculate_weights(self, reflections):
        """Statistical weights for X, Y. Weights for DeltaPsi must be already
        provided in the reflection table"""

        # call parent class method to set X and Y weights
        reflections = super().calculate_weights(reflections)

        if "delpsical.weights" not in reflections:

            raise DialsRefineConfigError(
                'The key "delpsical.weights" is expected within the input reflections'
            )

        return reflections


class ConstantWeightingStrategy:
    def __init__(self, wx, wy, wz, stills=False):
        self._wx = wx
        self._wy = wy
        self._wz = wz
        self._stills = stills

    def calculate_weights(self, reflections):
        """Set weights to constant terms. If stills, the z weights are
        the 'delpsical.weights' attribute of the reflection table. Otherwise, use
        the usual 'xyzobs.mm.weights'"""

        wx = flex.double(len(reflections), self._wx)
        wy = flex.double(len(reflections), self._wy)
        wz = flex.double(len(reflections), self._wz)
        if self._stills:
            null = flex.double(len(reflections), 0)
            reflections["xyzobs.mm.weights"] = flex.vec3_double(wx, wy, null)
            reflections["delpsical.weights"] = wz
        else:
            reflections["xyzobs.mm.weights"] = flex.vec3_double(wx, wy, wz)

        return reflections
