from __future__ import annotations

import os
from dataclasses import dataclass
from os.path import isfile, join
from typing import Dict, List

import procrunner
from algorithm_types import AlgorithmType
from dash import html

from dxtbx.model import Experiment
from dxtbx.serialize import load


@dataclass
class DIALSAlgorithm:

    """
    Holds basic information of the requirments and history of an algorithm
    """

    name: AlgorithmType
    command: str
    args: Dict[str, str]
    log: str
    required_files: List[str]


class ActiveFile:

    """
    Manages all data relating to a file imported in the via the GUI
    """

    def __init__(self, file_dir: str, filename: str) -> None:
        self.file_dir = file_dir
        self.filename = filename
        self.file_path = join(file_dir, filename)
        self.setup_algorithms(filename)

    def setup_algorithms(self, filename):
        self.algorithms = {
            AlgorithmType.dials_import: DIALSAlgorithm(
                name=AlgorithmType.dials_import,
                command="dials.import",
                args={},
                log="",
                required_files=[filename],
            ),
            AlgorithmType.dials_find_spots: DIALSAlgorithm(
                name=AlgorithmType.dials_find_spots,
                command="dials.find_spots",
                args={},
                log="",
                required_files=["imported.expt"],
            ),
            AlgorithmType.dials_index: DIALSAlgorithm(
                name=AlgorithmType.dials_index,
                command="dials.index",
                args={},
                log="",
                required_files=["imported.expt", "strong.refl"],
            ),
            AlgorithmType.dials_refine: DIALSAlgorithm(
                name=AlgorithmType.dials_refine,
                command="dials.refine",
                args={},
                log="",
                required_files=["indexed.expt", "indexed.refl"],
            ),
            AlgorithmType.dials_integrate: DIALSAlgorithm(
                name=AlgorithmType.dials_integrate,
                command="dials.integrate",
                args={},
                log="",
                required_files=["refined.expt", "refined.refl"],
            ),
            AlgorithmType.dials_scale: DIALSAlgorithm(
                name=AlgorithmType.dials_scale,
                command="dials.scale",
                args={},
                log="",
                required_files=["integrated.expt", "integrated.refl"],
            ),
            AlgorithmType.dials_export: DIALSAlgorithm(
                name=AlgorithmType.dials_scale,
                command="dials.export",
                args={},
                log="",
                required_files=["scaled.expt", "scaled.refl"],
            ),
        }

    def _get_experiment(self) -> Experiment:
        file_path = join(self.file_dir, "imported.expt")
        experiment = load.experiment_list(file_path)[0]
        assert experiment is not None
        return experiment

    def get_image_range(self):
        try:
            image_range = self._get_experiment().imageset.get_array_range()
            return (image_range[0] + 1, image_range[1])
        except AttributeError:
            return (1, len(self._get_experiment().imageset))

    def get_experiment_params(self):
        experiment = self._get_experiment()
        experiment_params = {}
        beam_params = {}
        beam_str = str(experiment.beam).split("\n")
        for i in beam_str[:-1]:
            name, val = i.split(":")
            beam_params[name.strip()] = val.strip()
        experiment_params["beam"] = beam_params
        return experiment_params

    def can_run(self, algorithm_type: AlgorithmType) -> bool:
        for i in self.algorithms[algorithm_type].required_files:
            if not isfile(join(self.file_dir, i)):
                return False
        return True

    def run(self, algorithm_type: AlgorithmType):

        """
        procrunner wrapper for dials commands.
        Converts log to html and returns it
        """

        def get_log_text(procrunner_result):
            stdout = procrunner_result.stdout.decode().split("\n")
            text = []
            for i in stdout:
                text.append(i)
                text.append(html.Br())
            return text

        assert self.can_run(algorithm_type)

        cwd = os.getcwd()
        os.chdir(self.file_dir)
        algorithm = self.algorithms[algorithm_type]
        dials_command = [algorithm.command]

        for i in algorithm.required_files:
            dials_command.append(i)

        for arg in algorithm.args:
            print(f" TESTESTESTESTESTESTEST {arg}={algorithm.args[arg]}")
            dials_command.append(f"{arg}={algorithm.args[arg]}")

        result = procrunner.run((dials_command))
        log = get_log_text(result)
        self.algorithms[algorithm_type].log = log
        os.chdir(cwd)
        return log

    def get_available_algorithms(self):

        """
        Dictionary of algorithms that can be run in the current state
        """

        available_algorithms = {}
        for i in self.algorithms.keys():
            available_algorithms[i] = self.can_run(i)
        return available_algorithms

    def get_logs(self):
        return [self.algorithms[i].log for i in AlgorithmType][:2]

    def update_arg(
        self, algorithm_type: AlgorithmType, param_name: str, param_value: str
    ) -> None:
        self.algorithms[algorithm_type].args[param_name] = param_value
