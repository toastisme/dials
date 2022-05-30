from __future__ import annotations

from dataclasses import dataclass
from os.path import isfile, join
from typing import List

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
    log: str
    required_files: List[str]


class ActiveFile:

    """
    Manages all data relating to a file imported in the via the GUI
    """

    algorithms = {
        AlgorithmType.dials_import: DIALSAlgorithm(
            name=AlgorithmType.dials_import,
            command="dials.import",
            log="",
            required_files=[],
        ),
        AlgorithmType.dials_find_spots: DIALSAlgorithm(
            name=AlgorithmType.dials_find_spots,
            command="dials.find_spots",
            log="",
            required_files=["imported.expt"],
        ),
        AlgorithmType.dials_index: DIALSAlgorithm(
            name=AlgorithmType.dials_index,
            command="dials.index",
            log="",
            required_files=["imported.expt", "strong.refl"],
        ),
        AlgorithmType.dials_refine: DIALSAlgorithm(
            name=AlgorithmType.dials_refine,
            command="dials.refine",
            log="",
            required_files=["indexed.expt", "indexed.refl"],
        ),
        AlgorithmType.dials_integrate: DIALSAlgorithm(
            name=AlgorithmType.dials_integrate,
            command="dials.integrate",
            log="",
            required_files=["refined.expt", "refined.refl"],
        ),
    }

    def __init__(self, file_dir: str, filename: str) -> None:
        self.file_dir = file_dir
        self.filename = filename
        self.file_path = join(file_dir, filename)
        self.algorithms[AlgorithmType.dials_import].required_files = [self.filename]

    def _get_experiment(self) -> Experiment:
        experiment = load.experiment_list(self.file_path)[0]
        assert experiment is not None
        return experiment

    def get_experiment_params(self):
        experiment = self._get_experiment()
        experiment_params = {}
        beam_params = {}
        beam_str = str(experiment.beam).split("\n")
        for i in beam_str:
            name, val = i.split(":")
            beam_params[name.lstrip()] = val.lstrip()
        experiment_params["beam"] = beam_params
        return experiment_params

    def can_run(self, algorithm_type: AlgorithmType) -> bool:
        for i in self.algorithms[algorithm_type].required_files:
            print(i)
            if not isfile(join(self.file_dir, i)):
                return False
        return True

    def run(self, algorithm_type: AlgorithmType, *args):

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
        algorithm = self.algorithms[algorithm_type]
        dials_command = [algorithm.command]

        for i in algorithm.required_files:
            dials_command.append(join(self.file_dir, i))

        for arg in args:
            dials_command.append(arg)

        result = procrunner.run((dials_command))
        log = get_log_text(result)
        self.algorithms[algorithm_type].log = log
        return log

    def get_available_algorithms(self):

        """
        Dictionary of algorithms that can be run in the current state
        """

        available_algorithms = {}
        for i in self.algorithms.keys():
            available_algorithms[i] = self.can_run(i)
        return available_algorithms
