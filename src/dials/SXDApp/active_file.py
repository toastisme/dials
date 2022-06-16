from __future__ import annotations

import json
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
    output_experiment_file: str
    output_reflections_file: str


class ActiveFile:

    """
    Manages all data relating to a file imported in the via the GUI
    """

    def __init__(self, file_dir: str, filename: str) -> None:
        self.file_dir = file_dir
        self.filename = filename
        self.file_path = join(file_dir, filename)
        self.current_expt_file = None
        self.current_refl_file = None
        self.setup_algorithms(filename)

    def setup_algorithms(self, filename):
        self.algorithms = {
            AlgorithmType.dials_import: DIALSAlgorithm(
                name=AlgorithmType.dials_import,
                command="dials.import",
                args={},
                log="",
                required_files=[filename],
                output_experiment_file="imported.expt",
                output_reflections_file=None,
            ),
            AlgorithmType.dials_find_spots: DIALSAlgorithm(
                name=AlgorithmType.dials_find_spots,
                command="dials.find_spots",
                args={},
                log="",
                required_files=["imported.expt"],
                output_experiment_file="imported.expt",
                output_reflections_file="strong.refl",
            ),
            AlgorithmType.dials_index: DIALSAlgorithm(
                name=AlgorithmType.dials_index,
                command="dials.index",
                args={},
                log="",
                required_files=["imported.expt", "strong.refl"],
                output_experiment_file="indexed.expt",
                output_reflections_file="indexed.refl",
            ),
            AlgorithmType.dials_refine: DIALSAlgorithm(
                name=AlgorithmType.dials_refine,
                command="dials.refine",
                args={},
                log="",
                required_files=["indexed.expt", "indexed.refl"],
                output_experiment_file="refined.expt",
                output_reflections_file="refined.refl",
            ),
            AlgorithmType.dials_integrate: DIALSAlgorithm(
                name=AlgorithmType.dials_integrate,
                command="dials.integrate",
                args={},
                log="",
                required_files=["refined.expt", "refined.refl"],
                output_experiment_file="integrated.expt",
                output_reflections_file="integrated.refl",
            ),
            AlgorithmType.dials_scale: DIALSAlgorithm(
                name=AlgorithmType.dials_scale,
                command="dials.scale",
                args={},
                log="",
                required_files=["integrated.expt", "integrated.refl"],
                output_experiment_file="scaled.expt",
                output_reflections_file="scaled.refl",
            ),
            AlgorithmType.dials_export: DIALSAlgorithm(
                name=AlgorithmType.dials_scale,
                command="dials.export",
                args={},
                log="",
                required_files=["scaled.expt", "scaled.refl"],
                output_experiment_file="exported.expt",
                output_reflections_file="exported.refl",
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

    def get_beam_params(self, expt_file):
        beam = expt_file["beam"][0]
        params = {}
        params["Sample to Source Direction"] = str(tuple(beam["direction"]))
        params["Sample to Moderator Distance (mm)"] = str(
            beam["sample_to_moderator_distance"]
        )
        return [params]

    def get_detector_params(self, expt_file):
        panels = expt_file["detector"][0]["panels"]
        params = []
        for i in range(len(panels)):
            panels[i]["fast_axis"] = [round(j, 3) for j in panels[i]["fast_axis"]]
            panels[i]["slow_axis"] = [round(j, 3) for j in panels[i]["slow_axis"]]
            panels[i]["origin"] = [round(j, 3) for j in panels[i]["origin"]]
            params.append(
                {
                    "Name": panels[i]["name"],
                    "Origin (mm)": str(tuple(panels[i]["origin"])),
                    "Fast Axis": str(tuple(panels[i]["fast_axis"])),
                    "Slow Axis": str(tuple(panels[i]["slow_axis"])),
                    "Pixels": str(tuple(panels[i]["image_size"])),
                    "Pixel Size (mm)": str(tuple(panels[i]["pixel_size"])),
                }
            )
        return params

    def get_sequence_params(self, expt_file):
        sequence = expt_file["sequence"][0]
        params = {}
        params["Image Range"] = str(tuple(sequence["image_range"]))
        min_tof = round(sequence["tof_in_seconds"][0], 3)
        max_tof = round(sequence["tof_in_seconds"][-1], 3)
        min_wavelength = round(sequence["wavelengths"][0], 3)
        max_wavelength = round(sequence["wavelengths"][-1], 3)
        params["ToF Range (s)"] = str((min_tof, max_tof))
        params["Wavelength Range (A)"] = str((min_wavelength, max_wavelength))
        return [params]

    def get_goniometer_params(self, expt_file):
        return [{"Orientation (deg)": "0"}]

    def get_crystal_params(self, expt_file):
        return [
            {
                "a": "-",
                "b": "-",
                "c": "-",
                "alpha": "-",
                "beta": "-",
                "gamma": "-",
                "Orientation": "-",
                "Space Group": "-",
            }
        ]

    def get_experiment_params(self):
        with open(self.current_expt_file, "r") as g:
            expt_file = json.load(g)
            experiment_params = []
            experiment_params.append(self.get_beam_params(expt_file))
            experiment_params.append(self.get_detector_params(expt_file))
            experiment_params.append(self.get_sequence_params(expt_file))
            experiment_params.append(self.get_goniometer_params(expt_file))
            experiment_params.append(self.get_crystal_params(expt_file))
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
            dials_command.append(f"{arg}={algorithm.args[arg]}")

        result = procrunner.run((dials_command))
        log = get_log_text(result)
        self.algorithms[algorithm_type].log = log
        expt_file = self.algorithms[algorithm_type].output_experiment_file
        self.current_expt_file = join(self.file_dir, expt_file)
        refl_file = self.algorithms[algorithm_type].output_reflections_file
        if refl_file is not None:
            self.current_refl_file = join(self.file_dir, refl_file)
        else:
            self.current_refl_file = None

        os.chdir(cwd)
        print(f"Ran command {dials_command}")
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
