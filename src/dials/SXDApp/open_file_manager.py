from __future__ import annotations

import base64
from io import FileIO
from os import mkdir, rmdir
from os.path import isdir, join, splitext

from active_file import ActiveFile
from algorithm_types import AlgorithmType


class OpenFileManager:

    """
    Manages a running list of ActiveFiles and which is currently selected
    """

    def __init__(self, working_directory: str = "tmp/") -> None:

        self.working_directory = working_directory
        self.active_files = {}
        self.selected_file = None

        if not isdir(working_directory):
            mkdir(working_directory)

    def add_active_file(self, filename: str, content: FileIO) -> None:
        name, ext = splitext(filename)
        file_dir = self.working_directory + name
        self.create_local_file(file_dir, filename, content)
        self.active_files[filename] = ActiveFile(file_dir, filename)
        self.selected_file = self.active_files[filename]

    def remove_active_file(self, filename: str) -> None:

        file_dir = self.active_files[filename]
        if self.selected_file == self.active_files[filename]:
            new_selected_idx = list(self.active_files.keys()).index(filename) - 1
            if new_selected_idx < 0:
                self.selected_file = None
            else:
                key = list(self.active_files.keys())[new_selected_idx]
                self.selected_file = self.active_files[key]
        del self.active_files[filename]
        rmdir(file_dir)

    def create_local_file(self, file_dir: str, filename: str, content: FileIO):

        if not isdir(file_dir):
            mkdir(file_dir)
        file_path = join(file_dir, filename)
        content_type, content_string = content.split(",")
        decoded = base64.b64decode(content_string)
        with open(file_path, "wb") as g:
            g.write(decoded)

    def run(self, algorithm_type: AlgorithmType) -> str:
        return self.selected_file.run(algorithm_type)

    def get_selected_filename(self):
        return self.selected_file.filename

    def get_experiment_params(self):
        return self.selected_file.get_experiment_params()

    def get_selected_file_image_range(self):
        return self.selected_file.get_image_range()

    def update_selected_file(self, idx: int) -> None:
        name = list(self.active_files.keys())[idx]
        self.selected_file = self.active_files[name]

    def update_selected_file_arg(
        self, algorithm_type: AlgorithmType, param_name: str, param_value: str
    ) -> None:
        return self.selected_file.update_arg(algorithm_type, param_name, param_value)

    def get_logs(self):
        if self.selected_file is not None:
            return self.selected_file.get_logs()
        return ["" for i in AlgorithmType][:2]
