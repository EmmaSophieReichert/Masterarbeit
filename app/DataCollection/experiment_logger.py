import csv
from typing import Optional
import time
import os

class ExperimentLogger:
    def __init__(
        self,
        file_path_performance: str,
        file_path_accuracy: str,
        file_path_real_head: Optional[str] = None,
        participant_id: Optional[int] = 0,
        voxel_size: Optional[float] = None,
        head_position: Optional[str] = None
    ):
        self.file_path_performance = self._get_new_path(file_path_performance)
        self.file_path_accuracy = self._get_new_path(file_path_accuracy)
        if file_path_real_head:
            self.file_path_real_head = self._get_new_path(file_path_real_head)
        else:
            self.file_path_real_head = None

        self.voxel_size = voxel_size
        self.head_position = head_position
        self.participant_id = participant_id

        self.performance_fieldnames = [
            "participant_id",
            "voxel_size",
            "head_position",
            "action_description",
            "time"
        ]

        self.accuracy_fieldnames = [
            "participant_id",
            "voxel_size",
            "head_position",
            "description",
            "fitness",
            "rmse"
        ]

        self.real_head_fieldnames = [
            "participant_id",
            "voxel_size",
            "head_position",
            "landmark_name",
            "color_u",
            "color_v",
            "color_x",
            "color_y",
            "color_z",
            "comp_u",
            "comp_v",
            "comp_x",
            "comp_y",
            "comp_z",
            "distance_3d_mm",
            "distance_2d_mm"
        ]

        self._performance_file = None
        self._performance_writer = None

        self._accuracy_file = None
        self._accuracy_writer = None

        self._real_head_file = None
        self._real_head_writer = None


    def _get_new_path(self, path):
        if os.path.exists(path):
            base, ext = os.path.splitext(path)

            new_path = f"{base}_new{ext}"
            counter = 1

            while os.path.exists(new_path):
                new_path = f"{base}_new_{counter}{ext}"
                counter += 1

            return new_path

        return path


    def open(self):
        # Performance CSV
        self._performance_file = open(
            self.file_path_performance,
            mode="w",
            newline="",
            encoding="utf-8"
        )

        self._performance_writer = csv.DictWriter(
            self._performance_file,
            fieldnames=self.performance_fieldnames
        )

        self._performance_writer.writeheader()

        # Accuracy CSV
        self._accuracy_file = open(
            self.file_path_accuracy,
            mode="w",
            newline="",
            encoding="utf-8"
        )

        self._accuracy_writer = csv.DictWriter(
            self._accuracy_file,
            fieldnames=self.accuracy_fieldnames
        )

        self._accuracy_writer.writeheader()

        # Real Head CSV
        if self.file_path_real_head:
            self._real_head_file = open(
                self.file_path_real_head,
                mode="w",
                newline="",
                encoding="utf-8"
            )

            self._real_head_writer = csv.DictWriter(
                self._real_head_file,
                fieldnames=self.real_head_fieldnames
            )

            self._real_head_writer.writeheader()


    def close(self):
        if self._performance_file and not self._performance_file.closed:
            self._performance_file.close()

        if self._accuracy_file and not self._accuracy_file.closed:
            self._accuracy_file.close()

        if self.file_path_real_head:
            if self._real_head_file and not self._real_head_file.closed:
                self._real_head_file.close()


    def set_metadata(
        self,
        voxel_size: Optional[float] = None,
        head_position: Optional[str] = None,
        participant_id: Optional[int] = 0
    ):
        if voxel_size is not None:
            self.voxel_size = voxel_size
        if head_position is not None:
            self.head_position = head_position
        if participant_id is not 0:
            self.participant_id = participant_id


    def log_action(self, action_description: str):
        if self._performance_writer is None or self._performance_file.closed:
            raise RuntimeError("Logger ist nicht geöffnet.")

        self._performance_writer.writerow({
            "participant_id": self.participant_id,
            "voxel_size": self.voxel_size,
            "head_position": self.head_position,
            "action_description": action_description,
            "time": time.perf_counter()
        })

        self._performance_file.flush()


    def log_result(self, description: str, fitness: float, rmse: float):
        if self._accuracy_writer is None or self._accuracy_file.closed:
            raise RuntimeError("Logger ist nicht geöffnet.")

        self._accuracy_writer.writerow({
            "participant_id": self.participant_id,
            "voxel_size": self.voxel_size,
            "head_position": self.head_position,
            "description": description,
            "fitness": fitness,
            "rmse": rmse
        })

        self._accuracy_file.flush()

    def log_real_head_data(
        self, 
        landmark_name: str,
        color_u: float,
        color_v: float,
        color_x: float,
        color_y: float,
        color_z: float,
        comp_u: float,
        comp_v: float,
        comp_x: float,
        comp_y: float,
        comp_z: float,
        distance_3d_mm: float,
        distance_2d_mm: float,
    ):
        if self._real_head_writer is None or self._accuracy_file.closed:
            raise RuntimeError("Logger ist nicht geöffnet.")

        self._real_head_writer.writerow({
            "participant_id": self.participant_id,
            "voxel_size": self.voxel_size,
            "head_position": self.head_position,
            "landmark_name": landmark_name,
            "color_u": color_u,
            "color_v": color_v,
            "color_x": color_x,
            "color_y": color_y,
            "color_z": color_z,
            "comp_u": comp_u,
            "comp_v": comp_v,
            "comp_x": comp_x,
            "comp_y": comp_y,
            "comp_z": comp_z,
            "distance_3d_mm": distance_3d_mm,
            "distance_2d_mm": distance_2d_mm,
        })

        self._real_head_file.flush()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()