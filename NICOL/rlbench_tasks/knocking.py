from typing import List, Tuple, Union
import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import DetectedCondition, Condition
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from rlbench.backend.task import Task
from pyrep.objects.shape import Shape
from playsound import playsound
import os


DESTINATION = "waypoint1"
FULL_COLORS = ("red", "green", "blue", "yellow", "purple", "orange")
COLOR_MAPS = {}
for color_name, color_rgb in colors:
    COLOR_MAPS[color_name] = color_rgb
SOUND_PATH = "/informatik3/wtm/home/zhao/Codes/material_sounds/"
# e.g. + "/train/ceramic/Clip_1.wav"

MATERIALS = [
    (
        "metal",
        ("yellow", "orange"),
        ("hard and cold", "rigid, cold, and smooth"),
        ("heavy", "300g"),
    ),
    (
        "glass",
        ("red", "blue", "yellow", "green"),
        ("hard", "hard and smooth", "cold and smooth"),
        ("a little bit heavy", "150g"),
    ),
    (
        "ceramic",
        FULL_COLORS,
        ("hard", "tough"),
        ("100g", "average weight", "not too light nor not too heavy"),
    ),
    ("plastic", FULL_COLORS, ("hard", "soft"), ("light", "30g")),
    (
        "fibre",
        FULL_COLORS,
        ("soft", "flexible"),
        ("lightweight", "underweight", "10g"),
    ),
]


shapes = ("block", "cylinder")


class NotInCondition(Condition):
    def __init__(self, obj: Object, boundary: Shape):
        self.obj = obj
        self.boundary = boundary

    def condition_met(self):
        # Only consider x, y (not z)
        N_only_consider = 2
        pos = self.obj.get_position()[:N_only_consider]
        bbox_pos = self.boundary.get_position()[:N_only_consider]
        bbox = self.boundary.get_bounding_box()[: 2 * N_only_consider]
        bbox_min = bbox_pos + bbox[::2]
        bbox_max = bbox_pos + bbox[1::2]
        met = np.any(pos < bbox_min) or np.any(pos > bbox_max)
        return met, False


class ReverseDetectedCondition(DetectedCondition):
    def condition_met(self):
        met, _ = super().condition_met()
        return not met, False


class ExtendedShape(Shape):
    def __init__(self, name_or_handle: Union[str, int]):
        super().__init__(name_or_handle)
        self._color_name = None
        self._material_name = None
        self._material_sound = None

    def set_color_name(self, color_name):
        self._color_name = color_name

    def get_color_name(self):
        return self._color_name

    def set_material_name(self, material_name):
        self._material_name = material_name

    def get_material_name(self):
        return self._material_name

    def set_material_sound(self, material_sound):
        self._material_sound = material_sound

    def get_material_sound(self):
        return self._material_sound

    def set_touch_data(self, touch_data):
        self._touch_data = touch_data

    def get_touch_data(self):
        return self._touch_data

    def set_weight(self, weight):
        self._weight = weight

    def get_weight(self):
        return self._weight


class Knocking(Task):
    def init_task(self) -> None:
        self._mode = "train"
        self._random = True
        self._records = []

        self.r_palm = Shape("r_palm")
        self.r_Tbumper = Shape("r_Tbumper_respondable")
        self.r_Ibumper = Shape("r_Ibumper_respondable")
        self.r_Mbumper = Shape("r_Mbumper_respondable")
        self.r_Rbumper = Shape("r_Rbumper_respondable")
        self.r_Lbumper = Shape("r_Lbumper_respondable")
        self.Container = Shape("Container")

        self.target = ExtendedShape("Target")
        self.distractors = [ExtendedShape(f"Distractor{i}") for i in range(2)]
        self.objs = [*self.distractors, self.target]
        self.register_graspable_objects(self.objs)
        self.frame = 0
        self.last_play_frame = 0

        # to normalize action
        # self.boundary = BoundaryObject(Shape('Boundary'))

        success_sensor = ProximitySensor("success")
        self.register_success_conditions(
            [DetectedCondition(self.target, success_sensor)]
        )
        self.boundary = Shape("Boundary")
        self.register_fail_conditions(
            [
                NotInCondition(self.target, self.boundary),
                ReverseDetectedCondition(self.target, success_sensor),
            ]
        )

    def init_episode(self, index: int) -> List[str]:
        def set_material_color_touch(
            obj: ExtendedShape,
            sample_color_name,
            sample_touch_data,
            sample_weight,
            material_name,
            log=True,
        ):
            sample_color_rgb = COLOR_MAPS[sample_color_name]
            material_sound_dir = SOUND_PATH + f"{material_name}/{self._mode}"
            if os.path.exists(material_sound_dir):
                material_sounds = [
                    f for f in os.listdir(material_sound_dir) if f.endswith(".wav")
                ]

                material_sound = (
                    material_sound_dir + "/" + np.random.choice(material_sounds)
                )
            else:
                material_sound = "None"
            obj.set_color(sample_color_rgb)
            obj.set_color_name(sample_color_name)
            obj.set_material_name(material_name)
            obj.set_material_sound(material_sound)
            obj.set_touch_data(sample_touch_data)
            obj.set_weight(sample_weight)
            log and print(
                f"{obj.get_name()}: {sample_color_name}, {sample_touch_data}, {sample_weight}, {material_name} ({material_sound})."
            )
            return

        def sample_material_color(
            obj: ExtendedShape, material_index, sample_color_index=None, log=True
        ):
            (
                material_name,
                material_colors,
                material_touchs,
                material_weights,
            ) = MATERIALS[material_index]
            if sample_color_index is None:
                sample_color_index = np.random.choice(len(material_colors))
                sample_touch_index = np.random.choice(len(material_touchs))
                sample_weight_index = np.random.choice(len(material_weights))
            sample_color_name = material_colors[sample_color_index]
            sample_touch_data = material_touchs[sample_touch_index]
            sample_weight = material_weights[sample_weight_index]
            set_material_color_touch(
                obj,
                sample_color_name,
                sample_touch_data,
                sample_weight,
                material_name,
                log,
            )

        if self.random:
            sample_material_color(self.target, index)
            # It is OK to use the same color but different material
            distractor_material_indices = np.random.choice(
                list(range(index)) + list(range(index + 1, len(MATERIALS))),
                size=len(self.distractors),
                replace=True,
            )
            for distractor, distractor_material_index in zip(
                self.distractors, distractor_material_indices
            ):
                sample_material_color(distractor, distractor_material_index)

            b = SpawnBoundary([self.boundary])
            for ob in [self.target, *self.distractors]:
                b.sample(
                    ob,
                    min_distance=0.15,
                    min_rotation=(0, 0, -3.14 / 4),
                    max_rotation=(0, 0, 3.14 / 4),
                )
        else:
            sample_material_color(self.target, material_index=0, sample_color_index=0)
            sample_material_color(
                self.distractors[0], material_index=1, sample_color_index=0
            )
            sample_material_color(
                self.distractors[1], material_index=0, sample_color_index=2
            )
        # shapes[0] is =block=, only one shape for now.
        self.frame = 0
        self.last_play_frame = 0
        return [f"pick up the {self.target.get_material_name()} {shapes[0]}"]

    def is_static_workspace(self) -> bool:
        return True

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return len(MATERIALS)

    def step(self) -> None:
        self.last_play_frame
        self.frame += 1
        # Called during each sim step. Remove this if not using.
        for obj in self.objs:
            if obj.check_collision(self.r_palm) or obj.check_collision(self.Container):
                # TODO:
                # - [X] Play sound in background
                # - [ ] Only play sound when intentionally knock on objects
                # For testing
                obj_sound = obj.get_material_sound()
                # Avoid play repeated sound too soon.
                if (
                    obj_sound is not None
                    and obj_sound != "None"
                    and self.frame - self.last_play_frame > 20
                ):
                    print(f"=====>>>> Playing {obj_sound}")
                    playsound(obj_sound, block=False)
                    self.last_play_frame = self.frame
                self.record(obj_sound)

            if (
                obj.check_collision(self.r_Tbumper)
                or obj.check_collision(self.r_Ibumper)
                or obj.check_collision(self.r_Mbumper)
                or obj.check_collision(self.r_Rbumper)
                or obj.check_collision(self.r_Lbumper)
            ):
                obj_touch = obj.get_touch_data()
                self.touch(obj_touch)

                obj_weight = obj.get_weight()
                self.weigh(obj_weight)
                # print(f"=====>>>> Feeling {obj_touch}")
                # print(f"=====>>>> Weighing {obj_weight}")

    def set_mode(self, mode):
        self._mode = mode

    def set_random(self, random):
        self._random = random

    @property
    def random(self):
        return self._random

    @property
    def mode(self):
        return self._mode

    @property
    def records(self):
        return self._records

    def record(self, sound):
        self._records.append(sound)

    def clear_records(self):
        self._records = []

    @property
    def touchs(self):
        return self._touchs

    def touch(self, touch_data):
        self._touchs.append(touch_data)

    def clear_touchs(self):
        self._touchs = []

    @property
    def weights(self):
        return self._weights

    def weigh(self, weight):
        self._weights.append(weight)

    def clear_weights(self):
        self._weights = []

    def cleanup(self) -> None:
        # Called during at the end of each episode. Remove this if not using.
        self.clear_records()
        self.clear_touchs()
        self.clear_weights()

    def base_rotation_bounds(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Defines how much the task base can rotate during episode placement.

        Default is set such that it can rotate any amount on the z axis.

        :return: A tuple containing the min and max (x, y, z) rotation bounds
            (in radians).
        """
        return (0.0, 0.0, 0), (0.0, 0.0, 0)

    def load(self) -> Object:
        if Object.exists(self.name):
            return Dummy(self.name)
        ttm_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            './task_ttms/%s.ttm' % self.name)
        if not os.path.isfile(ttm_file):
            raise FileNotFoundError(
                'The following is not a valid task .ttm file: %s' % ttm_file)
        self._base_object = self.pyrep.import_model(ttm_file)
        return self._base_object