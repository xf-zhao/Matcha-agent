from typing import Dict, Tuple
from gym import spaces
from rlbench.gym.rlbench_env import RLBenchEnv
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.observation_config import ObservationConfig, CameraConfig
import numpy as np
from loguru import logger
from NICOL.nicol_base import NicolEnvironment
from NICOL.nicol_arm import PalmDiscrete
from NICOL.rlbench_tasks.knocking import Knocking


# Right on top of the wooden box
DESTINATION = "waypoint1"

SKILLS = {"knock_on": 1, "touch": 2, "pick_up": 3, "weigh": 4}
PALM_POSE = {"keep": 0, "make_fist": 1, "point": 2, "grasp": 3, "release": 4}
ARM_POSE = {
    "fist_up": np.array([0.9, 0, 0, 0.8509035, 0.525322, PALM_POSE["make_fist"]]),
    "fist_down": np.array([0.81, 0, 0, 0.8509035, 0.525322, PALM_POSE["make_fist"]]),
    "grasp": np.array(
        [
            0.83443451,
            0.76801813,
            -0.61692601,
            -0.13173638,
            0.11043578,
            PALM_POSE["release"],
        ]
    ),
    "hang": np.array(
        [
            0.9,
            0.76801813,
            -0.61692601,
            -0.13173638,
            0.11043578,
            PALM_POSE["release"],
        ]
    ),
}


class ActionScaleWrapper:
    """Scale only (x, y) according to the workspace boundary."""

    def __init__(self, env, ignore_z=True, mini=-1.0, maxi=1.0):
        self._env = env
        boundary = Shape("Plane")
        bbox_pos = boundary.get_position()
        bbox = boundary.get_bounding_box()
        orig_min = bbox_pos + bbox[::2]
        orig_max = bbox_pos + bbox[1::2]

        scale = (orig_max - orig_min) / (maxi - mini)

        if ignore_z:
            scale = scale[:-1]
            orig_min = orig_min[:-1]

        def transform(action):
            action = orig_min + scale * (action - mini)
            return action

        self._transform = transform

    def reset(self):
        return self._env.reset()

    def step(self, action):
        action[:2] = self._transform(action[:2])
        rtn = self._env.step(action)
        return rtn

    def __getattr__(self, name):
        return getattr(self._env, name)


class SkillWrapper:
    def __init__(self, env, max_steps=30):
        self._env = env
        self._nsteps = 0
        self.max_steps = max_steps
        self.skill_modes = SKILLS

    def reset(self):
        self._nsteps = 0
        return self._env.reset()

    def _hang_pose(self, action, pose):
        hang_action = np.r_[action[:2], ARM_POSE["hang"][:-1], PALM_POSE[pose]]
        rtn = self._env.step(hang_action)
        return rtn

    def _grasp_pose(self, action, pose):
        _action = np.r_[action[:2], ARM_POSE["grasp"][:-1], PALM_POSE[pose]]
        rtn = self._env.step(_action)
        return rtn

    def _hang_fist_pose(self, action, pose):
        # Only use the first two: (x, y)
        hang_action = np.r_[action[:2], ARM_POSE[pose]]
        rtn = self._env.step(hang_action)
        return rtn

    def step(self, action):
        assert action.shape[0] == 8
        if action[-1] == self.skill_modes["knock_on"]:
            # make a fist before movement
            self._hang_fist_pose(action, "fist_up")
            self._hang_fist_pose(action, "fist_down")
            rtn = self._hang_fist_pose(action, "fist_up")
        elif action[-1] == self.skill_modes["touch"]:
            self._hang_pose(action, "keep")
            self._grasp_pose(action, "grasp")
            self._grasp_pose(action, "release")
            rtn = self._hang_pose(action, "keep")
        elif action[-1] == self.skill_modes["pick_up"]:
            self._hang_pose(action, "keep")
            self._grasp_pose(action, "grasp")
            rtn = self._hang_pose(action, "keep")
            # only move to destination when grasped something
            if len(self._scene.robot.gripper._grasped_objects) > 0:
                destination = Dummy(DESTINATION)
                destination_action = np.r_[
                    destination.get_position() + np.array([0, 0, 0.2]),
                    destination.get_quaternion(),
                    PALM_POSE["release"],
                ]
                rtn = self._env.step(destination_action)
            else:
                rtn = self._hang_pose(action, "release")
        elif action[-1] == self.skill_modes["weigh"]:
            self._hang_pose(action, "keep")
            self._grasp_pose(action, "grasp")
            if len(self._scene.robot.gripper._grasped_objects) > 0:
                self._hang_pose(action, "keep")
                self._grasp_pose(action, "release")
                rtn = self._hang_pose(action, "keep")
            else:
                logger.warning("no grasped objs.")
                rtn = self._hang_pose(action, "release")
        else:
            raise NotImplementedError
        obs, reward, terminate = rtn
        self._nsteps += 1
        # Terminations
        if self._nsteps >= self.max_steps:
            terminate = True
        return obs, reward, terminate

    def __getattr__(self, name):
        return getattr(self._env, name)


class NICOLEnv(RLBenchEnv):
    def __init__(
        self,
        task_class=None,
        observation_mode="state",
        headless=False,
        render_mode=None,
        variation=True,
        absolute_action_mode=True,
        crop=True,
    ):
        self._observation_mode = observation_mode
        self._render_mode = render_mode
        self._absolute_action_mode = absolute_action_mode
        self._crop = crop
        self.variation = True
        self.config = ObservationConfig(
            front_camera=CameraConfig(
                mask=False,
                image_size=(512, 512),
                # Use OPENGL instead of OPENGL3 to ignore shadows
                # see https://www.coppeliarobotics.com/helpFiles/en/visionSensorPropertiesDialog.htm
                render_mode=RenderMode.OPENGL,
            ),
            joint_positions=False,
            joint_velocities=False,
            gripper_open=False,
            gripper_pose=False,
            gripper_matrix=False,
        )
        obs_config = self.config
        action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(),
            gripper_action_mode=PalmDiscrete(),
        )

        if observation_mode == "state":
            obs_config.set_all_high_dim(False)
            obs_config.set_all_low_dim(False)
        elif observation_mode == "vision":
            obs_config.front_camera.set_all(True)
            obs_config.overhead_camera.set_all(False)
            obs_config.left_shoulder_camera.set_all(False)
            obs_config.right_shoulder_camera.set_all(False)
            obs_config.wrist_camera.set_all(False)
        else:
            raise ValueError("Unrecognised observation_mode: %s." % observation_mode)

        self.env = NicolEnvironment(
            action_mode=action_mode,
            obs_config=obs_config,
            robot_setup="nicol",
            headless=headless,
        )
        self.env.launch()

        # This order matters
        self.task_class = task_class
        task = self.env.get_task(self.task_class)
        task = SkillWrapper(task)
        if not self._absolute_action_mode:
            task = ActionScaleWrapper(task, mini=0.0, maxi=1.0)
        self.task = task

        info, obs = self.task.reset()
        self.info = {"instruction": info[0]}

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,)
        )  # (x, y, z, *quaterion, hand gesture)

        if observation_mode == "state":
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs.get_low_dim_data().shape
            )
        elif observation_mode == "vision":
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Box(
                        low=-np.inf, high=np.inf, shape=obs.get_low_dim_data().shape
                    ),
                    "front_rgb": spaces.Box(low=0, high=1, shape=(128, 128, 3)),
                }
            )

        if render_mode is not None:
            # Add the camera to the scene
            cam_placeholder = Dummy("cam_cinematic_placeholder")
            self._gym_cam = VisionSensor.create([640, 640])
            self._gym_cam.set_pose(cam_placeholder.get_pose())
            # self._gym_cam = VisionSensor('render')
            if render_mode == "human":
                self._gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
            else:
                self._gym_cam.set_render_mode(RenderMode.OPENGL3)
        self.states = None

    def _extract_obs(self, obs) -> Dict[str, np.ndarray]:
        if self._observation_mode == "state":
            return obs.get_low_dim_data()
        elif self._observation_mode == "vision":
            # crop vision
            if self._crop:
                obs.front_rgb = obs.front_rgb[90 : 90 + 205, 125 : 125 + 205, :]
            return {
                "state": obs.get_low_dim_data(),
                "front_rgb": obs.front_rgb,
            }

    def reset(self) -> Dict[str, np.ndarray]:
        if self.variation:
            self.task.sample_variation()
        _, obs = self.task.reset()
        states = self._extract_obs(obs)
        self.states = states
        return states

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        self.task._task.cleanup()
        obs, reward, terminate = self.task.step(action)
        states = self._extract_obs(obs)
        self.states = states
        return states, reward, terminate, self.info


class NICOLKnockingEnv(NICOLEnv):
    def __init__(self, *args, **kwargs):
        self.task_class = Knocking
        super().__init__(
            task_class=Knocking, absolute_action_mode=False, *args, **kwargs
        )

    def reset(self) -> Dict[str, np.ndarray]:
        if self.variation:
            self.task.sample_variation()
        info, obs = self.task.reset()
        self.info = {"instruction": info[0], "sounds": None}
        states = self._extract_obs(obs)
        self.states = states
        return states

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        self.task._task.cleanup()
        obs, reward, terminate = self.task.step(action)
        states = self._extract_obs(obs)
        self.info["sounds"] = self.task._task.records
        self.info["touchs"] = self.task._task.touchs
        self.info["weights"] = self.task._task.weights
        self.states = states
        return states, reward, terminate, self.info

    def set_mode(self, mode):
        # Note that this function should be called every after =init_task()= is called.
        self.task._task.set_mode(mode)

    def set_random(self, random):
        # Note that this function should be called every after =init_task()= is called.
        self.task._task.set_random(random)
