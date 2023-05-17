from typing import Union, Dict, Tuple
from pathlib import Path
import numpy as np
from gym import spaces
from rlbench.gym.rlbench_env import RLBenchEnv
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import (
    JointVelocity,
    EndEffectorPoseViaPlanning,
)
from rlbench.action_modes.gripper_action_modes import (
    GripperActionMode,
    assert_action_shape,
    PalmDiscrete,
)
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import Knocking
import numpy as np
from pyrep.const import RenderMode
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import (
    JointVelocity,
    EndEffectorPoseViaPlanning,
)
from rlbench.action_modes.gripper_action_modes import (
    GripperActionMode,
    assert_action_shape,
    PalmDiscrete,
)
from rlbench.backend.spawn_boundary import SpawnBoundary, BoundaryObject
from rlbench.observation_config import ObservationConfig, CameraConfig
import os
import uuid
# from rlbench.tasks.knocking import DESTINATION

DESTINATION = 'waypoint1'
SKILLS = {"ask_human": 0, "knock_on": 1, "touch": 2, "pick_up": 3, "weigh": 4}
# z and quatenion
DEFAULT_POSE = np.array([0.83443451, 0.76801813, -0.61692601, -0.13173638, 0.11043578, 4]) # the last "4" is for release
# DEFAULT_POSE = np.array([0.83443451, 0.76801813, -0.61692601, -0.13173638, 0.11043578])
FIST_POSE = np.array([0.9, 0, 0, 0.8509035, 0.525322, 1]) # the last 1 indicates 'knock_on'
FIST_POSE_DOWN = np.array([0.81, 0, 0, 0.8509035, 0.525322, 1]) # the last 1 indicates 'knock_on'


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
        self.last_action = None

    def reset(self):
        self._nsteps = 0
        self.last_action = None
        return self._env.reset()

    def _hang_pose(self, action):
        hang_action = np.r_[action[:2], FIST_POSE[0], DEFAULT_POSE[1:]]
        rtn = self._env.step(hang_action)
        return rtn

    def _hang_fist_pose(self, action):
        hang_action = np.r_[action[:2], FIST_POSE]
        rtn = self._env.step(hang_action)
        return rtn

    def _prepare_normal_pose(self):
        # to prepare and really execute after this
        if self.last_action is not None:
            make_normal_action = np.r_[self.last_action[:3], DEFAULT_POSE[1:-1], 4] # release and adjust pose first
        rtn = self._env.step(make_normal_action)
        return rtn

    def step(self, action):
        RATIO = 1 / 0.9
        assert action.shape[0] == 8
        if action[-1] == self.skill_modes["ask_human"]:
            # copied from "knock" action
            _action = np.r_[action[:-1], 2]  # point
            _action = (
                _action
                + np.r_[
                    -0.15 * RATIO,
                    -0.05 * RATIO,
                    0.05 * RATIO,
                    [0] * (_action.shape[0] - 3),
                ]
            )
            rtn = self._env.step(_action)
            _action[-1] = 4  # release
            _action[2] = FIST_POSE[0]
            rtn = self._env.step(_action)
        elif action[-1] == self.skill_modes["knock_on"]:
            # make a fist before movement
            self._hang_fist_pose(action)
            _action = np.r_[action[:2], FIST_POSE]
            rtn = self._env.step(_action)
            _action = np.r_[action[:2], FIST_POSE_DOWN]
            rtn = self._env.step(_action)
            _action = np.r_[action[:2], FIST_POSE]
            rtn = self._env.step(_action)
        elif action[-1] == self.skill_modes["touch"]:
            self._hang_pose(action)
            _action = np.r_[action[:-1], 3]  # grasp
            rtn = self._env.step(_action)
            # release first
            _action[-1] = 4
            rtn = self._env.step(_action)
            _action[-1] = 0  # keep
            _action[2] = FIST_POSE[0]
            rtn = self._env.step(_action)
        elif action[-1] == self.skill_modes["pick_up"]:
            self._hang_pose(action)
            # to execute
            _action = np.r_[action[:-1], 3]  # grasp
            rtn = self._env.step(_action)
            # only move to destination when grasped something
            if len(self._scene.robot.gripper._grasped_objects) > 0:
                _action[2] += 0.1 * RATIO
                rtn = self._env.step(_action)
                destination = Dummy(DESTINATION)
                destination_action = np.r_[
                    destination.get_position(), destination.get_quaternion(), 4
                ]
                rtn = self._env.step(destination_action)
                _action =destination_action
            else:
                _action[2] = FIST_POSE[0]
                _action = np.r_[_action[:-1], 4]
                rtn = self._env.step(_action)
        elif action[-1] == self.skill_modes["weigh"]:
            # self._prepare_normal_pose()
            self._hang_pose(action)
            _action = np.r_[action[:-1], 3]  # grasp
            rtn = self._env.step(_action)
            # only lift when grasped something
            if len(self._scene.robot.gripper._grasped_objects) > 0:
                _action[2] += 0.1 * RATIO
                rtn = self._env.step(_action)
                # release
                _action[2] -= 0.1 * RATIO
                _action[-1] = 4
                rtn = self._env.step(_action)
                # lift
                _action[2] = FIST_POSE[0]
                _action[-1] = 0  # keep
                rtn = self._env.step(_action)
            else:
                # _action[2] += 0.1 * RATIO
                _action[2] = FIST_POSE[0]
                _action = np.r_[_action[:-1], 4]
                rtn = self._env.step(_action)
        else:
            raise NotImplementedError
        self.last_action = _action
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
        task_class=Knocking,
        observation_mode="state",
        headless=False,
        render_mode=None,
        variation=True,
        absolute_action_mode=False,
    ):
        self._observation_mode = observation_mode
        self._render_mode = render_mode
        self._absolute_action_mode = absolute_action_mode
        self.variation = True
        self.config = ObservationConfig(
            # left_shoulder_camera=CameraConfig(mask=False, image_size=(512, 512)),
            #                            right_shoulder_camera=CameraConfig(mask=False, image_size=(512, 512)),
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
            # obs_config.set_all(True)
        else:
            raise ValueError("Unrecognised observation_mode: %s." % observation_mode)

        self.env = Environment(
            action_mode=action_mode,
            obs_config=obs_config,
            robot_setup="nicol",
            headless=headless,
        )
        self.env.launch()

        # This order matters
        task = self.env.get_task(task_class)
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
                    # "left_shoulder_rgb": spaces.Box(
                    #     low=0, high=1, shape=obs.left_shoulder_rgb.shape),
                    # "right_shoulder_rgb": spaces.Box(
                    #     low=0, high=1, shape=obs.right_shoulder_rgb.shape),
                    # "wrist_rgb": spaces.Box(
                    #     low=0, high=1, shape=obs.wrist_rgb.shape),
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
            # obs.front_rgb = obs.front_rgb[240:240 + 128, 105:105 + 128, :]
            obs.front_rgb = obs.front_rgb[90 : 90 + 205, 125 : 125 + 205, :]
            return {
                "state": obs.get_low_dim_data(),
                "front_rgb": obs.front_rgb,
                # "left_shoulder_rgb": obs.left_shoulder_rgb,
                # "right_shoulder_rgb": obs.right_shoulder_rgb,
                # "wrist_rgb": obs.wrist_rgb,
            }

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


import random
import openai
import numpy as np
import imageio


###################### MODULES #################################################

import json
import requests


class Client:
    def __init__(self) -> None:
        self.headers = {"content-type": "application/json"}
        self.address = None

    def call(self, **kwargs):
        data = json.dumps(kwargs)
        response = requests.post(self.address, data=data, headers=self.headers)
        result = json.loads(response.text)
        return result


class ViLDClient(Client):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.address = "http://134.100.10.107:8848/api/vild"
        self.CATEGORY_NAMES = [
            "red block",
            "green block",
            "blue block",
            "orange block",
            "yellow block",
            "purple block",
            # "pink block",
            # "cyan block",
            # "brown block",
            # "blue bowl",
            # "red bowl",
            # "green bowl",
            # "orange bowl",
            # "yellow bowl",
            # "purple bowl",
            # "pink bowl",
            # # "cyan bowl",
            # "brown bowl",
        ]

    def call(self, category_names=None, **kwargs):
        if category_names is None:
            category_names = self.CATEGORY_NAMES
        return super().call(category_names=category_names, **kwargs)


class SoundClient(Client):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.address = "http://134.100.10.107:8848/api/sound"


class Assistant:
    def __init__(
        self, switch_vision=True, swith_sound=True, sound_use_adjective=True
    ) -> None:
        if swith_sound:
            self.sounder = SoundClient()
            self.sound_use_adjective = sound_use_adjective
        if switch_vision:
            self.vilder = ViLDClient()

    def vision(self, image_path=None, category_names=None, plot_on=False):
        found_objects = self.vilder.call(
            image_path=image_path, category_names=category_names, plot_on=plot_on
        )

        objects = ", ".join([k for k in found_objects])
        caption = f"[{objects}]"

        self.caption = caption
        self.found_objects = found_objects
        return caption

    def sound(self, sounds):
        if len(sounds) == 0:
            sound_like = (
                f"The robot is currently not able to knock on the targeted object."
            )
            return sound_like
        top_probs, top_materials, top_adjectives = self.sounder.call(
            sound_path=sounds[0]
        )
        prob0, material0, adj0 = (
            int(top_probs[0] * 100),
            top_materials[0],
            top_adjectives[0],
        )
        prob1, material1, adj1 = (
            int(top_probs[1] * 100),
            top_materials[1],
            top_adjectives[1],
        )
        if self.sound_use_adjective:
            if prob0 > 50:
                sound_like = f"It sounds {adj0}."  # or f'sounds {sharp}'
            else:
                sound_like = f"It sounds {adj0} mostly and also a little bit {adj1}."  # or f'sounds {sharp}'
        else:
            if prob0 > 85:
                sound_like = f"It is made of {material0}."  # or f'sounds {sharp}'
            elif prob0 > 50:
                sound_like = (
                    f"It is probably made of {material0}."  # or f'sounds {sharp}'
                )
            else:
                sound_like = f"The material cannot be certainly confirmed according to the impact sound. It could be {material0} with a {prob0}% chance, or {material1} with a {prob1}% chance."  # or f'sounds {sharp}'
        return sound_like

    def touch(self, touchs):
        if len(touchs) > 0:
            feeling = f"It feels {touchs[0]}."
        else:
            feeling = f"Cannot touch it."
        return feeling

    def weigh(self, weights):
        if len(weights) > 0:
            weighing = f"It weighs {weights[0]}."
        else:
            weighing = f"Not able to weigh it now."
        return weighing

    def feedback(self):
        return

    def target_to_normalized_coordinates(self, target):
        invalid_target = False
        if target not in self.found_objects:
            target = random.choice([k for k in self.found_objects])
            invalid_target = True
        return (
            self.found_objects[target]["normalized_coordinates"],
            target,
            invalid_target,
        )



class Agent:
    def __init__(self, assistant: Assistant, action_tolerance=5) -> None:
        self.actions = {
            "knock_on": self.knock_on,
            "touch": self.touch,
            "weigh": self.weigh,
            "ask_human": self.ask_human,
            "pick_up": self.pick_up,
            "terminate": self.terminate,
        }
        self.assistant = assistant
        self.action_indicator = " robot."
        self.action_tolerance = action_tolerance
        self._clear()

    def reset(self):
        self._clear()

    def execute(self, environment, command):
        # command in a format: "AI: > robot.knock(blue block)"
        if self.action_indicator not in command:
            return '', self.explain(command), "explain",0, False

        normal = False
        command = command.replace('[', '(').replace(']', ')')
        infos = command.split(self.action_indicator)[1].split("(")
        target, *_explanation = infos[1].split(")")
        if isinstance(_explanation, str):
            explanation = _explanation
        else:
            explanation = "".join(_explanation)
        skill = infos[0].lower()
        if skill not in SKILLS:
            if skill in ['knockon', 'knock on', 'knock up', 'knock_up']:
                skill = 'knock_on'
            elif skill in ['pickup', 'pick up', 'pick on', 'pick_on']:
                skill = 'pick_up'
            elif skill in ['touchon', 'touch_on', 'touch up', 'touch_up']:
                skill = 'touch'
            elif skill in ['weighon', 'weigh_up', 'weigh up', 'weigh_on']:
                skill = 'weigh'
            else:
                skill = 'knock_on'
                invalid_skill = True
        action, chosen_target, invalid_target = self._target_to_action(target, skill)
        if self.many_duplicates:
            skill = "pick_up"
            reason = "Too many duplicated actions."
        elif self.invalid_count >= self.action_tolerance:
            skill = "pick_up"
            reason = "Too many invalid actions."
        else:
            if invalid_target:
                reason = "Invalid target."
                # print(reason)
                self.invalid_count += 1
            else:
                normal = True

        skill_func = self.actions[skill]
        description, explanation, reward, done = skill_func(environment, action)
        if normal:
            return description, explanation, skill, reward, done
        # description_post, done = skill_func(environment, action)
        skill_natural = skill.replace("_", " ")
        description_pre = (
            f"Human: {reason} Randomly {skill_natural} the {chosen_target} instead."
        )
        description = description.replace("Human:", "")
        description = description_pre + description
        # count duplicated actions
        if description not in self.executions:
            self.executions[description] = 0
        else:
            self.executions[description] += 1
            if self.executions[description] >= self.action_tolerance:
                self.many_duplicates = True
            else:
                pass
        return description, explanation, skill, reward, done

    def knock_on(self, environment, action):
        action = self._update_action_skill(action, skill="knock_on")
        env = environment.env
        obs, reward, terminate, info = env.step(action)
        sounds = info["sounds"]
        sound_like = self.assistant.sound(sounds)
        description = f"Human: {sound_like}\nAI:"
        return description, None, reward, False or terminate

    def pick_up(self, environment, action):
        env = environment.env
        action = self._update_action_skill(action, skill="pick_up")
        obs, reward, terminate, info = env.step(action)
        description = "Human: Explain why.\nAI:"
        return description, None, reward, True or terminate

    def explain(self, command):
        print(command)
        description = "Human: go on.\nAI:"
        return description, command, 0, False

    def vision(self, *args, **kwargs):
        return self.assistant.vision(*args, **kwargs)

    def touch(self, environment, action):
        action = self._update_action_skill(action, skill="touch")
        env = environment.env
        obs, reward, terminate, info = env.step(action)
        touchs = info["touchs"]
        feeling = self.assistant.touch(touchs)
        description = f"Human: {feeling}\nAI:"
        return description, None, reward, False or terminate

    def ask_human(self, target):
        answer = self.assistant.feedback()
        return answer, None, 0, False

    def weigh(self, environment, action):
        action = self._update_action_skill(action, skill="weigh")
        env = environment.env
        obs, reward, terminate, info = env.step(action)
        weights = info["weights"]
        feeling = self.assistant.weigh(weights)
        description = f"Human: {feeling}\nAI:"
        return description, None, reward, False or terminate

    def terminate(self):
        reward = 0
        return "", None, reward, True

    def _clear(self):
        self.invalid_count = 0
        self.executions = {}
        self.many_duplicates = False

    def _target_to_action(self, target, skill):
        (
            normmalized_coordinates,
            target,
            invalid_target,
        ) = self.assistant.target_to_normalized_coordinates(target)
        return (
            np.r_[normmalized_coordinates, DEFAULT_POSE[:-1], SKILLS[skill]],
            target,
            invalid_target,
        )

    def _update_action_skill(self, action, skill):
        action[-1] = SKILLS[skill]
        return action


class ChatEnvironment:
    def __init__(
        self, env_cls, mode="test", headless=False, temp_directory="./temp", debug=True, render_mode=None
    ) -> None:
        if render_mode=='None':
            render_mode=None
        env = env_cls(observation_mode="vision", headless=headless, render_mode=render_mode)
        env.set_mode(mode)  # either 'train' or 'test'
        env.set_random(not debug)
        self.env = env
        self.temp_directory = Path(temp_directory).absolute()
        self.temp_files = []
        self.instruction = None

    def reset(self):
        try:
            obs = self.env.reset()
        except RuntimeError:
            print("Bad initialization, reset.")
        except Exception as e:
            raise e
        # image = obs["front_rgb"]
        self.instruction = self.env.info["instruction"]
        return

    def instruct(self):
        return self.instruction

    def instruct_with_caption(self, caption=None):
        if caption is None:
            instruction = f'Human: "{self.instruction}".\n'
        else:
            instruction = (
                f'Human: "{self.instruction}" in the scene that contains {caption}.\n'
            )
        return instruction

    def render(self):
        env = self.env
        image = env.states["front_rgb"]
        temp_image_name = str(uuid.uuid4())[-8:] + ".jpg"
        if not self.temp_directory.exists():
            self.temp_directory.mkdir()
        temp_image_path = str(self.temp_directory / temp_image_name)
        imageio.imwrite(temp_image_path, image)
        self.temp_files.append(temp_image_path)
        return temp_image_path

    def clean_up(self):
        for temp_file in self.temp_files:
            os.remove(temp_file)
            print(f"Removed {temp_file}.")
        self.temp_files = []
        return


LLM_CACHE = {}


class LLM:
    def __init__(
        self, openai_api_key_path, prompt_path=None, engine="text-ada-001", 
        max_tokens=128, temperature=0.7,
    ) -> None:
        with open(openai_api_key_path) as f:
            openai_api_key = f.read()
            openai.api_key = openai_api_key
        if prompt_path is not None:
            with open(prompt_path) as f:
                prompt = f.read() + "\n"
        else:
            prompt = ""
        self._prompt = prompt
        self.engine = engine
        self.max_tokens = max_tokens
        self.temperature = temperature

    def reset(self):
        pass

    def feed(self, prompt=""):
        if 'explain' in prompt.lower():
            max_tokens = 128
        else:
            # For actions, use small tokens.
            max_tokens = 16
        prompt = self._prompt + prompt
        response = self.gpt3_call(prompt, max_tokens=max_tokens)
        command = response["choices"][0]["text"]
        # command = 'robot.knock(blue block)\nother things\n'
        if "\n" in command:
            command = command.split("\n")[0]
        command = command + "\n"
        return command

    def gpt3_call(self, prompt,max_tokens=128, logprobs=1, echo=False):
        max_tokens = min(self.max_tokens, max_tokens)
        temperature = self.temperature
        id = tuple((self.engine, prompt, max_tokens, temperature, logprobs, echo))
        if id in LLM_CACHE.keys():
            print("cache hit, returning")
            response = LLM_CACHE[id]
        else:
            response = openai.Completion.create(
                engine=self.engine,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=logprobs,
                echo=echo,
            )
            LLM_CACHE[id] = response
        return response


class FakeLLM(LLM):
    def __init__(self, *args, **kwargs):
        self.count = 0
        self.diversity = [
 "> robot.knock_on(random object)\n",
 "> robot.touch(random object)\n",
 "> robot.weigh(random object)\n",
 "> robot.weigh(random object)\n",
 "> robot.touch(random object)\n",
 "> robot.touch(random object)\n",
 "> robot.touch(random object)\n",
 "> robot.knock_on(random object)\n",
 "> robot.weigh(random object)\n",
 "> robot.knock_on(random object)\n",
        ]

    def reset(self):
        self.count = 0

    def feed(self, *args, **kwargs):
        rtn =  self.diversity[self.count]
        self.count+=1
        return rtn