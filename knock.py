import gym
from typing import Union, Dict, Tuple
import numpy as np
from gym import spaces
from rlbench.gym.rlbench_env import RLBenchEnv
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import GripperActionMode, assert_action_shape, PalmDiscrete
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import Knocking
import numpy as np
from pyrep.const import RenderMode
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import GripperActionMode, assert_action_shape, PalmDiscrete
from rlbench.backend.spawn_boundary import SpawnBoundary,  BoundaryObject
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.tasks import Knocking

from tqdm import tqdm


DESTINATION = 'waypoint1'





class FinishGraspingWrapper:

    def __init__(self, env, max_steps=30):
        self._env = env
        destination = Dummy(DESTINATION)
        destination_action = np.array([*destination.get_position(), *destination.get_quaternion(), 0, 0])
        self._nsteps = 0
        self.max_steps = max_steps

        def transform_grasp(action):
            action = np.r_[action, 0, 1]
            return action

        def transform_lift(action):
            # x, y, z
            action[2] += 0.15
            return action

        def transform_lift_open(action):
            # x, y, z
            action[2] += 0.15
            # release
            action[-2], action[-1] = 0, 0
            return action

        def transform_destination(*args):
            return destination_action

        self._transform_grasp = transform_grasp
        self._transform_lift = transform_lift
        self._transform_lift_open = transform_lift_open
        self._transform_destination = transform_destination

    def reset(self):
        self._nsteps = 0
        return self._env.reset()

    def step(self, action):
        action = self._transform_grasp(action)
        rtn = self._env.step(action)

        # only move to destination when grasped something
        if len(self._scene.robot.gripper._grasped_objects)>0:
            action = self._transform_lift(action)
            rtn = self._env.step(action)
            action = self._transform_destination(action)
            rtn = self._env.step(action)
        else:
            action = self._transform_lift_open(action)
            rtn = self._env.step(action)
        self._nsteps += 1
        obs, reward, terminate = rtn
        # Terminations
        if self._nsteps >= self.max_steps:
            terminate = True
        return obs, reward, terminate

    def __getattr__(self, name):
        return getattr(self._env, name)



class ActionScaleWrapper:

    def __init__(self, env, ignore_z =True, mini=-1.0, maxi=1.0):
        self._env = env
        boundary = Shape('Boundary')
        bbox_pos = boundary.get_position()
        # bbox_pos[2] += 0.07 # Go to the top of the bbox for grasping
        # bbox  = BoundaryObject(boundary)._boundary_bbox
        # orig_min = np.array([bbox.min_x, bbox.min_y, bbox.min_z]) + bbox_pos
        # orig_max = np.array([bbox.max_x, bbox.max_y, bbox.max_z]) + bbox_pos
        bbox = boundary.get_bounding_box()
        orig_min = bbox_pos + bbox[::2]
        orig_max = bbox_pos + bbox[1::2]

        scale = (orig_max - orig_min)/ (maxi - mini)

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
        action = self._transform(action)
        rtn = self._env.step(action)
        return rtn

    def __getattr__(self, name):
        return getattr(self._env, name)

class ActionMidWrapper:

    def __init__(self, env):
        self._env = env
        action_mid = np.array([
            0.83443451,
            0.76801813, -0.61692601, -0.13173638, 0.11043578,
        ])
        def transform(action):
            action = np.r_[action, action_mid]
            return action
        self._transform = transform

    def reset(self):
        return self._env.reset()

    def step(self, action):
        if action.shape[0] <= 4:
            action = self._transform(action)
        rtn = self._env.step(action)
        return rtn

    def __getattr__(self, name):
        return getattr(self._env, name)



# env = Environment(
#     action_mode=MoveArmThenGripper(
#         arm_action_mode=EndEffectorPoseViaPlanning(), gripper_action_mode=PalmDiscrete()),
#     obs_config=ObservationConfig(),
#     robot_setup='nicol',
#     headless=False)
# env.launch()
# task = env.get_task(Knocking)
#
# # This order matters
# task = FinishGraspingWrapper(task)
# task = ActionMidWrapper(task)
# task = ActionScaleWrapper(task)
# env =task


class Agent(object):

    def __init__(self, action_shape):
        self.action_shape = action_shape

    def act(self, obs):
        arm = np.random.normal(0.0, 0.1, size=(self.action_shape[0] - 1,))
        gripper = [1.0]  # Always open
        return np.concatenate([arm, gripper], axis=-1)


class NICOLEnv(RLBenchEnv):
    def __init__(self, task_class, observation_mode='state', headless=False, render_mode=None, variation=True):
        self._observation_mode = observation_mode
        self._render_mode = render_mode
        self.config = ObservationConfig(
            # left_shoulder_camera=CameraConfig(mask=False, image_size=(512, 512)),
            #                            right_shoulder_camera=CameraConfig(mask=False, image_size=(512, 512)),
                                        front_camera=CameraConfig(mask=False, image_size=(512, 512)),
                                        joint_positions=True, joint_velocities=False, gripper_open=False,
                                        gripper_pose=False, gripper_matrix=False, )
        obs_config = self.config
        action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(), gripper_action_mode=PalmDiscrete())

        self.variation = variation
        if observation_mode == 'state':
            obs_config.set_all_high_dim(False)
            obs_config.set_all_low_dim(True)
        elif observation_mode == 'vision':
            obs_config.front_camera.set_all(True)
            obs_config.overhead_camera.set_all(False)
            obs_config.left_shoulder_camera.set_all(False)
            obs_config.right_shoulder_camera.set_all(False)
            obs_config.wrist_camera.set_all(False)
            # obs_config.set_all(True)
        else:
            raise ValueError(
                'Unrecognised observation_mode: %s.' % observation_mode)

        self.env = Environment(
            action_mode= action_mode,
            obs_config=obs_config,
            robot_setup='nicol',
            headless=headless,
        )
        self.env.launch()

        # This order matters
        task = self.env.get_task(task_class)
        task = FinishGraspingWrapper(task)
        task = ActionMidWrapper(task)
        task = ActionScaleWrapper(task)
        self.task = task

        _, obs = self.task.reset()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,))

        if observation_mode == 'state':
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs.get_low_dim_data().shape)
        elif observation_mode == 'vision':
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                # "left_shoulder_rgb": spaces.Box(
                #     low=0, high=1, shape=obs.left_shoulder_rgb.shape),
                # "right_shoulder_rgb": spaces.Box(
                #     low=0, high=1, shape=obs.right_shoulder_rgb.shape),
                # "wrist_rgb": spaces.Box(
                #     low=0, high=1, shape=obs.wrist_rgb.shape),
                "front_rgb": spaces.Box(
                    low=0, high=1, shape=(128,128,3)),
            })

        if render_mode is not None:
            # Add the camera to the scene
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            self._gym_cam = VisionSensor.create([640, 360])
            self._gym_cam.set_pose(cam_placeholder.get_pose())
            if render_mode == 'human':
                self._gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
            else:
                self._gym_cam.set_render_mode(RenderMode.OPENGL3)

    def _extract_obs(self, obs) -> Dict[str, np.ndarray]:
        if self._observation_mode == 'state':
            return obs.get_low_dim_data()
        elif self._observation_mode == 'vision':
            obs.front_rgb = obs.front_rgb[240:240 + 128, 105:105 + 128, :]
            return {
                "state": obs.get_low_dim_data(),
                "front_rgb": obs.front_rgb,
                # "left_shoulder_rgb": obs.left_shoulder_rgb,
                # "right_shoulder_rgb": obs.right_shoulder_rgb,
                # "wrist_rgb": obs.wrist_rgb,
            }


# agent = Agent(env.action_shape)

training_epoches = 20
episode_length = 2
obs = None

# cuboid position is known
# target = Dummy('waypoint0')
# distractor = Dummy('waypoint0')

env = NICOLEnv(task_class=Knocking, observation_mode='vision', headless=False, render_mode=None, variation = True)

obs = env.reset()
action = np.random.rand(2)
action
obs, reward, terminate, info = env.step(action)
print(obs)
print(info)
import matplotlib.pyplot as plt
for key, value in obs.items():
    if len(value.shape) <= 1:
        continue
    plt.figure()
    plt.imshow(value)
plt.show()


for ep in tqdm(range(training_epoches)):
    print('Reset Episode')
    # DONE: add try-catch to reset()
    try:
        obs = env.reset()
        action = np.array([*target.get_position(), *target.get_quaternion(), 0, 1])
        # distract_action = np.array([*distractor.get_position(), *distractor.get_quaternion(), 0, 1])
    except RuntimeError:
        print('Bad initialization, reset.')
        continue
    except Exception as e:
        raise e

    for i in range(50):
        action = np.random.rand(2)
        print(action)
        obs, reward, terminate, _ = env.step(action)
        print(reward)
        if terminate:
            break

print('Done')
env.close()
# from stable_baselines3 import A2C
# 
# model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log='./a2c_nicol/')
# model.learn(total_timesteps=10_0000, progress_bar=True, log_interval=20)
# 