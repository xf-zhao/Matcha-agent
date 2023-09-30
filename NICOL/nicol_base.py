from os.path import dirname, abspath, join
from pyrep import PyRep
from rlbench.backend.const import *
from rlbench.action_modes.action_mode import ActionMode
from rlbench.backend.const import *
from rlbench.backend.robot import Robot
from rlbench.backend.scene import Scene
from rlbench.const import SUPPORTED_ROBOTS
from rlbench.observation_config import ObservationConfig
from rlbench.sim2real.domain_randomization import RandomizeEvery, \
    VisualRandomizationConfig, DynamicsRandomizationConfig
from rlbench.sim2real.domain_randomization_scene import DomainRandomizationScene
from rlbench.environment import Environment
from NICOL.nicol_arm import NicolRightArm, NicolRightPalm

DIR_PATH = dirname(abspath(__file__))
SUPPORTED_ROBOTS.update({'nicol': (NicolRightArm, NicolRightPalm, 8),})


class NicolEnvironment(Environment):
    """Each environment has a scene."""

    def __init__(self,
                 action_mode: ActionMode,
                 dataset_root: str = '',
                 obs_config: ObservationConfig = ObservationConfig(),
                 headless: bool = False,
                 static_positions: bool = False,
                 robot_setup: str = 'panda',
                 randomize_every: RandomizeEvery = None,
                 frequency: int = 1,
                 visual_randomization_config: VisualRandomizationConfig = None,
                 dynamics_randomization_config: DynamicsRandomizationConfig = None,
                 attach_grasped_objects: bool = True,
                 shaped_rewards: bool = False
                 ):

        self._dataset_root = dataset_root
        self._action_mode = action_mode
        self._obs_config = obs_config
        self._headless = headless
        self._static_positions = static_positions
        self._robot_setup = robot_setup.lower()

        self._randomize_every = randomize_every
        self._frequency = frequency
        self._visual_randomization_config = visual_randomization_config
        self._dynamics_randomization_config = dynamics_randomization_config
        self._attach_grasped_objects = attach_grasped_objects
        self._shaped_rewards = shaped_rewards

        if robot_setup not in SUPPORTED_ROBOTS.keys():
            raise ValueError('robot_configuration must be one of %s' %
                             str(SUPPORTED_ROBOTS.keys()))

        if (randomize_every is not None and
                visual_randomization_config is None and
                dynamics_randomization_config is None):
            raise ValueError(
                'If domain randomization is enabled, must supply either '
                'visual_randomization_config or dynamics_randomization_config')

        self._check_dataset_structure()
        self._pyrep = None
        self._robot = None
        self._scene = None
        self._prev_task = None

    def launch(self):
        if self._pyrep is not None:
            raise RuntimeError('Already called launch!')
        self._pyrep = PyRep()
        self._pyrep.launch(join(DIR_PATH, TTT_FILE), headless=self._headless)

        arm_class, gripper_class, _ = SUPPORTED_ROBOTS[
            self._robot_setup]

        # We assume the panda is already loaded in the scene.
        if self._robot_setup != 'nicol':
            # Remove the panda from the scene
            # If panda doesn't exist, do nothing (suppose we already have our own robot in the scene now)
            arm = NicolRightArm()
            pos = arm.get_position()
            arm.remove()
            arm_path = join(DIR_PATH, 'robot_ttms', self._robot_setup + '.ttm')
            self._pyrep.import_model(arm_path)
            arm, gripper = arm_class(), gripper_class()
            arm.set_position(pos)
        else:
            arm, gripper = arm_class(), gripper_class()

        self._robot = Robot(arm, gripper)
        if self._randomize_every is None:
            self._scene = Scene(
                self._pyrep, self._robot, self._obs_config, self._robot_setup)
        else:
            self._scene = DomainRandomizationScene(
                self._pyrep, self._robot, self._obs_config, self._robot_setup,
                self._randomize_every, self._frequency,
                self._visual_randomization_config,
                self._dynamics_randomization_config)

        self._action_mode.arm_action_mode.set_control_mode(self._robot)
