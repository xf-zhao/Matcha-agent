from nicol_env import NICOLEnv
from pyrep.objects.shape import Shape
from rlbench.tasks import LiftNumberedBlock
import numpy as np
from typing import List
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition, GraspedCondition


def quick_fix(blocks):
    for block in blocks:
        block.set_quaternion([0.76801813, -0.61692601, -0.13173638, 0.11043578])


def init_episode(self, index: int) -> List[str]:
    block_num = index + 1
    target_block = self._blocks[index]
    self._boundary.clear()
    for block in self._blocks:
        self._boundary.sample(block, min_distance=0.2)
    quick_fix(self._blocks)
    self._w1.set_pose(self._anchor[index].get_pose())

    self.register_success_conditions(
        [
            GraspedCondition(self.robot.gripper, target_block),
            DetectedCondition(target_block, self._success_detector),
        ]
    )

    return [
        "pick up the block with the number %d" % block_num,
        "grasp the %d numbered block and lift" % block_num,
        "lift the %d numbered block" % block_num,
    ]


LiftNumberedBlock.init_episode = init_episode

env = NICOLEnv(task_class=LiftNumberedBlock, observation_mode="vision", headless=False)


SKILLS = { "weigh": 1, "knock_on": 2, "touch": 3, "pick_up": 4}
dest = Shape("block1")
for i in range(1, 5):
    env.reset()
    action = np.r_[dest.get_position(), dest.get_quaternion(), i]
    print(action)
    rtn = env.step(action)
