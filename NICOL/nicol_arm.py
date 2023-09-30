from pyrep.robots.end_effectors.gripper import Gripper
from pyrep.robots.robot_component import RobotComponent
from pyrep.robots.arms.arm import Arm
from rlbench.action_modes.gripper_action_modes import (
    GripperActionMode,
    assert_action_shape,
)
from rlbench.backend.scene import Scene
import numpy as np

POSITION_ERROR = 0.001


class Palm(RobotComponent):
    pass


class NicolRightArm(Arm):
    def __init__(self, count: int = 0):
        super().__init__(count, "r", num_joints=8)


class NicolRightPalm(Gripper):
    def __init__(
        self,
        count: int = 0,
        name="r_palm",
        joint_names=["r_jointI2", "r_jointM2", "r_jointR2", "r_jointL2", "r_jointT2"],
    ):
        super().__init__(count, name, joint_names)
        self.joint_counts = len(joint_names)

    def actuate(self, amount: np.array, velocity: float) -> bool:
        _, joint_intervals_list = self.get_joint_intervals()
        joint_intervals = np.array(joint_intervals_list)
        amount = np.array(amount).reshape(-1)
        if len(amount) == 1:
            amount = np.repeat(amount, self.joint_counts)

        # Decide on if we need to open or close
        joint_range = joint_intervals[:, 1] - joint_intervals[:, 0]
        target_pos = joint_intervals[:, 0] + (joint_range * amount)
        current_positions = self.get_joint_positions()
        done = True
        for i, (j, target, cur, prev) in enumerate(
            zip(self.joints, target_pos, current_positions, self._prev_positions)
        ):
            # Check if the joint has moved much
            not_moving = prev is not None and np.fabs(cur - prev) < POSITION_ERROR
            reached_target = np.fabs(target - cur) < POSITION_ERROR
            vel = -velocity if cur - target > 0 else velocity
            oscillating = self._prev_vels[i] is not None and vel != self._prev_vels[i]
            if not_moving or reached_target or oscillating:
                j.set_joint_target_velocity(0)
                continue
            done = False
            self._prev_vels[i] = vel  # type: ignore
            j.set_joint_target_velocity(vel)
        self._prev_positions = current_positions  # type: ignore

        if done:
            self._prev_positions = [None] * self._num_joints
            self._prev_vels = [None] * self._num_joints
            self.set_joint_target_velocities([0.0] * self._num_joints)
        return done

    def operate(self, skill):
        if skill == "close_hand":
            pos = 0.65
        elif skill == "open_hand":
            pos = 0.3
        elif skill == "knock_on":
            pos = 1
        elif skill == "knock_off":
            pos = 1
        elif skill == "point":
            pos = np.r_[0, [1] * (self.joint_counts - 1)]
        elif skill == "fist":
            pos = 1
        else:
            pos = -1
        done = self.actuate(pos, 1)
        return done


class PalmDiscrete(GripperActionMode):
    """Control if the gripper is open or closed in a discrete manner.

    Action values > 0.5 will be discretised to 1 (open), and values < 0.5
    will be  discretised to 0 (closed).
    """

    def __init__(
        self, attach_grasped_objects: bool = True, detach_before_open: bool = True
    ):
        self._attach_grasped_objects = attach_grasped_objects
        self._detach_before_open = detach_before_open

        # keep consitent with `knock_env.py` line 90 in `class ActionScaleWrapper`
        self.action_encodes = {
            "keep": np.array([0]),
            "make_fist": np.array([1]),
            "point": np.array([2]),
            "grasp": np.array([3]),
            "release": np.array([4]),
        }

    def _actuate(self, action, scene):
        done = False
        if all(action == self.action_encodes["keep"]):
            pass
        elif all(action == self.action_encodes["point"]):
            while not done:
                done = scene.robot.gripper.operate("point")
                scene.pyrep.step()
                scene.task.step()
        elif all(action == self.action_encodes["release"]):
            while not done:
                done = scene.robot.gripper.operate("open_hand")
                scene.pyrep.step()
                scene.task.step()
        elif all(action == self.action_encodes["make_fist"]):
            while not done:
                done = scene.robot.gripper.operate("fist")
                scene.pyrep.step()
                scene.task.step()
            done = False
        elif all(action == self.action_encodes["grasp"]):
            while not done:
                done = scene.robot.gripper.operate("close_hand")
                scene.pyrep.step()
                scene.task.step()
        elif all(action == self.action_encodes["grasp_and_lift"]):
            raise NotImplementedError
        else:
            raise NotImplementedError

    def action(self, scene: Scene, action: int):
        assert_action_shape(action, self.action_shape(scene.robot))
        if True:
            done = False
            if not self._detach_before_open:
                self._actuate(action, scene)
            if (
                all(action == self.action_encodes["grasp"])
                and self._attach_grasped_objects
            ):
                # If gripper close action, the check for grasp.
                for g_obj in scene.task.get_graspable_objects():
                    scene.robot.gripper.grasp(g_obj)
            elif all(action == self.action_encodes["release"]):
                # If gripper open action, the check for un-grasp.
                scene.robot.gripper.release()
            else:
                pass
            if self._detach_before_open:
                self._actuate(action, scene)
            if all(action == self.action_encodes["release"]):
                # Step a few more times to allow objects to drop
                for _ in range(10):
                    scene.pyrep.step()
                    scene.task.step()

    def action_shape(self, scene: Scene) -> tuple:
        return (1,)
