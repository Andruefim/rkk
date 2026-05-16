import pybullet as pb
import pybullet_data as pbd
import time
import math
import numpy as np
import os

pb.connect(pb.DIRECT)
pb.setAdditionalSearchPath(pbd.getDataPath())
robot_id = pb.loadURDF("engine/data/humanoid/humanoid.urdf", [0, 0, 1])

# Find joints
n_joints = pb.getNumJoints(robot_id)
spine_idx = -1
head_idx = -1
for i in range(n_joints):
    info = pb.getJointInfo(robot_id, i)
    name = info[1].decode("utf-8")
    if name == "spine":
        spine_idx = i
    if name == "head":
        head_idx = i

def get_head_x():
    st = pb.getLinkState(robot_id, head_idx)
    return st[4][0]

print("Neutral head X:", get_head_x())

# Set spine pitch with POSITIVE Z rotation
q_pos = pb.getQuaternionFromEuler([0, 0, 1.0])
pb.setJointMotorControlMultiDof(robot_id, spine_idx, pb.POSITION_CONTROL, targetPosition=list(q_pos), force=[500, 500, 500])
for _ in range(100): pb.stepSimulation()
print("Positive Z rotation head X:", get_head_x())

# Set spine pitch with NEGATIVE Z rotation
q_neg = pb.getQuaternionFromEuler([0, 0, -1.0])
pb.setJointMotorControlMultiDof(robot_id, spine_idx, pb.POSITION_CONTROL, targetPosition=list(q_neg), force=[500, 500, 500])
for _ in range(100): pb.stepSimulation()
print("Negative Z rotation head X:", get_head_x())

pb.disconnect()
