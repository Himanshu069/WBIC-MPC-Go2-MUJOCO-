from robot_model import PinModel
import numpy as np

robot = PinModel()

print("=== All frames with position ===")
for i, frame in enumerate(robot.model.frames):
    pos = robot.data.oMf[i].translation
    print(f"  [{i:3d}] {frame.name:<35s}  z={pos[2]:.4f}  pos={np.round(pos,4)}")