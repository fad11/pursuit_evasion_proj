import time
from pathlib import Path

import pybullet as p
import pybullet_data


def main():
    client = p.connect(p.GUI)
    if client < 0:
        raise RuntimeError("Could not connect to PyBullet in GUI mode")

    try:
        p.setGravity(0, 0, 0, physicsClientId=client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        p.loadURDF("plane.urdf", [0, 0, 0], physicsClientId=client)
        urdf_path = Path(__file__).resolve().parent / "assets" / "drone.urdf"
        p.loadURDF(str(urdf_path), [0, 0, 0], physicsClientId=client)

        for _ in range(100):
            p.stepSimulation(physicsClientId=client)
            time.sleep(1/240)

        print("Completed 100 simulation steps; close the PyBullet window to exit.")
        while p.isConnected(client):
            time.sleep(0.1)
    finally:
        if p.isConnected(client):
            p.disconnect(physicsClientId=client)


if __name__ == "__main__":
    main()
