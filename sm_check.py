from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from isaaclab.devices.spacemouse import Se3SpaceMouse, Se3SpaceMouseCfg
import time
sm = Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=0.4, rot_sensitivity=0.8, gripper_term=True, sim_device="cpu"))
print(sm)
sm.reset()
for i in range(200): print(sm.advance()); time.sleep(0.05)
app.close()
