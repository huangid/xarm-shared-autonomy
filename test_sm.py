from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from isaaclab.devices import Se3SpaceMouse
import time
sm = Se3SpaceMouse(pos_sensitivity=0.4, rot_sensitivity=0.8)
sm.reset()
for i in range(40): print(sm.advance()); time.sleep(0.05)
app.close()
