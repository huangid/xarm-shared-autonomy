from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import isaaclab
print("isaaclab path:", isaaclab.__file__)
import pkgutil, isaaclab
for m in pkgutil.iter_modules(isaaclab.__path__): print(" submodule:", m.name)
app.close()
