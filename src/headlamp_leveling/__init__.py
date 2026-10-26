"""IMU-based automatic headlamp leveling.

Sub-packages:
    sim       vehicle pitch / road / driving / IMU simulation
    data      trip storage, windowing and the PyTorch dataset
    models    physics baseline and neural estimators
    training  training loop, metrics and evaluation
    control   leveling controller that turns pitch estimates into actuator commands
"""

__version__ = "0.1.0"

# Input channel order used everywhere (simulator, real-log loader, models).
CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz", "v_wheel")
G = 9.80665
