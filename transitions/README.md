# Transitions

These scripts animate transitions between different sets of particles.

They rely on the [EnergyFlow](https://github.com/thaler-lab/EnergyFlow) package to determine the optimal transport plans between key frames.

These scripts were written by Austine Zhang during a Brown University UTRA in Fall '25.

## Running

You'll need to set up a virtual environment

```
# Just the first time
python -m venv ani.venv

# Every time
source ani.venv/bin/activate
pip install requirements.txt
```