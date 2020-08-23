# Inverse reinforcement learning for dexterous hand manipulation

This code accompanies the paper "Inverse reinforcement learning for dexterous hand manipulation" to enable reproduction of the presented results.

##Installation steps (Linux or MacOS):
### For mujoco-py:
- Install mujoco libraries: https://www.roboti.us/index.html
- `sudo apt install libglew-dev patchelf libosmesa6-dev`

#### If running on Linux
Add to .zshrc:  
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<user>/.mujoco/mujoco200/bin`  
`export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so`

#### If running on MacOS
Need to install gcc for mujoco-py 2.0:  
`brew install gcc`

### Install package
`pip3 install -e . && pip3 install -e git+https://github.com/Jendker/mj_envs.git#egg=mj_envs`

## Results reproduction
Training can be started with:  
`cd inverse_rl_dexterous_hand/training`  
`python training.py --configs IRL`

Please see the 'inverse_rl_dexterous_hand/training/README.md' for the detailed commands for results reproduction.
