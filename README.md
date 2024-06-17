# pyMAC

This code is an implementation designed to enable the usage of the code for MASS [Lee et al, 2019], Generative GaitNet [Park et al, 2022], and Bidirectional GaitNet [Park et al, 2023] solely through Python libraries without the need for C++.

We checked this code works in Python 3.8, ray(rllib) 2.0.1 and Cluster Server (64 CPUs (128 threads) and 1 GPU (RTX 3090) per node).


## Installation

```bash
# Simulation and viwer libraries
pip3 install dartpy 
pip3 install imgui 
pip3 install glfw 
pip3 install numpy 
pip3 install numba 
pip3 install gym 
pip3 install bvh 
pip3 install numpy-quaternion

# DeepRL library 
pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip3 install ray==2.0.1 
pip3 install ray[rllib] 
pip3 install ray[default]

# (Optional) if not working with previous installation
pip3 install "pydantic<2"
```
## Render 

```bash
cd {project folder}/
python3 viewer.py
```

## Learning 

```bash
cd {project folder}/
python3 ray_train.py --config={configuration}
```

## Update Log
- [x] Implementation of imitation learning (deep mimic / scadiver) torque version (2023.03.05 (Verified))

- [ ] Implementation of imitation learning muscle version 

- [ ] Attach video2motion frameworks

- [ ] Fast rendering of muscle 

- [ ] Fast rendering of high-resolution mesh

- [ ] Support MacOS (Converting glfw to glut)