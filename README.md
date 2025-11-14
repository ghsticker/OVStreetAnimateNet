# Deep learning solver unites SDGSAT-1 observations and Navier-Stokes theory for oceanic vortex streets

This repository contains the implementation for the paper: **Deep learning solver unites SDGSAT-1 observations and Navier-Stokes theory for oceanic vortex streets**.

Our work introduces a novel self-supervised neural network framework that integrates real-world satellite observations with the physical principles of Navier-Stokes equations to simulate oceanic vortex streets. This model leverages data from the world's first scientific satellite for sustainable development goals (SDGSAT-1) to achieve high-fidelity simulations with lower computational costs compared to traditional methods.

## Installation

1. **Create a Conda environment:**

   ```bash
   conda create --name ocean_vortex_env python=3.8
   conda activate ocean_vortex_env
   ```

2. **Install dependencies:**
   This project requires PyTorch 1.8.1. Please install it following the official instructions for your specific hardware (CPU/GPU).

   ```bash
   # Example for a specific CUDA version, please adjust to your setup
   conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
   
   # Install other packages
   pip install matplotlib numpy opencv-python
   ```

## Usage

### Interactive Demo

Run the interactive demo to visualize fluid simulations with a pre-trained model. The following command runs a simulation with specific physical parameters (viscosity `mu`, density `rho`, and time step `dt`).

```bash
python demo_interactive.py --mu=0.1 --rho=4 --dt=4
```

### Training

To train your own model from scratch, use the `train.py` script. You can specify physical parameters and the integration scheme. For example, to train a model using the IMEX (Implicit-Explicit) integration scheme:

```bash
python train.py --mu=0.1 --rho=4 --dt=4 --integrator=imex
```

For more information on available training parameters, you can run:

```bash
python train.py --help
```

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{gao2024deep,
  title={Deep learning solver unites SDGSAT-1 observations and Navier--Stokes theory for oceanic vortex streets},
  author={Gao, He and Huang, Baoxiang and Chen, Ge and Xia, Linghui and Radenkovic, Milena},
  journal={Remote Sensing of Environment},
  volume={315},
  pages={114425},
  year={2024},
  publisher={Elsevier}
}
```
