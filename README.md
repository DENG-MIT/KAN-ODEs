# KAN-ODEs
The code is associated with the paper entitled "KAN-ODEs: Kolmogorov-Arnold Network Ordinary Differential Equations for Learning Dynamical Systems and Hidden Physics" ([CMAME](https://www.sciencedirect.com/science/article/pii/S0045782524006522), [Arxiv](https://arxiv.org/abs/2407.04192)).

# Lotka-Volterra

Please find the sources codes in the folder "Lotka-Volterra".

# PDE examples

Please find the source codes in the folder "PDE examples".

# Auxillary Pytorch code

The results in the corresponding manuscript are generated exclusively in Julia. We strongly recommend using the Julia code for speed, convergence, and robustness. However, we provide Pytorch code as well for users who may be interested in experimenting with KAN-ODEs in Python. Please find these in the folder "Lotka-Volterra-Pytorch".

# Optimized LeanKAN implementation 

Please see the following paper and GitHub repository for LeanKAN, our optimized KAN layer structure that includes multiplication: [Neural Networks paper](https://doi.org/10.1016/j.neunet.2025.107883), [Github](https://github.com/DENG-MIT/LeanKAN).

# Citation

If you use the code in your research or if you find our paper useful, please cite [this paper](https://www.sciencedirect.com/science/article/pii/S0045782524006522):

```
@article{koenig2024kanodes,
title = {KAN-ODEs: Kolmogorov–Arnold network ordinary differential equations for learning dynamical systems and hidden physics},
journal = {Computer Methods in Applied Mechanics and Engineering},
volume = {432},
pages = {117397},
year = {2024},
issn = {0045-7825},
doi = {https://doi.org/10.1016/j.cma.2024.117397},
url = {https://www.sciencedirect.com/science/article/pii/S0045782524006522},
author = {Benjamin C. Koenig and Suyong Kim and Sili Deng},
}
```
