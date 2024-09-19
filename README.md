# KAN-ODEs
The code is associated with the paper entitled "KAN-ODEs: Kolmogorov-Arnold Network Ordinary Differential Equations for Learning Dynamical Systems and Hidden Physics" ([CMAME](https://www.sciencedirect.com/science/article/pii/S0045782524006522), [Arxiv](https://arxiv.org/abs/2407.04192)).

# Lotka-Volterra
The Lotka-Volterra directory contains six julia codes:

- LV_driver_KANODE.jl runs KAN-ODE training on the LV example, with customizable KAN size and options for sparsification and pruning.
- LV_driver_MLP.jl runs MLP-NODE training on the LV example, with customizable MLP size.
- Activation_getter.jl is a function called throughout training and plotting to extract individual activation functions from the global KAN-ODE.
- Plotting_standard.jl includes code used to generate Fig. 3(A, B1, B2) in the manuscript, using saved checkpoints from the above two codes.
- Symbolic_reg.jl takes a pruned KAN-ODE save and breaks it into individual activations, allowing for sparse regression on each activation function (sparse regression by direct function calling in the REPL).
- Then, Plotting_symbolic.jl generates the subplots of Fig. 4, using saved checkpoints from KAN-ODE and MLP-NODE training as well as symbolic results copied from Symbolic_reg.jl.

Additionally, trend_plotter.py is the simple code used to generate Fig. 3(C), via hand-saved results from the two driver codes.

Direct training results and plots are saved in the two corresponding /results_kanode/ and /results_mlp/ directories. Plotting.jl saves into /post_plots/.

- The LV example was uploaded to github with all figures already generated (and checkpoints saved). 
- These runs were done shortly before upload to verify the functionality of the code, and are not the exact runs reported in the manuscript.
- The exception to this are the figures in /post_plots/activation_plots/ and /post_plots/contour_compare/, which correspond directly to the subfigures used in Fig. 4 of the manuscript. The checkpoint corresponding to these plots is saved in /results_kanode/checkpoints/LV_kanode_results_pruned_3nodes_trunc.mat.

**We recommend starting from LV_driver_KANODE.jl. Many of the other codes provided here are tailored specifically for the results shown in the manuscript, and are included largely just for replicability of our results. For general use, the training loop of LV_driver_KANODE.jl will be of most use, and  includes its own plotting and visualization loops during the training cycle (saved to /results_kanode/figs).**

As noted in the /src/ folder, the KolmogorovArnold.jl package used here is from https://github.com/vpuri3/KolmogorovArnold.jl, a source which is also referenced in the manuscript.

# PDE examples

Please find the source codes in the folder "PDE examples".

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
