
# Lotka-Volterra
This directory contains six julia codes. **LV_driver_KANODE.jl is the main training driver, and includes plotting into /results_kanode/figs/. The remaining codes are included for replication of the results in the manuscript, but are not essential for KAN-ODE training. We recommend adaptation of LV_driver_KANODE.jl for future applications.**


- LV_driver_KANODE.jl runs KAN-ODE training on the LV example, with customizable KAN size and options for sparsification and pruning.
- LV_driver_MLP.jl runs MLP-NODE training on the LV example, with customizable MLP size.
- Activation_getter.jl is a function called throughout pruning and plotting to extract individual activation functions from the global KAN-ODE.
- Plotting_standard.jl includes code used to generate Fig. 3(A, B1, B2) in the manuscript, using saved checkpoints from the above two codes.
- Symbolic_reg.jl takes a pruned KAN-ODE save and breaks it into individual activations, allowing for sparse regression on each activation function (sparse regression by direct function calling in the REPL).
- Then, Plotting_symbolic.jl generates the subplots of Fig. 4, using saved checkpoints from KAN-ODE and MLP-NODE training as well as symbolic results copied from Symbolic_reg.jl.

Additionally, trend_plotter.py is the simple code used to generate Fig. 3(C), via hand-saved results from the two driver codes.

Direct training results and plots are saved in the two corresponding /results_kanode/ and /results_mlp/ directories. Plotting.jl saves into /post_plots/.

- The LV example was uploaded to github with all figures already generated (and checkpoints saved). 
- These runs were done shortly before upload to verify the functionality of the code, and are not the exact runs reported in the manuscript.
- The exception to this are the figures in /post_plots/activation_plots/ and /post_plots/contour_compare/, which correspond directly to the subfigures used in Fig. 4 of the manuscript. The checkpoint corresponding to these plots is saved in /results_kanode/checkpoints/LV_kanode_results_pruned_3nodes_trunc.mat.

**We recommend starting from LV_driver_KANODE.jl. Many of the other codes provided here are tailored specifically for the results shown in the manuscript, and are included largely just for replicability of our results. For general use, the training loop of LV_driver_KANODE.jl will be of most use, and  includes its own plotting and visualization loops during the training cycle (saved to /results_kanode/figs/).**

As noted in the /src/ folder, the KolmogorovArnold.jl package used here is from https://github.com/vpuri3/KolmogorovArnold.jl, a source which is also referenced in the manuscript.

Julia 1.11.1 was used to run LV_driver_KANODE.jl in the current repository. Package information is available in Project.toml and Manifest.toml.