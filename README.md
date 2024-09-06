# KAN-ODEs
The code is associated with the paper entitled "KAN-ODEs: Kolmogorov-Arnold Network Ordinary Differential Equations for Learning Dynamical Systems and Hidden Physics"

# Lotka-Volterra
The Lotka-Volterra directory contains three julia codes:

- LV_driver_KANODE.jl runs KANODE training on the LV example, with customizable KAN size.
- LV_driver_MLP.jl runs MLP-NODE training on the LV example, with customizable MLP size.
- Plotting.jl includes code used to generate Figs. 3 and 4 in the main manuscript. Please see comments for more details on the plotting contained here.

trend_plotter.py is the simple code used to generate Fig. 3(C).

Direct training results and plots are saved in the two corresponding /results_kanode/ and /results_mlp/ directories. Plotting.jl saves into /post_plots/.

- These three directories were uploaded to github with all figures already generated (and checkpoints saved). 
- These runs were done shortly before upload to verify the functionality of the code, and are not the exact runs reported in the manuscript.
- The exception to this are the results in /post_plots/activation_plots/ and /post_plots/contour_compare/, which correspond directly to the subfigures used in Fig. 4 of the manuscript (the checkpoint for this run is not uploaded due to its size, but is available on request).

As noted in the /src/ folder, the KolmogorovArnold.jl package used here is from https://github.com/vpuri3/KolmogorovArnold.jl, a source which is also referenced in the manuscript.


# PDE examples

Please find the source codes in the folder "PDE examples".

# Citation

To be included after acceptance.
