using Random, Lux, LinearAlgebra
using NNlib, ConcreteStructs, WeightInitializers, ChainRulesCore
using ComponentArrays
using BenchmarkTools
using OrdinaryDiffEq, Plots, DiffEqFlux, ForwardDiff
using Flux: ADAM, mae, update!
using Flux
using Optimisers
using MAT
using Plots
using ProgressBars
using Zygote: gradient as Zgrad

using SymbolicRegression
using SymbolicUtils

##See the very bottom of this code for the snippets to be run by hand to extract the symbolic expressions
##that are used in Plotting_symbolic.jl.
##Automation of this process is possible but not needed for the sparse KAN-ODEs studied in this work.
##Snippets at the bottom expect a [2, 3, 5], [3, 2, 5] KAN-ODE, although the index values can be modified
##trivially to generalize them.


#this is a fix for an issue with an author's computer. Feel free to remove.
ENV["GKSwstype"] = "100"

# Load the KAN package from https://github.com/vpuri3/KolmogorovArnold.jl
include("src/KolmogorovArnold.jl")
using .KolmogorovArnold

#load the activation function getter (written for this project):
include("Activation_getter.jl")

##########same initializtion as in the KANODE driver##########
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end


function LV(u, p)
    α, β, γ, δ = p
    du = [α * u[1] - β * u[2] * u[1], γ * u[1] * u[2] - δ * u[2]]
    return du
end

#if we prune, then the loss minimum might be before pruning
#but we obviously want to plot the pruned KANODE
#so update this based on training:
#i.e. here the major pruning occurred at ~49000 epochs,
#so we will search for the loss minimum after this point.
#(written as 5000 because reporting was every 10 epochs in this truncated save)
is_pruned=false
loss_minimum_truncation=5000

timestep=0.1
n_plot_save=100
rng = Random.default_rng()
Random.seed!(rng, 0)
tspan = (0.0, 14)
tspan_train=(0.0, 3.5)
u0 = [1, 1]
p_ = Float32[1.5, 1, 1, 3]
prob = ODEProblem(lotka!, u0, tspan, p_)
solution = solve(prob, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = timestep)
end_index=Int64(floor(length(solution.t)*tspan_train[2]/tspan[2]))
t = solution.t #full dataset
t_train=t[1:end_index] #training cut
X = Array(solution)

dir         = @__DIR__
dir         = dir*"/"
cd(dir)
fname       = "LV_kanode"
fname_mlp       = "LV_MLP"
add_path    = "post_plots/"
add_path_kan    = "results_kanode/"
add_path_mlp    = "results_mlp/"
figpath=dir*add_path*"figs"
mkpath(figpath)

##pruned save, provided in this repo
load_file=dir*add_path_kan*"checkpoints/"*fname*"_results_pruned_3nodes_trunc.mat"

loss_list_kan=matread(load_file)["loss"]
loss_list_test_kan=matread(load_file)["loss_test"]

t_kan=matread(load_file)["kan_pred_t"]
u1_kan=matread(load_file)["kan_pred_u1"]
u2_kan=matread(load_file)["kan_pred_u2"]
size_kan=matread(load_file)["size_KAN"]
p_list=matread(load_file)["p_list"]
num_layers=size_kan[1]
layer_width=size_kan[2]
grid_size=size_kan[3]

##re-initialize KAN-ODE from the saved parameters
basis_func = rbf      
normalizer = tanh_fast 
kan1 = Lux.Chain(
    KDense( 2, layer_width, grid_size; use_base_act = true, basis_func, normalizer),
    KDense(layer_width,  2, grid_size; use_base_act = true, basis_func, normalizer),
)

pM , stM  = Lux.setup(rng, kan1) 
pM_data = getdata(ComponentArray(pM))
pM_axis = getaxes(ComponentArray(pM))

##more loading and processing
param_count_prune=layer_width*grid_size*2*2+2*layer_width*2
l_min=minimum(loss_list_kan[5000:end]) 
if is_pruned
    l_min=minimum(loss_list_kan[loss_minimum_truncation:end])
end
idx_min = findfirst(x -> x == l_min, loss_list_kan)
p_curr = p_list[idx_min,1:param_count_prune,1]
pM_    = ComponentArray(p_curr,pM_axis)
pM_new = [pM_.layer_1, pM_.layer_2]






#this calls the code from Activation_getter.jl to compute the individual activation function values (rather than the matrix multiplied outputs):
activations_x, activations_y, activations_second, LV_samples_lay1,=activation_getter(pM_new, kan1, grid_size)

#######symbolic regression:
inputs_lay1=X
activations_lay1_x=activations_x
activations_lay1_y=activations_y
activations_lay2=activations_second
inputs_lay2=LV_samples_lay1

#set the four binary operators
options = SymbolicRegression.Options(
    binary_operators=(+, *, /, -),
    npopulations=20
)

#run EquationSearch for each activation to get the symbolic functions
#After running Plotting.jl, these can be called in the repl individually for clearest results.
#=
#These two have three equations each (first layer)
hallOfFame = EquationSearch(inputs_lay1[1, :]', activations_lay1_x', niterations=100, options=options)
hallOfFame = EquationSearch(inputs_lay1[2, :]', activations_lay1_y', niterations=100, options=options)
#These six have one equation each (second layer)
hallOfFame = EquationSearch(inputs_lay2[1, :]', activations_lay2[1, :]', niterations=100, options=options)
hallOfFame = EquationSearch(inputs_lay2[1, :]', activations_lay2[2, :]', niterations=100, options=options)
hallOfFame = EquationSearch(inputs_lay2[2, :]', activations_lay2[3, :]', niterations=100, options=options)
hallOfFame = EquationSearch(inputs_lay2[2, :]', activations_lay2[4, :]', niterations=100, options=options)
hallOfFame = EquationSearch(inputs_lay2[3, :]', activations_lay2[5, :]', niterations=100, options=options)
hallOfFame = EquationSearch(inputs_lay2[3, :]', activations_lay2[6, :]', niterations=100, options=options)
=#



