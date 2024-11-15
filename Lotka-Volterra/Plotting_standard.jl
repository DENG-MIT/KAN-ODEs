using Random, Lux, LinearAlgebra
using NNlib, ConcreteStructs, WeightInitializers, ChainRulesCore
using ComponentArrays
using BenchmarkTools
using OrdinaryDiffEq, Plots, DiffEqFlux, ForwardDiff
using Flux: Adam, mae, update!
using Flux
using Optimisers
using MAT
using Plots
using ProgressBars
using Zygote: gradient as Zgrad
using Zygote

using SymbolicRegression
using SymbolicUtils

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

#if we are plotting the pruned case, turn is_pruned=true 
#and truncate the saved loss profile after the pruning event
#we are plotting the minimum loss KAN-ODE, but for many pruned cases the loss minimum is before pruning
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

##loading most recent saves 

load_file=dir*add_path_kan*"checkpoints/"*fname*"_results.mat"
load_file_mlp=dir*add_path_mlp*"checkpoints/"*fname_mlp*"_results_MLP.mat"

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
l_min=minimum(loss_list_kan)
if is_pruned
    l_min=minimum(loss_list_kan[loss_minimum_truncation:end])
end
idx_min = findfirst(x -> x == l_min, loss_list_kan)
p_curr = p_list[idx_min,1:param_count_prune,1]
pM_    = ComponentArray(p_curr,pM_axis)
pM_new = [pM_.layer_1, pM_.layer_2]

#this calls the code from Activation_getter.jl to compute the individual activation function values (rather than the matrix multiplied outputs):
activations_x, activations_y, activations_second, LV_samples_lay1, lay2, K=activation_getter(pM_, pM_new, kan1, grid_size)


xsort=sortperm(X[1, :])
ysort=sortperm(X[2, :])

sort_second=Int64.(zeros(lay2.in_dims, K))
for i in 1:layer_width
    sort_second[i, :]=sortperm(LV_samples_lay1[i, :])
end
width=70
height=75


sf=2
scalefontsizes()
scalefontsizes(1/sf)



l_min=minimum(loss_list_kan)
if is_pruned
    l_min=minimum(loss_list_kan[loss_minimum_truncation:end])
end
idx_min = findfirst(x -> x == l_min, loss_list_kan)
p_curr = p_list[idx_min,1:param_count_prune,1]
train_node_ = NeuralODE(kan1, tspan, Tsit5(), saveat = timestep); #neural ode
pred_sol_kan = train_node_(u0, ComponentArray(p_curr,pM_axis), stM)[1]

plt=scatter(solution.t[1:end_index],reduce(hcat,solution.u)'[1:end_index, 1], margin=3Plots.mm, legend=(0.6,0.97), alpha = 0.75, label = "Train x",ylims=(0,10),dpi=1000,size=(475, 200), grid=false, color=:mediumseagreen)
scatter!(solution.t[1:end_index], reduce(hcat,solution.u)'[1:end_index, 2], alpha = 0.75, label = "Train y", markershape=:pentagon, color=:cornflowerblue)

plot!(pred_sol_kan.t, reduce(hcat,pred_sol_kan.u)'[:, 2], linewidth=2, label="KAN-ODE y", color=:midnightblue)
plot!(pred_sol_kan.t, reduce(hcat,pred_sol_kan.u)'[:, 1], linewidth=2, label="KAN-ODE x", color=:darkgreen)

plot!(solution.t[end_index+1:end],reduce(hcat,solution.u)'[end_index+1:end, 1],  label = "Test x", linestyle=:dot, color=:mediumseagreen, linewidth=3, thickness_scaling = 1)
plot!(solution.t[end_index+1:end], reduce(hcat,solution.u)'[end_index+1:end, 2],  label = "Test y", linestyle=:dot, color=:cornflowerblue, linewidth=3, thickness_scaling = 1)
vline!([3.5], color=:darkorange1, label = "Train/test split", legend_columns=2, linewidth=2, thickness_scaling = 1, linestyle=:dot)
xlabel!("Time [s]")
ylabel!("x,y ")
png(plt, string(figpath, "/paper_reconstruction.png"))


#Plot the KANODE and MLP loss profiles, starting with MLP loading:

load_file_mlp=dir*add_path_mlp*"checkpoints/"*fname_mlp*"_results_MLP.mat"
loss_list_mlp=matread(load_file_mlp)["loss"]
loss_list_test_mlp=matread(load_file_mlp)["loss_test"]
t_mlp=matread(load_file_mlp)["kan_pred_t"]
p_list_mlp=matread(load_file_mlp)["p_list"]
u1_mlp=matread(load_file_mlp)["kan_pred_u1"]
u2_mlp=matread(load_file_mlp)["kan_pred_u2"]
l_min=minimum(loss_list_mlp)
idx_min = findfirst(x -> x == l_min, loss_list_mlp)
p_mlp = p_list_mlp[idx_min,:,1]
#MLP = DiffEqFlux.Chain(DiffEqFlux.Dense(2 => 50, tanh), DiffEqFlux.DiffEqFlux.Dense(50 => 2))

#MLP[1].init_weight.=reshape(p_mlp[1:100], 50, 2)
#MLP[1].bias.=reshape(p_mlp[101:150], 50)
#MLP[2].weight.=reshape(p_mlp[151:250], 2, 50)
#MLP[2].bias.=reshape(p_mlp[251:252], 2)

plt=Plots.plot(loss_list_kan, yaxis=:log, label="KAN train", dpi=600, size=(325, 290), xticks=LinRange(0, round(length(loss_list_kan), sigdigits=1), 3), grid=false)
plot!(ylims=[minimum(loss_list_kan)*0.9, maximum([maximum(loss_list_kan), maximum(loss_list_mlp), maximum(loss_list_test_kan), maximum(loss_list_test_mlp)])*1.1])
plot!(yticks=[1e-6, 1e-4, 1e-2, 1], margin=3.5Plots.mm)
plot!(loss_list_test_kan, yaxis=:log, label="KAN test")
xlabel!("Epoch")
ylabel!("Loss")
png(plt, string(figpath, "/loss_kan.png"))

#Plot the MLP loss profile
plt=Plots.plot(loss_list_mlp, yaxis=:log, label="MLP train", dpi=600, size=(325, 290), xticks=LinRange(0, round(length(loss_list_mlp), sigdigits=1), 3), grid=false)
plot!(ylims=[minimum(loss_list_kan)*0.9, maximum([maximum(loss_list_kan), maximum(loss_list_mlp), maximum(loss_list_test_kan), maximum(loss_list_test_mlp)])*1.1])
plot!(loss_list_test_mlp, yaxis=:log, label="MLP test")
plot!(yticks=[1e-6, 1e-4, 1e-2, 1], margin=3.5Plots.mm)
xlabel!("Epoch")
ylabel!("Loss")
png(plt, string(figpath, "/loss_mlp.png"))




