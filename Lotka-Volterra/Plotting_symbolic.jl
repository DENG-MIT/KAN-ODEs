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

############run Symbolic_reg.jl, and copy the results into this code directly below. Then run this to generate symbolic plots.
############need to change the loaded KAN-ODE model to generate the dense and sparse contours: see is_pruned below.

function symb_comp_getter(xc, yc)
    ##for forward passes of the symbolic KAN
    ##These are also copied later in the code (~line 220) to plot the activations
    symb_acts=zeros(12)
    symb_acts[1]=0.545*xc-0.204
    symb_acts[4]=-0.605*yc+0.220-0.120/yc
    symb_acts[2]=-0.277*xc+0.794/xc+0.425
    symb_acts[5]=-0.506*yc+1.864-0.108/yc
    symb_acts[3]=0.334*xc-0.635
    symb_acts[6]=-0.780*yc-0.102
    n1c=symb_acts[1]+symb_acts[4]
    n2c=symb_acts[2]+symb_acts[5]
    n3c=symb_acts[3]+symb_acts[6]
    symb_acts[7]=-0.090*n1c^3+0.520n1c^2+0.890*n1c-0.411
    symb_acts[9]=-n2c^2+0.439*n2c/(n2c^2+0.0695)+4.99*n2c-1.652
    symb_acts[11]=2.929*n3c/(n3c^2+1.307*n3c+1.41)+0.684*n3c-0.490
    symb_acts[8]=0.168*n1c^3-0.887n1c^2+2.242*n1c+0.748
    symb_acts[10]=-3.201*n2c/(n2c^2+0.719*n2c+0.675)-n2c
    symb_acts[12]=0.353-1.281*n3c/(n3c^2+0.800)

    dxdt=symb_acts[7]+symb_acts[9]+symb_acts[11]
    dydt=symb_acts[8]+symb_acts[10]+symb_acts[12]
return dxdt, dydt
end


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

is_pruned=false #CHANGE THIS IF PLOTTING DENSE OR SPARSE KAN CONTOURS
#true=sparse kan plotting (pretrained checkpoint provided, see load_file below)
#false=dense kan plotting
loss_minimum_truncation=5000

if is_pruned==true
    load_file=dir*add_path_kan*"checkpoints/"*fname*"_results_pruned_3nodes_trunc.mat"
elseif is_pruned==false
    load_file=dir*add_path_kan*"checkpoints/"*fname*"_results.mat"
end
load_file_mlp=dir*add_path_mlp*"checkpoints/"*fname_mlp*"_results_MLP.mat"

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







loss_list_mlp=matread(load_file_mlp)["loss"]
loss_list_test_mlp=matread(load_file_mlp)["loss_test"]
t_mlp=matread(load_file_mlp)["kan_pred_t"]
p_list_mlp=matread(load_file_mlp)["p_list"]
u1_mlp=matread(load_file_mlp)["kan_pred_u1"]
u2_mlp=matread(load_file_mlp)["kan_pred_u2"]
l_min=minimum(loss_list_mlp)
idx_min = findfirst(x -> x == l_min, loss_list_mlp)
p_mlp = p_list_mlp[idx_min,:,1]
MLP = DiffEqFlux.Chain(DiffEqFlux.Dense(2 => 50, tanh), DiffEqFlux.DiffEqFlux.Dense(50 => 2))
MLP[1].weight.=reshape(p_mlp[1:100], 50, 2)
MLP[1].bias.=reshape(p_mlp[101:150], 50)
MLP[2].weight.=reshape(p_mlp[151:250], 2, 50)
MLP[2].bias.=reshape(p_mlp[251:252], 2)
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
    #skip over the intial dense training period
    l_min=minimum(loss_list_kan[loss_minimum_truncation:end])
end
idx_min = findfirst(x -> x == l_min, loss_list_kan)
p_curr = p_list[idx_min,1:param_count_prune,1]
pM_    = ComponentArray(p_curr,pM_axis)
pM_new = [pM_.layer_1, pM_.layer_2]

#this calls the code from Activation_getter.jl to compute the individual activation function values (rather than the matrix multiplied outputs):
activations_x, activations_y, activations_second, LV_samples_lay1, lay2, K=activation_getter(pM_new, kan1, grid_size)

xsort=sortperm(X[1, :])
ysort=sortperm(X[2, :])

sort_second=Int64.(zeros(lay2.in_dims, K))
for i in 1:layer_width
    sort_second[i, :]=sortperm(LV_samples_lay1[i, :])
end
width=70
height=75

#for activation plots, alpha is a function of the output to input range
beta=1
top_marg=-1.5Plots.mm
bot_marg=-4.5Plots.mm
left_marg=-1Plots.mm
right_marg=-1Plots.mm
sf=2
scalefontsizes()
scalefontsizes(1/sf)

train_node_ = NeuralODE(kan1, tspan, Tsit5(), saveat = timestep); #neural ode
pred_sol_kan = train_node_(u0, ComponentArray(p_curr,pM_axis), stM)[1]


############Now contour plots, for Fig. 4(C)

x = LinRange(0.25, 7.5, 100)
y = LinRange(0.1, 5, 100)
xmesh = x' .* ones(length(y))
ymesh = ones(length(x))' .* y
xdot = zeros(size(xmesh))
ydot = zeros(size(xmesh))
xdot_kan = zeros(size(xmesh))
ydot_kan = zeros(size(xmesh))
xdot_mlp = zeros(size(xmesh))
ydot_mlp = zeros(size(xmesh))

xdot_symb = zeros(size(xmesh))
ydot_symb = zeros(size(xmesh))

xdot_symb_comp = zeros(size(xmesh))
ydot_symb_comp = zeros(size(xmesh))

#evaluate the gradients in the prescribed (x, y) domain:
for i = 1:length(x)
    for j = 1:length(y)
        xdot[i,j], ydot[i,j] = LV([xmesh[i,j], ymesh[i,j]], p_) #actual
        xdot_kan[i,j], ydot_kan[i,j] = kan1([xmesh[i,j], ymesh[i,j]], pM_, stM)[1] #kan
        xdot_mlp[i,j], ydot_mlp[i,j] = MLP([xmesh[i,j], ymesh[i,j]]) #mlp
        xdot_symb[i,j]=1.495*xmesh[i,j]-0.986*xmesh[i,j]*ymesh[i,j] #global symbolic representation (far right in Fig. 4(C))
        ydot_symb[i,j]=.970*xmesh[i,j]*ymesh[i,j]-2.929*ymesh[i,j] 

        xdot_symb_comp[i, j], ydot_symb_comp[i, j]=symb_comp_getter(xmesh[i,j], ymesh[i,j]) #local symbolic representation ("Symbolic KAN-ODE" in Fig. 4(C))
    end
end

#take the differences to plot the error
xdot_kan=xdot-xdot_kan
ydot_kan=ydot-ydot_kan
xdot_symb=xdot-xdot_symb
ydot_symb=ydot-ydot_symb
xdot_mlp=xdot-xdot_mlp
ydot_mlp=ydot-ydot_mlp
xdot_symb_comp=xdot-xdot_symb_comp
ydot_symb_comp=ydot-ydot_symb_comp
xdot_kan=clamp.(xdot_kan, minimum(xdot_mlp), maximum(xdot_mlp))
ydot_kan=clamp.(ydot_kan, minimum(ydot_mlp), maximum(ydot_mlp))

#finally, generate the plots
plt=Plots.contourf(xmesh[1, :], ymesh[:, 1], xdot, clim=(minimum(xdot),maximum(xdot)), c=:amp,size=(300, 225))
xlabel!("x")
ylabel!("y")
scatter!(reduce(hcat,solution.u)'[1:end_index, 1], reduce(hcat,solution.u)'[1:end_index, 2], c=:darkgoldenrod1, markersize=1.)
png(plt, string(dir*add_path, "contour_compare/xdot_actual.png"))

plt=Plots.contourf(xmesh[1, :], ymesh[:, 1], ydot, clim=(minimum(ydot),maximum(ydot)), c=:amp,size=(300, 225))
xlabel!("x")
ylabel!("y")
scatter!(reduce(hcat,solution.u)'[1:end_index, 1], reduce(hcat,solution.u)'[1:end_index, 2], c=:darkgoldenrod1, markersize=1.)

png(plt, string(dir*add_path, "contour_compare/ydot_actual.png"))

plt=Plots.contourf(xmesh[1, :], ymesh[:, 1], xdot_kan, clim=((-1, 1).*maximum(abs, 15)), c=:balance, levels=240,size=(150, 120), dpi=500, linewidth=0, legend=false)
xlabel!("x")
ylabel!("y")
scatter!(reduce(hcat,solution.u)'[1:end_index, 1], reduce(hcat,solution.u)'[1:end_index, 2], c=:darkgoldenrod1, markersize=1.)

png(plt, string(dir*add_path, "contour_compare/xdot_kan.png"))
plt=Plots.contourf(xmesh[1, :], ymesh[:, 1], ydot_kan, clim=((-1, 1).*maximum(abs, 15)), c=:balance, levels=240,size=(150, 120), dpi=500, linewidth=0, legend=false)
xlabel!("x")
ylabel!("y")
scatter!(reduce(hcat,solution.u)'[1:end_index, 1], reduce(hcat,solution.u)'[1:end_index, 2], c=:darkgoldenrod1, markersize=1.)

png(plt, string(dir*add_path, "contour_compare/ydot_kan.png"))

plt=Plots.contourf(xmesh[1, :], ymesh[:, 1], xdot_symb, clim=((-1, 1).*maximum(abs, 15)), c=:balance, levels=240,size=(150, 120), dpi=500, linewidth=0, legend=false)
xlabel!("x")
ylabel!("y")
scatter!(reduce(hcat,solution.u)'[1:end_index, 1], reduce(hcat,solution.u)'[1:end_index, 2], c=:darkgoldenrod1, markersize=1.)

png(plt, string(dir*add_path, "contour_compare/xdot_symb.png"))
plt=Plots.contourf(xmesh[1, :], ymesh[:, 1], ydot_symb, clim=((-1, 1).*maximum(abs, 15)), c=:balance, levels=240,size=(150, 120), dpi=500, linewidth=0, legend=false)
xlabel!("x")
ylabel!("y")
scatter!(reduce(hcat,solution.u)'[1:end_index, 1], reduce(hcat,solution.u)'[1:end_index, 2], c=:darkgoldenrod1, markersize=1.)

png(plt, string(dir*add_path, "contour_compare/ydot_symb.png"))

plt=Plots.contourf(xmesh[1, :], ymesh[:, 1], xdot_symb_comp, clim=((-1, 1).*maximum(abs, 15)), c=:balance, levels=240,size=(150, 120), dpi=500, linewidth=0, legend=false)
xlabel!("x")
ylabel!("y")
scatter!(reduce(hcat,solution.u)'[1:end_index, 1], reduce(hcat,solution.u)'[1:end_index, 2], c=:darkgoldenrod1, markersize=1.)

png(plt, string(dir*add_path, "contour_compare/xdot_symb_comp.png"))
plt=Plots.contourf(xmesh[1, :], ymesh[:, 1], ydot_symb_comp, clim=((-1, 1).*maximum(abs, 15)), c=:balance, levels=240,size=(150, 120), dpi=500, linewidth=0, legend=false)
xlabel!("x")
ylabel!("y")
scatter!(reduce(hcat,solution.u)'[1:end_index, 1], reduce(hcat,solution.u)'[1:end_index, 2], c=:darkgoldenrod1, markersize=1.)

png(plt, string(dir*add_path, "contour_compare/ydot_symb_comp.png"))

plt=Plots.contourf(xmesh[1, :], ymesh[:, 1], xdot_mlp, clim=((-1, 1).*maximum(abs, 15)), c=:balance, levels=240,size=(150, 120), dpi=500, linewidth=0, legend=false)
xlabel!("x")
ylabel!("y")
scatter!(reduce(hcat,solution.u)'[1:end_index, 1], reduce(hcat,solution.u)'[1:end_index, 2], c=:darkgoldenrod1, markersize=1.)

png(plt, string(dir*add_path, "contour_compare/xdot_mlp.png"))
plt=Plots.contourf(xmesh[1, :], ymesh[:, 1], ydot_mlp, clim=((-1, 1).*maximum(abs, 15)), c=:balance, levels=240,size=(150, 120), dpi=500, linewidth=0, legend=false)
xlabel!("x")
ylabel!("y")
scatter!(reduce(hcat,solution.u)'[1:end_index, 1], reduce(hcat,solution.u)'[1:end_index, 2], c=:darkgoldenrod1, markersize=1.)

png(plt, string(dir*add_path, "contour_compare/ydot_mlp.png"))


#########################################
#Up to here, the plotting script can be run on most KANODE and MLP checkpoints.
#Below is the plotting code for the symbolic regression portion of Section A2.
#This code is not general, and expects a KANODE with a size of [2, 3, 5], [3, 2, 5]
#i.e., 6 activations per layer, and 2 layers.
#We have included a pretrained pruned KANODE checkpoint in the results_kanode/checkpoints folder.
#Replacing LV_kanode_results.mat with LV_kanode_results_pruned_3nodes.mat will load this pruned KANODE,
#with which the below code can be uncommented and run.
#########################################

##########Below, the best symbolic fits for each activation (from Symbolic_reg.jl) are copied for plotting. 

symb_acts=zeros(K, layer_width*2*2) #2d input and output
for i in 1:K #hard coded for 3-wide hidden layer
    xc=X[1, i]
    yc=X[2, i]
    symb_acts[i, 1]=0.545*xc-0.204
    symb_acts[i, 4]=-0.605*yc+0.220-0.120/yc
    symb_acts[i, 2]=-0.277*xc+0.794/xc+0.425
    symb_acts[i, 5]=-0.506*yc+1.864-0.108/yc
    symb_acts[i, 3]=0.334*xc-0.635
    symb_acts[i, 6]=-0.780*yc-0.102
    n1c=LV_samples_lay1[1, i]
    n2c=LV_samples_lay1[2, i]
    n3c=LV_samples_lay1[3, i]
    symb_acts[i, 7]=-0.090*n1c^3+0.520n1c^2+0.890*n1c-0.411
    symb_acts[i, 9]=-n2c^2+0.439*n2c/(n2c^2+0.0695)+4.99*n2c-1.652
    symb_acts[i, 11]=2.929*n3c/(n3c^2+1.307*n3c+1.41)+0.684*n3c-0.490
    symb_acts[i, 8]=0.168n1c^3-0.887n1c^2+2.242n1c+0.748
    symb_acts[i, 10]=-3.201*n2c/(n2c^2+0.719*n2c+0.675)-n2c
    symb_acts[i, 12]=0.353-1.281*n3c/(n3c^2+0.800)

end

activations_x_symb=symb_acts[:, 1:3]
activations_y_symb=symb_acts[:, 4:6]
activations_second_symb=symb_acts[:, 7:12]'

##########Finally, generate the subfigures for Fig. 4(A-B)

if layer_width==3 
##only plot the activations for the pruned network.
##otherwise skip directly to contour plots
    #plot first layer, with curve transparency depending on the magnitude of the inputs/outputs
    for i in 1:layer_width
        input_range=X[1, xsort][end]-X[1, xsort][1]
        output_range=maximum(activations_x[xsort, i])-minimum(activations_x[xsort, i])
        acts_scale=output_range/input_range
        trans_curr=tanh(beta*acts_scale) #the more this activation changes the range passing through, the darker the line gets
        plt=Plots.plot(X[1, xsort], activations_x[xsort, i], color = :black,  legend=false, size = (width, height), dpi=500, grid=false,xticks=[ceil(minimum(X[1, xsort]), sigdigits=1), floor(maximum(X[1, xsort]), sigdigits=1)], yticks=false, alpha=trans_curr, framestyle = :box, bottom_margin=bot_marg, top_margin=top_marg, thickness_scaling=sf,xguidefontsize=18, left_margin=left_marg, right_margin=right_marg )
        png(plt, string(dir*add_path, "activation_plots/X", string(i), ".png"))

        input_range=X[2, ysort][end]-X[2, ysort][1]
        output_range=maximum(activations_y[ysort, i])-minimum(activations_y[ysort, i])
        acts_scale=output_range/input_range
        trans_curr=tanh(beta*acts_scale) #the more this activation changes the range passing through, the darker the line gets
        plt=Plots.plot(X[2, ysort], activations_y[ysort, i], color = :black,  legend=false, size = (width, height), dpi=500, grid=false,xticks=[ceil(minimum(X[2, ysort]), sigdigits=1), floor(maximum(X[2, ysort]), sigdigits=1)], yticks=false, alpha=trans_curr, framestyle = :box, bottom_margin=bot_marg, top_margin=top_marg, thickness_scaling=sf,xguidefontsize=18, left_margin=left_marg, right_margin=right_marg )
        png(plt, string(dir*add_path, "activation_plots/Y", string(i), ".png"))

        input_range=X[1, xsort][end]-X[1, xsort][1]
        output_range=maximum(activations_x_symb[:, i])-minimum(activations_x_symb[:, i])
        acts_scale=output_range/input_range
        trans_curr=tanh(beta*acts_scale) #the more this activation changes the range passing through, the darker the line gets
        plt=Plots.plot(X[1, xsort], activations_x_symb[xsort, i], color = :black,  legend=false, size = (width, height), dpi=500, grid=false,xticks=[ceil(minimum(X[1, xsort]), sigdigits=1), floor(maximum(X[1, xsort]), sigdigits=1)], yticks=false, alpha=trans_curr, framestyle = :box, bottom_margin=bot_marg, top_margin=top_marg, thickness_scaling=sf,xguidefontsize=18, left_margin=left_marg, right_margin=right_marg )
        png(plt, string(dir*add_path, "activation_plots/X_symb_", string(i), ".png"))

        input_range=X[2, ysort][end]-X[2, ysort][1]
        output_range=maximum(activations_y_symb[:, i])-minimum(activations_y_symb[:, i])
        acts_scale=output_range/input_range
        trans_curr=tanh(beta*acts_scale) #the more this activation changes the range passing through, the darker the line gets
        plt=Plots.plot(X[2, ysort], activations_y_symb[ysort, i], color = :black,  legend=false, size = (width, height), dpi=500, grid=false,xticks=[ceil(minimum(X[2, ysort]), sigdigits=1), floor(maximum(X[2, ysort]), sigdigits=1)], yticks=false, alpha=trans_curr, framestyle = :box, bottom_margin=bot_marg, top_margin=top_marg, thickness_scaling=sf,xguidefontsize=18, left_margin=left_marg, right_margin=right_marg )
        png(plt, string(dir*add_path, "activation_plots/Y_symb_", string(i), ".png"))
    end

    #plot second layer, with curve transparency depending on the magnitude of the inputs/outputs
    for i in 1:layer_width
        input_range=LV_samples_lay1[i, sort_second[i, :]][end]-LV_samples_lay1[i, sort_second[i, :]][1]
        output_range=maximum(activations_second[2*i-1, sort_second[i, :]])-minimum(activations_second[2*i-1, sort_second[i, :]])
        acts_scale=output_range/input_range
        trans_curr=tanh(beta*acts_scale) #the more this activation changes the range passing through, the darker the line gets
        plt=Plots.plot(LV_samples_lay1[i, sort_second[i, :]], activations_second[2*i-1, sort_second[i, :]], color = :black, legend=false, size = (width, height), dpi=500, grid=false,xticks=[ceil(minimum(LV_samples_lay1[i, sort_second[i, :]]), sigdigits=1), floor(maximum(LV_samples_lay1[i, sort_second[i, :]]), sigdigits=1)], yticks=false, alpha=trans_curr, framestyle = :box, bottom_margin=bot_marg, top_margin=top_marg, thickness_scaling=sf,xguidefontsize=18, left_margin=left_marg, right_margin=right_marg )
        png(plt, string(dir*add_path, "activation_plots/second_", string(i), "_to_X.png"))

        input_range=LV_samples_lay1[i, sort_second[i, :]][end]-LV_samples_lay1[i, sort_second[i, :]][1]
        output_range=maximum(activations_second[2*i, sort_second[i, :]])-minimum(activations_second[2*i, sort_second[i, :]])
        acts_scale=output_range/input_range
        trans_curr=tanh(beta*acts_scale) #the more this activation changes the range passing through, the darker the line gets
        plt=Plots.plot(LV_samples_lay1[i, sort_second[i, :]], activations_second[2*i, sort_second[i, :]], color = :black, legend=false, size = (width, height), dpi=500, grid=false,xticks=[ceil(minimum(LV_samples_lay1[i, sort_second[i, :]]), sigdigits=1), floor(maximum(LV_samples_lay1[i, sort_second[i, :]]), sigdigits=1)], yticks=false, alpha=trans_curr, framestyle = :box, bottom_margin=bot_marg, top_margin=top_marg, thickness_scaling=sf,xguidefontsize=18, left_margin=left_marg, right_margin=right_marg )
        png(plt, string(dir*add_path, "activation_plots/second_", string(i), "_to_Y.png"))

        input_range=LV_samples_lay1[i, sort_second[i, :]][end]-LV_samples_lay1[i, sort_second[i, :]][1]
        output_range=maximum(activations_second_symb[2*i-1, sort_second[i, :]])-minimum(activations_second_symb[2*i-1, :])
        acts_scale=output_range/input_range
        trans_curr=tanh(beta*acts_scale) #the more this activation changes the range passing through, the darker the line gets
        plt=Plots.plot(LV_samples_lay1[i, sort_second[i, :]], activations_second_symb[2*i-1, sort_second[i, :]], color = :black, legend=false, size = (width, height), dpi=500, grid=false,xticks=[ceil(minimum(LV_samples_lay1[i, sort_second[i, :]]), sigdigits=1), floor(maximum(LV_samples_lay1[i, sort_second[i, :]]), sigdigits=1)], yticks=false, alpha=trans_curr, framestyle = :box, bottom_margin=bot_marg, top_margin=top_marg, thickness_scaling=sf,xguidefontsize=18, left_margin=left_marg, right_margin=right_marg )
        png(plt, string(dir*add_path, "activation_plots/second_symb_", string(i), "_to_X.png"))

        input_range=LV_samples_lay1[i, sort_second[i, :]][end]-LV_samples_lay1[i, sort_second[i, :]][1]
        output_range=maximum(activations_second[2*i, sort_second[i, :]])-minimum(activations_second[2*i, sort_second[i, :]])
        acts_scale=output_range/input_range
        trans_curr=tanh(beta*acts_scale) #the more this activation changes the range passing through, the darker the line gets
        plt=Plots.plot(LV_samples_lay1[i, sort_second[i, :]], activations_second_symb[2*i, sort_second[i, :]], color = :black, legend=false, size = (width, height), dpi=500, grid=false,xticks=[ceil(minimum(LV_samples_lay1[i, sort_second[i, :]]), sigdigits=1), floor(maximum(LV_samples_lay1[i, sort_second[i, :]]), sigdigits=1)], yticks=false, alpha=trans_curr, framestyle = :box, bottom_margin=bot_marg, top_margin=top_marg, thickness_scaling=sf,xguidefontsize=18, left_margin=left_marg, right_margin=right_marg )
        png(plt, string(dir*add_path, "activation_plots/second_symb_", string(i), "_to_Y.png"))

    end
end






