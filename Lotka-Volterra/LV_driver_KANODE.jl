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


#this is a fix for an issue with an author's computer. Feel free to remove.
ENV["GKSwstype"] = "100"

is_restart=false
is_prune=false

sparse_on=0 #set this to 1 and see reg_loss() and prune() functions if sparsity is desired 

# Directories
dir         = @__DIR__
dir         = dir*"/"
cd(dir)
fname       = "LV_kanode"
add_path    = "results_kanode/"
figpath=dir*add_path*"figs"
ckptpath=dir*add_path*"checkpoints"
mkpath(figpath)
mkpath(ckptpath)

# Load the KAN package from https://github.com/vpuri3/KolmogorovArnold.jl
include("src/KolmogorovArnold.jl")
using .KolmogorovArnold

#define LV
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end


function prune(p, kan_curr, layer_width, grid_size, pM_axis, theta=1e-2)
    #pruning function used to sparsify KAN-ODEs (i.e. delete negligible connections)
    #theta corresponds to gamma_pr in the manuscript (value of 1e-2)
    #not optimized - only runs a few times per training cycle, so extreme efficiency is not important

    #load current save
    load_file=dir*add_path*"checkpoints/"*fname*"_results.mat"
    p_list_=matread(load_file)["p_list"]
    p_list=[]
    for j = 1:size(p_list_,1)
        append!(p_list, [p_list_[j, :, 1]])
    end
    p=p_list_[end, :, 1]
    l=matread(load_file)["loss"]
    l_test=matread(load_file)["loss_test"]

    ##long block of code here extracts the intermediate values in the KAN
    ##because a forward pass returns only the matrix multiplication results,
    ##and does not let us dig deeper into the actual activations.
    pM_= ComponentArray(p,pM_axis)
    pM_new = [pM_.layer_1, pM_.layer_2]
    lay1=kan_curr[1]
    lay2=kan_curr[2]
    st=stM[1]
    pc1=pM_new[1].C
    pc2=pM_new[2].C
    pc1x=pc1[:, 1:grid_size]
    pc1y=pc1[:, grid_size+1:end]
    pw1=pM_new[1].W
    pw2=pM_new[2].W
    pw1x=pw1[:, 1]
    pw1y=pw1[:, 2]

        
    size_in  = size(X)                          # [I, ..., batch,]
    size_out = (lay1.out_dims, size_in[2:end]...,) # [O, ..., batch,]

    x = reshape(X, lay1.in_dims, :)
    K = size(x, 2)

    x_norm = lay1.normalizer(x)              # ∈ [-1, 1]
    x_resh = reshape(x_norm, 1, :)                        # [1, K]
    basis  = lay1.basis_func(x_resh, st.grid, lay1.denominator) # [G, I * K]
    basisx=basis[:, 1:2:end] #odds are x
    basisy=basis[:, 2:2:end] #evens are y
    activations_x=basisx'*pc1x'
    activations_y=basisy'*pc1y'
    activations_x+=lay1.base_act.(x[1, :]).*pw1x'
    activations_y+=lay1.base_act.(x[2, :]).*pw1y'

    ##second layer
    LV_samples_lay1=kan_curr[1](X, pM_.layer_1, stM[1])[1] #this is the activation function results for the first layer

        
    x = reshape(LV_samples_lay1, lay2.in_dims, :)
    K = size(x, 2)

    x_norm = lay2.normalizer(x)              # ∈ [-1, 1]
    x_resh = reshape(x_norm, 1, :)                        # [1, K]
    basis  = lay2.basis_func(x_resh, st.grid, lay2.denominator) # [G, I * K]
    activations_second=zeros(lay2.in_dims*2, K)

    ##we now have the first and second layer activations (activations_x and activations_y for first, activations_second for second)
    ##now delete any negligible nodes:
    
    for i in 1:lay2.in_dims
        basis_curr=basis[:, i:lay2.in_dims:end]
        pc_curr=pc2[:, (i-1)*grid_size+1:i*grid_size]
        activations_curr=basis_curr'*pc_curr'
        activations_curr+=(lay2.base_act.(x[i, :]).*pw2[:, i]')
        activations_second[2*i-1:2*i, :]=activations_curr'
    end
    nodes_to_eval=1:layer_width
    nodes_to_keep=[]
    for i in nodes_to_eval
        input_score=maximum([maximum(abs.(activations_x[i, :])), maximum(abs.(activations_y[i, :]))])
        output_score=maximum(maximum(abs.(activations_second[(i-1)*2+1:(i-1)*2+2, :])))
        if minimum([input_score, output_score])>theta
            append!(nodes_to_keep, i)
        end
    end


    ##re-initialize KAN, but with the smaller size

    layer_width=length(nodes_to_keep)
    kan1 = Lux.Chain(
        KDense( 2, layer_width, grid_size; use_base_act = true, basis_func, normalizer),
        KDense(layer_width,  2, grid_size; use_base_act = true, basis_func, normalizer),
    )

    ##and save only those parameters into it
    pm1c=pM.layer_1.C[nodes_to_keep, :]
    pm1w=pM.layer_1.W[nodes_to_keep, :]
    pm2c=zeros(2, grid_size*layer_width)
    count=0
    for i in nodes_to_keep
        count+=1
        pm2c[:, (count-1)*grid_size+1:count*grid_size]=pM.layer_2.C[:, (i-1)*grid_size+1:i*grid_size]
    end
    pm2w=pM.layer_2.C[:, nodes_to_keep]

    pM_new=(layer_1=(C=pm1c, W=pm1w), layer_2=(C=pm2c, W=pm2w))
    return pM_new, kan1, layer_width
end

#data generation parameters
timestep=0.1
n_plot_save=1000
rng = Random.default_rng()
Random.seed!(rng, 0)
tspan = (0.0, 14)
tspan_train=(0.0, 3.5)
u0 = [1, 1]
p_ = Float32[1.5, 1, 1, 3]
prob = ODEProblem(lotka!, u0, tspan, p_)

#generate training data, split into train/test
solution = solve(prob, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = timestep)
end_index=Int64(floor(length(solution.t)*tspan_train[2]/tspan[2]))
t = solution.t #full dataset
t_train=t[1:end_index] #training cut
X = Array(solution)
Xn = deepcopy(X) 


basis_func = rbf      # rbf, rswaf
normalizer = tanh_fast # sigmoid(_fast), tanh(_fast), softsign

# Define KAN-ODEs
###layer_width and grid_size can be modified here to replicate the testing in section A2 of the manuscript

num_layers=2 #defined just to save into .mat for plotting
layer_width=10
grid_size=5
kan1 = Lux.Chain(
    KDense( 2, layer_width, grid_size; use_base_act = true, basis_func, normalizer),
    KDense(layer_width,  2, grid_size; use_base_act = true, basis_func, normalizer),
)
pM , stM  = Lux.setup(rng, kan1)

#Can restart from a previous training result
if is_restart==true
    load_file=dir*add_path*"checkpoints/"*fname*"_results.mat"
    p_list_=matread(load_file)["p_list"]
    p_list=[]
    for j = 1:size(p_list_,1)
        append!(p_list, [p_list_[j, :, 1]])
    end
    p=p_list_[end, :, 1]
    l=matread(load_file)["loss"]
    l_test=matread(load_file)["loss_test"]
else 
    l = []
    l_test=[]
    p_list = []
end

pM_axis = getaxes(ComponentArray(pM))

###pruning occurs when the code starts running.
###So to prune at epoch=1000, run it once with 1000 epochs
###then restart the driver with is_restart=true, and is_prune=true
###and when re-initializing, the below snippet will run pruning. 
if is_prune==true
    pM, kan1, layer_width=prune(p, kan1, layer_width, grid_size, pM_axis, 1e-1)
end


pM_data = getdata(ComponentArray(pM))
pM_axis = getaxes(ComponentArray(pM))
p = (deepcopy(pM_data))


###########KAN is fully defined. Now, define the KAN ODE wrapped around it.

train_node = NeuralODE(kan1, tspan_train, Tsit5(), saveat = t_train); 
train_node_test = NeuralODE(kan1, tspan, Tsit5(), saveat = t); #only difference is the time span
function predict(p)
    Array(train_node(u0, p, stM)[1])
end

#regularization loss (see Eq. 12)
function reg_loss(p, act_reg=1.0, entropy_reg=1.0)
    l1_temp=(abs.(p))
    activation_loss=sum(l1_temp)
    entropy_temp=l1_temp/activation_loss
    entropy_loss=-sum(entropy_temp.*log.(entropy_temp))
    total_reg_loss=activation_loss*act_reg+entropy_loss*entropy_reg
    return total_reg_loss
end

#overall loss
function loss(p)
    loss_temp=mean(abs2, Xn[:, 1:end_index].- predict(ComponentArray(p,pM_axis)))
    if sparse_on==1
        loss_temp+=reg_loss(p, 5e-4, 0) #if we have sparsity enabled, add the reg loss
    end
    return loss_temp
end

function predict_test(p)
    Array(train_node_test(u0, p, stM)[1])
end

function loss_train(p)
    mean(abs2, Xn[:, 1:end_index].- predict(ComponentArray(p,pM_axis)))
end
function loss_test(p)
    mean(abs2, Xn .- predict_test(ComponentArray(p,pM_axis)))
end


# TRAINING
du = [0.0; 0.0]
opt = ADAM(5e-4)

N_iter = 1e5
i_current = 1


function plot_save(l, l_test, p_list, epoch)
    
    l_min = minimum(l)
    idx_min = findfirst(x -> x == l_min, l)
    plt=Plots.plot(l, yaxis=:log, label="train", dpi=600)
    plot!(l_test, yaxis=:log, label="test")
    xlabel!("Epoch")
    ylabel!("Loss")
    png(plt, string(figpath, "/loss.png"))
    print("minimum train loss: ")
    print(minimum(l))
    print("          minimum test loss: ")
    print(minimum(l_test))


    p_curr = p_list[end]
    train_node_ = NeuralODE(kan1, tspan, Tsit5(), saveat = timestep); #neural ode
    pred_sol_kan = train_node_(u0, ComponentArray(p_curr,pM_axis), stM)[1]
    pred_sol_true = solve(ODEProblem(lotka!, u0, tspan, p_), Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = timestep)
    plt=scatter(pred_sol_true, alpha = 0.75)
    plot!(pred_sol_kan)
    vline!([3.5], color=:black, label = "train/test split")
    xlabel!("Time [s]")
    ylabel!("x, y")
    png(plt, string(figpath, "/training/results.png"))

    #packaging various quantities, then saving them to a .mat file
    p_list_ = zeros(size(p_list,1),size(p_list[1],1),size(p_list[1],2))
    for j = 1:size(p_list,1)
        p_list_[j,1:length(p_list[j]),:] = p_list[j]
    end
    l_ = zeros(size(p_list,1))
    for j = 1:size(l,1)
        l_[j] = l[j]
    end
    l_test_ = zeros(size(p_list,1))
    for j = 1:size(l,1)
        l_test_[j] = l_test[j]
    end
    file = matopen(dir*add_path*"checkpoints/"*fname*"_results.mat", "w")
    write(file, "p_list", p_list_)
    write(file, "loss", l_)
    write(file, "loss_test", l_test_)
    write(file, "kan_pred_t", pred_sol_kan.t)
    write(file, "kan_pred_u1", reduce(hcat,pred_sol_kan.u)'[:, 1])
    write(file, "kan_pred_u2", reduce(hcat,pred_sol_kan.u)'[:, 2])
    write(file, "size_KAN", [num_layers, layer_width, grid_size])
    close(file)

end



##Actual training loop:
iters=tqdm(1:N_iter-i_current)
 for i in iters
    global i_current
    
    # gradient computation
    grad = Zgrad(loss, p)[1]

    #model update
    update!(opt, p, grad)

    #loss metrics
    loss_curr=deepcopy(loss_train(p))
    loss_curr_test=deepcopy(loss_test(p))
    append!(l, [loss_curr])
    append!(l_test, [loss_curr_test])
    append!(p_list, [deepcopy(p)])

    set_description(iters, string("Loss:", loss_curr))
    i_current = i_current + 1


    if i%n_plot_save==0
        plot_save(l, l_test, p_list, i)
    end

    
end









