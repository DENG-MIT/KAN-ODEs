using Random, LinearAlgebra
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

# Directories
dir         = @__DIR__
dir         = dir*"/"
cd(dir)
fname       = "LV_MLP"
add_path    = "results_mlp/"
add_path_kan    = "results_kanode/"
figpath=dir*add_path*"figs"
ckptpath=dir*add_path*"checkpoints"
mkpath(figpath)
mkpath(ckptpath)


#define LV
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

#data generation parameters
timestep=0.1
n_plot=1000
n_save=50
rng = Random.default_rng()
Random.seed!(rng, 0)
tspan = Float32.([0.0, 14])
tspan_train=Float32.([0.0, 3.5])
u0 = Float32.([1., 1.])
p_ = Float32[1.5, 1, 1, 3]
prob = ODEProblem(lotka!, Float64.(u0), tspan, Float64.(p_))

#generate training data, split into train/test
solution = solve(prob, Tsit5(), abstol = 1e-12, reltol = 1e-12, saveat = timestep)
end_index=Int64(floor(length(solution.t)*tspan_train[2]/tspan[2]))
t = Float32.(solution.t) #full dataset
t_train=t[1:end_index] #training cut
X = Array(solution)
Xn = deepcopy(X) 



# Define MLP 
###As in KANODE code, the layers can be modified here to recreate the testing in section A2 of the manuscript
MLP = Chain(Dense(2 => 50, tanh), Dense(50 => 2)) #like in https://github.com/RajDandekar/MSML21_BayesianNODE/blob/main/BayesiaNODE_SGLD_LV.jl
## **also note here that the KANODE and MLP-NODE codes use different packages.
## **so if the Dense command fails, make sure to use a different REPL (i.e. one that you did not previously run the KANODE code on)


# Define Neural ODE with MLP

train_node = NeuralODE(MLP, tspan_train, Tsit5(), saveat = t_train); #neural ode
train_node_test = NeuralODE(MLP, tspan, Tsit5(), saveat = t); #neural ode

function predict(p)
    Array(train_node(u0, p))
end
function loss(p)
    mean(abs2, Xn[:, 1:end_index].- predict(p))
end

function predict_test(p)
    Array(train_node_test(u0, p))
end
function loss_test(p)
    mean(abs2, Xn .- predict_test(p))
end

# TRAINING
du = [0.0; 0.0]
p = deepcopy(train_node.p)
print("parameter size:")
print(length(p))
opt = ADAM(1e-2)
l = []
l_test=[]

p_list = []
N_iter = 1e5
i_current = 1



function plotter(l, p_list, epoch)
    l_min = minimum(l)
    idx_min = findfirst(x -> x == l_min, l)
    plt=Plots.plot(l, yaxis=:log, label="train")
    plot!(l_test, yaxis=:log, label="test")
    xlabel!("Epoch")
    ylabel!("Loss")
    png(plt, string(figpath, "/loss.png"))
    print("minimum train loss: ")
    print(minimum(l))
    print("minimum test loss: ")
    print(minimum(l_test))

    p_opt = p_list[idx_min]
    train_node_ = NeuralODE(MLP, tspan, Tsit5(), saveat = timestep); #neural ode
    pred_sol_true = solution
    p_curr = p_list[end]
    pred_sol_kan = train_node_(u0, p_curr)
    plt=scatter(pred_sol_true, alpha = 0.75)
    plot!(pred_sol_kan)
    vline!([3.5], color=:black, label = "train/test split")
    xlabel!("Time [s]")
    ylabel!("x, y")
    png(plt, string(figpath, "/training/results.png"))

    #packaging various quantities, then saving them to a .mat file
    p_list_ = zeros(size(p_list,1),size(p_list[1],1),size(p_list[1],2))
    for j = 1:size(p_list,1)
        p_list_[j,:,:] = p_list[j]
    end
    l_ = zeros(size(p_list,1))
    for j = 1:size(l,1)
        l_[j] = l[j]
    end
    
    l_test_ = zeros(size(p_list,1))
    for j = 1:size(l,1)
        l_test_[j] = l_test[j]
    end
    pred_sol_kan = train_node_(u0, p_opt)
    file = matopen(dir*add_path*"checkpoints/"*fname*"_results_MLP.mat", "w")
    write(file, "p_list", p_list_)
    write(file, "loss", l_)
    write(file, "loss_test", l_test_)
    write(file, "kan_pred_t", pred_sol_kan.t)
    write(file, "kan_pred_u1", reduce(hcat,pred_sol_kan.u)'[:, 1])
    write(file, "kan_pred_u2", reduce(hcat,pred_sol_kan.u)'[:, 2])
    close(file)

end

iters=tqdm(1:N_iter-i_current)
 for i in iters
    global i_current
    
    # gradient computation
    grad = Zgrad(loss, p)[1] 

    #model update
    update!(opt, p, grad)

    #loss metrics
    loss_curr=deepcopy(loss(p))
    loss_curr_test=deepcopy(loss_test(p))
    append!(l, [loss_curr])
    append!(l_test, [loss_curr_test])
    append!(p_list, [deepcopy(p)])
    set_description(iters, string("Loss:", loss_curr))
    i_current = i_current + 1


    if i%n_plot==0
        plotter(l, p_list, i)
    end

    
end







