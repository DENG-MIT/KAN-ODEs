# KAN-ODEs: Kolmogorov-Arnold Network Ordinary Differential Equations for Learning Dynamical Systems and Hidden Physics
# Allen-Cahn equation
# Source term modeling (Sec. C.1.)

# PACKAGES AND INCLUSIONS
using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots, LinearAlgebra
using Random
using ModelingToolkit
using MAT
using NNlib, ConcreteStructs, WeightInitializers, ChainRulesCore
using ComponentArrays
using Random
using ForwardDiff
using Flux: ADAM, mae, update!, mean
using Flux
using Optimisers
using MethodOfLines

# DIRECTORY
dir         = @__DIR__
dir         = dir*"/"
cd(dir)
fname       = "Allen_Cahn_Source"
add_path    = "test/"
mkpath(dir*add_path*"figs")
mkpath(dir*add_path*"checkpoints")
mkpath(dir*add_path*"results")

# KAN PACKAGE LOAD
include("src/KolmogorovArnold.jl")
using .KolmogorovArnold

# GRID SETTINGS
xspan   = (-1.0,1.0)
tspan   = (0.0,1.0)
dx      = 0.05
dt      = 0.01
x       = collect(xspan[1]:dx:xspan[2]);
t       = collect(tspan[1]:dt:tspan[2]);
Nx      = Int64(length(x));
Nt      = Int64(length(t));

# INITIAL CONDITION
u0      = x.^(2).*cos.(pi.*x)

# SOURCE TERM IN THE ALLEN-CAHN EQUATION
reaction(u) = - 5.0*u^(3) + 5.0*u

# DISCRETIZATION
lap     = diagm(0 => -2.0 * ones(Nx), 1=> ones(Nx-1), -1 => ones(Nx-1)) ./ dx^2

# PERIODIC BC
lap[1,end]  = 1.0/dx^2
lap[end,1]  = 1.0/dx^2

# DISCRETIZED PDES OF THE ALLEN-CAHN EQUATION
function rc_ode(u, p, t)
    -0.0001 * lap * u + reaction.(u)
end

# GENERATE TRAINING DATA
prob        = ODEProblem(rc_ode, u0, tspan, saveat=dt)
sol         = solve(prob, Tsit5());
ode_data    = hcat([sol[:,i]  for i in 1:size(sol,2)]...)
Xₙ          = deepcopy(ode_data)'

# PLOT TRAINING DATA
xgrid       = deepcopy(x)
tgrid       = deepcopy(t)
pythonplot()
contourf(tgrid, xgrid, ode_data, color=:turbo, levels=201, size=(600, 250), xlims=(0.0,1.0))
xlabel!("x")
ylabel!("t")

# DEFINE THE NETWORK OF KAN-ODEs
basis_func  = rbf
normalizer  = softsign
KANgrid     = 10;
kan1        = Lux.Chain(
    KDense(1, 1, KANgrid; use_base_act = true, basis_func, normalizer),
)
rng         = Random.default_rng()
Random.seed!(rng, 0)
pM , stM    = Lux.setup(rng, kan1)
pM_data = getdata(ComponentArray(pM))
pM_axis = getaxes(ComponentArray(pM))
u0 = deepcopy(prob.u0)

# CONSTRUCT KAN-ODES
function rc_kanode(u, p, t)
    kan1_(x) = kan1([x], ComponentArray(p,pM_axis), stM)[1][1]
    -0.0001 * lap * u + kan1_.(u)
end

# PREDICTION FUNCTION
function predict(p)
    prob = ODEProblem(rc_kanode, u0, tspan, p, saveat=dt)
    sol = Array(solve(prob, Tsit5()));
end

# LOSS FUNCTION
function loss(p)
    mean(abs2, Xₙ .- predict(p)')
end

# CALLBACK FUNCTION
function callback(i)
    if i%1000 == 0
        # SAVE PARAMETERS AND LOSS
        # p_list in mat form
        p_list_ = zeros(size(p_list,1),size(p_list[1],1),size(p_list[1],2))
        for j = 1:size(p_list,1)
            p_list_[j,:,:] = p_list[j]
        end
        # loss in mat form
        l_ = zeros(size(p_list,1))
        for j = 1:size(l,1)
            l_[j] = l[j]
        end
        file = matopen(dir*add_path*"/checkpoints/"*fname*"_results.mat", "w")
        write(file, "p_list", p_list_)
        write(file, "loss", l_)
        close(file)

        # SAVE SUMMARY PLOTS
        l_min = minimum(l)
        idx_min = findfirst(x -> x == l_min, l)

        p1 = plot(l, yaxis=:log, label=:none)
        xlabel!("Epoch")
        ylabel!("Loss")

        p0 = contourf(xgrid, tgrid, ode_data', title="Ground truth", xlabel="x", ylabel="t", color=:turbo, levels=201)#, aspect_ratio=:equal)

        p_opt = p_list[1]
        pred_sol_kan = predict(p_opt)
        p2 = contourf(xgrid, tgrid, pred_sol_kan', title="Initial guess", xlabel="x", ylabel="t", color=:turbo, levels=201)#, aspect_ratio=:equal)

        p_opt = p_list[idx_min]
        pred_sol_kan = predict(p_opt)
        p3 = contourf(xgrid, tgrid, pred_sol_kan', title="Learned field", xlabel="x", ylabel="t", color=:turbo, levels=201)#, aspect_ratio=:equal)

        pt = plot(p0, p1, p2, p3, layout=(2,2))
        savefig(pt, dir*add_path*"figs/"*fname*"_result.png")

        # SAVE A PLOT OF THE REACTION FUNCTION 
        ρgrid = -1.0:0.05:1.0
        pρ = plot(ρgrid, reaction.(ρgrid), linewidth=2.0, label="True")
        kan1_(x) = kan1([x], ComponentArray(p,pM_axis), stM)[1][1]
        plot!(pρ, ρgrid, kan1_.(ρgrid), linewidth=2.0, label="Learned")
        xlabel!("u")
        ylabel!("KAN(u)")
        savefig(dir*add_path*"figs/"*fname*"_rho_profile.png")
    end
end

# TRAINING SETUP
isrestart   = false
du          = zeros(length(u0))
p           = deepcopy(pM_data)
opt         = ADAM(1e-2)
l           = []
p_list      = []
N_iter      = 5e4
i_current   = 1
append!(p_list, [deepcopy(p)])
append!(l, [deepcopy(loss(p))])

if isrestart == true
    file    = matopen(dir*add_path*"/checkpoints/"*fname*"_results.mat")
    p_list_ = read(file, "p_list")
    l_      = read(file, "loss")
    close(file)
    i_current = length(l_)

    for i = 1:i_current
        append!(p_list, [p_list_[i,:,:][:]])
        append!(l, [l_[i]])
    end

    p       = p_list[end]
    p_list_ = nothing
    l_      = nothing
end

# TRAINING LOOP
using Zygote
for i = 1:N_iter-i_current
    global i_current

    # GRADIENT COMPUTATION
    grad = Zygote.gradient(x->loss(x), p)[1]

    # UPDATE WITH ADAM OPTIMIZER
    update!(opt, p, grad)

    # PARAM, LOSS
    append!(l, [deepcopy(loss(p))])
    append!(p_list, [deepcopy(p)])

    # CALLBACK
    println("Iteration: ", Int32(i_current),"/", Int32(N_iter), ",\t Loss: ", l[end])
    i_current = i_current + 1

    # SAVE
    callback(i)
end

# DERIVE SYMBOLIC REGRESSION
using SymbolicRegression, MLJ
model = SRRegressor(
    binary_operators=[+, -, *, /],
    niterations=30
)
X       = zeros(length(ρgrid),1)
X[:,1]  = collect(ρgrid)
mach    = machine(model, X, kan1_.(ρgrid))
fit!(mach)

# REPORT
r       = report(mach)
r.equations
bar(r[4], yaxis=:log10, ylabel="Loss")

# FROM THE REPORT, THE BEST EQUATION IS
r.equations[r.best_idx]
fitted_eq(x) = 5.675949973338312e-5 - (((x * (x * x)) - x) * 5.000357135982538)

# PLOT THE REACTION FUNCTION
ρgrid       = -1.0:0.05:1.0
kan1_(x)    = kan1([x], ComponentArray(p,pM_axis), stM)[1][1]
pρ          = plot(ρgrid, kan1_.(ρgrid), linewidth=2.0, color=:red, label="Learned")
plot!(ρgrid, fitted_eq.(ρgrid), 
                titlefontsize=18, guidefontsize=18, tickfontsize=16, legendfontsize=12, grid=false, framestyle = :box, size=(450,350),
                xlabel="u", ylabel="KAN(u)", xlim=(0.0,1.0), ylim=(-3.5, 3.5), linewidth=6.0, color=:blue, alpha = 0.35, label="Symbolic regression")
savefig(dir*add_path*"figs/"*fname*"_rho_profile_symbolic.pdf")