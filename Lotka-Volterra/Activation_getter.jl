#Lots of the drivers and plotters have to extract the activations from the KAN-ODE
#for plotting, visualization, pruning, etc. This shared function enables this. 
function activation_getter(pM_, pM_new, kan1, grid_size)
    lay1=kan1[1]
    lay2=kan1[2]
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

    ##sanity check: run the actual spline formulation and make sure they match
    #basis  = reshape(basis, lay1.grid_len * lay1.in_dims, K)    # [G * I, K]
    #spline = pc1*basis+pw1*lay1.base_act.(x)                                  # [O, K]
    #sum(abs.(spline.-((activations_x+activations_y)'[:, :])).<1e-10)==length(spline) #make sure it's all equal 

    ##second layer
    LV_samples_lay1=kan1[1](X, pM_.layer_1, stM[1])[1] #this is the activation function results for the first layer

    x = reshape(LV_samples_lay1, lay2.in_dims, :)
    K = size(x, 2)

    x_norm = lay2.normalizer(x)              # ∈ [-1, 1]
    x_resh = reshape(x_norm, 1, :)                        # [1, K]
    basis  = lay2.basis_func(x_resh, st.grid, lay2.denominator) # [G, I * K]
    activations_second=zeros(lay2.in_dims*2, K)
    for i in 1:lay2.in_dims
        basis_curr=basis[:, i:lay2.in_dims:end]
        pc_curr=pc2[:, (i-1)*grid_size+1:i*grid_size]
        activations_curr=basis_curr'*pc_curr'
        activations_curr+=(lay2.base_act.(x[i, :]).*pw2[:, i]')
        activations_second[2*i-1:2*i, :]=activations_curr'
    end
    ##sanity check: run the actual spline formulation and make sure they match 
    #basis  = reshape(basis, lay2.grid_len * lay2.in_dims, K)    # [G * I, K]
    #spline = pc2*basis+pw2*lay2.base_act.(x)                                  # [O, K]
    ##activation_compare=zeros(2, K)
    #activation_compare[1, :]=sum(activations_second[1:2:end, :], dims=1)
    #activation_compare[2, :]=sum(activations_second[2:2:end, :], dims=1)
    #sum(abs.(spline.-((activation_compare))).<1e-10)==length(spline) #make sure it's all equal 
    return activations_x, activations_y, activations_second, LV_samples_lay1, lay2, K
end