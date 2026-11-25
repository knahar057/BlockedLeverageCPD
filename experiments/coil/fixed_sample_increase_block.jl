include("$(@__DIR__)/read_coil.jl")
results_dir = joinpath(
    @__DIR__,
    "..","..",
    "results",
)
mkpath(results_dir)

path = joinpath(@__DIR__, "..", "..", "data","coil-100")
image_names = readdir(path)

imgs = 100
coil = readcoil(path, imgs, Float64);

rng = RandomDevice()
init_guess = ITensorCPD.random_CPD(coil, 20; rng);

ncoil = sqrt(sum(hadamard_product(coil,coil)))


#################################
### Solution error vs time #####
#################################
    # Computes the CP-ALS using an efficient version of the normal equations.
    # Stops when change in fit falls below 1e-3 
    algNormal = ITensorCPD.KRPFreeNormal()
    check_exact = ITensorCPD.FitCheck(1e-3, 20, ncoil)

    ## Create a random initial guess with rank 20
    samples = [500,1000,2000, 4000]
    blocks = 20
    times = Array{Float64}(undef, (length(samples), blocks))
    fits = Array{Float64}(undef, (length(samples), blocks))
    
    # precompile the optimizer
    als = ITensorCPD.compute_als(coil, init_guess; alg=ITensorCPD.BlockLevScoreSampled(100,2), 
    check=ITensorCPD.NoCheck(20), normal=false,);
    opt = ITensorCPD.optimize(init_guess, als; verbose=false)
    
    for (j,s) in enumerate(samples)
        println("Starting Sample :$(s)")
        for i in 1:20
            println("Block Size: $(i)")
            als = ITensorCPD.compute_als(coil, init_guess; alg=ITensorCPD.BlockLevScoreSampled(s,i), 
            check=ITensorCPD.NoCheck(20), normal=false);
            times[j,i] = @elapsed opt = ITensorCPD.optimize(init_guess, als; verbose=false)
            fits[j,i] = check_fit(als, opt)
        end
    end

    results_file = joinpath(
        results_dir,
        "coil_fixed_sample_increase_block.jld2",
    )

    jldopen(results_file, "w+") do f
        f["times"] = times
        f["fits"] = fits
    end
    
    #### To gather data
    # f = jldopen(results_file, "r")
    # times = f["times"]
    # fits = f["fits"]

    # lw=4
    # ms=7
    # plot((1 .- times[1,:] ./ times[1,1]) .* 100, marker=:circle,
    # label="500 Samples";ms,lw)
    # plot!((1 .- times[2,:] ./ times[2,1]) .* 100, marker=:circle,
    # label="1000 Samples";ms,lw)
    # plot!((1 .- times[3,:] ./ times[3,1]) .* 100, 
    # marker=:circle, label="2000 Samples"; lw,ms)
    # plot!((1 .- times[4,:] ./ times[4,1]) .* 100, 
    # marker=:circle, label="4000 Samples"; lw,ms)
    # plot!(
    #     legend=:bottomright, label=nothing, 
    # ylabel="Percent Time Reduction", xlabel="Block Size",  
    # title="Acceleration from Block Sampling"
    # )
    #savefig("$(@__DIR__)/../plots/coil/$(imgs)_images/block_time_acceleration_multiple.pdf")

    # plot(fits[1,:] ./ fits[1,1] .* 100, marker=:circle, label="500 Samples"; lw,ms) 
    # plot!(fits[2,:] ./ fits[2,1] .* 100, marker=:circle, label="1000 Samples"; lw,ms) 
    # plot!(fits[3,:] ./ fits[3,1] .* 100, marker=:circle, label="2000 Samples"; lw,ms) 
    # plot!(fits[4,:] ./ fits[4,1] .* 100, marker=:circle, label="4000 Samples"; lw,ms) 
    # plot!(legend=:bottomleft, label=nothing, 
    # ylabel="Relative Accuracy of L2 Fit", 
    # xlabel="Block Size", 
    # color=:red,
    # lw=4, ms=7, 
    # title="Error Associated with Block Sampling")
    #savefig("$(@__DIR__)/../plots/coil/$(imgs)_images/block_error_multiple.pdf")

