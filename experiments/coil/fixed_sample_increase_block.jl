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
    samples = [1000]
    blocks = 20
    repeat = 5
    times = Array{Float64}(undef, (length(samples), blocks))
    fits = Array{Float64}(undef, (length(samples), blocks))
    
    # precompile the optimizer
    als = ITensorCPD.compute_als(coil, init_guess; alg=ITensorCPD.BlockLevScoreSampled(100,2), 
    check=ITensorCPD.NoCheck(20), normal=false,);
    opt = ITensorCPD.optimize(init_guess, als; verbose=false)
    
    for (j,s) in enumerate(samples)
        println("Starting Sample :$(s)")
        for i in 1:20
            best_fit = 0
            ts = []
            println("Block Size: $(i)")
            for _ in 1:repeat
                als = ITensorCPD.compute_als(coil, init_guess; alg=ITensorCPD.BlockLevScoreSampled(s,i), 
                check=ITensorCPD.NoCheck(20), normal=false);
                push!(ts, @elapsed opt = ITensorCPD.optimize(init_guess, als; verbose=false))
                f = check_fit(als, opt)
                best_fit = f > best_fit ? f : best_fit
            end
            times[j,i] = mean(ts)
            fits[j,i] = best_fit
        end
    end

    results_file = joinpath(
        results_dir,
        "coil_fixed_sample_increase_block.jld2",
    )

    jldopen(results_file, "w+") do f
        f["times"] = times
        f["fits"] = fits
        f["samples"] = samples
        f["blocks"] = blocks
    end
    