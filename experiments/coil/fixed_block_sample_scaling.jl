include("$(@__DIR__)/read_coil.jl")
using StatsBase
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

als = ITensorCPD.compute_als(coil, init_guess; alg=ITensorCPD.BlockLevScoreSampled(100,2), 
check=ITensorCPD.NoCheck(20), normal=false,);
opt = ITensorCPD.optimize(init_guess, als; verbose=false)

################################################
### Fixed block size checking sample scaling #####
#################################################

    block_sizes = [1,2,6,10,]
    repeat = 1
    fits_scaled = Array{Float64}(undef, length(block_sizes), length(block_sizes))
    times2 = similar(fits_scaled)
    
    sample = 2000
    for (i,block) in enumerate(block_sizes)
        samp_vals = @view fits_scaled[i,:]
        t = @view times2[i,:]
        println("Starting block: $(block)")
        for (j,scale) in enumerate(block_sizes)
            println("Scaling is $(scale)")
            best_fit = 0
            algSampl = ITensorCPD.BlockLevScoreSampled(sample * scale, block)
            ts = []
            for i in 1:repeat
                als = ITensorCPD.compute_als(coil, init_guess; alg=algSampl, check=ITensorCPD.NoCheck(20));
                init_guess = ITensorCPD.random_CPD(coil, 20; rng)
                
                push!(ts, @elapsed opt = ITensorCPD.optimize(init_guess, als; verbose=false));
                f = check_fit(als, opt)
                best_fit = f > best_fit ? f : best_fit
            end
            t[j] = mean(ts)
            samp_vals[j] = best_fit
        end
    end

    results_file = joinpath(
        results_dir,
        "coil_fixed_block_sample_scaling.jld2",
    )