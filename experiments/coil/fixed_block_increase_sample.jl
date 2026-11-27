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
### Fixed block size and increase sampling #####
#################################################

    ## Compute a sampled CP-ALS using random single samples of the khatri-rao product.
    # check = ITensorCPD.FitCheck(1e-3, 50, ncoil);
    ## Using 100 samples
    block_sizes = [1,2,6,10,14]
    samples = [400:400:5000...]
    repeat = 5
    fits = Array{Float64}(undef, length(block_sizes), length(samples))
    times = Array{Float64}(undef, length(block_sizes), length(samples))
    check=ITensorCPD.CPDiffCheck(1e-10, 20)
    
    ### precompile
    als = ITensorCPD.compute_als(coil, init_guess; 
    alg=ITensorCPD.BlockLevScoreSampled(10, 2), check,);
    init_guess = ITensorCPD.random_CPD(coil, 20; rng)
    ITensorCPD.optimize(init_guess, als; verbose=false);
    
    for (i,block) in enumerate(block_sizes)
        samp_vals = @view fits[i,:]
        t = @view times[i,:]
        println("Starting block: $(block)")
        for (j,sample) in enumerate(samples)
            println("Starting sample $(sample)")
            best_fit = 0
            algSampl = ITensorCPD.BlockLevScoreSampled(sample, block)
            ts = []
            for i in 1:repeat
                als = ITensorCPD.compute_als(coil, init_guess; 
                alg=algSampl, check, stop_resample=-1, variance_truncation=false);

                init_guess = ITensorCPD.random_CPD(coil, 20; rng)

                push!(ts, @elapsed opt = ITensorCPD.optimize(init_guess, als; verbose=false));
                f = check_fit(als, opt)
                best_fit = f > best_fit ? f : best_fit
            end
            samp_vals[j] = best_fit
            t[j] = mean(ts)
        end
    end
    

    results_file = joinpath(
        results_dir,
        "coil_fixed_block_increase_sample.jld2",
    )

    jldopen(results_file, "w+") do f
        f["times"] = times
        f["fits"] = fits
        f["block_sizes"] = block_sizes
        f["samples"] = samples
    end
