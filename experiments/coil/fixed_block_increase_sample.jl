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
    repeat = 1
    fits = Array{Float64}(undef, length(block_sizes), length(samples))
    times = Array{Float64}(undef, length(block_sizes), length(samples))
    check=ITensorCPD.CPDiffCheck(1e-10, 20)
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
    end

    # xdata = samples
    # lw = 2
    # ms = 8
    # p = plot(xdata, fits[1,:], label="Block Size: 1", lw=lw, marker=:circle, ms=ms)
    # for (i, j) in zip(2:length(fits),  [2,6,10,14])
    #     plot!(xdata, fits[i,:], label="Block Size: $(j)",lw=lw,marker=:circle, ms=ms)
    # end
    # plot!(legendcolumns=2,
    # title="Accuracy of Block Sampled CPD\nData: COIL",
    # xlabel="Number of Samples",
    # ylabel="Fit",
    # legend=:topleft,
    # yrange=[0.6,0.71], 
    # legendfontsize=10,
    # xtickfontsize=9,
    # ytickfontsize=9,
    # labelfontsize=12,
    # lw=4)
    # # savefig("$(@__DIR__)/../plots/coil/block_coil_data.pdf")

    # p = plot(xdata, times[1,:] ./ 30, label="Block Size: 1", lw=lw, marker=:circle, ms=ms)
    # for (i,j) in zip(2:length(fits),  [2,6,10,14])
    #     plot!(xdata, times[i,:] ./ 30, label="Block Size: $(j)", lw=lw, marker=:circle, ms=ms)
    # end

    # plot!(legendcolumns=2,
    # title="Time of Block Sampled CPD\nData: COIL",
    # xlabel="Number of Samples",
    # ylabel="Time per ALS Iteration (s)",
    # legendfontsize=10,
    # xtickfontsize=9,
    # ytickfontsize=9,
    # labelfontsize=12,)
    # savefig("$(@__DIR__)/../plots/coil/block_coil_time_data.pdf")
