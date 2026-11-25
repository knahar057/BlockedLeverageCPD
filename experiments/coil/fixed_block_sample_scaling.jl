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
                als = ITensorCPD.compute_als(coil, init_guess; alg=algSampl, check);
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
    jldopen(results_file, "w+") do f
        f["scaled_samples_times"] = times2
        f["scaled_samples_fits"] = fits_scaled
    end

    xdata = sample .* block_sizes[1:end]
    p = plot(block_vals[1:end], fits_scaled[1,1] .* ones(length(xdata)), label="Block Size: 1"; lw, marker=:circle, ms=ms)
    for i in 2:length(fits_scaled)
        plot!( block_vals[1:end], fits_scaled[i,:], label="Block Size: $(block_vals[i])"; lw,  marker=:circle, ms=ms)
    end
    plot!(legendcolumns=2,
    title="Accuracy of Block Sampled CPD\nSamples = $(sample)",
    xlabel="Factor times Single Block Sample Size",
    legendfontsize=10,
    xtickfontsize=9,
    ytickfontsize=9,
    labelfontsize=12,
    yrange=[0.62,0.681],
    ylabel="Fit")
    savefig("$(@__DIR__)/../plots/coil/scaled_block_coil_data_$(sample).pdf")

    lw=2
    using Plots
    p = plot(block_vals[1:end], times2[1][1]/30 * ones(length(block_vals[1:end])), label="Block Size: 1"; lw, marker=:circle, ms=ms)
    for i in 2:length(fits_scaled)
        plot!(block_vals[1:end], times2[i][1:3:end] ./ 30, label="Block Size: $(block_vals[i])"; lw,  marker=:circle, ms=ms)
    end
    plot!(legendcolumns=2,
    title="Time per Iteration of Block Sampled CPD\nSamples = $(sample)",
    xlabel="Factor times Single Block Sample Size",
    ylabel="Time per Iteration (s)",
    yrange=[0.1, 2.3],
    legendfontsize=10,
    xtickfontsize=9,
    ytickfontsize=9,
    labelfontsize=12,
    )
    savefig("$(@__DIR__)/../plots/coil/scaled_block_coil_time_$(sample).pdf")

