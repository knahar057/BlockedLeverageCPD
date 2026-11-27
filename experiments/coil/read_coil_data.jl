using Pkg
Pkg.activate(joinpath(@__DIR__,"..",".."))
include(joinpath(@__DIR__, "..","..", "src", "blocked_cpd.jl"))
using Images, FileIO, ITensors, ITensorCPD, Random, Plots
using JLD2, StatsBase

# ============================================================
# Results file
# ============================================================

#results_file = "coil_fixed_block_increase_sample.jld2"
#results_file = "coil_fixed_sample_increase_block.jld2"
results_file = "coil_variance.jld2"
#results_file = "coil_fixed_block_sample_scaling.jld2"


# ============================================================
# File locations
# ============================================================

results_path = joinpath(
    @__DIR__,
    "..","..",
    "results",
    results_file,
)

figure_dir = joinpath(
    @__DIR__,
    "..","..",
    "figures",
    "coil",
)

mkpath(figure_dir)

data = jldopen(results_path)

if results_file == "coil_variance.jld2"
    fits = f["fits"]
    repeat = f["repeat"]
    blocks = f["blocks"]
    samples = f["samples"]
    
    fs = map(x -> mean.(eachrow(fits[x,:,:])), 1:3)
    f1 = hcat(fits[1,:,:])
    f2 = hcat(fits[2,:,:])
    f3 = hcat(fits[3,:,:])
    fsstd = map(x->std.(eachrow(fits[x,:,:])) ./ mean.(eachrow(fits[x,:,:])), 1:3)

    lw = 4
    plot(ref_loss .* ones(length(fs[1])), ylabel="Standard ALS";lw)
    plot!(fs[1], marker=:circle,
    legend=:topright, label="500 Samples", ms=6, color=palette(:default)[2]) 
    scatter!(Base.repeat(1:blocks, inner=repeat), reshape(f1, (repeat * length(1:blocks))), label=nothing, color=palette(:default)[2], ms=6, alpha = 0.2)
    plot!(fs[2], marker=:square,
    legend=:topright, label="1000 Samples", ms=6, color=palette(:default)[3]) 
    scatter!(Base.repeat(1:blocks, inner=repeat), reshape(f2, (repeat * length(1:blocks))), 
    label=nothing, color=palette(:default)[3], ms=6, alpha = 0.2, marker=:square)
    plot!(fs[3], marker=:utriangle,
    legend=:topright, label="2000 Samples", ms=6, color=palette(:default)[4]) 
    scatter!(Base.repeat(1:blocks, inner=repeat), reshape(f3, (repeat * length(1:blocks))), 
    label=nothing, color=palette(:default)[4], ms=6, alpha = 0.2, marker=:utriangle)
    #  plot!((fs[4] .- ref[4]) ./ ref[4], marker=:circle,
    #  legend=:topright, label="Mode 4", ms=6) 
    #  scatter!(Base.repeat(1:20, inner=repeat), reshape(f4, (repeat * length(1:20))), label=nothing, color=:red, ms=6, alpha = 0.4)
    plot!(
    # title="Relative Standard Deviation in Loss Function", 
    title="Loss Function Versus Block Size",
    xlabel="Block Size", 
    #  ylabel="Relative Standard Deviation",
    ylabel="Loss Function",
    color=:red,
    lw=2, ms=7, 
    legend=:topleft,
    # yscale=:log10,
    yrange=[4*10^6, 1.5*10^7],
    legendfontsize=8,
    xtickfontsize=9,
    ytickfontsize=9,
    labelfontsize=12,
    )
    savefig(joinpath(
        figure_dir,
        "block_dev_loss.pdf",
        ),
    )

    plot(fsstd[1], label="500 Samples", ms=6, color=palette(:default)[2], marker=:circle)
    plot!(fsstd[2],label="1000 Samples", ms=6, color=palette(:default)[3], marker=:square)
    plot!(fsstd[3], label="2000 Samples", ms=6, color=palette(:default)[4], marker= :utriangle)
    plot!(title="Relative Standard Deviation in Loss Function", 
    ylabel="Relative Standard Deviation",
    xlabel="Block Size",
    legendfontsize=8,
    xtickfontsize=9,
    ytickfontsize=9,
    labelfontsize=12,)
    savefig(joinpath(
        figure_dir,
        "block_dev_rel_std.pdf",
        ),
    )
elseif results_file == "coil_fixed_sample_increase_block.jld2"
    times = data["times"]
    fits = data["fits"]
    
    lw=4
    ms=7

    plot((1 .- times[1,:] ./ times[1,1]) .* 100, marker=:circle,
    label="500 Samples";ms,lw)
    plot!((1 .- times[2,:] ./ times[2,1]) .* 100, marker=:circle,
    label="1000 Samples";ms,lw)
    plot!((1 .- times[3,:] ./ times[3,1]) .* 100, 
    marker=:circle, label="2000 Samples"; lw,ms)
    plot!((1 .- times[4,:] ./ times[4,1]) .* 100, 
    marker=:circle, label="4000 Samples"; lw,ms)
    plot!(
        legend=:bottomright, label=nothing, 
    ylabel="Percent Time Reduction", xlabel="Block Size",  
    title="Acceleration from Block Sampling"
    )
    savefig("$(@__DIR__)/../plots/coil/$(imgs)_images/block_time_acceleration_multiple.pdf")

    plot(fits[1,:] ./ fits[1,1] .* 100, marker=:circle, label="500 Samples"; lw,ms) 
    plot!(fits[2,:] ./ fits[2,1] .* 100, marker=:circle, label="1000 Samples"; lw,ms) 
    plot!(fits[3,:] ./ fits[3,1] .* 100, marker=:circle, label="2000 Samples"; lw,ms) 
    plot!(fits[4,:] ./ fits[4,1] .* 100, marker=:circle, label="4000 Samples"; lw,ms) 
    plot!(legend=:bottomleft, label=nothing, 
    ylabel="Relative Accuracy of L2 Fit", 
    xlabel="Block Size", 
    color=:red,
    lw=4, ms=7, 
    title="Error Associated with Block Sampling")
    savefig("$(@__DIR__)/../plots/coil/$(imgs)_images/block_error_multiple.pdf")
elseif results_file == "coil_fixed_block_increase_sample.jld2"
    times = data["times"]
    fits = data["fits"]
    samples=data["samples"]
    blocks = data["blocks"]
    
    lw=3
    ms=7

    p = plot()
    for i in 1:length(blocks)
        plot!((1 .- times[i,:] ./ times[i,1]) .* 100, marker=:circle,
        label=length(blocks) == 1 ? nothing : "$(samples[1]) Samples";ms,lw)
    end
    plot!(
        legend=:bottomright, label=nothing, 
    ylabel="Percent Time Reduction", xlabel="Block Size",  
    title="Acceleration from Block Sampling",
    legendfontsize=10,
    xtickfontsize=9,
    ytickfontsize=9,
    labelfontsize=12,
    )
    savefig(
    joinpath(
        figure_dir,
        "block_time_acceleration.pdf",
        ),
    )

    p = plot()
    for i in 1:length(blocks)
        plot(fits[i,:] ./ fits[i,1] .* 100, marker=:circle, label=length(blocks) == 1 ? nothing : "$(samples[1]) Samples"; lw,ms) 
    end
    plot!(legend=:bottomleft, label=nothing, 
    ylabel="Relative Accuracy of L2 Fit", 
    xlabel="Block Size", 
    color=:red,
    xtickfontsize=9,
    ytickfontsize=9,
    labelfontsize=12,
    title="Error Associated with Block Sampling")
     savefig(joinpath(
        figure_dir,
        "block_error.pdf",
        ),
    )
elseif results_file == "fixed_block_sample_scaling.jld2"
    jldopen(results_file, "w+") do f
        f["times"] = times2
        f["fits"] = fits_scaled
        f["block_sizes"] = block_sizes
    end

    ms = 8
    xdata = sample .* block_sizes[1:end]
    p = plot(block_sizes[1:end], fits_scaled[1,1] .* ones(length(xdata)), label="Block Size: 1"; lw, marker=:circle, ms=ms)
    for i in 2:size(fits_scaled)[1]
        @show i
        plot!( block_sizes[1:end], fits_scaled[i,:], label="Block Size: $(block_sizes[i])"; lw,  marker=:circle, ms=ms)
    end
    plot!(legendcolumns=2,
    title="Accuracy of Block Sampled CPD\nSamples = $(sample)",
    xlabel="Factor times Single Block Sample Size",
    legendfontsize=10,
    xtickfontsize=9,
    ytickfontsize=9,
    labelfontsize=12,
    yrange=[0.63,0.691],
    ylabel="Fit")
    # savefig("$(@__DIR__)/../plots/coil/scaled_block_coil_data_$(sample).pdf")

    lw=2
    using Plots
    p = plot(block_sizes[1:end], times2[1][1]/30 * ones(length(block_sizes[1:end])), label="Block Size: 1"; lw, marker=:circle, ms=ms)
    for i in 2:size(fits_scaled)[1]
        plot!(block_sizes, times2[i,:] ./ 30, label="Block Size: $(block_sizes[i])"; lw,  marker=:circle, ms=ms)
    end
    plot!(legendcolumns=2,
    title="Time per Iteration of Block Sampled CPD\nSamples = $(sample)",
    xlabel="Factor times Single Block Sample Size",
    ylabel="Time per Iteration (s)",
    yrange=[0.01, 2.3],
    legendfontsize=10,
    xtickfontsize=9,
    ytickfontsize=9,
    labelfontsize=12,
    )
    # savefig("$(@__DIR__)/../plots/coil/scaled_block_coil_time_$(sample).pdf")
end