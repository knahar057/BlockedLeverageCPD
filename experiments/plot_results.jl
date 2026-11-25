using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using DelimitedFiles
using Plots

# ============================================================
# Plot saved CPD experiment results
# ============================================================
#
# Specify the results file below. The dataset name is obtained
# automatically from the filename.
#
# Example:
#   miranda_results.csv  -> dataset name: miranda
#   chicago_results.csv  -> dataset name: chicago
#   coil_results.csv     -> dataset name: coil
#
# The script generates plots for:
#   - CPD fit
#   - Runtime
#   - Allocated memory
# ============================================================


# ============================================================
# Results file
# ============================================================

# results_file = "miranda_results.csv"
results_file = "chicago_results.csv"


# ============================================================
# File locations
# ============================================================

results_path = joinpath(
    @__DIR__,
    "..",
    "results",
    results_file,
)

figure_dir = joinpath(
    @__DIR__,
    "..",
    "figures",
)

mkpath(figure_dir)


# ============================================================
# Determine dataset name
# ============================================================

dataset = replace(
    results_file,
    "_results.csv" => "",
)

dataset_title = Dict(
    "chicago" => "Chicago Crime",
    "miranda" => "Miranda",
    "coil" => "COIL",
)[dataset]
#titlecase(dataset)


# ============================================================
# Read experiment results
# ============================================================

data, header = readdlm(
    results_path,
    ',',
    header=true,
)

# Regular leverage-score sampling
time_exact = Float64(data[1, 3])
fit_exact = Float64(data[1, 4])
memory_exact = Float64(data[1, 5])

# Blocked leverage-score sampling
block_sizes = Int.(data[2:end, 2])
block_times_mean = Float64.(data[2:end, 3])
block_times_std = Float64.(data[2:end, 4])
block_memory_mean = Float64.(data[2:end, 5])
block_memory_std= Float64.(data[2:end, 6])
block_fits_max = Float64.(data[2:end, 7])
block_fits_min = Float64.(data[2:end, 8])
block_fits_diff = Float64.(data[2:end, 9])

lw = 3
ms = 8
# ============================================================
# Plot 1: CPD fit
# ============================================================

p_fit = hline(
    block_sizes,
    [fit_exact],
    linestyle = :dot,
    label = "Standard ALS";
    lw,ms
)

x = block_sizes
y1 = block_fits_max
y2 = block_fits_min
mid = (y1 .+ y2) ./ 2   #the midpoints (usually representing mean values)
w = (y2 .- y1) ./ 2     #the vertical deviation around the means

plot!(x, mid, ribbon = w , 
fillalpha = 0.35, c = 2, marker=:circle,
legend = :topleft, label = "Block Sampled ALS";
lw,ms)
# plot!(x,y1, msw = 0, ms = 2.5, label = "Lower bound")
# plot!(x,y2, msw = 0, ms = 2.5, label = "Upper bound")

plot!(
    xlabel = "Block Size",
    ylabel = "Fit",
    title = "Accuracy of Block Sampled ALS
    Data: $(dataset_title)",
    legendfontsize=10,
    xtickfontsize=9,
    ytickfontsize=9,
    labelfontsize=12,
    yrange=dataset == "chicago" ? [0,0.251] : [0.98,1],
    legend = :bottomleft)

savefig(
    p_fit,
    joinpath(
        figure_dir,
        "$(dataset)_fit.pdf",
    ),
)


# ============================================================
# Plot 2: Runtime
# ============================================================

p_time = plot(
    block_sizes,
    time_exact .* ones(length(block_sizes)),
    linestyle = :dot,
    label = "Standard ALS";
    lw,ms
)

plot!(
    block_sizes,
    block_times_mean,
    # yerr = block_times_std,
    marker = :circle,
    label = "Block Sampled ALS";
    lw,ms
)
plot!(
    xlabel = "Block Size",
    ylabel = "Time (s)",
    title = "Time of Block Sampled CPD
    Data: $(dataset_title)",
    legend = :right,
    legendfontsize=10,
    xtickfontsize=9,
    ytickfontsize=9,
    labelfontsize=12,)

savefig(
    p_time,
    joinpath(
        figure_dir,
        "$(dataset)_time.pdf",
    ),
)


# ============================================================
# Plot 3: Allocated memory
# ============================================================

p_memory = hline(
    block_sizes,
    [memory_exact],
    linestyle = :dot,
    label = "Standard ALS";
    lw,ms
)

plot!(
    block_sizes,
    block_memory_mean,
    # yerr = block_memory_std,
    marker = :circle,
    label = "Block Sampled ALS";
    lw,ms
)
plot!(
    xlabel = "Block Size",
    ylabel = "Allocated Memory (MB)",
    title = "Memory Allocation for Block Sampled CPD
    Data: $(dataset_title)",
    legend = :right,
    legendfontsize=10,
    xtickfontsize=9,
    ytickfontsize=9,
    labelfontsize=12,)

savefig(
    p_memory,
    joinpath(
        figure_dir,
        "$(dataset)_memory.pdf",
    ),
)


println("\nFigures saved to:")
println(figure_dir)