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

dataset_title = titlecase(dataset)


# ============================================================
# Read experiment results
# ============================================================

data, header = readdlm(
    results_path,
    ',',
    header=true,
)

# Regular leverage-score sampling
regular_time = Float64(data[1, 3])
regular_fit = Float64(data[1, 4])
regular_memory = Float64(data[1, 5])

# Blocked leverage-score sampling
block_sizes = Int.(data[2:end, 2])
block_times = Float64.(data[2:end, 3])
block_fits = Float64.(data[2:end, 4])
block_memory = Float64.(data[2:end, 5])


# ============================================================
# Plot 1: CPD fit
# ============================================================

p_fit = plot(
    block_sizes,
    block_fits;
    marker=:circle,
    linewidth=3,
    markersize=5,
    label="Blocked",
    xlabel="Block Size",
    ylabel="Fit",
    title="$(dataset_title): CPD Fit",
    legend=:bottomleft,
)

hline!(
    p_fit,
    [regular_fit];
    linestyle=:dash,
    linewidth=2,
    label="Regular Sampling",
)

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
    block_times;
    marker=:circle,
    linewidth=3,
    markersize=5,
    label="Blocked",
    xlabel="Block Size",
    ylabel="Time (s)",
    title="$(dataset_title): Runtime",
    legend=:topright,
)

hline!(
    p_time,
    [regular_time];
    linestyle=:dash,
    linewidth=2,
    label="Regular Sampling",
)

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

p_memory = plot(
    block_sizes,
    block_memory;
    marker=:circle,
    linewidth=3,
    markersize=5,
    label="Blocked",
    xlabel="Block Size",
    ylabel="Allocated Memory (MB)",
    title="$(dataset_title): Memory Allocation",
    legend=:topright,
)

hline!(
    p_memory,
    [regular_memory];
    linestyle=:dash,
    linewidth=2,
    label="Regular Sampling",
)

savefig(
    p_memory,
    joinpath(
        figure_dir,
        "$(dataset)_memory.pdf",
    ),
)


println("\nFigures saved to:")
println(figure_dir)