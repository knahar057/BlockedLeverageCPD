using DelimitedFiles
using Plots

# ============================================================
# Plot CPD scalability results
# ============================================================
#
# Specify the scaling results file below.
#
# Expected columns:
#   tensor_size, method, block_size, time_seconds, fit, memory_mb
#
# The script generates:
#   - Runtime vs. tensor size
#   - Fit vs. tensor size
#   - Allocated memory vs. tensor size
# ============================================================


# ============================================================
# Results file
# ============================================================

results_file = "miranda_scaling_results.csv"


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
    "_scaling_results.csv" => "",
)

dataset_title = titlecase(dataset)


# ============================================================
# Read results
# ============================================================

data, header = readdlm(
    results_path,
    ',',
    header=true,
)

# Tensor sizes
tensor_sizes = unique(Int.(data[:, 1]))

# Regular leverage-score results
regular_rows = data[:, 2] .== "Regular"

regular_times = Float64.(data[regular_rows, 4])
regular_fits = Float64.(data[regular_rows, 5])
regular_memory = Float64.(data[regular_rows, 6])

# Block sizes used in the experiment
blocked_rows = data[:, 2] .== "Blocked"

block_sizes = unique(
    Int.(data[blocked_rows, 3])
)


# ============================================================
# Plot 1: Runtime
# ============================================================

p_time = plot(
    tensor_sizes,
    regular_times;
    marker=:circle,
    linewidth=3,
    markersize=6,
    label="Regular",
    xlabel="Tensor Size",
    ylabel="Time (s)",
    title="$(dataset_title): Runtime Scalability",
)

for block_size in block_sizes

    rows =
        (data[:, 2] .== "Blocked") .&
        (data[:, 3] .== block_size)

    block_times = Float64.(data[rows, 4])

    plot!(
        p_time,
        tensor_sizes,
        block_times;
        marker=:circle,
        linewidth=3,
        markersize=6,
        label="Block $(block_size)",
    )
end

savefig(
    p_time,
    joinpath(
        figure_dir,
        "$(dataset)_scaling_time.pdf",
    ),
)


# ============================================================
# Plot 2: CPD fit
# ============================================================

p_fit = plot(
    tensor_sizes,
    regular_fits;
    marker=:circle,
    linewidth=3,
    markersize=6,
    label="Regular",
    xlabel="Tensor Size",
    ylabel="Fit",
    title="$(dataset_title): Fit Scalability",
)

for block_size in block_sizes

    rows =
        (data[:, 2] .== "Blocked") .&
        (data[:, 3] .== block_size)

    block_fits = Float64.(data[rows, 5])

    plot!(
        p_fit,
        tensor_sizes,
        block_fits;
        marker=:circle,
        linewidth=3,
        markersize=6,
        label="Block $(block_size)",
    )
end

savefig(
    p_fit,
    joinpath(
        figure_dir,
        "$(dataset)_scaling_fit.pdf",
    ),
)


# ============================================================
# Plot 3: Allocated memory
# ============================================================

p_memory = plot(
    tensor_sizes,
    regular_memory;
    marker=:circle,
    linewidth=3,
    markersize=6,
    label="Regular",
    xlabel="Tensor Size",
    ylabel="Allocated Memory (MB)",
    title="$(dataset_title): Memory Scalability",
)

for block_size in block_sizes

    rows =
        (data[:, 2] .== "Blocked") .&
        (data[:, 3] .== block_size)

    block_memory = Float64.(data[rows, 6])

    plot!(
        p_memory,
        tensor_sizes,
        block_memory;
        marker=:circle,
        linewidth=3,
        markersize=6,
        label="Block $(block_size)",
    )
end

savefig(
    p_memory,
    joinpath(
        figure_dir,
        "$(dataset)_scaling_memory.pdf",
    ),
)


println("\nScaling figures saved to:")
println(figure_dir)