using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using Random
using Plots
using StatsBase
using MAT

# Load the common regular and blocked CP-ALS experiment.
include(joinpath(@__DIR__, "..", "src", "blocked_cpd.jl"))


# ============================================================
# Load Miranda data
# ============================================================

"""
    read_miranda(path, nfirst; elt=Float64)

Load the Miranda density tensor from a MAT file and select the first
`nfirst` entries along the first mode.

The resulting dense array is converted to an ITensor.
"""
function read_miranda(path, nfirst; elt=Float64)

    file = matopen(path)

    # Load the density tensor.
    density = read(file, "density")

    close(file)

    println("Full Miranda size     = ", size(density))

    # Select the requested portion of the first mode.
    X = convert.(elt, density[1:nfirst, :, :])

    println("Selected Miranda size = ", size(X))

    # Convert the dense Julia array to an ITensor.
    inds = Index.(size(X))

    return itensor(X, inds)
end


# ============================================================
# Data location
# ============================================================

data_path = joinpath(
    @__DIR__,
    "..",
    "data",
    "density.mat",
)


# ============================================================
# Experiment parameters
# ============================================================

nfirst = 256

rank = 20
samples = 2000
maxiter = 20
seed = 1234
block_sizes = 1:20


# ============================================================
# Load data
# ============================================================

println("\nLoading Miranda tensor...")

miranda = read_miranda(
    data_path,
    nfirst;
    elt=Float64,
)


# ============================================================
# Run CPD experiments
# ============================================================

results = run_blocked_cpd(
    miranda;
    rank=rank,
    samples=samples,
    block_sizes=block_sizes,
    maxiter=maxiter,
    seed=seed,
)


# ============================================================
# Print final results
# ============================================================

println("\n========================================")
println("MIRANDA RESULTS")
println("========================================")

println("\nCanonical ALS")
println("Time   = ", results.regular.time, " s")
println("Fit    = ", results.regular.fit)
println("Memory = ", results.regular.memory, " MB")

println("\nBlocked leverage-score sampling")

block_times_mean = mean.(eachcol(results.blocked.times))
block_times_std = std.(eachcol(results.blocked.times))
block_memory_mean = mean.(eachcol(results.blocked.memory))
block_memory_std = std.(eachcol(results.blocked.memory))
block_fits_max = maximum.(eachcol(results.blocked.fits))
block_fits_min = minimum.(eachcol(results.blocked.fits))
block_fits_mean = mean.(eachcol(results.blocked.fits))
diff = block_fits_max - block_fits_min

for i in eachindex(results.blocked.block_sizes)

    println(
        "Block size = ",
        results.blocked.block_sizes[i],
    )

    println(
        "  Time   = ",
        block_times_mean[i],
        " s",
    )

    println(
        "  Fit    = ",
        block_fits_mean[i],
    )

    println(
        "  Memory = ",
        block_memory_mean[i],
        " MB",
    )
end

# ============================================================
# Save results
# ============================================================

results_dir = joinpath(
    @__DIR__,
    "..",
    "results",
)

mkpath(results_dir)

results_file = joinpath(
    results_dir,
    "miranda_results.csv",
)

open(results_file, "w") do file

    # Column names
    println(
        file,
        "method,block_size,time_seconds,fit,memory_mb",
    )

    # Regular leverage-score sampling
    println(
        file,
        "Regular,NA,",
        results.regular.time, ",",
        results.regular.fit, ",",
        results.regular.memory,
    )

    # Blocked leverage-score sampling
    for i in eachindex(results.blocked.block_sizes)

        println(
            file,
            "Blocked,",
            results.blocked.block_sizes[i], ",",
            block_times_mean[i], ",",
            block_times_std[i], ",",
            block_memory_mean[i], ",",
            block_memory_std[i], ",",
            block_fits_max[i], ",",
            block_fits_min[i], ",",
            diff[i]
        )
    end
end

println("\nResults saved to:")
println(results_file)