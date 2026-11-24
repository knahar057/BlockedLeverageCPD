# Load the common regular and blocked CP-ALS experiment.
using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using Random
using Plots
using StatsBase

include(joinpath(@__DIR__, "..", "src", "blocked_cpd.jl"))


# ============================================================
# Load Chicago Crime data
# ============================================================
#
# Tensor dimensions:
#
#     day x hour x community x crime type
#
# The input file contains:
#
#     day  hour  community  crime_type  value
#
# ============================================================

function read_chicago(path, ndays; elt=Float64)

    X = zeros(
        elt,
        ndays,
        24,
        77,
        32,
    )

    open(path, "r") do file

        for line in eachline(file)

            vals = split(line)

            day = parse(Int, vals[1])

            if day <= ndays

                hour = parse(Int, vals[2])
                community = parse(Int, vals[3])
                crime = parse(Int, vals[4])
                value = parse(elt, vals[5])

                X[
                    day,
                    hour,
                    community,
                    crime,
                ] = value
            end
        end
    end

    # Convert the dense array to an ITensor.
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
    "chicago-crime-comm.tns",
)


# ============================================================
# Experiment parameters
# ============================================================

ndays = 365

rank = 15
samples = 10000
maxiter = 20
seed = 1234

block_sizes = [1, 2, 3, 4, 5, 6]


# ============================================================
# Load Chicago Crime tensor
# ============================================================

println("\nLoading Chicago Crime tensor...")

chicago = read_chicago(
    data_path,
    ndays;
    elt=Float64,
)

println("Tensor size = ", size(chicago))


# ============================================================
# Run CPD experiments
# ============================================================

results = run_blocked_cpd(
    chicago;
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
println("CHICAGO CRIME RESULTS")
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
    "chicago_results.csv",
)

open(results_file, "w") do file

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