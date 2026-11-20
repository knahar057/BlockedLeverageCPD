using MAT
using ITensors
using ITensorCPD

# Load the common regular and blocked CP-ALS experiment.
include(joinpath(@__DIR__, "..", "src", "blocked_cpd.jl"))


# ============================================================
# Load Miranda data
# ============================================================

function read_miranda(path, nfirst; elt=Float64)

    file = matopen(path)

    # Load the density tensor.
    density = read(file, "density")

    close(file)

    println("Full Miranda size     = ", size(density))

    # Select the requested portion along the first mode.
    X = convert.(elt, density[1:nfirst, :, :])

    println("Selected Miranda size = ", size(X))

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
    "density.mat",
)


# ============================================================
# Experiment parameters
# ============================================================

tensor_sizes = [256, 512, 1024, 2048]

rank = 20
samples = 2000
maxiter = 20
seed = 1234

block_sizes = [1, 5, 10, 15, 20]


# ============================================================
# Output file
# ============================================================

results_dir = joinpath(
    @__DIR__,
    "..",
    "results",
)

mkpath(results_dir)

results_file = joinpath(
    results_dir,
    "miranda_scaling_results.csv",
)


# ============================================================
# Run scalability experiment
# ============================================================

open(results_file, "w") do file

    # Column names
    println(
        file,
        "tensor_size,method,block_size,time_seconds,fit,memory_mb",
    )

    for nfirst in tensor_sizes

        println("\n========================================")
        println("Tensor size = ", nfirst, " x 256 x 256")
        println("========================================")

        # Load the requested Miranda tensor size.
        miranda = read_miranda(
            data_path,
            nfirst;
            elt=Float64,
        )

        # Run regular and blocked leverage-score sampled CP-ALS.
        results = run_blocked_cpd(
            miranda;
            rank=rank,
            samples=samples,
            block_sizes=block_sizes,
            maxiter=maxiter,
            seed=seed,
        )


        # ----------------------------------------------------
        # Save regular leverage-score result
        # ----------------------------------------------------

        println(
            file,
            nfirst, ",",
            "Regular,NA,",
            results.regular.time, ",",
            results.regular.fit, ",",
            results.regular.memory,
        )


        # ----------------------------------------------------
        # Save blocked leverage-score results
        # ----------------------------------------------------

        for i in eachindex(results.blocked.block_sizes)

            println(
                file,
                nfirst, ",",
                "Blocked,",
                results.blocked.block_sizes[i], ",",
                results.blocked.times[i], ",",
                results.blocked.fits[i], ",",
                results.blocked.memory[i],
            )
        end


        # Free the current tensor before loading the next size.
        miranda = nothing
        GC.gc()
    end
end


println("\n========================================")
println("MIRANDA SCALABILITY EXPERIMENT COMPLETE")
println("========================================")

println("\nResults saved to:")
println(results_file)