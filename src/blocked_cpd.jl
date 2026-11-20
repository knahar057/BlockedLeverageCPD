using Random
using ITensorCPD
using ITensors

# ============================================================
# CPD fit
# ============================================================
#
# Compute the CP decomposition fit:
#
#     fit = 1 - ||X - X_hat|| / ||X||
#
# where X is the target tensor and X_hat is the CP approximation.
#
# This function is adapted from the fit calculation used in the
# ITensorCPD test environment.
# ============================================================

function check_fit(als, cpd)

    factors = cpd.factors
    cprank = ITensorCPD.cp_rank(cpd)
    λ = cpd.λ
    fact = 1

    target = als.target

    # Inner product between the target tensor and CP approximation.
    inner_prod = (
        ITensorCPD.had_contract(
            [target, dag.(factors)...],
            cprank,
        ) * dag(λ)
    )[]

    # Compute the squared norm of the CP approximation.
    partial_gram = [
        factor * dag(prime(factor; tags=tags(cprank)))
        for factor in factors
    ]

    fact_square = ITensorCPD.norm_factors(
        partial_gram,
        λ,
    )

    # Compute the norm of the target tensor.
    ref_norm = sqrt(sum(target .^ 2))

    # Compute the residual norm.
    norm_residual = sqrt(
        abs(
            ref_norm * ref_norm +
            fact_square -
            2 * abs(inner_prod)
        )
    )

    fit = 1.0 - norm_residual / norm(ref_norm)

    return fit
end

# ============================================================
# Regular and blocked leverage-score sampled CP-ALS
# ============================================================
#
# run_blocked_cpd(X; rank, samples, block_sizes, maxiter=20, seed=1234)
#
# Runs regular leverage-score sampled CP-ALS and blocked
# leverage-score sampled CP-ALS on an input tensor.
#
# The same random seed is used for each experiment so that the
# regular and blocked methods start from the same random CP
# initialization.
#
# Inputs:
#   X           - Input tensor represented as an ITensor.
#   rank        - Target CP rank.
#   samples     - Number of samples used by the randomized ALS method.
#   block_sizes - Block sizes used for blocked leverage-score sampling.
#   maxiter     - Maximum number of ALS iterations (default: 20).
#   seed        - Random seed for the initial CP factors (default: 1234).
#
# Output:
#   Returns the runtime, fit, and allocated memory for the regular
#   leverage-score method and each blocked leverage-score experiment.
# ============================================================

function run_blocked_cpd(
    X;
    rank,
    samples,
    block_sizes,
    maxiter=20,
    seed=1234,
)

    # ============================================================
    # Regular leverage-score sampled CP-ALS
    # ============================================================

    println("\n========================================")
    println("Regular Leverage-Score Sampling")
    println("========================================")

    # Initialize the CP factors.
    rng = MersenneTwister(seed)

    init_regular = ITensorCPD.random_CPD(
        X,
        rank;
        rng,
    )

    # Define the regular leverage-score sampling algorithm.
    alg_regular = ITensorCPD.LevScoreSampled(samples)

    # Construct the ALS problem.
    als_regular = ITensorCPD.compute_als(
        X,
        init_regular;
        alg=alg_regular,
        check=ITensorCPD.NoCheck(maxiter),
        normal=true,
        stop_resample=-1,
        variance_truncation=false,
    )

    # Run CP-ALS and record runtime and allocated memory.
    stats_regular = @timed opt_regular = ITensorCPD.optimize(
        init_regular,
        als_regular;
        verbose=false,
    )

    # Compute the final CPD fit.
    fit_regular = check_fit(
        als_regular,
        opt_regular,
    )

    time_regular = stats_regular.time
    memory_regular = stats_regular.bytes / 1024^2

    println("Time   = ", time_regular, " s")
    println("Fit    = ", fit_regular)
    println("Memory = ", memory_regular, " MB")


    # ============================================================
    # Blocked leverage-score sampled CP-ALS
    # ============================================================

    println("\n========================================")
    println("Blocked Leverage-Score Sampling")
    println("========================================")

    block_times = Float64[]
    block_fits = Float64[]
    block_memory = Float64[]

    for block_size in block_sizes

        println("\n----------------------------------------")
        println("Block size = ", block_size)

        # Reset the random seed so that each experiment uses the
        # same random CP initialization.
        rng = MersenneTwister(seed)

        init_block = ITensorCPD.random_CPD(
            X,
            rank;
            rng,
        )

        # Define the blocked leverage-score sampling algorithm.
        alg_block = ITensorCPD.BlockLevScoreSampled(
            samples,
            block_size,
        )

        # Construct the ALS problem.
        als_block = ITensorCPD.compute_als(
            X,
            init_block;
            alg=alg_block,
            check=ITensorCPD.NoCheck(maxiter),
            normal=true,
            stop_resample=-1,
            variance_truncation=false,
        )

        # Run CP-ALS and record runtime and allocated memory.
        stats_block = @timed opt_block = ITensorCPD.optimize(
            init_block,
            als_block;
            verbose=false,
        )

        # Compute the final CPD fit.
        fit_block = check_fit(
            als_block,
            opt_block,
        )

        time_block = stats_block.time
        memory_block = stats_block.bytes / 1024^2

        push!(block_times, time_block)
        push!(block_fits, fit_block)
        push!(block_memory, memory_block)

        println("Time   = ", time_block, " s")
        println("Fit    = ", fit_block)
        println("Memory = ", memory_block, " MB")
    end


    # ============================================================
    # Return experiment results
    # ============================================================

    return (
        regular=(
            time=time_regular,
            fit=fit_regular,
            memory=memory_regular,
        ),
        blocked=(
            block_sizes=collect(block_sizes),
            times=block_times,
            fits=block_fits,
            memory=block_memory,
        ),
    )
end
