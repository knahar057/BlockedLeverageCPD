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
    miranda_scaling = false
)

    # ============================================================
    # Regular leverage-score sampled CP-ALS
    # ============================================================

    println("\n========================================")
    println("Canonical ALS")
    println("========================================")

    # Initialize the CP factors.
    rng = MersenneTwister(seed)

    init = ITensorCPD.random_CPD(
        X,
        rank;
        rng,
    )
    
    alg_exact = ITensorCPD.KRPFreeNormal()

    # Construct the canonical ALS problem.
    als_exact = ITensorCPD.compute_als(
        X,
        init;
        alg = alg_exact,
        check = ITensorCPD.NoCheck(maxiter)
    );

    if miranda_scaling
        als_exact.additional_items[:mttkrp_contract_sequences] = [Any[1, Any[3, 2]], Any[2, Any[3, 1]], Any[2, Any[3, 1]]]
    end
    # Precompile the optimization
    opt_exact = ITensorCPD.optimize(
        init,
        als_exact;
        verbose = false
    );

    # Run CP-ALS and record runtime and allocated memory.
    stats_exact = @timed opt_exact = ITensorCPD.optimize(
        init,
        als_exact;
        verbose = false
    );

    # Compute the final CPD fit.
    fit_exact = check_fit(als_exact, opt_exact)

    time_exact = stats_exact.time
    memory_exact = stats_exact.bytes / 1024^2

    println("Time   = ", time_exact, " s")
    println("Fit    = ", fit_exact)
    println("Memory = ", memory_exact, " MB")


    # ============================================================
    # Blocked leverage-score sampled CP-ALS
    # ============================================================

    println("\n========================================")
    println("Blocked Leverage-Score Sampling")
    println("========================================")

    ntimes = 10
    block_times = Matrix{Float64}(undef, (ntimes, length(block_sizes)))
    block_fits = similar(block_times)
    block_memory = similar(block_times)

    # this precompiles the optimization loop.
    for (block, i) in zip(block_sizes[1], 1:length(block_sizes[1]))
        println("\n----------------------------------------")
        println("Block size = ", block)
        for rep in 1:ntimes

            alg_block = ITensorCPD.BlockLevScoreSampled(
                samples,
                block
            )

            als_block = ITensorCPD.compute_als(
                X,
                init;
                alg = alg_block,
                check = ITensorCPD.NoCheck(maxiter),
                normal = false,
            )

            try
                stats_block = @timed opt_block = ITensorCPD.optimize(
                    init,
                    als_block;
                    verbose = false
                )

                fit_block = check_fit(
                    als_block,
                    opt_block
                )

                time_block = stats_block.time
                memory_block = stats_block.bytes / 1024^2

            catch
                rep -= 1
            end
        end
    end

    for (block, i) in zip(block_sizes, 1:length(block_sizes))
        println("\n----------------------------------------")
        println("Block size = ", block)
        for rep in 1:ntimes

            alg_block = ITensorCPD.BlockLevScoreSampled(
                samples,
                block
            )

            als_block = ITensorCPD.compute_als(
                X,
                init;
                alg = alg_block,
                check = ITensorCPD.NoCheck(maxiter),
                normal = false,
            )

            try
                stats_block = @timed opt_block = ITensorCPD.optimize(
                    init,
                    als_block;
                    verbose = false
                )

                fit_block = check_fit(
                    als_block,
                    opt_block
                )

                time_block = stats_block.time
                memory_block = stats_block.bytes / 1024^2

                block_times[rep, i] = time_block
                block_fits[rep, i] = fit_block
                block_memory[rep,i] = memory_block

                println("Time   = ", time_block, " s")
                println("Fit    = ", fit_block)
                println("Memory = ", memory_block, " MB")
            catch
                rep -= 1
            end
        end
    end


    # ============================================================
    # Return experiment results
    # ============================================================

    return (
        regular=(
            time=time_exact,
            fit=fit_exact,
            memory=memory_exact,
        ),
        blocked=(
            block_sizes=block_sizes,
            times=block_times,
            fits=block_fits,
            memory=block_memory,
        ),
    )
end
