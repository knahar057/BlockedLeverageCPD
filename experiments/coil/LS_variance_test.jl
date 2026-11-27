include("$(@__DIR__)/read_coil.jl")

using StatsBase
include(joinpath(@__DIR__, "..","..", "src", "single_solve.jl"))

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

################################
### Variance in loss function ##
################################

    algNormal = ITensorCPD.KRPFreeNormal()
    check_exact = ITensorCPD.FitCheck(1e-3, 20, ncoil)
    als = ITensorCPD.compute_als(coil, init_guess; alg=algNormal, check=check_exact);
    cpd_good = ITensorCPD.optimize(init_guess, als; verbose=true);
    
    _,_, ref_loss, t_loss = single_solve(als, cpd_good, 1);

    als = ITensorCPD.compute_als(coil, init_guess; alg=ITensorCPD.BlockLevScoreSampled(2000,1), check=check_exact, normal=true);
    single_solve(als, cpd_good, 1);

    samples = [500,1000,2000]
    repeat = 5
    blocks = 20
    fits = Array{Float64}(undef, length(samples), blocks, repeat)
    for (fact,s) in enumerate(samples)
        ft = @view fits[fact,:,:]
        println("Starting sample $(s)")
        for i in 1:blocks
            for j in 1:repeat
                als = ITensorCPD.compute_als(coil, cpd_good; alg=ITensorCPD.BlockLevScoreSampled(s,i), 
                check=ITensorCPD.NoCheck(20), normal=true, stop_resample=-1, variance_truncation=false);
                
                _,_,f,t = single_solve(als, cpd_good, 1);
                
                ft[i,j] = f
            end
        end
    end

    results_dir = joinpath(
    @__DIR__,
    "..","..",
    "results",
)

    results_file = joinpath(
        results_dir,
        "coil_variance.jld2",
    )

    jldopen(results_file, "w+") do f
        f["fits"] = fits
        f["repeat"] = repeat
        f["blocks"] = blocks
        f["samples"] = samples
    end

    # plot((1 .- time ./ time[1]) .* 100, marker=:circle,
    # legend=:topright, label=nothing, 
    # ylabel="Percent Time Reduction", xlabel="Block Size", 
    # lw=2, ms=7, 
    # title="Acceleration from Block Sampling")
    # savefig("$(@__DIR__)/../plots/coil/$(imgs)_images/block_time_acceleration.pdf")