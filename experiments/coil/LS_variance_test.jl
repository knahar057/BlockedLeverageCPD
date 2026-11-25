include("$(@__DIR__)/read_coil.jl")
using StatsBase
include(joinpath(@__DIR__, "..","..", "src", "single_solve.jl"))
results_dir = joinpath(
    @__DIR__,
    "..","..",
    "results",
)
mkpath(results_dir)

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
    
    als = ITensorCPD.compute_als(coil, init_guess; alg=ITensorCPD.BlockLevScoreSampled(2000,1), check=check_exact, normal=true);
    _,_, l, t = single_solve(als, cpd_good, 1);

    samples = [500,1000,2000]
    repeat = 2
    fit = Array{Float64}(undef, length(samples), repeat)
    ref = []
    for (fact,s) in enumerate(samples)
        fact = 1
        ft = @view fit[fact]
        println("Starting sample $(s)")
        for i in 1:20
            lt = []
            lf = []
            for j in 1:5
            als = ITensorCPD.compute_als(coil, cpd_good; alg=ITensorCPD.BlockLevScoreSampled(s,i), 
            check=ITensorCPD.NoCheck(20), normal=true, stop_resample=-1, variance_truncation=false);
            
            _,_,f,t = single_solve(als, cpd_good, 1);
            
            ft[i] = f
            #push!(lt, t)
            # push!(lf, f)
            end
        end
    end

    plot((1 .- time ./ time[1]) .* 100, marker=:circle,
    legend=:topright, label=nothing, 
    ylabel="Percent Time Reduction", xlabel="Block Size", 
    lw=2, ms=7, 
    title="Acceleration from Block Sampling")
    savefig("$(@__DIR__)/../plots/coil/$(imgs)_images/block_time_acceleration.pdf")

    fs = map(x -> mean.(eachcol(hcat(x...))), fit)
    f1 = hcat(fit[1]...)
    f2 = hcat(fit[2]...)
    f3 = hcat(fit[3]...)
    fsstd = map(x -> std.(eachcol(hcat(x...)) ./ mean.(eachcol(hcat(x...)))), fit)

    plot(fs[1], marker=:circle,
    legend=:topright, label="500 Samples", ms=6) 
    scatter!(Base.repeat(1:20, inner=repeat), reshape(f1, (repeat * length(1:20))), label=nothing, color=:lightblue, ms=6, alpha = 0.4)
    plot!(fs[2], marker=:circle,
    legend=:topright, label="1000 Samples", ms=6, color=:green) 
    scatter!(Base.repeat(1:20, inner=repeat), reshape(f2, (repeat * length(1:20))), label=nothing, color=:lightgreen, ms=6, alpha = 0.4)
    plot!(fs[3], marker=:circle,
    legend=:topright, label="2000 Samples", ms=6, color=:darkgoldenrod) 
    scatter!(Base.repeat(1:20, inner=repeat), reshape(f3, (repeat * length(1:20))), label=nothing, color=:lightyellow, ms=6, alpha = 0.4)
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
    legend=:bottomleft,
    yrange=[2.6*10^7, 3.8*10^7],
    legendfontsize=8,
    xtickfontsize=9,
    ytickfontsize=9,
    labelfontsize=12,
    )
    savefig("$(@__DIR__)/../plots/coil/$(imgs)_images/block_dev_loss.pdf")

