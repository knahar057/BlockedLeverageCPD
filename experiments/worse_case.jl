using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
include(joinpath(@__DIR__,"..","src","blocked_cpd.jl"))
using LinearAlgebra, Statistics, DataFrames, CategoricalArrays, StatsPlots
using LaTeXStrings, ITensorCPD, ITensors
using Random, JLD2

results_dir = joinpath(
    @__DIR__,
    "..",
    "results",
)

figure_dir = joinpath(
    @__DIR__,
    "..",
    "figures",
)

function make_data(elt, ids, bad, r, n, rep =20)
    name = "$(n)"
    cpd_exact = ITensorCPD.random_CPD(zeros(ids...), r);
    if n != 0
        for (x, is) in zip(n, ids)
            array(cpd_exact[x])[1:bad, :] .*= 100;
            array(cpd_exact[x]) .= array(cpd_exact[x])[randperm(is), :]
            tmp, lam = ITensorCPD.row_norm(cpd_exact[x], ind(cpd_exact[x], 1))
            array(cpd_exact[x]) .= tmp
            cpd_exact[] .= lam
        end
    end

    r = ITensorCPD.cp_rank(cpd_exact);
    T = ITensorCPD.had_contract(cpd_exact[1], cpd_exact[2], r) * ITensorCPD.had_contract(cpd_exact[3], cpd_exact[], r)


    verbose= true
    samples = [400,500,600,800,1000,1200,1500,2000,2500,3000,3500,4000]

    lev_error_mat1 = Matrix{Float64}(undef, 20, length(samples))
    lev_error_mat5 = Matrix{Float64}(undef, 20, length(samples))
    lev_error_mat10 = Matrix{Float64}(undef, 20, length(samples))
    lev_error_mat50 = Matrix{Float64}(undef, 20, length(samples))

    check_piv = ITensorCPD.CPDiffCheck(1e-2, 100)
    check_direct = ITensorCPD.FitCheck(1e-5, 50,sqrt(sum(T.^2)) )

    lev_error_mat = Matrix{Float64}(undef, 20, length(samples))
    rng=RandomDevice()
    cp_T = nothing
    blocks = [1, 5, 10, 20]

    cp_T = ITensorCPD.random_CPD(T,20; rng)
    alslev = ITensorCPD.compute_als(T,cp_T;
        alg = ITensorCPD.BlockLevScoreSampled(100, 2),check = check_piv,);
    int_opt_T = ITensorCPD.optimize(cp_T, alslev; verbose=false);

    for rk in [dim(r)]
        r = Index(rk, "CP_rank")
        err_leverage = Vector{Float64}()
        cp_T = ITensorCPD.random_CPD(T,r; rng)

        lev_mats = Array{Float64}(undef, length(blocks), rep, length(samples))
        for (bi,b) in enumerate(blocks)
            println("Starting block $(b)")
            lm = @view lev_mats[bi,:,:]
            
           for (i,sample) in enumerate(samples)
                println("Starting sample $(sample)")
                for q=1:rep
                    println("iter $(q)")
                    alslev = ITensorCPD.compute_als(T,cp_T;alg = ITensorCPD.BlockLevScoreSampled(sample, b),check = check_piv);
                    int_opt_T = ITensorCPD.optimize(cp_T, alslev; verbose=false);
                    lm[q,i] = check_fit(alslev, int_opt_T)
                end
            end
        end
        
        println("Finished sample stuff")

        alsNormal = ITensorCPD.compute_als(T, cp_T; alg=ITensorCPD.KRPFreeNormal(), check = check_direct);
        opt_T = ITensorCPD.optimize(cp_T, alsNormal; verbose=true);
        direct_error = check_fit(alsNormal, opt_T)

        results_file = joinpath(
            results_dir,
            "worse_case_$(ids[1])_$(length(ids)).jld2",
        )

        jldopen(results_file, "w+") do f
            f["lev_mats"] = lev_mats
            f["blocks"] = blocks
            f["samples"] = samples
            f["direct_error"] = direct_error
        end
    end
end
 
function make_figs(elt, ids, bad, r, n, rep =20)
    name = "$(n)"
    rk = dim(r)
     ll = "$(ids[1])"
    for i in ids[2:end]
        ll *= " " * L"\times" * " $(i)"
    end

    results_file = "worse_case_$(ids[1])_$(length(ids)).jld2"
    results_path = joinpath(
        @__DIR__,
        "..",
        "results",
        results_file,
    )
    data = jldopen(results_path)
    lev_mats = data["lev_mats"]
    blocks = data["blocks"]
    samples = data["samples"]
    direct_error = data["direct_error"]
    err_leverages = [maximum.(eachcol(lev_mats[x,:,:])) for x in 1:length(blocks)]
    markers = [:circle, :cross, :diamond,  :xcross, :utriangle]
    ms = 8
    lw=3
    plt2 = plot(samples, direct_error .* ones(length(samples)), ls=:dot, label="Standard ALS"; markersize=ms, lw)
    for (i,b, marker) in zip(1:length(blocks), blocks,markers)
        plot!(samples, err_leverages[i], marker=marker, label="Block Size: $(b)"; ms,lw, alpha =0.8)
    end
    plot!(
    xticks=samples[1]:samples[1]:samples[end],
    yrange=[0.9,1.01],
    yticks=0.0:0.05:1.0,
    legendfontsize=10,
    xtickfontsize=9,
    ytickfontsize=9,
    labelfontsize=12,
    xlabel="Number of Samples",
    ylabel="Fit",
    title="Accuracy of the Sampled ALS
    $(ll)"
    )

    n = nothing
    if elt == Float64
        n = "F64"
    else
        n = "F32"
    end

    savefig(
        joinpath(
            figure_dir,
            "worse_case",
            "mode_$(name)_rank_$(rk)_modified_test_$(n)_$(ids[1])_$(length(ids)).pdf",
        ),
    )


    #mid = [mean.(eachcol(lev_mats[i,:,:])) for i in 1:4]
    #var = [std.(eachcol(lev_mats[i,:,:])) for i in 1:4]
    mn = [minimum.(eachcol(lev_mats[i,:,:])) for i in 1:4]
    mx = [maximum.(eachcol(lev_mats[i,:,:])) for i in 1:4]
    plot()
    plt2 = plot(samples, direct_error .* ones(length(samples)), ls=:dot, label="Standard ALS"; markersize=ms, lw)
    for (marker,i) in zip(markers, 1:4)
        w = (mn[i] .- mx[i]) ./ 2     #the vertical deviation around the means
        mid = (mn[i] .+ mx[i]) ./ 2
        plot!(samples, mid, ribbon = w , 
        fillalpha = 0.35, marker=marker, color=palette(:default)[i+1],
        label = "Block Sampled ALS"; lw,ms)
    end
    plot!(
    xticks=samples[1]:samples[1]:samples[end],
    yrange=[0.5,1.01],
    yticks=0.0:0.1:1.0,
    legendfontsize=10,
    xtickfontsize=9,
    ytickfontsize=9,
    labelfontsize=12,
    xlabel="Number of Samples",
    ylabel="Fit",
    title="Accuracy of the Sampled ALS
    $(ll)",
    legend=:bottomright
    )

    savefig(
        joinpath(
            figure_dir,
            "worse_case",
            "var_mode_$(name)_rank_$(rk)_modified_test_$(n)_$(ids[1])_$(length(ids)).pdf",
        ),
    )

    df = DataFrame(
        sample = vcat(repeat(samples, inner=rep), repeat(samples, inner=rep), repeat(samples, inner=rep), repeat(samples, inner=rep)),
        method = vcat(fill("Block Size 1", rep*length(samples)),
                    fill("Block Size 5", rep*length(samples)),
                    fill("Block Size 10", rep*length(samples)),
                    fill("Block Size 20", rep*length(samples))),
        error  = error  = vcat(vec(lev_mats[1,:,:]), vec(lev_mats[2,:,:]), vec(lev_mats[3,:,:]), vec(lev_mats[4,:,:]))
    )

    df.sample = CategoricalArray(string.(df.sample), ordered=true, levels=string.(samples))

    bs_1 = filter(row -> row.method == "Block Size 1", df)
    bs_5 = filter(row -> row.method == "Block Size 5", df)
    bs_10 = filter(row -> row.method == "Block Size 10", df)
    bs_50 = filter(row -> row.method == "Block Size 20", df)

    p = @df bs_1 violin(:sample, :error, side=:left, label="Block Size 1", color=:blue, alpha=0.2)
    @df bs_5 violin!(:sample, :error, side=:left, label="Block Size 5", color=:red,alpha=0.2)
    @df bs_10 violin!(:sample, :error, side=:left, label="Block Size 10", color=:orange,alpha=0.2)
    @df bs_50 violin!(:sample, :error, side=:left, label="Block Size 20", color=:green,alpha=0.2)
    xlabel!("Number of Samples")
    ylabel!("Distribution of Fits")
    title!("Distribution of CPD Fits\n$(ll)")
    plot!(legend=ids[1] == 400 ? :topleft : :bottomright,
    yticks=[0.5,0.6,0.7,0.8,0.9,1.0], 
    yrange=[0.5, 1.01])
    plot!(legendfontsize=10,
    xtickfontsize=9,
    ytickfontsize=9,
    labelfontsize=12)
    savefig(
        joinpath(
            figure_dir,
            "worse_case",
            "violin_mode_$(name)_rank_$(rk)_modified_test_$(n)_$(ids[1])_$(length(ids)).pdf",
        ),
    )

end
elt = Float64 
ids = (400,400,400)
bad = 3
r = 10
n = [1,2,3,4]

make_data(elt, ids, bad, r, n)
make_figs(elt, ids, bad, r, n)


# ### Compute gifs of leverage score sampling
# cpd_exact = ITensorCPD.random_CPD(zeros(ids...), r);
# # main(elt, ids, bad, r, n)
# for (x, is) in zip(n, ids)
#     array(cpd_exact[x])[1:bad, :] .*= 100;
#     array(cpd_exact[x]) .= array(cpd_exact[x])[randperm(is), :]
#     tmp, lam = ITensorCPD.row_norm(cpd_exact[x], ind(cpd_exact[x], 1))
#     array(cpd_exact[x]) .= tmp
#     cpd_exact[] .= lam
# end
# cpr = ITensorCPD.cp_rank(cpd_exact)

# i23 = Index(90 * 90, "234")
# krp23 = ITensorCPD.had_contract([cpd_exact[2], cpd_exact[3]], cpr)
# krp23 = itensor((reshape(array(krp23), (90 * 90, r))), i23, cpr)

# #levs1 = ITensorCPD.compute_leverage_score_probabilitiy(cpd_exact[1], ind(cpd_exact[1],1))
# levs23 = ITensorCPD.compute_leverage_score_probabilitiy(krp23, i23)
# plot(levs23)
# sum(levs23 .> 1e-2)

# it = itensor(zeros(ids...), inds(cpd_exact))
# als = ITensorCPD.compute_als(it, cpd_exact; alg=ITensorCPD.LevScoreSampled(1));

# plot(als.additional_items[:factor_weights][2], lw=3, label="Mode 2")
# plot!(als.additional_items[:factor_weights][3], lw=3, label="Mode 3")
# plot!(title="Factor-wise Leverage Score Sampling Probabilities",
# xlabel="Column index", ylabel="Sample Probability", legend=:bottomright,
# yrange=[0.001, 0.035])
# savefig("LevScoreDistribution.pdf")

# for bs in [1,2,5,10]
#     ps = []
#     for i in 1:100
#         sam_block = ITensorCPD.block_sample_factor_matrices(10, als.additional_items[:factor_weights], bs, 1)
#         prob_per_factor = hcat(als.additional_items[:factor_weights][2][eachcol(sam_block)[1]], als.additional_items[:factor_weights][3][eachcol(sam_block)[2]])'
#         pos1 = ITensorCPD.multi_coords_to_column((90,90), sam_block)
#         p = plot(eachrow(prob_per_factor)[1], lw=3, label="Mode 2")
#         plot!(eachrow(prob_per_factor)[2], lw=3, label="Mode 3")
#         plot!(levs23[pos1], lw=3, label="KRP")
#         plot!(title="Factor-wise Sampled Leverage Scores\nBlock Size = $(bs)", xlabel="Sample Number",
#         ylabel="Sample Probability", legend=:topleft)
#         push!(ps, p)
#     end
#     anim = @animate for i in ps
#         plot(i)
#     end
#     gif(anim, "samples_block_$(bs).gif", fps=2)
# end