include("$(@__DIR__)/../test_env.jl")
using LinearAlgebra, Statistics, DataFrames, CategoricalArrays, StatsPlots
using LaTeXStrings
using Random

function main(elt, ids, bad, r, n)
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

    for rk in [dim(r)]
        r = Index(rk, "CP_rank")
        err_leverage = Vector{Float64}()
        cp_T = ITensorCPD.random_CPD(T,r; rng)

        lev_error_mat1 = Matrix{Float64}(undef, 20, length(samples))
        lev_error_mat5 = Matrix{Float64}(undef, 20, length(samples))
        lev_error_mat10 = Matrix{Float64}(undef, 20, length(samples))
        lev_error_mat50 = Matrix{Float64}(undef, 20, length(samples))
        lev_mats = [lev_error_mat1, lev_error_mat5, lev_error_mat10, lev_error_mat50]
        for (b, lm) in zip(blocks, lev_mats)
           for (s, i) in zip(samples[1:end], 1:length(samples))
                lev_error_vect = Vector{Float64}()
                for q=1:20
                    @show b, s
                    # success = false
                    # while !success
                        # try
                            alslev = ITensorCPD.compute_als(T,cp_T;alg = ITensorCPD.BlockLevScoreSampled(s, b),check = check_piv, variance_truncation=false);
                            int_opt_T = ITensorCPD.optimize(cp_T, alslev; verbose=false);
                            push!(lev_error_vect,check_fit(T, int_opt_T)) 
                            # success = true
                        # catch
                        # end
                    # end
                end
                lm[:,i] .= lev_error_vect
            end
        end
        
        println("Finished sample stuff")
        err_leverages = [maximum.(eachcol(lm)) for lm in lev_mats]

        alsNormal = ITensorCPD.compute_als(T, cp_T; alg=ITensorCPD.KRPFreeNormal(), check = check_direct);
        opt_T = ITensorCPD.optimize(cp_T, alsNormal; verbose=true);
        direct_error = check_fit(alsNormal, opt_T.factors, r, opt_T.λ, 1)

        markersize = 8
        plt2 = plot(samples, direct_error .* ones(length(samples)), marker=:o, label="Normal Equations"; markersize)
        for (b,i) in zip(blocks, 1:length(blocks))
            plot!(samples, err_leverages[i], marker=:o, label="Block Size: $(b)"; markersize)
        end
        plot!(
        xticks=samples[1]:samples[1]:samples[end],
        yrange=[0.2,1.1],
        yticks=0.0:0.2:1.0,
        )

        xlabel!("Number of Samples")
        ylabel!("Best CPD Fit")
        ll = "$(ids[1])"
        for i in ids[2:end]
            ll *= " " * L"\times" * " $(i)"
        end
        title!("Convergence of the Best CPD\n$(ll)")
        plot!(legendfontsize=10,
        xtickfontsize=9,
        ytickfontsize=9,
        labelfontsize=12)

        n = nothing
        if elt == Float64
            n = "F64"
        else
            n = "F32"
        end

        savefig("$(@__DIR__)/../plots/worse_case/mode_$(name)_rank_$(rk)_modified_test_$(n)_$(ids[1])_$(length(ids)).pdf")

        # display(plt2)

        # ## Violin Plot of modified tensor
        df = DataFrame(
            sample = vcat(repeat(samples, inner=20), repeat(samples, inner=20), repeat(samples, inner=20), repeat(samples, inner=20)),
            method = vcat(fill("Block Size 1", 20*length(samples)),
                        fill("Block Size 5", 20*length(samples)),
                        fill("Block Size 10", 20*length(samples)),
                        fill("Block Size 20", 20*length(samples))),
            error  = error  = vcat(vec(lev_error_mat1), vec(lev_error_mat5), vec(lev_error_mat10), vec(lev_error_mat50))
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
        ylabel!("Distribution of CPD Fits")
        title!("Distribution of Optimized CPD\n$(ll)")
        plot!(legend=ids[1] == 400 ? :topleft : :bottomright,
        yticks=[0.5,0.6,0.7,0.8,0.9,1.0], 
        yrange=[0.5, 1.01])
        plot!(legendfontsize=10,
        xtickfontsize=9,
        ytickfontsize=9,
        labelfontsize=12)
        savefig("$(@__DIR__)/../plots/worse_case/mode_$(name)_distribution_rank_$(rk)_modified_test_$(n)_$(ids[1])_$(length(ids)).pdf")
    end
end
 
elt = Float64 
ids = (400,400,400)
bad = 3
r = 10
n = [1,2,3]

main(elt, ids, bad, r, n)
### Compute gifs of leverage score sampling
cpd_exact = ITensorCPD.random_CPD(zeros(ids...), r);
# main(elt, ids, bad, r, n)
for (x, is) in zip(n, ids)
    array(cpd_exact[x])[1:bad, :] .*= 100;
    array(cpd_exact[x]) .= array(cpd_exact[x])[randperm(is), :]
    tmp, lam = ITensorCPD.row_norm(cpd_exact[x], ind(cpd_exact[x], 1))
    array(cpd_exact[x]) .= tmp
    cpd_exact[] .= lam
end
cpr = ITensorCPD.cp_rank(cpd_exact)

i23 = Index(90 * 90, "234")
krp23 = ITensorCPD.had_contract([cpd_exact[2], cpd_exact[3]], cpr)
krp23 = itensor((reshape(array(krp23), (90 * 90, r))), i23, cpr)

#levs1 = ITensorCPD.compute_leverage_score_probabilitiy(cpd_exact[1], ind(cpd_exact[1],1))
levs23 = ITensorCPD.compute_leverage_score_probabilitiy(krp23, i23)
plot(levs23)
sum(levs23 .> 1e-2)

it = itensor(zeros(ids...), inds(cpd_exact))
als = ITensorCPD.compute_als(it, cpd_exact; alg=ITensorCPD.LevScoreSampled(1));

plot(als.additional_items[:factor_weights][2], lw=3, label="Mode 2")
plot!(als.additional_items[:factor_weights][3], lw=3, label="Mode 3")
plot!(title="Factor-wise Leverage Score Sampling Probabilities",
xlabel="Column index", ylabel="Sample Probability", legend=:bottomright,
yrange=[0.001, 0.035])
savefig("LevScoreDistribution.pdf")

for bs in [1,2,5,10]
    ps = []
    for i in 1:100
        sam_block = ITensorCPD.block_sample_factor_matrices(10, als.additional_items[:factor_weights], bs, 1)
        prob_per_factor = hcat(als.additional_items[:factor_weights][2][eachcol(sam_block)[1]], als.additional_items[:factor_weights][3][eachcol(sam_block)[2]])'
        pos1 = ITensorCPD.multi_coords_to_column((90,90), sam_block)
        p = plot(eachrow(prob_per_factor)[1], lw=3, label="Mode 2")
        plot!(eachrow(prob_per_factor)[2], lw=3, label="Mode 3")
        plot!(levs23[pos1], lw=3, label="KRP")
        plot!(title="Factor-wise Sampled Leverage Scores\nBlock Size = $(bs)", xlabel="Sample Number",
        ylabel="Sample Probability", legend=:topleft)
        push!(ps, p)
    end
    anim = @animate for i in ps
        plot(i)
    end
    gif(anim, "samples_block_$(bs).gif", fps=2)
end