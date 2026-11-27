using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using LinearAlgebra, Statistics, DataFrames, CategoricalArrays, StatsPlots, Random
using LaTeXStrings

## ijk are the dimensions bad are the number of large leverage scores per factor r is the Cp rank n is which modes are embedded with large leverage scores
## λ is the random value added to the factors to bump them away from the true answer and f is the factor being updated with single solve
function main(elt, i,j,k, bad, r, n, λ, f)
    name = "$(n)"
    cpd_exact = ITensorCPD.random_CPD(zeros(i,j,k), r);
    if n != 0
        for (x, is) in zip(n, k .* ones(Int, length(n)))
            @show x
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
    check_direct = ITensorCPD.FitCheck(1e-3, 1, norm(T))
    
    lev_error_mat = Matrix{Float64}(undef, 20, length(samples))
    rng=RandomDevice()
    cp_T = nothing
    blocks = [1, 5, 10, 20]

    for rk in [dim(r)]
        r = Index(rk, "CP_rank")
        err_leverage = Vector{Float64}()
        cp_T = deepcopy(cpd_exact)
        for i in cp_T.factors
            array(i) .+= λ .* randn(size(i))
        end

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
                            #_,_,fit = single_solve(alslev, cp_T, f)
                            push!(lev_error_vect,check_fit(alslev, int_opt_T)) 
                            # push!(lev_error_vect, fit)
                            # success = true
                        # catch
                        # end
                    # end
                end
                lm[:,i] .= lev_error_vect
            end
        end
        
        println("Finished sample stuff")
        err_leverages = [median.(eachcol(lm)) for lm in lev_mats]

        alsNormal = ITensorCPD.compute_als(T, cp_T; alg=ITensorCPD.KRPFreeNormal(), check = check_piv);
        # opt_T = ITensorCPD.optimize(cp_T, alsNormal; verbose=true);
        # direct_error = check_fit(alsNormal, opt_T.factors, r, opt_T.λ, 1)
        _,_,direct_error = single_solve(alsNormal, cp_T, f)

        plt2 = plot(samples, direct_error .* ones(length(samples)), marker=:star, label="Normal Equations", markersize=7)
        #plt2 = plot()
        for (b,i) in zip(blocks, 1:length(blocks))
            plot!(samples, err_leverages[i], marker=:o, label="Block Size: $(b)", markersize=7)
            # yrange=[0, 1.1], yticks=-0.5:0.1:1.1)
        end
        plot!(
        legendtitlefontsize=7,
        xticks=samples[1]:samples[1]:samples[end],
         yscale=:log10
        # xrotation=60
        )

        xlabel!("Number of Samples")
        ylabel!("Loss Function, " * L"$||T_2 - B K ||^2$")
        title!("Modified Synthetic Tensor Test:\n Rank $rk")

        et = nothing
        if elt == Float64
            et = "F64"
        else
            et = "F32"
        end

        figure_dir = joinpath(
            @__DIR__,
            "..",
            "figures",
            "worse_case"
        )
        savefig("$(figure_dir)/single_mode_$(f)_$(name)_rank_$(rk)_modified_test_$(et).pdf")

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
        ylabel!("Distribution of CPD Loss Function Values")
        title!("Modified Synthetic tensor test: \n Rank $rk")
        plot!(legend=:topright, yrange=[10,4000])
        # yticks=-0.5:0.1:1.1, yrange=[0, 1.1])
        savefig("$(@__DIR__)/../plots/worse_case/single_mode_$(f)_$(name)_distribution_rank_$(rk)_modified_test_$(n).pdf")
    end
end
 
λ = 0.01
elt = Float64 
i,j,k = 90,90,90
bad = 3
r = 30
n = [1,2,3]
main(elt, i,j,k, bad, r, n, λ, 3)