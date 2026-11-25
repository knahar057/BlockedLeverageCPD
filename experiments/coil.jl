using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using Images, FileIO

function readcoil(coil_path,numobjects, elt=Float32)
# 	% Read COIL-100 images, given a number of objects
	X = Array{elt}(undef, 72*numobjects,128,128,3);
	for obj_num = 1:numobjects
		for i = 1:72
		    degrees = 5*(i-1);
		    filename = joinpath(coil_path, "obj$(obj_num)__$(degrees).png")
            # sprintf('%s/obj%d__%d.png',coil_path,obj_num,degrees);
		    # %X(i,:,:,obj_num) = rgb2gray(imread(filename, 'png'));
            img = load(filename)
			m = convert.(elt, Array(channelview(img)))
		    X[i + 72*(obj_num-1),:,:,:] .= permutedims(m, (2,3,1))
		end
	end
    is = Index.(size(X))
	return itensor(X, is);
end

#path = joinpath(@__DIR__, "..", "..", "data", "coil-100")
path="/Users/kpierce/Workspace/ipam-rnla-2025-kmgroup/julia/coil/coil-100/"
image_names = readdir(path)

imgs = 100
coil = readcoil(path, imgs, Float64);
rng = RandomDevice()
init_guess = ITensorCPD.random_CPD(coil, 20; rng);

ncoil = sqrt(sum(hadamard_product(coil,coil)))
rng = Random.RandomDevice()

#################################
### Solution error vs time #####
#################################
    # Computes the CP-ALS using an efficient version of the normal equations.
    # Stops when change in fit falls below 1e-3 
    algNormal = ITensorCPD.KRPFreeNormal()
    check_exact = ITensorCPD.FitCheck(1e-3, 20, ncoil)
    # als = ITensorCPD.compute_als(coil, init_guess; alg=algNormal, check_exact);
    ## Create a random initial guess with rank 20
    # ITensorCPD.decompose(coil, 0.01, 5; verbose=true, alg=algNormal, check=check_exact);
    time_500 = []
    fit_500 = []
    time_1000 = []
    fit_1000 = []
    time_2000 = []
    fit_2000 = []
    time_4000 = []
    fit_4000 = []
    for i in 1:20
        # als = ITensorCPD.compute_als(coil, init_guess; alg=ITensorCPD.BlockLevScoreSampled(500,i), 
        # check=ITensorCPD.NoCheck(20), normal=true, stop_resample=-1, variance_truncation=false);
        # push!(time_500, @elapsed opt = ITensorCPD.optimize(init_guess, als; verbose=false));
        # push!(fit_500, check_fit(als, opt))

        # als = ITensorCPD.compute_als(coil, init_guess; alg=ITensorCPD.BlockLevScoreSampled(1000,i), 
        # check=ITensorCPD.NoCheck(20), normal=true, stop_resample=-1, variance_truncation=false);
        # push!(time_1000, @elapsed opt = ITensorCPD.optimize(init_guess, als; verbose=false));
        # push!(fit_1000, check_fit(als, opt))

        # als = ITensorCPD.compute_als(coil, init_guess; alg=ITensorCPD.BlockLevScoreSampled(2000,i), 
        # check=ITensorCPD.NoCheck(20), normal=true, stop_resample=-1, variance_truncation=false);
        # push!(time_2000, @elapsed opt = ITensorCPD.optimize(init_guess, als; verbose=false));
        # push!(fit_2000, check_fit(als, opt))

        als = ITensorCPD.compute_als(coil, init_guess; alg=ITensorCPD.BlockLevScoreSampled(4000,i), 
        check=ITensorCPD.NoCheck(20), normal=true, stop_resample=-1, variance_truncation=false);
        push!(time_4000, @elapsed opt = ITensorCPD.optimize(init_guess, als; verbose=false));
        push!(fit_4000, check_fit(als, opt))
    end

    lw=4
    ms=7
    plot((1 .- time_500 ./ time_500[1]) .* 100, marker=:circle,
    label="500 Samples";ms,lw)
    plot!((1 .- time_1000 ./ time_1000[1]) .* 100, marker=:circle,
    label="1000 Samples";ms,lw)
    plot!((1 .- time_2000 ./ time_2000[1]) .* 100, 
    marker=:circle, label="2000 Samples"; lw,ms)
    plot!((1 .- time_4000 ./ time_4000[1]) .* 100, 
    marker=:circle, label="4000 Samples"; lw,ms)
    plot!(
        legend=:bottomright, label=nothing, 
    ylabel="Percent Time Reduction", xlabel="Block Size",  
    title="Acceleration from Block Sampling"
    )
    savefig("$(@__DIR__)/../plots/coil/$(imgs)_images/block_time_acceleration_multiple.pdf")

    plot(fit_500 ./ fit_500[1] .* 100, marker=:circle, label="500 Samples"; lw,ms) 
    plot!(fit_1000 ./ fit_1000[1] .* 100, marker=:circle, label="1000 Samples"; lw,ms) 
    plot!(fit_2000 ./ fit_2000[1] .* 100, marker=:circle, label="2000 Samples"; lw,ms) 
    plot!(fit_4000 ./ fit_4000[1] .* 100, marker=:circle, label="4000 Samples"; lw,ms) 
    plot!(legend=:bottomleft, label=nothing, 
    ylabel="Relative Accuracy of L2 Fit", 
    xlabel="Block Size", 
    color=:red,
    lw=4, ms=7, 
    title="Error Associated with Block Sampling")
    savefig("$(@__DIR__)/../plots/coil/$(imgs)_images/block_error_multiple.pdf")


################################
### Variance in loss function ##
################################

    algNormal = ITensorCPD.KRPFreeNormal()
    check_exact = ITensorCPD.FitCheck(1e-3, 20, ncoil)
    als = ITensorCPD.compute_als(coil, init_guess; alg=algNormal, check=check_exact);
    cpd_good = ITensorCPD.optimize(init_guess, als; verbose=true);
    als = ITensorCPD.compute_als(coil, init_guess; alg=ITensorCPD.BlockLevScoreSampled(2000,1), check=check_exact, normal=true);
    _,_, l, t = single_solve(als, cpd_good, 1);
    ## Create a random initial guess with rank 20
    # ITensorCPD.decompose(coil, 0.01, 5; verbose=true, alg=algNormal, check=check_exact);
    time = []
    fit = [[],[],[],[]]
    ref = []
    for (s,fact) in zip([500,1000,2000],1:4)
        fact = 1
        # als = ITensorCPD.compute_als(coil, init_guess; alg=ITensorCPD.KRPFreeNormal(), check=check_exact);
        # _,_, ref_loss, ref_time = single_solve(als, cpd_good, fact);
        # push!(ref, ref_loss)
        for i in 1:20
            lt = []
            lf = []
            for j in 1:5
            als = ITensorCPD.compute_als(coil, cpd_good; alg=ITensorCPD.BlockLevScoreSampled(s,i), 
            check=ITensorCPD.NoCheck(20), normal=true, stop_resample=-1, variance_truncation=false);
            # push!(time, @elapsed opt = ITensorCPD.optimize(init_guess, als; verbose=false));
            # push!(fit, check_fit(als, opt))
            _,_,f,t = single_solve(als, cpd_good, 1);
            #(init_guess, als; verbose=false);
            push!(lt, t)
            push!(lf, f)
            end
            # push!(time, maximum(lt))
            push!(fit[fact], lf)
            # push!(fit, check_fit(als, opt))
        end
    end

    plot((1 .- time ./ time[1]) .* 100, marker=:circle,
    legend=:topright, label=nothing, 
    ylabel="Percent Time Reduction", xlabel="Block Size", 
    lw=2, ms=7, 
    title="Acceleration from Block Sampling")
    savefig("$(@__DIR__)/../plots/coil/$(imgs)_images/block_time_acceleration.pdf")

    repeat = 5
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

################################################
### Fixed block size and increase sampling #####
#################################################

    plot((fit .- ref_loss))
    ## Compute a sampled CP-ALS using random single samples of the khatri-rao product.
    # check = ITensorCPD.FitCheck(1e-3, 50, ncoil);
    ## Using 100 samples
    bvals = []
    times = []
    check=ITensorCPD.CPDiffCheck(1e-10, 30)
    for block in [1,2,6,10,14]
        samp_vals = []
        time = []
        println("Starting block: $(block)")
        for sample in 400:200:5000
            best_fit = 0
            algSampl = ITensorCPD.BlockLevScoreSampled(sample, block)
            for i in 1:1
                als = ITensorCPD.compute_als(coil, init_guess; 
                alg=algSampl, check, stop_resample=-1, variance_truncation=false);

                init_guess = ITensorCPD.random_CPD(coil, 20; rng)

                push!(time, @elapsed opt = ITensorCPD.optimize(init_guess, als; verbose=false));
                f = check_fit(als, opt.factors, ITensorCPD.cp_rank(opt), opt.λ, 1)
                best_fit = f > best_fit ? f : best_fit
            end
            push!(samp_vals, best_fit)
        end
        push!(bvals, samp_vals)
        push!(times, time)
    end
    # f = open("$(@__DIR__)/scaled_coil_values.txt", "w")
    # for i in bvals
    #     println(f, Vector{Float64}(i))
    # end
    # close(f)

    using Plots
    xdata = [400:200:5000...]
    lw = 2
    ms = 8
    p = plot(xdata, bvals[1], label="Block Size: 1", lw=lw, marker=:circle, ms=ms)
    for (i, j) in zip(2:length(bvals),  [2,6,10,14])
        plot!(xdata, bvals[i], label="Block Size: $(j)",lw=lw,marker=:circle, ms=ms)
    end
    plot!(legendcolumns=2,
    title="Accuracy of Block Sampled CPD\nData: COIL",
    xlabel="Number of Samples",
    ylabel="Fit",
    legend=:bottomright,
    yrange=[0.6,0.71], 
    legendfontsize=10,
    xtickfontsize=9,
    ytickfontsize=9,
    labelfontsize=12,
    lw=4)
    savefig("$(@__DIR__)/../plots/coil/block_coil_data.pdf")

    using Plots
    # xdata = [450:50:5000...]
    p = plot(xdata, times[1] ./ 30, label="Block Size: 1", lw=lw, marker=:circle, ms=ms)
    for (i,j) in zip(2:length(bvals),  [2,6,10,14])
        plot!(xdata, times[i] ./ 30, label="Block Size: $(j)", lw=lw, marker=:circle, ms=ms)
    end
    # plot!(xrange = [400,2000], yrange =[0,1.0])
    plot!(legendcolumns=2,
    title="Time of Block Sampled CPD\nData: COIL",
    xlabel="Number of Samples",
    ylabel="Time per ALS Iteration (s)",
    legendfontsize=10,
    xtickfontsize=9,
    ytickfontsize=9,
    labelfontsize=12,)
    savefig("$(@__DIR__)/../plots/coil/block_coil_time_data.pdf")

    bvals2 = []
    times2 = []
    block_vals = [1,2,6,10,]
    sample = 2000
    for block in block_vals
        samp_vals = []
        time = []
        println("Starting block: $(block)")
        for scale in block_vals[1:end]
            best_fit = 0
            algSampl = ITensorCPD.BlockLevScoreSampled(sample * scale, block)
            for i in 1:3
                als = ITensorCPD.compute_als(coil, init_guess; alg=algSampl, check);
                init_guess = ITensorCPD.random_CPD(coil, 20; rng)
                #als = ITensorCPD.update_samples(als, 5000; reshuffle=false);
                push!(time, @elapsed opt = ITensorCPD.optimize(init_guess, als; verbose=false));
                f = check_fit(als, opt.factors, ITensorCPD.cp_rank(opt), opt.λ, 1)
                best_fit = f > best_fit ? f : best_fit
            end
            push!(samp_vals, best_fit)
        end
        push!(bvals2, samp_vals)
        push!(times2, time)
    end

    using Plots
    #xdata = [450:50:1000...]
    xdata = 500 .* block_vals[1:end]
    p = plot(block_vals[1:end], bvals2[1][1] .* ones(length(xdata)), label="Block Size: 1"; lw, marker=:circle, ms=ms)
    for i in 2:length(bvals2)
        plot!( block_vals[1:end], bvals2[i], label="Block Size: $(block_vals[i])"; lw,  marker=:circle, ms=ms)
        # plot!( block_vals[1:end-1], bvals[i], label="$(block_vals[i])", markers=:circle)
    end
    plot!(legendcolumns=2,
    title="Accuracy of Block Sampled CPD\nSamples = $(sample)",
    xlabel="Factor times Single Block Sample Size",
    # legendtitle="Block Size",
    # legendtitlefontsize=9,
    legendfontsize=10,
    xtickfontsize=9,
    ytickfontsize=9,
    labelfontsize=12,
    yrange=[0.62,0.681],
    ylabel="Fit")
    savefig("$(@__DIR__)/../plots/coil/scaled_block_coil_data_$(sample).pdf")

    lw=2
    using Plots
    p = plot(block_vals[1:end], times2[1][1]/30 * ones(length(block_vals[1:end])), label="Block Size: 1"; lw, marker=:circle, ms=ms)
    for i in 2:length(bvals2)
        plot!(block_vals[1:end], times2[i][1:3:end] ./ 30, label="Block Size: $(block_vals[i])"; lw,  marker=:circle, ms=ms)
    end
    plot!(legendcolumns=2,
    title="Time per Iteration of Block Sampled CPD\nSamples = $(sample)",
    xlabel="Factor times Single Block Sample Size",
    ylabel="Time per Iteration (s)",
    yrange=[0.1, 2.3],
    legendfontsize=10,
    xtickfontsize=9,
    ytickfontsize=9,
    labelfontsize=12,
    )
    savefig("$(@__DIR__)/../plots/coil/scaled_block_coil_time_$(sample).pdf")


########################################
### Checking if more iterations help ###
########################################

    # Computes the CP-ALS using an efficient version of the normal equations.
    # Stops when change in fit falls below 1e-3 
    algNormal = ITensorCPD.KRPFreeNormal()
    check_exact = ITensorCPD.FitCheck(1e-3, 20, ncoil)
    als = ITensorCPD.compute_als(coil, init_guess; alg=algNormal, check=check_exact);
    ITensorCPD.optimize(init_guess, als; verbose=true);
    ref_acc = als.check.final_fit
    #ref_acc = 0.6843696156160473 
    als = nothing
    GC.gc()
    
    repeats = 1
    checks = [20,50,100,250,500,1000,1500,2000]
    samples = [500,1000,1500,2000,4000]
    fits_10 = Array{Float64}(undef, (length(samples), repeats, length(checks)));
    fits_20 = similar(fits_10)
    ## Create a random initial guess with rank 20
    
    for (s_idx, s) in enumerate(samples)
        @show s
        f10 = @view fits_10[s_idx,:,:]
        f20 = @view fits_20[s_idx,:,:]
        for (check_idx, c) in enumerate(checks)
            @show c
            for rep in 1:repeats
                init_guess = ITensorCPD.random_CPD(coil, 20; rng)
                als = ITensorCPD.compute_als(coil, init_guess; 
                    alg=ITensorCPD.BlockLevScoreSampled(s,10), 
                check=ITensorCPD.NoCheck(c), normal=true);
                opt = ITensorCPD.optimize(init_guess, als; verbose=false)
                f10[rep,check_idx] = check_fit(als, opt)

                als = ITensorCPD.compute_als(coil, init_guess; 
                    alg=ITensorCPD.BlockLevScoreSampled(s,20), 
                check=ITensorCPD.NoCheck(c), normal=true);
                opt = ITensorCPD.optimize(init_guess, als; verbose=false)
                f20[rep,check_idx] = check_fit(als, opt)
                GC.gc()
            end
        end
    end

    lw=4
    ms=7
    hline(checks, [ref_acc], label="Canonical ALS", ls=:dash;lw,ms)
    for (s_idx, s) in enumerate(samples)
        plot!(checks, fits_10[s_idx,1,:], label="Samples: $(s)", marker=:square, c=s_idx+1; lw,ms)
        plot!(checks, fits_20[s_idx,1,:], label="Samples: $(s)", marker=:circle,c=s_idx+1; lw,ms)
    end
    plot!(
        xlabel="Number of ALS iterations",
        ylabel="Fit",
        title="CPD Fit versus number of ALS iterations
        Data: COIL-100",
        legend=:right
    )
    savefig("COIL-ALS-iter-test-block-10-20.pdf")

    plot(fit_500 ./ fit_500[1] .* 100, marker=:circle, label="500 Samples"; lw,ms) 
    plot!(fit_1000 ./ fit_1000[1] .* 100, marker=:circle, label="1000 Samples"; lw,ms) 
    plot!(fit_2000 ./ fit_2000[1] .* 100, marker=:circle, label="2000 Samples"; lw,ms) 
    plot!(fit_4000 ./ fit_4000[1] .* 100, marker=:circle, label="4000 Samples"; lw,ms) 
    plot!(legend=:bottomleft, label=nothing, 
    ylabel="Relative Accuracy of L2 Fit", 
    xlabel="Block Size", 
    color=:red,
    lw=4, ms=7, 
    title="Error Associated with Block Sampling")
    savefig("$(@__DIR__)/../plots/coil/$(imgs)_images/block_error_multiple.pdf")
