using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using DelimitedFiles
using Plots

# ============================================================
# Plot CPD scalability results
# ============================================================
#
# Specify the scaling results file below.
#
# Expected columns:
#   tensor_size, method, block_size, time_seconds, fit, memory_mb
#
# The script generates:
#   - Runtime vs. tensor size
#   - Fit vs. tensor size
#   - Allocated memory vs. tensor size
# ============================================================


# ============================================================
# Results file
# ============================================================

results_file = "miranda_scaling_results.csv"


# ============================================================
# File locations
# ============================================================

results_path = joinpath(
    @__DIR__,
    "..",
    "results",
    results_file,
)

figure_dir = joinpath(
    @__DIR__,
    "..",
    "figures",
)

mkpath(figure_dir)


# ============================================================
# Determine dataset name
# ============================================================

dataset = replace(
    results_file,
    "_scaling_results.csv" => "",
)

dataset_title = Dict(
    "chicago" => "Chicago Crime",
    "miranda" => "Miranda",
    "coil" => "COIL",
)[dataset]


lw = 3
ms = 8

# ============================================================
# Read results
# ============================================================

data, header = readdlm(
    results_path,
    ',',
    header=true,
)

# Tensor sizes
tensor_sizes = unique((data[:, 1]))

# Regular leverage-score results
regular_rows = data[:, 2] .== "Regular"

regular_times = Float64.(data[regular_rows, 4])
regular_fits = Float64.(data[regular_rows, 5])
regular_memory = Float64.(data[regular_rows, 6])

# Block sizes used in the experiment
blocked_rows = data[:, 2] .== "Blocked"
blocked_data = data[blocked_rows,:]

block_sizes = unique(
    Int.(data[blocked_rows, 3])
)

### This matrix is dimension of index x number of blocks
block_times_mean = Array{Float64}(undef, sum(regular_rows), length(block_sizes))
block_mem_mean = similar(block_times_mean)
block_fits_max = similar(block_times_mean)
block_fits_min = similar(block_times_mean)
for i in 1:5:20
    block_times_mean[i÷5+1,:]  = blocked_data[i:i+4,4]
    block_mem_mean[i÷5+1,:]  = blocked_data[i:i+4,6]
    block_fits_max[i÷5+1,:]  = blocked_data[i:i+4,8]
    block_fits_min[i÷5+1,:]  = blocked_data[i:i+4,9]
end

# ============================================================
# Plot 1: Runtime
# ============================================================

markers = [:circle, :cross, :diamond,  :xcross, :utriangle]
p_time = plot(
    tensor_sizes,
    regular_times,
    ls=:dot,
    label = "Stanard ALS";
    lw,ms
)

for (i,block) in enumerate(block_sizes)
    plot!(
        tensor_sizes,
        block_times_mean[:,i],
        marker = markers[i],
        label = "Block Size: $(block)",
        ;
        lw,ms
    )
end

plot!(
    xlabel = "Tensor Size (n × 256 × 256)",
    ylabel = "Time (s)",
    title = "Runtime Scaling:
    Data: Miranda",
    legendfontsize=9,
    xtickfontsize=9,
    ytickfontsize=9,
    labelfontsize=12,
    yrange=[10^-1,10^2],
    yscale=:log10,
    legend=:topleft
)

savefig(
    p_time,
    joinpath(
        figure_dir,
        "$(dataset)_scaling_time.pdf",
    ),
)


# ============================================================
# Plot 2: CPD fit
# ============================================================

p_fit = plot(
    tensor_sizes,
    regular_fits,
    ls=:dot,
    label = "Standard ALS";
    ms,lw
)

for (i,block) in enumerate(block_sizes)
    x = tensor_sizes
    y1 = block_fits_max[:,i]
    y2 = block_fits_min[:,i]
    mid = (y1 .+ y2) ./ 2   #the midpoints (usually representing mean values)
    w = (y2 .- y1) ./ 2     #the vertical deviation around the means

    plot!(x, mid, ribbon = w , 
    fillalpha = 0.35, c = i+1,
    marker=markers[i],
    legend = :topleft, label = "Block Size: $(block)";
    ms,lw)

end

plot!(
    xlabel = "Tensor Size (n × 256 × 256)",
    ylabel = "Fit",
    title = "Accuracy Scaling:
Data: Miranda",
    legendfontsize=10,
    xtickfontsize=9,
    ytickfontsize=9,
    labelfontsize=12,
    legend=:topright
)

savefig(
    p_fit,
    joinpath(
        figure_dir,
        "$(dataset)_scaling_fit.pdf",
    ),
)


# ============================================================
# Plot 3: Allocated memory
# ============================================================

p_memory = plot(
    tensor_sizes,
    regular_memory,
    ls=:dot,
    label = "Standard ALS";
    ms,lw
)

for (i, block) in enumerate(block_sizes)
    plot!(
        tensor_sizes,
        block_mem_mean[:,i],
        marker = markers[i],
        label = "Block Size: $(block)";
        ms,lw
    )
end


plot!(
    xlabel = "Tensor Size (n × 256 × 256)",
    ylabel = "Allocated Memory (MB)",
    title = "Memory Scaling:
Data: Miranda",
    legendfontsize=10,
    xtickfontsize=9,
    ytickfontsize=9,
    labelfontsize=12,
    # yscale=:log10,
    legend=:topleft
)

savefig(
    p_memory,
    joinpath(
        figure_dir,
        "$(dataset)_scaling_memory.pdf",
    ),
)


println("\nScaling figures saved to:")
println(figure_dir)