using ITensorCPD: compute_krp, matricize_tensor, solve_ls_problem, row_norm, post_solve

function check_loss(target, cpd, grammian, fact, updated_factor)
    factors = cpd.factors
    cpr = ITensorCPD.cp_rank(cpd)
    n2 = (target * target)[]
    mttkrp = ITensorCPD.had_contract([target, factors[1:end .!= fact]...], cpr)
    inner = (mttkrp * updated_factor)[]

    g = itensor(ones(dim(grammian[1])), inds(grammian[1]))
    for x in grammian[1:end .!= fact]
        g = ITensors.hadamard_product!(g, g, x)
    end

    cpd2 = (g * (updated_factor * prime(updated_factor, tags=tags(cpr))))[]


    loss_function = n2 - 2 * inner + cpd2
    println("CPD rank\tMode\tLoss")
    println("$(dim(cpr))\t\t$(fact)\t$(loss_function)")
    return loss_function
end

function single_solve(als, cpd, fact)
   time = @elapsed begin
        factors = deepcopy(cpd.factors)
        λ = deepcopy(cpd.λ)
        rank = ITensorCPD.cp_rank(cpd)
        target_ind = ind(cpd, fact)
        
        krp = compute_krp(als.mttkrp_alg, als, factors, cpd, rank, fact)

        mtkrp = matricize_tensor(als.mttkrp_alg, als, factors, cpd, rank, fact)

        solution = solve_ls_problem(als.mttkrp_alg, als, krp, mtkrp, rank)
                
        factors[fact], λ = ITensorCPD.row_norm(solution, target_ind)

        #array(λ) .*= 1 ./ array(normalizers)

        post_solve(als.mttkrp_alg, als, factors, λ, cpd, rank, fact)
    end

    
    #fit = check_fit(alsRef, ITensorCPD.CPD{ITensor}(factors, λ))
    gram = [f * prime(f, tags=tags(rank)) for f in factors]
    fit = check_loss(als.target, cpd, gram, fact, solution)

    return ITensorCPD.CPD{ITensor}(factors, λ), als, fit, time
end