using ITensors, ITensorMPS
using DataFrames

function compute_correlation_paulix_norm(psi::MPS)
    xxcorr = correlation_matrix(complex(psi), "Sx", "Sx")
    dig = diag(xxcorr)
    dig = sqrt.(dig)
    for i=1:length(dig)
        xxcorr[i,:] = xxcorr[i,:] ./ dig[i]
        xxcorr[:,i] = xxcorr[:,i] ./ dig[i]
    end
    return vec(real(xxcorr))

end;


function compute_correlation_pauliy_norm(psi)
    yycorr = correlation_matrix(complex(psi), "Sy", "Sy")
    dig = diag(yycorr)
    dig = sqrt.(dig)
    for i=1:length(dig)
        yycorr[i,:] = yycorr[i,:] ./ dig[i]
        yycorr[:,i] = yycorr[:,i] ./ dig[i]
    end
    return vec(real(yycorr))

end;


function compute_correlation_pauliz_norm(psi)
    zzcorr = correlation_matrix(complex(psi), "Sz", "Sz")
    dig = diag(zzcorr)
    dig = sqrt.(dig)
    for i=1:length(dig)
        zzcorr[i,:] = zzcorr[i,:] ./ dig[i]
        zzcorr[:,i] = zzcorr[:,i] ./ dig[i]
    end
    return vec(real(zzcorr))
end;

function compute_correlation_rydberg(corr)
    return "[" * join(vec(corr), ", ") * "]"
end;

# computing entanglement von Neumann entropy for 1-dimension model
function compute_vonnu_entropy_1d(psi, b)
    psi = orthogonalize(psi, b)
    U,S,V = svd(psi[b], (linkinds(psi, b-1)..., siteinds(psi, b)...))
    SvN = 0.0     
    for n=1:ITensors.dim(S, 1)
        p = S[n,n]^2 # probability of n-th Schmidt coefficient
        SvN -= p * log(p) 
    end
    return SvN
end;

# computing entanglement 2-order Renyi entropy for 1-dimension model
function compute_renyi_entropy(psi, b, alpha=2)
    psi = orthogonalize(psi, b)
    s = siteinds(psi)
    U,S,V = svd(psi[b], (linkinds(psi, b-1)..., siteinds(psi, b)...))
    Sr = 0.0
    for i=1:ITensors.dim(S, 1)
        p = S[i,i]^2 
        Sr += p^alpha
    end
    Sr = 1 / (1 - alpha) * log(Sr)
    return Sr
end;


# Expected Values of H (energy)
function compute_expectation_value(psi, H)
    ex_W = inner(psi', H, psi)
    return ex_W
end;


function exact_renyi_entropy_size_two(psi::MPS, a::Int, b::Int)
    N = length(psi)
    sites = siteinds(psi)
    @assert 1 ≤ a < b ≤ N
    @assert b == a+1
    orthogonalize!(psi,a)
    psidag=prime(dag(psi),linkinds(psi))
    rho_ab = prime(psi[a], linkinds(psi, a-1)) * prime(psidag[a], sites[a])
    rho_ab *= prime(psi[b], linkinds(psi, b)) * prime(psidag[b], sites[b])
    D1,_ = eigen(rho_ab)
    return -log2(sum(real(diag(D1).*diag(D1))))
end;