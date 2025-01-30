using Hadamard
using TensorOperations
using cuTENSOR
using Tullio
include("utils.jl")


function snapshot_state(obs_sample)
    num_qubits = size(obs_sample)[1]
    # local qubit unitaries
    phase_z = [1 0; 0 -1im]
    hadamard = complex(Hadamard.hadamard(2)) / sqrt(2)
    identi = [1 0; 0 1]
    unitaries = [hadamard, hadamard * phase_z, identi]

    # reconstructing the snapshot state from local Pauli measurements
    rho_snapshot = [1]
    for i=1:num_qubits
        Ui = unitaries[div(obs_sample[i], 2) + 1]
        obs_sample[i] % 2 == 0 ? state = [1 0; 0 0] : state = [0 0; 0 1]
        local_rho = 3 * (adjoint(Ui) * state * Ui) - identi
        rho_snapshot = kron(rho_snapshot, local_rho)
    end
    return rho_snapshot
end;



function snapshot_state_local(obs_sample_one)
    # input: obs_sample_one: 2x2 matrix
    phase_z = [1 0; 0 -1im]
    hadamard = complex(Hadamard.hadamard(2)) / sqrt(2)
    identi = [1 0; 0 1]
    unitaries = [hadamard, hadamard .* phase_z, identi]
    
    rho_snapshot = [1]
    Ui = unitaries[div(obs_sample_one, 2) + 1]
    obs_sample_one % 2 == 0 ? state = [1 0; 0 0] : state = [0 0; 0 1]
    local_rho = 3 * (adjoint(Ui) * state * Ui) - identi
    rho_snapshot = kron(rho_snapshot, local_rho)
    return rho_snapshot
end;



function reconstruct_state_from_shadow(obs)
    num_shots, num_qubits = size(obs)[1], size(obs)[2]
    obs_samples = obs2samples(num_qubits, num_shots, obs)
    shadow_rho = complex(zeros(2 ^ num_qubits, 2 ^ num_qubits))
    for i=1:num_shots
        shadow_rho += snapshot_state(obs_samples[i])
    end;
    res_mat = shadow_rho / num_shots
    res_real, res_imag = vec(real(res_mat)), vec(imag(res_mat))
    res_vec = vcat(res_real, res_imag)
    return res_mat, res_vec # 2^n x 2^n complex matrix, 1 x 2^{n+2} float matrix
end;


function reconstruct_state_from_shadow_local(obs)
    num_shots, num_qubits = size(obs)[1], size(obs)[2]
    obs_samples = obs2samples(num_qubits, num_shots, obs)
    local_state = complex(zeros(2, 2))
    local_snapshots = complex(zeros(num_shots, num_qubits, 2, 2))

    for i=1:num_shots
        for j=1:num_qubits
            local_state = snapshot_state_local(obs_samples[i][j])
            local_snapshots[i, j, :, :] = local_state
        end;
    end;
    return local_snapshots
end;


## Adapted from https://github.com/lllewis234/improved-ml-algorithm/blob/master/dataloader.py
function shadow_alignment(m, n)
    if m > n
        return shadow_alignment(n, m) 
    end;
    if m == n
        return 3  # same basis and outcome
    elseif m % 2 == 0 && m == n - 1
        return -3  # same basis but different outcome
    else
        return 0
    end;
end;


## Adapted from https://github.com/lllewis234/improved-ml-algorithm/blob/master/dataloader.py
## calculate the approximate correlation matrix via classical shadow
function cal_shadow_correlation(obs)
    num_shots, num_qubits = size(obs)[1], size(obs)[2]
    approx_correlation = []
    samples = obs2samples(num_qubits, num_shots, obs)
    for i = 1:num_qubits
        for j = 1:num_qubits
            if i==j
                push!(approx_correlation, 1)
                continue
            end;
            corr = 0.0
            for measurement in samples
                corr += shadow_alignment(measurement[i], measurement[j])
            end;
            push!(approx_correlation, corr / length(samples))
        end;
    end;
    return [Int.(samples[i]) for i = 1:num_shots], float.(approx_correlation)
end


# Adapted from Wang et al. (2022)
# Renyi entropy of the subsystem of size at most 1
function cal_shadow_entropy(obs)
    # 2-order renyi entropy 
    num_shots, num_qubits = size(obs)[1], size(obs)[2]
    local_snapshots = reconstruct_state_from_shadow_local(obs) # shots x qubits x 2 x 2
    @tullio temp[i] := local_snapshots[t, i, a, b] * local_snapshots[m, i, b, a] 
    entropies = -log.(temp) .+ 2 * log(num_shots)
    return real(entropies)
end;


# Adapted from Huang et al.(2020)
# Renyi entropy of the subsystem of size at most 2
function cal_shadow_renyi_entropy(obs, subsystem)
    M, N = size(obs)[1], size(obs)[2]
    subsystem_size = length(subsystem)
    renyi_sum_of_binary_outcome = zeros(Int, 1 << (2 * subsystem_size))
    renyi_number_of_outcomes = zeros(Int, 1 << (2 * subsystem_size))
    for t in 1:M
        encoding, cumulative_outcome = 0, 1
        renyi_sum_of_binary_outcome[1] += 1
        renyi_number_of_outcomes[1] += 1
                
        for b in 1:(1 << subsystem_size) - 1
            change_i = trailing_zeros(b)  # Count trailing zeros to determine the index of the change
            index_in_original_system = subsystem[change_i + 1]  # Convert to 1-based index      
            pauli_basis, pauli_outcome = obs[t,:][index_in_original_system][1], obs[t,:][index_in_original_system][2]
            cumulative_outcome *= (pauli_outcome == 1 ? -1 : 1)  # Pauli outcome
            encoding ⊻= (pauli_basis == "X" ? 1 : (pauli_basis == "Y" ? 2 : 3)) << (2 * (change_i))  # Encoding Pauli basis
            
            renyi_sum_of_binary_outcome[encoding+1] += cumulative_outcome
            renyi_number_of_outcomes[encoding+1] += 1
        end;
    end;
    # Initialize level count and total
    level_cnt, level_ttl = zeros(Int, subsystem_size+1), zeros(Int, subsystem_size+1)

    # Calculate level counts
    for c in 0:(1 << (2 * subsystem_size)) - 1
        nonId = count(i -> ((c >> (2 * i)) & 3) != 0, 0:subsystem_size-1)
        if renyi_number_of_outcomes[c+1] >= 2
            level_cnt[nonId + 1] += 1
        end;
        level_ttl[nonId + 1] += 1
    end;

    # Calculate predicted entropy
    predicted_entropy = 0.0
    for c in 0:(1 << (2 * subsystem_size)) - 1
        if renyi_number_of_outcomes[c+1] <= 1
            continue
        end;
        nonId = count(i -> ((c >> (2 * i)) & 3) != 0, 0:subsystem_size-1)
        if level_cnt[nonId + 1] > 0
            predicted_entropy += 1.0 / (renyi_number_of_outcomes[c + 1] * (renyi_number_of_outcomes[c + 1] - 1)) *
                                     (renyi_sum_of_binary_outcome[c + 1]^2 - renyi_number_of_outcomes[c + 1]) /
                                     (1 << subsystem_size) * level_ttl[nonId + 1] / level_cnt[nonId + 1]
        end;
    end;
    res = -log2(min(max(predicted_entropy, 1.0 / 2^subsystem_size), 1.0 - 1e-9))
    #println("subsystem: ", subsystem, " entropy: ", @sprintf("%.8f", res))
    return res

end;