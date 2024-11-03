using Hadamard
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
end


function reconstruct_state_from_shadow(obs)
    num_shots, num_qubits = size(obs)[1], size(obs)[2]
    obs_samples = obs2samples(num_qubits, num_shots, obs)
    shadow_rho = complex(zeros(2 ^ num_qubits, 2 ^ num_qubits))
    for i=1:num_shots
        shadow_rho += snapshot_state(obs_samples[i])
    end;
    res_mat = shadow_rho / num_shots

    # res 2^n x 2^n complex matrix 
    # ----> 1 x 2^{n+2} float matrix
    res_real, res_imag = vec(real(res_mat)), vec(imag(res_mat))
    res_vec = vcat(res_real, res_imag)
    return res_mat, res_vec
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