using CSV
using DataFrames



function write_dataset_to_csv_rydberg(dataset, filename, attributes)
    df = DataFrame(nsites=dataset[1][1], unit_disk_radius=dataset[1][2], C6=dataset[1][3],
                    t=dataset[1][4], omega=dataset[1][5], delta=dataset[1][6], R_0=dataset[1][7],
                    energy=dataset[1][8], entropy=dataset[1][9], corr_matrix=dataset[1][10])
    CSV.write(filename, df)
end;


function write_dataset_to_csv_tf_ising(dataset, filename)
    df = DataFrame(qubits_num=dataset[1][1], J=dataset[1][2], energy=dataset[1][3], entropy=dataset[1][4], corrzz=dataset[1][5])
    CSV.write(filename, df)
end;


# Adatped from Huang et al. 2023
function make_coupling_strength_huang(Nx, Ny)
    N = 2 * Nx * Ny - Nx - Ny
    coupling_strength = [2.0 * (rand() + 1e-6) for i in 1:N]
    return coupling_strength
end;

function make_coupling_strength_huang__ood(Nx, Ny)
    N = 2 * Nx * Ny - Nx - Ny
    coupling_strength = [-2.0 * (rand() + 1e-6) for i in 1:N]
    return coupling_strength
end;

function make_coupling_matrix_tfim(N)
    # uniformly sample the coupling strength from [0, 2]
    coupling_strength = [2.0 * (rand() + 1e-6) for i in 1:N-1]
    return coupling_strength
end;


function make_coupling_matrix_tfim_ood(N)
    # uniformly sample the coupling strength from [0, 2]
    coupling_strength = [4.0 * (rand() + 1e-6)+2.0 for i in 1:N-1]
    return coupling_strength
end;

# Adatped from Tang et al. ICLR 2024
function make_coupling_matrix_tang(N, a, J)
    # uniformly sample the coupling strength from [0, 2]
    coupling_matrix = zeros(N, N)
    for i in 1:N-1
        for j in i+1:N
            prob = rand()
            if j - i > 1
                if prob > 0.5
                    coupling_matrix[i, j] = coupling_matrix[j, i] = 0.0
                elseif coupling_matrix[i, j - 1] == 0.0
                    coupling_matrix[i, j] = coupling_matrix[j, i] = 0.0
                else
                    coupling_matrix[i, j] = coupling_matrix[j, i] = J / (abs(j-i) ^ a) / 3
                end
            else 
                coupling_matrix[i, j] = coupling_matrix[j, i] = J / (abs(j-i) ^ a) / 3
            end
        end
    end
    return coupling_matrix
end;




function base2num(base)
    # map "X" to 1, "Y" to 2, "Z" to 3
    base_num = []
    for i in 1:length(base)
        push!(base_num, base[i][1] - 'X' + 1)
    end
    return base_num
end;



function obs2samples(qubits_num, shots, obs)
    samples = []
    # map the measurement results to samples
    # "X" => 0, "X" => 1, "Y" => 0, "Y" => 1, "Z" => 0, "Z" => 1  ===> 0, 1, 2, 3, 4, 5
    for i = 1:shots
        single_sample = []
        for j = 1:qubits_num
            # map "X", "Y", "Z" to 0, 1, 2
            base = obs[i,:][j][1][1] - 'X'
            # observation results: 0 or 1
            obs_val = obs[i,:][j][2]
            push!(single_sample, base * 2 + obs_val)
        end;
        push!(samples, single_sample)
    end;
    return samples
end;

function cosine_similarity(u, v)
    return dot(u, v) / (norm(u) * norm(v))
end;