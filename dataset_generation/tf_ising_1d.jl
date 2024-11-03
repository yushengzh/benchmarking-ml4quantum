using ITensors, ITensorMPS
using CUDA:cu
using ArgParse
using Distributions
include("Hamiltonian.jl")
include("mps_utils.jl")
include("utils.jl")

##### Parameters #####

s = ArgParseSettings()

@add_arg_table! s begin
    "--coupling_strenth_origin", "-j"
    help = "coupling strenth generatation item"
    required = true
    default = "J0"
end
args = parse_args(s)

N = 5 # Number of sites
qubits_num = [2, 5, 9, 13, 17, 20, 25, 31, 48, 51, 63, 72, 81, 100, 127] # Number of qubits
spin = "1/2"
nsweeps = 10
maxdim = [20, 60, 80, 100, 100]
cutoff = [1E-8]
noise = [1E-6]

J = J0 = zeros(15); # J ∈ {0.0, 0.0, ..., 0.0}
J1 = collect(range(0.1, step=0.1, stop=1.5));  # J ∈ {0.1, 0.2, ..., 1.5}
J2 = collect(range(-1.5, step=0.1, stop=-0.1)); # J ∈ {−1.5, −1.4, . . . , −0.1}

sigma = 0.1;
type = "J0"
if args["coupling_strenth_origin"] == "J0"
    J = J0
    type = "J0"
elseif args["coupling_strenth_origin"] == "J1"
    J = J1
    type = "J1"
elseif args["coupling_strenth_origin"] == "J2"
    J = J2
    type = "J2"
end


##### Data Generation #####

for qubits in qubits_num
    J_list_samples = []
    for J_prime in J
        dist = Normal(J_prime, sigma)
        J_val = []
        for i = 1: (qubits-1)
            push!(J_val, rand(dist))
        end
        push!(J_list_samples, J_val)
    end

    qubits_list = [qubits for i=1:length(J_list_samples)]
    features_list = []
    energy_list, entropy_list, corr_list = [], [], []
    dataset = []
    for J_list in J_list_samples
        ground_state_energy, psi1, first_excited_energy, psi2 = transverse_field_ising_1d(qubits, spin, nsweeps, maxdim, cutoff, Float32.(J_list));
        energy = [ground_state_energy, first_excited_energy]
        entropy = [compute_renyi_entropy(psi1, b) for b in 1:qubits]
        corr_matrix = compute_correlation_pauliz(psi1)

        push!(features_list, J_list)
        push!(energy_list, energy)
        push!(entropy_list, entropy)
        push!(corr_list, corr_matrix)
    end
    push!(dataset, (qubits_list, features_list, energy_list, entropy_list, corr_list))
    println("#### Writing in datasets of $qubits qubits ####")
    write_dataset_to_csv_tf_ising(dataset, "ising_1d/TF_n($qubits)_X(J)_y(energy,entropy,corrzz)_$type.csv")
end