using ITensors, ITensorMPS
using DataFrames
using CSV
include("mps_utils.jl")
include("utils.jl")
include("Hamiltonian.jl")


##### Parameters #####
samples_num = 50
N = 9
spin = "1/2";
nsweeps = 5;
maxdim = [20, 60, 80, 100, 100];
cutoff = [1E-8];
noise = [1E-6];
h1 = collect(range(0.025, step=0.025, stop=1.6));
h2 = collect(range(-1.55, step=0.05, stop=1.6));

qubits_num = [2, 5, 8, 9, 10, 12, 15, 20, 25, 31, 48, 51, 63, 72, 81, 100, 127]

##### Generate dataset #####
attributes = ["N", "h1", "h2", "energy", "entropy", "corrzz"]

for qubits in qubits_num
    N_list = [qubits for i in 1:length(h1)*length(h2)]
    energy_list = []# zeros(length(h1)*length(h2), 1)  
    entropy_list = []
    corrzz_list = []
    h1_list, h2_list = [], []
    dataset = []
    for i=1:length(h1)
        for j=1:length(h2)
            energy, psi = cluster_ising_1d(qubits, spin, nsweeps, maxdim, cutoff, h1[i], h2[j]);
            push!(h1_list, h1[i])
            push!(h2_list, h2[j])
            push!(energy_list, energy)
            push!(corrzz_list, compute_correlation_pauliz(psi))
            push!(entropy_list, [compute_renyi_entropy(psi, b) for b in 1:qubits])
        end
    end
    push!(dataset, (N_list, h1_list, h2_list, energy_list, entropy_list, corrzz_list))
    println("#### Writing datasets of $qubits qubits in files... ####")
    write_dataset_to_csv_cluster_ising(dataset, "ising_1d/Ci_n($qubits)_X(h1, h2)_y(energy,entropy,corrzz).csv")
end
