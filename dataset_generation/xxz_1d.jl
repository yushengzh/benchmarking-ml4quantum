using ITensors, ITensorMPS
using ArgParse
include("Hamiltonian.jl")
include("mps_utils.jl")
include("utils.jl")

##### Parameters #####
s = ArgParseSettings()

@add_arg_table! s begin
    "--samples_num", "-n"
    help = "number of samples"
    required = true
    default = 100
end

args = parse_args(s)


qubits_num = [6, 9, 13, 17, 20, 25, 31, 48, 51, 63, 72, 81, 100, 127] # Number of qubits
spin = "1/2"
nsweeps = 10
maxdim = [20, 60, 80, 100, 100]
cutoff = [1E-8]
noise = [1E-6]

samples_num = 1000

for qubits in qubits_num
    nsites_list = [qubits for i=1:samples_num]
    feature_list = []
    energy_list, entropy_list, corr_list = [], [], []
    dataset = []
    for i = 1:samples_num
        delta = 2 * rand(qubits - 1) .- 1;
        energy, psi = xxz_1d(qubits, spin, nsweeps, maxdim, cutoff, delta);
        corr = compute_correlation_pauliz(psi)
        entropy = [compute_renyi_entropy(psi, b) for b in 1:qubits]
        push!(feature_list, delta)
        push!(energy_list, energy)
        push!(entropy_list, entropy)   
        push!(corr_list, corr)
    end
    push!(dataset, (nsites_list, feature_list, energy_list, entropy_list, corr_list))
    println("Writing dataset (qubits:$qubits, samples:$samples_num) to csv...")
    write_dataset_to_csv_xxz(dataset, "xxz_1d/q($qubits)_n($samples_num)_X(delta)_y(energy_entropy_corr).csv")
end

