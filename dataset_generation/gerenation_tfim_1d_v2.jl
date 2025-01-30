using ITensors, ITensorMPS
using CUDA:cu
using ArgParse
using Distributions
using DataFrames
using CSV
using PastaQ
using LinearAlgebra

include("mps_utils.jl")
include("utils.jl")
include("Hamiltonian.jl")
include("shadow.jl")

spin = "1/2"
nsweeps = 10
maxdim = [20, 60, 80, 100, 100]
cutoff = [1E-8]
noise = [1E-6]

s = ArgParseSettings()
@add_arg_table! s begin
    "--coupling_strenth_origin", "-j"
    help = "coupling strenth generatation item"
    required = false
    default = "J0"

    "--samples_num", "-n"
    help = "the number of samples"
    required = true
    default = 100

    "--shots", "-s"
    help = "the number of shots"
    required = true
    default = 1000

    "--qubits", "-q"
    help = "the number of qubits"
    required = false
    default = 8
end
args = parse_args(s)

N = 5 # Number of sites
qubits_list = [8, 10, 12, 16, 25, 31, 48, 63, 100, 127]
samples_num = parse(Int, args["samples_num"])
shots = parse(Int, args["shots"])
# qubit = parse(Int, args["qubits"])

eigsolve_krylovdim = 5

##### Data Generation #####
for qubits in qubits_list
    println("Generating dataset for TFIM with $samples_num samples with $qubits qubits and $shots shots.")
    coupling_matrix_list, meas_sample_list = [], []
    approx_correlation_matrix_list, exact_correlation_list = [],[]
    approx_entropy_list, exact_entropy_list = [], []
    ground_state_energy_list, first_excited_energy_list, entropy_list, exact_correlation_matrix_list = [], [], [], []
    approx_energy_list = []
    for i in 1:samples_num
        coupling_matrix = make_coupling_matrix_tfim(qubits)
        push!(coupling_matrix_list, vec((real(coupling_matrix))))
        ground_state_energy, psi, first_excited_energy, psi2 = transverse_field_ising_1d(qubits, spin, nsweeps, maxdim, cutoff, Float32.(coupling_matrix),eigsolve_krylovdim);
        push!(ground_state_energy_list, ground_state_energy)
        push!(first_excited_energy_list, first_excited_energy)
        exact_z = compute_correlation_pauliz_norm(psi)
        exact_x = compute_correlation_paulix_norm(psi)
        exact_y = compute_correlation_pauliy_norm(psi)
        push!(exact_correlation_list, (exact_z .+ exact_x .+ exact_y) ./ 3.0)

        obs = getsamples(psi, randombases(qubits, shots, local_basis=["X", "Y", "Z"]));
        meas_samples, approx_correlation = cal_shadow_correlation(obs)

        exact_entropy = [exact_renyi_entropy_size_two(psi, a, a+1) for a in 1:qubits-1];
        approx_entropy = [cal_shadow_renyi_entropy(obs, [a, a+1]) for a in 1:qubits-1];

        push!(meas_sample_list, meas_samples)
        push!(approx_correlation_matrix_list, approx_correlation)
        push!(exact_entropy_list, exact_entropy)
        push!(approx_entropy_list, approx_entropy)
       
    end;

    df = DataFrame(coupling_matrix=coupling_matrix_list,
                    measurement_samples=meas_sample_list,
                    ground_state_energy=ground_state_energy_list, 
                    first_excited_energy=first_excited_energy_list,
                    approx_correlation=approx_correlation_matrix_list,
                    approx_entropy=approx_entropy_list,
                    exact_correlation=exact_correlation_list,
                    exact_entropy=exact_entropy_list         
            );
    println("Writing dataset to CSV file... qubits: $qubits, samples: $samples_num, shots: $shots. ...Success!")
    CSV.write("dataset_generation/tf_ising_1d/n$samples_num|X(coupling, meas$shots)_y(energy,entropy,corrs)_q$qubits.csv", df);
end;