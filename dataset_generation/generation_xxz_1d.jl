using ITensors, ITensorMPS
using DataFrames
using CSV
using ArgParse
using PastaQ
using CUDA
using LinearAlgebra
include("mps_utils.jl")
include("utils.jl")
include("Hamiltonian.jl")
include("shadow.jl")

spin = "1/2"
nsweeps = 10
maxdim = [20, 60, 80, 100, 100]
cutoff = 1E-5

##### Parameters #####
s = ArgParseSettings()
@add_arg_table! s begin
    "--samples_num", "-n"
    help = "number of samples"
    required = true
    default = 100

    "--shots", "-s"
    help = "the number of shots"
    required = true
    default = 1000

    "--qubits", "-q"
    help = "the number of qubits"
    required = true
    default = 8
end

args = parse_args(s)

samples_num = parse(Int, args["samples_num"])
shots = parse(Int, args["shots"])
qubits = parse(Int, args["qubits"])

# qubits_num = [6, 9, 13, 17, 20, 25, 31, 48, 51, 63, 72, 81, 100, 127] # Number of qubits

noise = [1E-6]
println("XXZ 1d model: generating dataset... qubits:$qubits, samples $samples_num, shots: $shots. ...")

nsites_list = [qubits for i=1:samples_num]
delta_list, meas_sample_list = [], []
ground_state_energy_list = []
approx_correlation_matrix_list = []
exact_correlation_matrix_list = []
approx_energy_list = []
exact_entropy_list, approx_entropy_list = [], []
for i = 1:samples_num
    delta = 2 * rand(qubits - 1) .- 1;
    push!(delta_list, vec(delta))

    energy, psi = xxz_1d(qubits, spin, nsweeps, maxdim, cutoff, delta);
    push!(ground_state_energy_list, Float32(energy))
    exact_z = compute_correlation_pauliz_norm(psi)
    exact_x = compute_correlation_paulix_norm(psi)
    exact_y = compute_correlation_pauliy_norm(psi)
    push!(exact_correlation_matrix_list, (exact_z .+ exact_x .+ exact_y) ./ 3.0)
    obs = getsamples(psi, randombases(qubits, shots, local_basis=["X", "Y", "Z"]));
    meas_samples, approx_correlation = cal_shadow_correlation(obs)
    exact_entropy = [exact_renyi_entropy_size_two(psi, a, a+1) for a in 1:qubits-1];
    approx_entropy = [cal_shadow_renyi_entropy(obs, [a, a+1]) for a in 1:qubits-1];
    push!(meas_sample_list, meas_samples) 
    push!(approx_correlation_matrix_list, approx_correlation)
    push!(exact_entropy_list, exact_entropy)
    push!(approx_entropy_list, approx_entropy)
end
df = DataFrame(coupling_matrix=delta_list,
                measurement_samples=meas_sample_list,
                ground_state_energy=ground_state_energy_list, 
                approx_correlation=approx_correlation_matrix_list,
                approx_entropy=approx_entropy_list,
                exact_correlation=exact_correlation_matrix_list,
                exact_entropy=exact_entropy_list);
println("XXZ 1d model: writing dataset to CSV file... qubits:$qubits, samples $samples_num, shots: $shots. ...Success!")
CSV.write("dataset_generation/xxz_1d/n$samples_num|X(delta, meas$shots)_y(energy,entropy,corrs)_q$qubits.csv", df);



