using ITensors, ITensorMPS
using DataFrames
using CSV
using ArgParse
using PastaQ
include("mps_utils.jl")
include("utils.jl")
include("Hamiltonian.jl")
include("shadow.jl")

spin = "1/2"
nsweeps = 5
maxdim = [20, 60, 80, 100, 100]
cutoff = 1E-5

s = ArgParseSettings()
@add_arg_table! s begin
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
    required = true
    default = 8
end
args = parse_args(s)

# qubits_list = [8, 10, 12, 16, 25, 31, 48, 63, 100, 127]
samples_num = parse(Int, args["samples_num"])
shots = parse(Int, args["shots"])
qubits = parse(Int, args["qubits"])

println("Generating dataset for Heisenberg 1D model with $samples_num samples with $qubits qubits and $shots shots.")

a_array = 1 .+ rand(samples_num);
J = [369.0 for i=1:samples_num];
coupling_matrix_list, meas_sample_list = [], []
approx_correlation_matrix_list = []
ground_state_energy_list, entropy_list, exact_correlation_zz_list = [], [], []
exact_correlation_xx_list, exact_correlation_yy_list = [], []
for i in 1:samples_num
    coupling_matrix = make_coupling_matrix_tang(qubits, a_array[i], J[i])
    push!(coupling_matrix_list, vec((real(coupling_matrix))))
    energy, psi = heisenberg_1d_tang(qubits, spin, nsweeps, maxdim, cutoff, coupling_matrix)
    push!(ground_state_energy_list, Float32(energy))
    obs = getsamples(psi, randombases(qubits, shots, local_basis=["X", "Y", "Z"]));
    meas_samples, approx_correlation = cal_shadow_correlation(obs)
    # shadow_state = reconstruct_state_from_shadow(obs)
    push!(meas_sample_list, meas_samples)
    push!(approx_correlation_matrix_list, approx_correlation)
    push!(entropy_list, [compute_renyi_entropy(psi, b) for b in 1:qubits])

    push!(exact_correlation_xx_list, compute_correlation_paulix_norm(psi))
    push!(exact_correlation_yy_list, compute_correlation_pauliy_norm(psi))
    push!(exact_correlation_zz_list, compute_correlation_pauliz_norm(psi))

end;
df = DataFrame(coupling_matrix=coupling_matrix_list,
            measurement_samples=meas_sample_list,
            ground_state_energy=ground_state_energy_list, 
            entropy=entropy_list, 
            exact_correlation_matrix_xx=exact_correlation_xx_list,
            exact_correlation_matrix_yy=exact_correlation_yy_list,
            exact_correlation_matrix_zz=exact_correlation_zz_list,
            approx_correlation_matrix=approx_correlation_matrix_list);
println("Writing dataset to CSV file... qubits: $qubits, samples: $samples_num, shots: $shots. ...Success!")
CSV.write("dataset_generation/heisenberg_1d/n$samples_num|X(coupling, meas$shots)_y(energy,entropy,corrs)_q$qubits.csv", df);
