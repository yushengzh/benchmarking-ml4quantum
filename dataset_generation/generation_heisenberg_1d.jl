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

# We follow settings from ICLR'24 (Tang et al.) for the Heisenberg 1D model
a_array = 1 .+ rand(samples_num);
J = [369.0 for i=1:samples_num];
coupling_matrix_list, meas_sample_list = [], []
approx_correlation_matrix_list, exact_correlation_list = [],[]
approx_entropy_list, exact_entropy_list = [], []
ground_state_energy_list = []
for i in 1:samples_num
    coupling_matrix = make_coupling_matrix_tang(qubits, a_array[i], J[i])
    push!(coupling_matrix_list, vec((real(coupling_matrix))))
    energy, psi, H = heisenberg_1d_tang(qubits, spin, nsweeps, maxdim, cutoff, coupling_matrix);
    push!(ground_state_energy_list, Float32(energy));
    obs = getsamples(psi, randombases(qubits, shots, local_basis=["X", "Y", "Z"]));
    
    exact_correlation = (compute_correlation_pauliz_norm(psi) .+ compute_correlation_paulix_norm(psi) .+ compute_correlation_pauliy_norm(psi)) ./ 3.0;
    meas_samples, approx_correlation = cal_shadow_correlation(obs);
    
    exact_entropy = [exact_renyi_entropy_size_two(psi, a, a+1) for a in 1:qubits-1];
    approx_entropy = [cal_shadow_renyi_entropy(obs, [a, a+1]) for a in 1:qubits-1];
    push!(meas_sample_list, meas_samples)
    push!(approx_correlation_matrix_list, approx_correlation)
    push!(approx_entropy_list, approx_entropy)
    push!(exact_correlation_list, exact_correlation)
    push!(exact_entropy_list, exact_entropy)

end;
df = DataFrame(coupling_matrix=coupling_matrix_list,
            measurement_samples=meas_sample_list,
            ground_state_energy=ground_state_energy_list, 
            approx_correlation=approx_correlation_matrix_list,
            approx_entropy=approx_entropy_list,
            exact_correlation=exact_correlation_list,
            exact_entropy=exact_entropy_list
            );
println("Writing dataset to CSV file... qubits: $qubits, samples: $samples_num, shots: $shots. ...Success!")
CSV.write("dataset_generation/heisenberg_1d/n$samples_num|X(coupling, meas$shots)_y(energy,entropy,corrs)_q$qubits.csv", df);
