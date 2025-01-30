using ITensors, ITensorMPS
using DataFrames
using CSV
using ArgParse
using PastaQ
include("mps_utils.jl")
include("utils.jl")
include("Hamiltonian.jl")
include("shadow.jl")


spin = "1/2";
nsweeps = 5;
maxdim = [20, 60, 80, 100, 100];
cutoff = 1E-5;
noise = [1E-6];

s = ArgParseSettings()
@add_arg_table! s begin

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

shots = parse(Int, args["shots"])
qubits = parse(Int, args["qubits"])

# We follow settings from NatComm'24 (Wu et al.)
h1 = collect(range(0.025, step=0.025, stop=1.6));
h2 = collect(range(-1.55, step=0.05, stop=1.6));


##### Generate dataset #####
# attributes = ["h1", "h2", "energy", "entropy", "corrzz"]


N_list = [qubits for i in 1:length(h1)*length(h2)]
energy_list = zeros(length(h1), length(h2))
entropy_list = []
exact_correlation_matrix_list = []
exact_correlation_zz_list, exact_correlation_xx_list = [], []
h1_list, h2_list, meas_sample_list = [], [], []
approx_correlation_matrix_list = []
approx_energy_list = []
for i=1:length(h1)
    for j=1:length(h2)
        energy, psi = cluster_ising_1d(qubits, spin, nsweeps, maxdim, cutoff, h1[i], h2[j]);
        energy_list[i, j] = energy
        push!(h1_list, h1[i])
        push!(h2_list, h2[j])
        push!(entropy_list, [compute_renyi_entropy(psi, b) for b in 1:qubits])
        exact_z = compute_correlation_pauliz_norm(psi)
        exact_x = compute_correlation_paulix_norm(psi)
        exact_y = compute_correlation_pauliy_norm(psi)
        push!(exact_correlation_matrix_list, (exact_z .+ exact_x .+ exact_y) ./ 3.0)
        obs = getsamples(psi, randombases(qubits, shots, local_basis=["X", "Y", "Z"]));
        meas_samples, approx_correlation = cal_shadow_correlation(obs)
        #res_mat, res_vec = reconstruct_state_from_shadow(obs)
        #elgvs = eigvals(real(res_mat))
        push!(meas_sample_list, meas_samples)
        push!(approx_correlation_matrix_list, approx_correlation)
        #push!(approx_energy_list, elgvs[1])
    end
end
df = DataFrame(h1 = h1_list, 
                h2 = h2_list, 
                measurement_samples = meas_sample_list,
                ground_state_energy = vec(energy_list), 
                entropy = entropy_list, 
                exact_correlation_matrix=exact_correlation_matrix_list,
                approx_correlation_matrix=approx_correlation_matrix_list);
println("#### cluster_ising_1d: writing datasets of $qubits qubits and $shots shots in file... ####")
CSV.write("dataset_generation/cluster_ising_1d/n4096|X(h1, h2, meas$shots)_y(energy,entropy,corrs)_q$qubits.csv", df);