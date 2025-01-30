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
maxerr = 1E-4
maxdim = [20, 60, 80, 100, 200]
cuttoff = 1E-5

s = ArgParseSettings()
@add_arg_table! s begin
    "--samples_num", "-n"
    help = "the number of samples"
    required = true
    default = 100

    "--shots", "-s"
    help = "the number of shots"
    required = true
    default = 10
end
args = parse_args(s)

samples_num = parse(Int, args["samples_num"])
shots = parse(Int, args["shots"])
println("Generating dataset for Heisenberg 2D model with $samples_num samples and $shots shots.")

nxy_list = [[7, 5], [8, 5], [9, 5], [8, 9],[10, 10]]; # [5, 5], [6, 5], 

for nxy in nxy_list
    Nx, Ny = nxy[1], nxy[2]
    coupling_strength_list, meas_sample_list = [], []
    ground_state_energy_list, entropy_list = [], []
    exact_correlation_list, exact_entropy_list = [], []
    approx_correlation_matrix_list, approx_entropy_list = [], []
    for i=1:samples_num
        coupling_strength = make_coupling_strength_huang(Nx, Ny)
        push!(coupling_strength_list, vec((real(coupling_strength))))

        energy, psi, H = heisenberg_2d_huang(
            Nx, Ny, spin, nsweeps, coupling_strength, maxdim, cuttoff
        );
        push!(ground_state_energy_list, Float32(energy))
        

        obs = getsamples(psi, randombases(Nx*Ny, shots, local_basis=["X", "Y", "Z"]));
        meas_samples, approx_correlation = cal_shadow_correlation(obs);
        exact_correlation = (compute_correlation_pauliz_norm(psi) .+ compute_correlation_paulix_norm(psi) .+ compute_correlation_pauliy_norm(psi)) ./ 3.0;
        approx_entropy = [cal_shadow_renyi_entropy(obs, [a, a+1]) for a in 1:Nx*Ny-1];
        exact_entropy = [exact_renyi_entropy_size_two(psi, a, a+1) for a in 1:Nx*Ny-1];
        push!(meas_sample_list, meas_samples)
        push!(approx_correlation_matrix_list, approx_correlation)
        push!(approx_entropy_list, approx_entropy)
        push!(exact_correlation_list, exact_correlation)
        push!(exact_entropy_list, exact_entropy)
    end
    df = DataFrame(coupling_matrix=coupling_strength_list,
                    measurement_samples=meas_sample_list,
                    ground_state_energy=ground_state_energy_list, 
                    approx_correlation=approx_correlation_matrix_list,
                    approx_entropy=approx_entropy_list,
                    exact_correlation=exact_correlation_list,
                    exact_entropy=exact_entropy_list
                    );
    println("Writing dataset to CSV file... Info: ($Nx, $Ny), samples $samples_num, shots: $shots. ...Success!")
    CSV.write("dataset_generation/heisenberg_2d/n$samples_num|X(coupling, meas$shots)_y(energy,entropy,corrs)_q($Nx, $Ny).csv", df);
    
end