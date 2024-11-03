using ITensors, ITensorMPS
using DataFrames
using CSV
using ArgParse
include("mps_utils.jl")
include("utils.jl")
include("Hamiltonian.jl")


spin = "1/2"
nsweeps = 5
maxerr = 1E-4
maxdim = [20, 60, 80, 100, 100]
cuttoff = 1E-5

s = ArgParseSettings()
@add_arg_table! s begin
    "--samples_num", "-s"
    help = "the number of samples"
    required = true
    default = 100
end
args = parse_args(s)

samples_num = parse(Int, args["samples_num"])
println("Generating dataset for Heisenberg 1D model with $samples_num samples")

nxy_list = [[4, 4], [4, 5], [5, 5], [6, 5], [6, 6], [7, 5], [8, 5], [8, 8], [9, 5], [10, 10], [15, 15], [31, 31]];

for nxy in nxy_list
    Nx, Ny = nxy[1], nxy[2]
    coupling_matrix_list = []
    energy_list, entropy_list, corrzz_list = [], [], []

    for i=1:samples_num
        coupling_strength = make_coupling_strength(Nx, Ny)
        push!(coupling_matrix_list, coupling_strength)

        energy, psi, H = heisenberg_2d_huang(
            Nx, Ny, spin, nsweeps, coupling_strength, maxdim, cuttoff
        );
        push!(energy_list, Float32(energy))
        push!(entropy_list, [compute_renyi_entropy(psi, b) for b in 1:Nx*Ny])
        push!(corrzz_list, compute_correlation_pauliz(psi))
    end
    df = DataFrame(coupling_matrix=coupling_matrix_list, 
                ground_state_energy=energy_list, 
                entropy=entropy_list, 
                correlation_matrix=corrzz_list);
    println("Writing dataset to CSV file... Info: ($Nx, $Ny), samples $samples_num. ")
    CSV.write("dataset_generation/heisenberg_2d/n[$samples_num]_X(coupling)_y(energy,entropy,corr)_q($Nx, $Ny).csv", df);
    
end