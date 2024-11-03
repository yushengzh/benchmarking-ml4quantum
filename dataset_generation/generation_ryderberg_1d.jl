using Bloqade
using CUDA
using DataFrames
using KrylovKit
using CSV

include("mps_utils.jl")
include("utils.jl")

samples_num = 50

###### Parameters ######
nsites_list = [2, 5, 8, 12, 19, 25, 31, 51]
C6s = collect(2*pi*range(0.5*10.0^6, step=0.1*10.0^6, stop=10.0^6)) # default: 862690 \times 2\pi
subspace_radius = unit_disk_radius = 6.9
C6 = 862690 * 2π
time_start, time_stop = 0.0, 10.0
omegas = piecewise_linear(clocks=collect(range(0, step=0.1, stop=10)), values=2π*collect(range(0, step=0.1, stop=10)))
deltas = piecewise_linear(clocks=collect(range(0, step=0.1, stop=10)), values=2π*collect(range(-10, step=0.25, stop=15)))

# R_0 = (C6 / omega)^(1/6)
global R_0, omega, delta


###### Generate dataset ######
# features: 
# - nsites
# - omega (time-dependent)
# - delta (time-dependent)
# - C6 (time-independent)
# - unit_disk_radius (time-independent)
attributes = ["nsites", "unit_disk_radius", "C6",
                 "t", "omega", "delta", "R_0", "energy", "entropy", "corr_matrix"]
            
for nsites in nsites_list
    atoms = generate_sites(ChainLattice(), nsites, scale=unit_disk_radius)
    h = rydberg_h(atoms; Δ=deltas, Ω=omegas);
    dataset = []

    nsites_list = [nsites for i = 1:samples_num]
    unit_disk_radius_list = [unit_disk_radius for i = 1:samples_num]
    C6_list = [C6 for i = 1:samples_num]
    t_list, omega_list, delta_list, R_0_list = [], [], [], []
    energy_list, entropy_list, corr_matrix_list = [], [], []
    
    for i = 1:samples_num
        t = round(rand()*time_stop, digits=1)
        ht= h |> attime(t)
        omega = omegas(t)
        delta = deltas(t)
        R_0 = (C6 / omega)^(1/6)
        push!(t_list, t)
        push!(omega_list, omega)
        push!(delta_list, delta)
        push!(R_0_list, R_0)
        mat_h = mat(ht)
        vals, vecs, info = KrylovKit.eigsolve(mat_h, 1, :SR)
        energy = real(vals[1])
        g_state = ArrayReg(vecs[1])
        corr_matrix = compute_correlation_norm(rydberg_corr(g_state))
        entropy = von_neumann_entropy([rydberg_density(g_state, j) for j in 1: nsites])
        push!(energy_list, energy)
        push!(entropy_list, entropy)
        push!(corr_matrix_list, corr_matrix)
        
        push!(dataset, (nsites_list, unit_disk_radius_list, C6_list, t_list, omega_list, delta_list, R_0_list, energy_list, entropy_list, corr_matrix_list))
        feature = [nsites, unit_disk_radius, C6, t, omega, delta, R_0]
        println("feature: ", feature)
    end;
    println("##### (nsites=$nsites)writing in file... #####")
    write_dataset_to_csv_rydberg(dataset, "rydberg_1d/n$(nsites)_r$(unit_disk_radius).csv", attributes)
end;
