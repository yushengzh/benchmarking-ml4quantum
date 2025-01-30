using ITensors, ITensorMPS

# Tang et al. 2024 ICLR
function heisenberg_1d_tang(N, spin, nsweeps, maxdim, cutoff, coupling_strength)
    J = coupling_strength
    sites = siteinds("S=$spin", N)
    os = OpSum()
    for i=1:N-1
        for j=i+1:N
            os += J[i, j], "Sx", i, "Sx", j
            os += J[i, j], "Sy", i, "Sy", j
            os += J[i, j], "Sz", i, "Sz", j
        end
    end
    H = MPO(os, sites)
    
    psi0 = MPS(sites, n -> isodd(n) ? "Up" : "Dn")
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, outputlevel=0)
    return energy, psi, H
end;

# Huang et al. 2022 Science
function heisenberg_2d_huang(Nx, Ny, spin, nsweeps, coupling_strength, maxdim, cutoff)
    N = Nx * Ny
    J = coupling_strength
    sites = siteinds("S=$spin", N)
    lattice = square_lattice(Nx, Ny; yperiodic=false)
    i = 1
    os = OpSum()
    for b in lattice
        os += J[i] / 2, "S+", b.s1, "S-", b.s2
        os += J[i] / 2, "S-", b.s1, "S+", b.s2
        os += J[i], "Sz", b.s1, "Sz", b.s2
        i += 1
    end
    H = MPO(os, sites)
    # H = map(os -> device(MPO(Float32, os, sites)), os)
    state = [isodd(n) ? "Up" : "Dn" for n=1:N]
    # psi0 = device(complex.(MPS(sites, state)))
    psi0 = MPS(sites, state)
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, outputlevel=0)
    
    return energy, psi, H
end;



# Zhu et al. 2022
function xxz_1d(N, spin, nsweeps, maxdim, cutoff, delta)
    sites = siteinds("S=$spin", N)
    os = OpSum()
    for i=1:N-1
        os -= delta[i] / 2, "Sx", i, "Sx", i+1
        os -= delta[i] / 2, "Sy", i, "Sy", i+1
        os -= 1, "Sz", i, "Sz", i+1
    end;
    H = MPO(os, sites)
    psi0 = MPS(sites, N -> isodd(N) ? "Up" : "Dn")
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff,  outputlevel=0);
    return energy, psi
end;


# Wu et al. 2023
function bond_alter_xxz_1d(N, spin, nsweeps, maxdim, cutoff, coupling_strength, coupling_strength2, delta)
    sites = siteinds("S=$spin", N)
    os = OpSum()
    J = coupling_strength
    J_prime = coupling_strength2
    
    for i=1:N/2
        os += J / 2, "S+", 2*i-1, "S-", 2*i
        os += J / 2, "S-", 2*i-1, "S+", 2*i
        os += J*delta, "Sz", 2*i-1, "Sz", 2*i
    end;
    
    for i=1:(N/2-1)
        os += J_prime / 2, "S+", 2*i, "S-", 2*i+1
        os += J_prime / 2, "S-", 2*i, "S+", 2*i+1
        os += J_prime*delta, "Sz", 2*i, "Sz", 2*i+1
    end;
    H = MPO(os, sites)
    psi0 = MPS(sites, N -> isodd(N) ? "Up" : "Dn")
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
    return energy, psi
end;


# Wu et al. 2023
function cluster_ising_1d(N, spin, nsweeps, maxdim, cutoff, h1, h2)
    sites = siteinds("S=$spin", N)
    os = OpSum() 
    for i=1:(N-2)
        os -= 1, "Sz", i, "Sx", i+1, "Sz", i+2 
    end
    for i=1:N
        os -= h1, "Sx", i
    end
    for i=1:N-1
        os -= h2, "Sx", i, "Sx", i+1
    end
    H = MPO(os, sites);
    psi0 = MPS(sites, N -> isodd(N) ? "Up" : "Dn");
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, outputlevel=0);
    return energy, psi 
end;

# Zhu et al. 2022
# ferromagnetic_ising_1d
function transverse_field_ising_1d(N, spin, nsweeps, maxdim, cutoff, coupling_strength, eigsolve_krylovdim)
    sites = siteinds("S=$spin", N)
    os = OpSum()
    J = coupling_strength
    h = 4.0 # theortical bound = 2*abs(h-1)
    weight = 200 * h
    noise = [1E-6]
    for i=1:N-1
        os -= J[i], "Sz", i, "Sz", i+1
    end;
    for i=1:N
        os -= 1, "Sx", i
    end;
    H = MPO(os, sites)
    psi_init = MPS(sites, N -> isodd(N) ? "Up" : "Dn")
    ground_state_energy, psi1 = dmrg(H, psi_init; nsweeps, maxdim, cutoff, eigsolve_krylovdim, outputlevel=0);
    psi_init2 = MPS(sites, N -> isodd(N) ? "Up" : "Dn")
    first_excited_energy, psi2 = dmrg(H, [psi1], psi_init2; nsweeps, maxdim, cutoff, weight, eigsolve_krylovdim, outputlevel=0);
    return ground_state_energy, psi1, first_excited_energy, psi2
end;


# Wu et al. 2023
function perturbed_cluster_ising_1d(N, spin, nsweeps, maxdim, cutoff, h1, h2, h3)
    sites = siteinds("S=$spin", N)
    os = OpSum()
    for i=1:N-2
        os -= 1, "Sz", i, "Sx", i+1, "Sz", i+2 
    end;
    for i=1:N
        os -= h1, "Sx", i
    end;
    for i=1:N-1
        os -= h2, "Sx", i, "Sx", i+1
    end;
    for i=1:N-2
        os += h3, "Sz", i, "Sz", i+2
    end;
    H = MPO(os, sites)
    psi0 = MPS(sites, N -> isodd(N) ? "Up" : "Dn")
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
    return energy, psi
end

