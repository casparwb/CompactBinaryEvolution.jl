using LinearAlgebra: dot, ⋅, cross, ×, norm
using StaticArrays
import RecursiveArrayTools

function strip_units(quantity)
    ustrip(upreferred(unit(quantity)), quantity)
end


function get_system_constants(binary::CompactBinary; verbose=false)
    m1, m2 = binary.m1, binary.m2
    M = m1 + m2

    G³m₁m₂Mc⁻⁵ = (G^3*m1*m2*M)/c^5
    β = 64/5*G³m₁m₂Mc⁻⁵
    Γ = 32/5*G^(7/2)/c^5*m1^2*m2^2*sqrt(m1 + m2)

    GM = G*M
    GMc⁻² = G/c^2*M
    sqrt_GM = sqrt(GM)

    if verbose
        @info " "  G³m₁m₂Mc⁻⁵ β Γ GMc⁻²

        @info " "  upreferred(G³m₁m₂Mc⁻⁵) upreferred(β) upreferred(Γ) upreferred(GMc⁻²)
    end

    return strip_units.((m1, m2, M, G³m₁m₂Mc⁻⁵, β, Γ, GM, GMc⁻², sqrt_GM))
end


function get_orbital_evolution_model(binary::CompactBinary, ::PetersModel)
    m1, m2, M, G³m₁m₂Mc⁻⁵, β, Γ, GM, GMc⁻², sqrt_GM = get_system_constants(binary)

    function f!(du, u, p, t)
        a = u[1,1]
        e = u[1,2]
        L̄  = SA[u[1,end], u[2,end], u[3,end]]

        (e < zero(e) || e >= one(e) || a < zero(a)) && return nothing
        e² = e^2
        one_min_e² = 1 - e²

        L = norm(L̄)
        L̂ = L̄/L

        da_dt_GW = -β/(a^3*one_min_e²^3.5)*(1 + div_73_24*e² + div_37_96*e²^2)

        de_dt_GW = -div_304_15*G³m₁m₂Mc⁻⁵/(a^4*one_min_e²^2.5)*(1 + div_121_304*e²)*e
        
        dL_dt_GW = -Γ/a^3.5*(1 + 0.875*e²)/one_min_e²^2 * L̂
        println(de_dt_GW)

        du[1,1] = da_dt_GW
        du[1,2] = de_dt_GW
        
        du[1,3] = dL_dt_GW[1]
        du[2,3] = dL_dt_GW[2]
        du[3,3] = dL_dt_GW[3]
        nothing
    end

    return f!
end

function get_orbital_evolution_model(binary::CompactBinary, ::PetersWithPrecession)
    m1, m2, M, G³m₁m₂Mc⁻⁵, β, Γ, GM, GMc⁻², sqrt_GM = get_system_constants(binary)


    function f!(du, u, p, t)
        a = u[1,1]
        ē = SA[u[1,2], u[2,2], u[3,2]]
        L̄  = SA[u[1,end], u[2,end], u[3,end]]

        e = norm(ē)
        (e < zero(e) || e >= one(e) || a < zero(a)) && return nothing
        e² = e^2
        one_min_e² = 1 - e²

        L = norm(L̄)
        L̂ = L̄/L

        da_dt_GW = -β/(a^3*one_min_e²^3.5)*(1 + div_73_24*e² + div_37_96*e²^2)

        de_dt_GW = -div_304_15*G³m₁m₂Mc⁻⁵/(a^4*one_min_e²^2.5)*(1 + div_121_304*e²)*ē
        de_dt_GR = (3GMc⁻²/(a*one_min_e²)*sqrt_GM/a^1.5*L̂) × ē

        dL_dt_GW = -Γ/a^3.5*(1 + 0.875*e²)/one_min_e²^2 * L̂

        de_dt = de_dt_GW + de_dt_GR
        du[1,1] = da_dt_GW
        du[1,2] = de_dt[1]
        du[2,2] = de_dt[2]
        du[3,2] = de_dt[3]
        
        du[1,3] = dL_dt_GW[1]
        du[2,3] = dL_dt_GW[2]
        du[3,3] = dL_dt_GW[3]
    end

    return f!
end

function get_orbital_evolution_model(binary::CompactBinary, ::BarkerOConnell{BackReaction})
    
    m1, m2, M, G³m₁m₂Mc⁻⁵, β, Γ, GM, GMc⁻², sqrt_GM = get_system_constants(binary)
    q = m2/m1
    q⁻¹ = 1/q

    function f!(du, u, p, t)
        a  = u[1,1]
        ē  = SA[u[1,2],   u[2,2],   u[3,2]]
        L̄  = SA[u[1,end], u[2,end], u[3,end]]

        e = norm(ē)
        (e < zero(e) || e >= one(e) || a < zero(a)) && return nothing
        e² = e^2

        L = norm(L̄)
        L̂ = L̄/L

        a³ = a^3

        one_min_e² = 1 - e²

        de_dt_GW = -div_304_15*G³m₁m₂Mc⁻⁵/(a^4*one_min_e²^2.5)*(1 + div_121_304*e²)*ē
        da_dt_GW = -β/(a³*one_min_e²^3.5)*(1 + div_73_24*e² + div_37_96*e²^2)
        dL_dt_GW = -Γ/a^3.5*(1 + 0.875*e²)/one_min_e²^2 * L̂
        de_dt_GR = (3GMc⁻²/(a*one_min_e²)*sqrt_GM/a^1.5*L̂) × ē

        
        fac = 1/(2a³*one_min_e²^1.5)
        Gc⁻²_fac = Gc⁻²*fac
        S̄₁ = SA[u[1,3], u[2,3], u[3,3]]
        S̄₂ = SA[u[1,4], u[2,4], u[3,4]]

        S₁ = norm(S̄₁)
        S₂ = norm(S̄₂)

        Ŝ₁ = S̄₁/S₁
        Ŝ₂ = S̄₂/S₂

        LS₁ = L̂ ⋅ Ŝ₁
        LS₂ = L̂ ⋅ Ŝ₂

        L⁻¹ = 1/L

        Ω1_dS = Gc⁻²_fac*(4 + 3q)*L*L̂
        Ω1_LT = Gc⁻²_fac*S₂*(Ŝ₂ - 3LS₂*L̂)
        Ω1_QM = Gc⁻²_fac*S₁*q*(Ŝ₁ - 3LS₁*L̂)

        Ω2_dS = Gc⁻²_fac*(4 + 3q⁻¹)*L*L̂
        Ω2_LT = Gc⁻²_fac*S₁*(Ŝ₁ - 3LS₁*L̂)
        Ω2_QM = Gc⁻²_fac*S₂*q⁻¹*(Ŝ₂ - 3LS₂*L̂)

        Ω1_dS_br =  Gc⁻²_fac*S₁*(4 + 3q)*(Ŝ₁ - 3LS₁*L̂)
        Ω2_dS_br =  Gc⁻²_fac*S₂*(4 + 3q⁻¹)*(Ŝ₂ - 3LS₂*L̂)
        Ω_LT_br  = -3Gc⁻²_fac*S₁*S₂*L⁻¹*(LS₁*Ŝ₂ + LS₂*Ŝ₁ + (Ŝ₁ ⋅ Ŝ₂ - 5LS₁*LS₂)*L̂)
        Ω1_QM_br = -3Gc⁻²_fac*S₁^2*0.5*L⁻¹*q*(2LS₁*Ŝ₁ + (1 - 5LS₁^2)*L̂)
        Ω2_QM_br = -3Gc⁻²_fac*S₂^2*0.5*L⁻¹*3q⁻¹*(2LS₂*Ŝ₂ + (1 - 5*LS₂^2)*L̂)

        dS₁_dt = (Ω1_dS + Ω1_LT + Ω1_QM) × S̄₁
        dS₂_dt = (Ω2_dS + Ω2_LT + Ω2_QM) × S̄₂

        dL_dt_dS_LT_QM = (Ω1_dS_br + Ω2_dS_br + Ω_LT_br + Ω1_QM_br + Ω2_QM_br) × L̄
        de_dt_dS_LT_QM = (Ω1_dS_br + Ω2_dS_br + Ω_LT_br + Ω1_QM_br + Ω2_QM_br) × ē

        de_dt = de_dt_GW + de_dt_GR + de_dt_dS_LT_QM
        dL_dt = dL_dt_GW + dL_dt_dS_LT_QM

        du[1,3] = dS₁_dt[1]
        du[2,3] = dS₁_dt[2]
        du[3,3] = dS₁_dt[3]

        du[1,4] = dS₂_dt[1]
        du[2,4] = dS₂_dt[2]
        du[3,4] = dS₂_dt[3]

        du[1,5] = dL_dt[1]
        du[2,5] = dL_dt[2]
        du[3,5] = dL_dt[3]

        du[1,2] = de_dt[1]
        du[2,2] = de_dt[2]
        du[3,2] = de_dt[3]

        du[1,1] = da_dt_GW
    end

    return f!
end

function get_orbital_evolution_model(binary::CompactBinary, ::BarkerOConnell{NoBackReaction})
    m1, m2 = binary.m1, binary.m2
    q = m2/m1
    q⁻¹ = 1/q

    M, G³m₁m₂Mc⁻⁵, β, Γ, GM, GMc⁻², sqrt_GM = get_system_constants(binary)

    function f!(du, u, p, t)
        a  = u[1,1]
        ē  = SA[u[1,2],   u[2,2],   u[3,2]]
        L̄  = SA[u[1,end], u[2,end], u[3,end]]

        e = norm(ē)
        (e < zero(e) || e >= one(e) || a < zero(a)) && return nothing
        e² = e^2

        L = norm(L̄)
        L̂ = L̄/L

        a³ = a^3

        one_min_e² = 1 - e²

        de_dt_GW = -div_304_15*G³m₁m₂Mc⁻⁵/(a^4*one_min_e²^2.5)*(1 + div_121_304*e²)*ē
        da_dt_GW = -β/(a³*one_min_e²^3.5)*(1 + div_73_24*e² + div_37_96*e²^2)
        dL_dt_GW = -Γ/a^3.5*(1 + 0.875*e²)/one_min_e²^2 * L̂
        de_dt_GR = (3GMc⁻²/(a*one_min_e²)*sqrt_GM/a^1.5*L̂) × ē
        
        fac = 1/(2a³*one_min_e²^1.5)
        Gc⁻²_fac = Gc⁻²*fac
        S̄₁ = SA[u[1,3], u[2,3], u[3,3]]
        S̄₂ = SA[u[1,4], u[2,4], u[3,4]]

        S₁ = norm(S̄₁)
        S₂ = norm(S̄₂)

        Ŝ₁ = S̄₁/S₁
        Ŝ₂ = S̄₂/S₂

        LS₁ = L̂ ⋅ Ŝ₁
        LS₂ = L̂ ⋅ Ŝ₂

        Ω1_dS = Gc⁻²_fac*(4 + 3q)*L*L̂
        Ω1_LT = Gc⁻²_fac*S₂*(Ŝ₂ - 3LS₂*L̂)
        Ω1_QM = Gc⁻²_fac*S₁*q*(Ŝ₁ - 3LS₁*L̂)

        Ω2_dS = Gc⁻²_fac*(4 + 3q⁻¹)*L*L̂
        Ω2_LT = Gc⁻²_fac*S₁*(Ŝ₁ - 3LS₁*L̂)
        Ω2_QM = Gc⁻²_fac*S₂*q⁻¹*(Ŝ₂ - 3LS₂*L̂)

        dS₁_dt = (Ω1_dS + Ω1_LT + Ω1_QM) × S̄₁
        dS₂_dt = (Ω2_dS + Ω2_LT + Ω2_QM) × S̄₂


        de_dt = de_dt_GW + de_dt_GR 
        dL_dt = dL_dt_GW 

        du[1,3] = dS₁_dt[1]
        du[2,3] = dS₁_dt[2]
        du[3,3] = dS₁_dt[3]

        du[1,4] = dS₂_dt[1]
        du[2,4] = dS₂_dt[2]
        du[3,4] = dS₂_dt[3]

        du[1,5] = dL_dt[1]
        du[2,5] = dL_dt[2]
        du[3,5] = dL_dt[3]

        du[1,2] = de_dt[1]
        du[2,2] = de_dt[2]
        du[3,2] = de_dt[3]

        du[1,1] = da_dt_GW
    end

    return f!
end