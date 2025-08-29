module CompactBinaryEvolution

export CompactBinary, do_orbital_evolution, newtonian_angular_momentum
export PetersModel, PetersWithPrecession, FumagelliModel, BarkerOConnell, BackReaction, NoBackReaction

include("units.jl")

function __init__()
    Unitful.register(CompactBinaryEvolution)
    merge!(Unitful.promotion, localpromotion)
end

using LinearAlgebra: dot, ⋅, cross, ×
using StaticArrays
using OrdinaryDiffEqTsit5
using SciMLBase: DiscreteCallback, VectorContinuousCallback, CallbackSet, RightRootFind
import RecursiveArrayTools

const G = upreferred(Unitful.G)
const c = upreferred(Unitful.c0)

const UNITLESS_G = ustrip(upreferred(dimension(G)), G)
const UNITLESS_c = ustrip(upreferred(dimension(c)), c)

const c² = UNITLESS_c^2
const Gc⁻² = UNITLESS_G/c²

const div_121_304 = 121/304
const div_73_24 = 73/24
const div_304_15 = 304/15
const div_37_96 = 37/96

abstract type OrbitalEvolutionModel end
abstract type SpinEvolutionModel end

struct PetersModel          <: OrbitalEvolutionModel end
struct PetersWithPrecession <: OrbitalEvolutionModel end
struct FumagelliModel       <: OrbitalEvolutionModel end
struct BarkerOConnell{T}    <: SpinEvolutionModel backreaction::T end
struct BackReaction end
struct NoBackReaction end

function newtonian_angular_momentum(m1, m2, a, e::Number)
    M = m1 + m2
    μ = (m1*m2)/M

    return μ*√(G*M*a*(1 - e^2))
end

function newtonian_angular_momentum(m1, m2, a, e::AbstractVector)
    newtonian_angular_momentum(m1, m2, a, norm(e))
end



struct CompactBinary{tT, mT, aT, eT, lT, ST}
    time::tT
    m1::mT
    m2::mT
    a::aT
    e::eT
    L::lT
    S1::ST
    S2::ST

    function CompactBinary(;m1::Unitful.Mass, m2::Unitful.Mass, 
                            a::Unitful.Length, e, 
                            time=0.0u"yr",
                            L = nothing,
                            S1=nothing, S2=nothing)
        tT = typeof(time)
        mT = typeof(m1)
        aT = typeof(a)
        eT = typeof(e)
        sT = typeof(S1)

        L = if isnothing(L) 
            newtonian_angular_momentum(m1, m2, a, e)*SA[0.0, 0.0, 1.0]
        elseif L isa Number
            L*SA[0.0, 0.0, 1.0]
        else 
            L
        end

        lT = typeof(L)
        return new{tT, mT, aT, eT, lT, sT}(time, m1, m2, a, e, L, S1, S2)
    end
end

struct CompactBinarySolution{tT, aT, eT, evT, lT, sT}
    ic::CompactBinary
    time::tT
    a::aT
    e::eT
    e_vec::evT
    L::lT
    S1::sT
    S2::sT

    function CompactBinarySolution(sol, binary)

        u = Array(sol)

        S1, S2 = if size(u, 2) == 5
            u[:,3,:] .* unit_spin, u[:,4,:] .* unit_spin
        else
            nothing, nothing
        end

        L = u[:,:,end] .* unit_spin

        a = u[1,1,:] .* unit_length
        e_vec = u[:,2,:]
        e = norm.(eachcol(e_vec))

        time = sol.t .* unit_time

        tT = typeof(time)
        aT = typeof(a)
        eT = typeof(e)
        evT = typeof(e_vec)
        sT = typeof(S1)
        lT = typeof(L)

        return new{tT, aT, eT, evT, lT, sT}(binary, time, a, e, e_vec, L, S1, S2)
    end
end

function Base.show(io::IO, sol::CompactBinarySolution)
    N = length(sol.time)
    a_final = sol.a[end]
    e_final = sol.e[end]
    t_final = sol.time[end]

    println("CompactBinarySolution: ")
    println("  N: ", N)
    println("  Final a [R⊙]: ", a_final)
    println("  Final e: ", e_final)
    println("  Final t [yr]: ", t_final)
end

include("equations.jl")

function get_u0(bin, ::OrbitalEvolutionModel)
    a = ustrip(unit_length, bin.a)
    a0 = [a, 0.0, 0.0]
    e0 = bin.e
    L = ustrip.(unit_spin, bin.L)
    return [a0 e0 L]
end

function get_u0(bin, ::PetersModel)
    a = ustrip(unit_length, bin.a)
    a0 = [a, 0.0, 0.0]
    e0 = [bin.e, 0.0, 0.0]
    L = ustrip.(unit_spin, bin.L)
    return [a0 e0 L]
end

function get_u0(bin, ::SpinEvolutionModel)
    if any(isnothing, (bin.S1, bin.S2))
        throw(ArgumentError("Model includes spin evolution but spins are not initialized."))
    end


    if dimension(bin.S1[1]) != spin_dim 
        throw(ErrorException("S1 has wrong dimensions."))
    elseif dimension(bin.S2[1]) != spin_dim 
        throw(ErrorException("S2 has wrong dimensions."))
    end

    a = ustrip(unit_length, bin.a)
    a0 = [a, 0.0, 0.0]
    e0 = bin.e
    L = ustrip.(unit_spin, bin.L)

    S1 = ustrip.(unit_spin, bin.S1)
    S2 = ustrip.(unit_spin, bin.S2)
    return [a0 e0 S1 S2 L]
end


function peak_f_GW(sqrt_GM_div_π::T, a::T, e::T)::T where T <: Real
    # (a < zero(a) || e > one(e)) && return 0.0
    num = sqrt_GM_div_π*(1 + e)^1.195
    denom = sqrt((a*(1 - e^2))^3)
    return num/denom
end


function setup_callbacks(bin, model; merger_callback=true, saving_callback=false, save_every=1,
                              peak_frequency_stop=10.0u"Hz",
                              a_min_rg=10, verbose=false)

    cbs = []

    GM = UNITLESS_G*ustrip(Float64, u"Msun", bin.m1 + bin.m2)
    sqrt_GM_div_π = sqrt(GM)/π
    
    min_a = a_min_rg*GM/UNITLESS_c^2
    peak_frequency_stop = ustrip(Float64, unit_time^-1, peak_frequency_stop)
    if merger_callback
        function condition_merger!(out, u, t, integrator) 

                a = u[1, 1]
                e = norm(SA[u[1, 2], u[2, 2], u[3, 2]])
                f_GW = peak_f_GW(sqrt_GM_div_π, a, e)
                out_freq = f_GW - peak_frequency_stop
                out_a = a - min_a
                out[1] = out_freq
                out[2] = out_a

                nothing
            end        
            
        function affect_merger!(integrator, idx) 
            if idx == 1
                verbose && println("Merger! (fGW)")
                terminate!(integrator)
            elseif idx == 2
                verbose && println("Merger! (a)")
                terminate!(integrator)
            end
            nothing
        end

        cb_merger = VectorContinuousCallback(condition_merger!, affect_merger!, 2, 
                                            save_positions=(false, false), 
                                            rootfind=RightRootFind,
                                            interp_points=100)

        push!(cbs, cb_merger)
    end

    if saving_callback
        condition_saving(u, t, integrator) = iszero(integrator.iter % save_every)
        function save_step!(integrator)
            savevalues!(integrator, true)
        end
        cb_saving = DiscreteCallback(condition_saving, save_step!, save_positions=(false, false))
        push!(cbs, cb_saving)
    end

    if isempty(cbs)
        return nothing
    elseif isone(length(cbs))
        return only(cbs)
    else
        return CallbackSet(cbs...)
    end
end

function do_orbital_evolution(bin::CompactBinary, model;
                              t_final=13.8u"Gyr", dtype=Float64,
                              alg=Tsit5(),
                              merger_callback=true, 
                              save_every=Inf,
                              peak_frequency_stop=10.0u"Hz",
                              a_min_rg=10, verbose=false,
                              args...)



    u0 = get_u0(bin, model)
    tspan = ustrip.(dtype, unit_time, (bin.time, t_final))
    
    callback_args = Dict(:merger_callback => merger_callback, 
                         :saving_callback => !isinf(save_every), 
                         :save_every => save_every,
                         :peak_frequency_stop => peak_frequency_stop,
                         :a_min_rg => a_min_rg, 
                         :verbose => verbose)

    # GM = UNITLESS_G*ustrip(Float64, u"Msun", bin.m1 + bin.m2)
    # sqrt_GM_div_π = sqrt(GM)/π
    
    # min_a = a_min_rg*GM/UNITLESS_c^2
    # peak_frequency_stop = ustrip(Float64, unit_time^-1, peak_frequency_stop)
    # function condition_merger!(out, u, t, integrator) 

    #         a = u[1, 1]
    #         e = norm(SA[u[1, 2], u[2, 2], u[3, 2]])
    #         f_GW = peak_f_GW(sqrt_GM_div_π, a, e)
    #         out_freq = f_GW - peak_frequency_stop
    #         out_a = a - min_a
    #         out[1] = out_freq
    #         out[2] = out_a

    #         nothing
    #     end        
        
    # function affect_merger!(integrator, idx) 
    #     if idx == 1
    #         verbose && println("Merger! (fGW)")
    #         terminate!(integrator)
    #     elseif idx == 2
    #         verbose && println("Merger! (a)")
    #         terminate!(integrator)
    #     end
    #     nothing
    # end

    # cb_merger = VectorContinuousCallback(condition_merger!, affect_merger!, 2, 
    #                                     save_positions=(false, false), 
    #                                     rootfind=RightRootFind,
    #                                     interp_points=100)

    callbacks = setup_callbacks(bin, model; callback_args...)
    abstol = get(args, :abstol, 1e-6)
    reltol = get(args, :reltol, 1e-6)
    dense = get(args, :dense, false)
    maxiters = get(args, :maxiters, 100_000)
    # save_everystep = get(args, :save_everystep, !saving_callback)
    save_everystep = !haskey(args, :saveat)
    save_everystep = !callback_args[:saving_callback]

    ode_func! = get_orbital_evolution_model(bin, model)
    prob = ODEProblem(ode_func!, u0, tspan)
    sol = solve(prob, alg; maxiters=maxiters, 
                           callback=callbacks,
                           abstol=abstol,
                           reltol=reltol, 
                           save_everystep=save_everystep,
                           dense=dense,
                           args...)

    return CompactBinarySolution(sol, bin)
end



end # end module