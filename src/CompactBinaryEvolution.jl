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

get_a_e(u, model) = u[1, 1], norm(SA[u[1,2], u[2,2], u[3,2]])
get_a_e(u, model::PetersModel) = u[1, 1], u[1, 2]

function peak_f_GW(sqrt_GM_div_π::T, a::T, e::T)::T where T <: Real
    # (a < zero(a) || e > one(e)) && return 0.0
    num = sqrt_GM_div_π*(1 + e)^1.195
    denom = sqrt((a*(1 - e^2))^3)
    return num/denom
end

# function get_merger_condition(bin, model::PetersModel; peak_frequency_stop=10.0u"Hz",
#                                                       a_min_rg=10)
#     GM = G*(bin.m1 + bin.m2)
#     min_a = a_min_rg*GM/c^2

#     sqrt_GM_div_π = sqrt(GM)/π
#     sqrt_GM_div_π = ustrip(upreferred(unit(sqrt_GM_div_π)), sqrt_GM_div_π)
#     get_a_e_(u) = get_a_e(u, model)

#     peak_frequency_stop = ustrip(unit_time^-1, peak_frequency_stop)
#     min_a = ustrip(unit_length, min_a)
#     min_a = min_a
#     function condition_merger!(out, u, t, integrator) 
#         a, e = u[1, 1], u[1, 2]
#         f_GW::Float64 = peak_f_GW(sqrt_GM_div_π, a, e)::Float64
#         out[1] = (f_GW - peak_frequency_stop)::Float64
#         out[2] = (a - min_a)::Float64
#         nothing
#     end

#     return condition_merger!
# end

# set_out!(out, val, idx) = out[idx] = val

# function get_merger_condition(bin, model; peak_frequency_stop=10.0u"Hz",
#                                                       a_min_rg=10)
#     GM = G*(bin.m1 + bin.m2)
#     min_a = a_min_rg*GM/c^2

#     sqrt_GM_div_π = sqrt(GM)/π
#     sqrt_GM_div_π = ustrip(upreferred(unit(sqrt_GM_div_π)), sqrt_GM_div_π)
#     # sqrt_GM_div_π::Float64 = Float64(sqrt_GM_div_π)
#     get_a_e_(u) = get_a_e(u, model)

#     peak_frequency_stop = ustrip(unit_time^-1, peak_frequency_stop)
#     min_a = ustrip(unit_length, min_a)
#     # min_a = min_a
#     function condition_merger!(out, u, t, integrator) 
#         a, e = u[1, 1], norm(SA[u[1, 2], u[2, 2], u[3, 2]])
#         f_GW::Float64 = peak_f_GW(sqrt_GM_div_π, a, e)::Float64
#         # out[1] = (f_GW - peak_frequency_stop)::Float64
#         # out[2] = (a - min_a)::Float64
#         out_freq::Float64 = f_GW - peak_frequency_stop
#         out_a::Float64 = a - min_a
#         set_out!(out, out_freq, 1)
#         set_out!(out, out_a, 2)

#         nothing
#     end

#     return condition_merger!
# end

function setup_callbacks(bin, model; merger_callback=true, saving_callback=false, save_every=1,
                              peak_frequency_stop=10.0u"Hz",
                              a_min_rg=10, verbose=false)

    cbs = []

    GM = G*(bin.m1 + bin.m2)
    min_a = a_min_rg*GM/c^2

    sqrt_GM_div_π = sqrt(GM)/π
    sqrt_GM_div_π = ustrip(upreferred(unit(sqrt_GM_div_π)), sqrt_GM_div_π)
    # sqrt_GM_div_π::Float64 = Float64(sqrt_GM_div_π)
    get_a_e_(u) = get_a_e(u, model)

    peak_frequency_stop = ustrip(unit_time^-1, peak_frequency_stop)
    min_a = ustrip(unit_length, min_a)
    if merger_callback
        function condition_merger!(out, u, t, integrator) 
                a, e = u[1, 1], norm(SA[u[1, 2], u[2, 2], u[3, 2]])
                f_GW::Float64 = peak_f_GW(sqrt_GM_div_π, a, e)::Float64
                # out[1] = (f_GW - peak_frequency_stop)::Float64
                # out[2] = (a - min_a)::Float64
                out_freq::Float64 = f_GW - peak_frequency_stop
                out_a::Float64 = a - min_a
                # set_out!(out, out_freq, 1)
                # set_out!(out, out_a, 2)
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
                              merger_callback=true, saving_callback=false, save_every=1,
                              peak_frequency_stop=10.0u"Hz",
                              a_min_rg=10, verbose=false,
                              args...)



    u0 = get_u0(bin, model)
    tspan = ustrip.(dtype, unit_time, (bin.time, t_final))
    
    callback_args = Dict(:merger_callback => merger_callback, 
                         :saving_callback => saving_callback, 
                         :save_every => save_every,
                         :peak_frequency_stop => peak_frequency_stop,
                         :a_min_rg => a_min_rg, 
                         :verbose => verbose)

    GM = G*(bin.m1 + bin.m2)
    min_a = a_min_rg*GM/c^2

    sqrt_GM_div_π = sqrt(GM)/π
    sqrt_GM_div_π = ustrip(upreferred(unit(sqrt_GM_div_π)), sqrt_GM_div_π)
    # sqrt_GM_div_π::Float64 = Float64(sqrt_GM_div_π)
    get_a_e_(u) = get_a_e(u, model)

    peak_frequency_stop = ustrip(unit_time^-1, peak_frequency_stop)
    min_a = ustrip(unit_length, min_a)
    function condition_merger!(out, u, t, integrator) 
            a::Float64 = u[1, 1]
            e::Float64 = norm(SA[u[1, 2], u[2, 2], u[3, 2]])
            f_GW::Float64 = peak_f_GW(sqrt_GM_div_π, a, e)::Float64
            # out[1] = (f_GW - peak_frequency_stop)::Float64
            # out[2] = (a - min_a)::Float64
            out_freq::Float64 = f_GW - peak_frequency_stop
            out_a::Float64 = a - min_a
            # set_out!(out, out_freq, 1)
            # set_out!(out, out_a, 2)
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

    # callbacks = setup_callbacks(bin, model; callback_args...)
    abstol = get(args, :abstol, 1e-6)
    reltol = get(args, :reltol, 1e-6)
    maxiters = get(args, :maxiters, 250_000)
    save_everystep = get(args, :save_everystep, !saving_callback)

    ode_func! = get_orbital_evolution_model(bin, model)
    prob = ODEProblem(ode_func!, u0, tspan)
    sol = solve(prob, alg; maxiters=maxiters, 
                           callback=cb_merger,
                           abstol=abstol,
                           reltol=reltol, 
                           save_everystep=save_everystep,
                           args...)

    return sol
end



end # end module