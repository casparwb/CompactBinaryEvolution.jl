using Unitful, UnitfulAstro

const time_dim = Unitful.𝐓
const length_dim = Unitful.𝐋
const mass_dim = Unitful.𝐌
const spin_dim = mass_dim*length_dim^2/time_dim

const unit_time = u"s"#upreferred(time_dim)
const unit_length = u"Rsun"#upreferred(length_dim)
const unit_mass = u"Msun"#upreferred(mass_dim)
const unit_spin = unit_mass*unit_length^2/unit_time

Unitful.preferunits(unit_time, unit_length, unit_mass)
const localpromotion = copy(Unitful.promotion)


# function get_nbody_units(masses, positions)

#     M = mass_unit = sum(masses)
#     E = potential_energy(positions, masses)

#     Rv = length_unit = -0.25 * GRAVCONST * M^2 / E
#     σ = sqrt(0.5 * GRAVCONST * M / Rv)
#     time_unit = Rv / (σ * √2)


#     @assert ustrip(length_unit^3 / mass_unit / time_unit^2, GRAVCONST) ≈ 1.0
#     return length_unit, mass_unit, time_unit
# end