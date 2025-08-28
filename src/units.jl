using Unitful, UnitfulAstro

const time_dim = Unitful.𝐓
const length_dim = Unitful.𝐋
const mass_dim = Unitful.𝐌
const spin_dim = mass_dim*length_dim^2/time_dim

const unit_time = u"yr"#upreferred(time_dim)
const unit_length = u"Rsun"#upreferred(length_dim)
const unit_mass = u"Msun"#upreferred(mass_dim)
const unit_spin = unit_mass*unit_length^2/unit_time

Unitful.preferunits(unit_time, unit_length, unit_mass)
const localpromotion = copy(Unitful.promotion)