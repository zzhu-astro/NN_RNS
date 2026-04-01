import numpy as np

#################
# unit conversion
#################
C_CGS = 29979245800.0  # Vacuum speed of light
G_CGS = 6.67428e-8  # Gravitational constant
M_SOL_CGS = 1.98892e33  # Solar mass
# convert cactus to milliseconds
cac2ms = 1000 * G_CGS * M_SOL_CGS / pow(C_CGS, 3)
# convert cactus to km
cac2km = G_CGS * M_SOL_CGS / (C_CGS * C_CGS) / 1.0e5
# convert cactus to g/cm^3 and kg/m^3
cac2den = pow(C_CGS, 6) / (G_CGS * G_CGS * G_CGS * M_SOL_CGS * M_SOL_CGS)
cac2pre = cac2den * C_CGS * C_CGS
cac2den_si = cac2den * 1000.0
cac2pre_si = cac2pre / 10.0

EV_SI = 1.602176634e-19  # electric charge
EV_CGS = EV_SI * 1.0e7  # ev in erg
# convert MeV-fm unit to cgs and si
mev2pre = 1.0e45 * EV_CGS
mev2den = mev2pre / C_CGS / C_CGS
mev2pre_si = mev2pre / 10.0
mev2den_si = mev2den * 1.0e3

# atomic mass in cgs (g)
atom_mass = 1.660539066e-24
# baryonic mass for nuclear matter
m_nucl = atom_mass
########################################
