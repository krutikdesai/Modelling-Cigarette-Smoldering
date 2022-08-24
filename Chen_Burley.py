import matplotlib.pyplot as plt
import numpy as np
plt.style.use("seaborn-dark")

# Thermal, Physical Parameters and Initial Values

eps = 0.97
phi = 0.822    # Void Fraction
Dp  = 3.95e-4  # m
r   = 4.0e-3   # m
d   = 3.7e-5   # m
R   = 8.314    # J/mol.K
ks  = 0.3163   # W/m.K
kg  = 0.0346   # W/m.K
kp  = 0.8368   # W/m.K

Ar  = 3.14159 * r**2
ma  = 2.5e-6   # kg/s of air
Ta  = 293      # Atmosphere (K)
Tpb = 294      # Penetration Boundary (K)
Tbf = 723      # Burning Front (K)
Tal = 923      # Ash Line (K)
LBR = 6.4e-5   # Burn Rate (m/s)

mesh  = 10000  # Mesh Size for PZ
meshB = 300    # Mesh Size for BZ
w0    = 0.119  # Moisture Content
Rho0  = 236    # Total (kg/m^3)
RhoC0 = 0      # Char (kg/m^3)
RhoV0 = Rho0 / (1 + w0)  # Tobacco (kg/m^3)
RhoW0 = RhoV0 * w0       # Water (kg/m^3)


# Returns Reaction Parameters [n, E (J/mol), Z (1/s), Cp (J/kg.K)]

dat = np.array([[1, 87027, 1.0e8, 1423],    # Tobacco Pyrolysis (0)
                [1, 107110, 7.31e8, 1423],  # Tobacco Pyrolysis (1)
                [1, 166523, 1.32e13, 1423], # Tobacco Pyrolysis (2)
                [3, 116734, 2.64e7, 1423],  # Tobacco Pyrolysis (3)
                [1, 6.6e4, 3.0e7, 4180],    # Water Evaporation (4)
                [1, 1.7e5, 4.0e6, 1046]])   # Char Combustion   (5)


# Defining functions needed in setting and solving up the PDEs

def k_eff(T):
    hr = 41840 * 5.422e-12 * eps * T**3
    tmp1 = kg + (2 * hr * Dp / 3)
    return ks * (1 - pow(phi, 2/3)) + tmp1 * pow(phi, 1/3)


def h_prime(Tsf):
    h = 41840 * (8e-5 * pow((Tsf - Ta)/r, 0.25) + 5.422e-12 * eps * pow((Tsf + Ta)/2, 3))
    tmp2 = 1 + (d * h / kp) + (0.693 * r * h / k_eff(Tsf))
    return h / tmp2


def D_x(Tsf):
    return 1.1e-5 * (Tsf/273)**1.75


def gill(dydx, x0, y0, n):
    y = np.empty(n)
    y[0] = y0
    h = Lc / (n - 1) if x0 == 0 else abs(x0 / (n - 1))

    for j in range(1, n):
        k1 = h * dydx(x0, y[j - 1])
        k2 = h * dydx(x0 + h / 2, y[j - 1] + 0.5 * k1)
        k3 = h * dydx(x0 + h / 2, y[j - 1] + (-0.5 + 2 ** -0.5) * k1 + k2 * (1 - 2 ** -0.5))
        k4 = h * dydx(x0 + h, y[j - 1] - (2 ** -0.5) * k2 + k3 * (1 + 2 ** -0.5))

        y[j] = y[j - 1] + (k1 + (2 - 2 ** 0.5) * k2 + (2 + 2 ** 0.5) * k3 + k4) / 6
        x0 = x0 + h

    return y


# Calculations in the Pyrolysis Zone

Sp = (dat[0,3] + w0 * dat[4,3]) * RhoV0         # Vol. Heat Capacity
A  = 3 * k_eff(Tbf) / Sp
B  = h_prime(Tbf) * (Tbf - Ta) / (2 * r * Sp * (Tbf - Tpb))
Lp = (-LBR + np.sqrt(LBR**2 + 4*A*B)) / (2*B)   # PZ Length at Steady State
Tavg = 0.75 * Tpb + 0.25 * Tbf                  # Average Temperature in PZ
q  = 3 * k_eff(Tavg) * (Tbf - Tpb) / Lp         # Heat Flux from Burning Zone to PZ


def T(x):  # Temperature in PZ at Steady State
    return Tpb + (Tbf - Tpb) * ((x + Lp) / Lp) ** 3


def pzl_V(x, y):  # Tobacco Pyrolysis
    tmp = 0
    sway = np.array([0.08, 0.32, 0.3, 0.3])  # Contributions from each reaction
    for i in range(4):
        tmp += -sway[i] * dat[i,2] * np.exp(-dat[i,1]/(R * T(x))) * (y/RhoV0)**dat[i,0]
    return tmp * RhoV0 / LBR


RhoV = gill(pzl_V, -Lp, RhoV0, 2*mesh-1)


def pzl_W(x, y):  # Water Evaporation
    i = int(round(2 * (mesh-1) * (1 + x/Lp)))
    return -dat[4,2] * np.exp(-dat[4,1]/(R * T(x))) * RhoW0 * (y*RhoV[i]/RhoW0)**dat[4,0] / LBR


RhoW = gill(pzl_W, -Lp, RhoW0, mesh)


def pzl_C(x, y):  # Char Formation
    i = int(round(2 * (mesh-1) * (1 + x/Lp)))
    return -0.34 * pzl_V(x, RhoV[i])


RhoC1 = gill(pzl_C, -Lp, RhoC0, mesh)
Rho1 = np.empty(mesh)
for j in range(mesh):
    Rho1[j] = RhoV[2*j] + RhoW[j] + RhoC1[j]


# Calculations in the Burning Zone

Tmax = 673 + 110 * (1e6*ma)**0.67 / (100*r)**0.75
F = (Tal - Tbf) / (Tmax - Tbf)
G = (((F ** 2 - 1) ** 0.5 - F + (0.5 / F)) / (4 * F ** 2)) ** (1 / 3)
G = G * (-1 - (-3) ** 0.5) / 2
gm = np.real(G + 1 / (4 * G * F ** 2) + 1 / (2 * F))
Tavg = Tbf + (Tmax - Tbf) * (3/gm - 0.5/gm**3) / 4
Lc = 3 * k_eff(Tbf)*(Tmax - Tbf)/(2*gm*q)


def T_bz(x):   # Temperature in BZ at Steady State
    tmp = x/(gm*Lc)
    return Tbf + (Tmax - Tbf) * (3*tmp - tmp**3) / 2


def bz_C(x, y):  # Char Combustion
    return -dat[5,2] * np.exp(-dat[5,1]/(R * T_bz(x))) * y / LBR


RhoC2 = gill(bz_C, 0, RhoC1[-1], 2*meshB-1)


def bz_A(x, y):  # Ash Formation
    i = int(round(2 * (meshB-1) * (1 - x/Lc)))
    return -0.38 * bz_C(x, RhoC2[i])


RhoA = gill(bz_A, 0, 0, meshB)
Rho2 = np.empty(meshB)
for j in range(meshB):
    Rho2[j] = RhoC2[2*j] + RhoA[j]

print('Pyrolysis Zone Length, Lp = ', 1000*Lp, ' mm')
print('Burning Zone Length, Lc = ', 1000*Lc, ' mm')
print('Maximum Temperature, Tmax = ', Tmax, ' K')
print('Gamma = ', gm)

# For graphing the obtained quantities

x1 = np.linspace(-1000*Lp, 0, mesh)
x2 = np.linspace(-1000*Lp, 0, 2*mesh-1)
x3 = np.linspace(0, 1000*Lc, meshB)
x4 = np.linspace(0, 1000*Lc, 2*meshB-1)

fig, ax1 = plt.subplots()

ax1.plot(x1, T(x1/1000), color='#D95319')
ax1.plot(x3, T_bz(x3/1000), color='#D95319')
ax1.set_xlabel('Distance from Burning Front (mm)', family='serif', size='medium', weight='semibold')
ax1.set_ylabel('Temperature (K)', color='#D95319', family='serif', size='medium', weight='semibold')
ax1.tick_params(labelsize='large')

ax2 = ax1.twinx()

ax2.plot(x2, RhoV/Rho0, label='Tobacco (V)', color='#7E2F8E')
ax2.plot(x1, RhoW/Rho0, label='Moisture (W)', color='#4DBEEE')

ax2.plot(x1, RhoC1/Rho0, label='Char (C)', color='#EDB120')
ax2.plot(x4, RhoC2/Rho0, color='#EDB120')
ax2.plot(x3, RhoA/Rho0, label='Ash (A)', color='slategrey')

ax2.plot(x1, Rho1/Rho0, label='Total Density', color='#77AC30')
ax2.plot(x3, Rho2/Rho0, color='#77AC30')

ax2.set_ylabel('Fractional Density', family='serif', size='medium', weight='semibold')
ax2.tick_params(labelsize='large')
ax2.legend(fontsize='large', frameon=1, framealpha=1, loc=6)

plt.show()
