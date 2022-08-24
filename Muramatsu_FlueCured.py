import matplotlib.pyplot as plt
import numpy as np
plt.style.use("seaborn-dark")

# Thermal, Physical Parameters and Initial Values

eps = 0.97
phi = 0.798    # Void Fraction (Real Density)
Dp  = 5.75e-4  # m
r   = 4.0e-3   # m
d   = 3.7e-5   # m
ks  = 0.3163   # W/m.K
kg  = 0.0346   # W/m.K
kp  = 0.8368   # W/m.K
R   = 8.314    # J/mol.K

Ta  = 293      # Atmosphere (K)
T0  = 294      # Penetration Boundary (K)
LBR = 4.5e-5   # Burn Rate (m/s)
w0  = 0.1311   # Moisture Content
RhoT0  = 281   # Total (kg/m^3)

RhoV0 = RhoT0/(1 + w0)  # Tobacco (kg/m^3)
RhoW0 = RhoV0 * w0      # Moisture (kg/m^3)
Ar    = 3.14159 * r**2  # Cross-Sectional Area
sig   = 1.38064852e-23  # Boltzmann Constant
phi_w = 0.659           # Void Fraction (Apparent Density)


# Returns Reaction Parameters [n, E (J/mol), Z (1/s), Cp (J/kg.K)]

dat = np.array([[1, 84516.8, 6.27e7, 1716],    # Tobacco Pyrolysis (0)
                [1, 102508, 1.69e8, 1716],     # Tobacco Pyrolysis (1)
                [1, 191208.8, 5.99e14, 1716],  # Tobacco Pyrolysis (2)
                [3, 105436.8, 4.69e6, 1716]])  # Tobacco Pyrolysis (3)


# Defining functions needed in setting up the PDEs

def k_e(T):
    hr = 41840 * 5.422e-12 * eps * T**3
    tmp1 = kg - (2 * hr * Dp / 3)
    return ks * (1 - pow(phi, 2/3)) + pow(phi, 1/3) * tmp1


def D_x(Tsf):
    return 1.1e-5 * (Tsf/273)**1.75


# Setting up the System of PDEs

def pzl_V(T, RhoV):  # Tobacco Pyrolysis

    tmp = 0
    sway = np.array([0.25, 0.28, 0.17, 0.3])  # Contributions from each reaction
    for i in range(4):
        tmp += -sway[i] * dat[i,2] * np.exp(-dat[i,1]/(R * T)) * (RhoV/RhoV0)**dat[i,0]
    return tmp * RhoV0 / LBR


def pzl_C(T, RhoV):  # Char Formation
    return -0.34 * pzl_V(T, RhoV)


def pzl_W(T, RhoW, Pw): # Water Evaporation

    Pws = 133.322 * 10**(7.991 - 1687/(T - 43))
    c1 = Pw/Pws
    w_eq = c1/(5.12 + 8.46*c1 - 14.16*(c1**2))
    c2 = 92.4015e10/LBR
    return -c2 * np.exp(-8430/T) * ((RhoW/RhoV0) - w_eq)**1.81


def pzl_Pw(T, Tx, RhoV, RhoC, RhoW, Pw):

    b1 = Pw * pzl_Tx(T, Tx, RhoV, RhoC, RhoW, Pw) / T
    b2 = 2 * 0.015 * (Pw - (1399.89 * T/Ta)) / (phi_w * r * d)
    b3 = LBR * R * T * pzl_W(T, RhoW, Pw) / (phi_w * 0.018 * D_x(T))
    b4 = LBR / D_x(T) + (0.25 * Tx / T)
    b5 = Pw * Tx / T
    return b5 - (b1 + b2 + b3) / b4


def pzl_Tx(T, Tx, RhoV, RhoC, RhoW, Pw):

    ke = k_e(T)
    h1 = 41840 * 8e-5 * pow((T - Ta)/r, 0.25)
    h2 = 41840 * 5.422e-12 * eps * T**3
    Qw = 2260871.99622  # J/kg
    Sp = RhoV * dat[0,3] + RhoW * 4184 + RhoC * 1046

    p1 = Tx * (LBR * Sp - (2 * Dp * pow(phi,1/3) * h2 * Tx/T)) / ke
    p2 = pzl_V(T, RhoV) * dat[0,3] + pzl_W(T, RhoW, Pw) * 4184 + pzl_C(T, RhoV) * 1046
    p3 = LBR * ((T-Ta) * p2 - (Qw * pzl_W(T, RhoW, Pw))) / ke
    p4 = 2 * (h1*(T-Ta) + sig*eps*(T**4 - Ta**4)) / (r * ke)

    return p1 + p3 + p4


# Solving the Nonlinear System of PDEs with the Runge-Kutta-Gill Method

mesh = 500
h = 2e-5  # Step Size (m)

T    = np.empty(mesh)
Tx   = np.empty(mesh)
Pw   = np.empty(mesh)
RhoT = np.empty(mesh)
RhoV = np.empty(mesh)
RhoC = np.empty(mesh)
RhoW = np.empty(mesh)

T[0]  = T0
Tx[0] = LBR * (T0 - Ta) * (dat[0,3] + w0 * 4184) * RhoV0 / k_e(T0)
Pw[0] = 1404  # Pascals
RhoV[0] = RhoV0
RhoT[0] = RhoT0
RhoW[0] = RhoW0

for j in range(1, mesh):

    a1 = h * Tx[j-1]
    b1 = h * pzl_V(T[j-1], RhoV[j-1])
    c1 = h * pzl_C(T[j-1], RhoV[j-1])
    d1 = h * pzl_W(T[j-1], RhoW[j-1], Pw[j-1])
    e1 = h * pzl_Pw(T[j-1], Tx[j-1], RhoV[j-1], RhoC[j-1], RhoW[j-1], Pw[j-1])
    f1 = h * pzl_Tx(T[j-1], Tx[j-1], RhoV[j-1], RhoC[j-1], RhoW[j-1], Pw[j-1])

    a2 = h * (Tx[j-1] + a1/2)
    b2 = h * pzl_V(T[j-1] + a1/2, RhoV[j-1] + b1/2)
    c2 = h * pzl_C(T[j-1] + a1/2, RhoV[j-1] + b1/2)
    d2 = h * pzl_W(T[j-1] + a1/2, RhoW[j-1] + d1/2, Pw[j-1] + e1/2)
    e2 = h * pzl_Pw(T[j-1] + a1/2, Tx[j-1] + f1/2, RhoV[j-1] + b1/2, RhoC[j-1] + c1/2, RhoW[j-1] + d1/2, Pw[j-1] + e1/2)
    f2 = h * pzl_Tx(T[j-1] + a1/2, Tx[j-1] + f1/2, RhoV[j-1] + b1/2, RhoC[j-1] + c1/2, RhoW[j-1] + d1/2, Pw[j-1] + e1/2)

    dA = (-0.5 + 2**-0.5) * a1 + (1 - 2**-0.5) * a2
    dB = (-0.5 + 2**-0.5) * b1 + (1 - 2**-0.5) * b2
    dC = (-0.5 + 2**-0.5) * c1 + (1 - 2**-0.5) * c2
    dD = (-0.5 + 2**-0.5) * d1 + (1 - 2**-0.5) * d2
    dE = (-0.5 + 2**-0.5) * e1 + (1 - 2**-0.5) * e2
    dF = (-0.5 + 2**-0.5) * f1 + (1 - 2**-0.5) * f2

    a3 = h * (Tx[j-1] + dA)
    b3 = h * pzl_V(T[j-1] + dA, RhoV[j-1] + dB)
    c3 = h * pzl_C(T[j-1] + dA, RhoV[j-1] + dB)
    d3 = h * pzl_W(T[j-1] + dA, RhoW[j-1] + dD, Pw[j-1] + dE)
    e3 = h * pzl_Pw(T[j-1] + dA, Tx[j-1] + dF, RhoV[j-1] + dB, RhoC[j-1] + dC, RhoW[j-1] + dD, Pw[j-1] + dE)
    f3 = h * pzl_Tx(T[j-1] + dA, Tx[j-1] + dF, RhoV[j-1] + dB, RhoC[j-1] + dC, RhoW[j-1] + dD, Pw[j-1] + dE)

    dA = -(2**-0.5) * a2 + (1 + 2**-0.5) * a3
    dB = -(2**-0.5) * b2 + (1 + 2**-0.5) * b3
    dC = -(2**-0.5) * c2 + (1 + 2**-0.5) * c3
    dD = -(2**-0.5) * d2 + (1 + 2**-0.5) * d3
    dE = -(2**-0.5) * e2 + (1 + 2**-0.5) * e3
    dF = -(2**-0.5) * f2 + (1 + 2**-0.5) * f3

    a4 = h * (Tx[j-1] + dA)
    b4 = h * pzl_V(T[j-1] + dA, RhoV[j-1] + dB)
    c4 = h * pzl_C(T[j-1] + dA, RhoV[j-1] + dB)
    d4 = h * pzl_W(T[j-1] + dA, RhoW[j-1] + dD, Pw[j-1] + dE)
    e4 = h * pzl_Pw(T[j-1] + dA, Tx[j-1] + dF, RhoV[j-1] + dB, RhoC[j-1] + dC, RhoW[j-1] + dD, Pw[j-1] + dE)
    f4 = h * pzl_Tx(T[j-1] + dA, Tx[j-1] + dF, RhoV[j-1] + dB, RhoC[j-1] + dC, RhoW[j-1] + dD, Pw[j-1] + dE)

    T[j] = T[j-1] + (a1 + (2 - 2**0.5) * a2 + (2 + 2**0.5) * a3 + a4) / 6
    Pw[j] = Pw[j-1] + (e1 + (2 - 2**0.5) * e2 + (2 + 2**0.5) * e3 + e4) / 6
    Tx[j] = Tx[j-1] + (f1 + (2 - 2**0.5) * f2 + (2 + 2**0.5) * f3 + f4) / 6
    RhoV[j] = RhoV[j-1] + (b1 + (2 - 2**0.5) * b2 + (2 + 2**0.5) * b3 + b4) / 6
    RhoC[j] = RhoC[j-1] + (c1 + (2 - 2**0.5) * c2 + (2 + 2**0.5) * c3 + c4) / 6
    RhoW[j] = RhoW[j-1] + (d1 + (2 - 2**0.5) * d2 + (2 + 2**0.5) * d3 + d4) / 6

for i in range(1, mesh):
    RhoT[i] = RhoV[i] + RhoW[i] + RhoC[i]

# For graphing the obtained quantities

x1 = np.linspace(-1000*h*mesh, 0, mesh)

ax1 = plt.subplot(222)
ax1.plot(x1, T, label='Temperature', color='#D95319')
ax1.set_ylabel('Temperature (K)', color='#D95319', family='serif', size='medium', weight='semibold')
ax1.tick_params(labelsize='large')

ax2 = plt.subplot(121)
ax2.plot(x1[250:], RhoV[250:]/RhoT0, label='Tobacco (V)', color='#7E2F8E')
ax2.plot(x1[250:], RhoW[250:]/RhoT0, label='Moisture (W)', color='#4DBEEE')
ax2.plot(x1[250:], RhoC[250:]/RhoT0, label='Char (C)', color='#EDB120')
ax2.plot(x1[250:], RhoT[250:]/RhoT0, label='Total Density', color='#77AC30')

ax2.set_xlabel('Position (mm)', family='serif', size='medium', weight='semibold')
ax2.set_ylabel('Fractional Density', family='serif', size='medium', weight='semibold')
ax2.tick_params(labelsize='large')
ax2.legend(fontsize='large', frameon=1, framealpha=1, loc=6)

ax3 = plt.subplot(224)
ax3.plot(x1, Pw/1000, label='Vapour Pressure', color='#0072BD')
ax3.set_xlabel('Position (mm)', family='serif', size='medium', weight='semibold')
ax3.set_ylabel('Vapour Pressure (kPa)', color='#0072BD', family='serif', size='medium', weight='semibold')
ax3.tick_params(labelsize='large')

ax1.grid(c='slategrey')
ax2.grid(c='slategrey')
ax3.grid(c='slategrey')
plt.show()