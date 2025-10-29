# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 12:12:32 2025

@author: Tahir Naseem
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import matplotlib.ticker as ticker

# ----------------------------
# Parameters
# ----------------------------
N = 50                 # Fock space cutoff
omega = 1.0            # Oscillator frequency

# Dissipation rates
gamma_L = 0.4        # Single-photon rate (left)
gamma_R = 0.4         # Single-photon rate (right)
Gamma_L = 0.01         # Two-photon rate (left)
Gamma_R = 0.0          # Two-photon rate (right, disabled)

# Fixed temperature of left bath
T_L = 0.25

# ----------------------------
# Helper: Bose-Einstein distribution
# ----------------------------
def bose(omega, T):
    return 1.0 / (np.exp(omega / T) - 1.0)

# ----------------------------
# Analytical expression using Gaussian approximation
# ----------------------------
def analytic_JR(nL, mL, nR):
    A = -2 * Gamma_L
    B = -1*(gamma_L + gamma_R) + 2 * Gamma_L * (4*mL+1)
    C =  (gamma_L * nL + gamma_R * nR + 4 * Gamma_L * mL)

    discriminant = B**2 - 4 * A * C
    if discriminant < 0:
        return None

    sqrt_disc = np.sqrt(discriminant)
    n1 = (-B + sqrt_disc) / (2 * A)
    n2 = (-B - sqrt_disc) / (2 * A)

    for sol in [n1, n2]:
        if np.isreal(sol) and sol > 0:
            return float(omega * gamma_R * (nR - np.real(sol)))
    return None

# ----------------------------
# Numerical solution using QuTiP
# ----------------------------
def numerical_JR(nL, nR, mL):
    a = destroy(N)
    adag = a.dag()
    num_op = adag * a

    c_ops = []

    # Left bath: two-photon processes (corrected with thermal dependence)
    c_ops.append(np.sqrt(Gamma_L * (mL + 1)) * a * a)   # emission
    c_ops.append(np.sqrt(Gamma_L * mL) * adag * adag)   # absorption

    # Left bath: single-photon processes
    c_ops.append(np.sqrt(gamma_L * (nL + 1)) * a)
    c_ops.append(np.sqrt(gamma_L * nL) * adag)

    # Right bath: single-photon processes only
    c_ops.append(np.sqrt(gamma_R * (nR + 1)) * a)
    c_ops.append(np.sqrt(gamma_R * nR) * adag)

    H = omega * num_op
    rho_ss = steadystate(H, c_ops)
    return omega * gamma_R * (nR - expect(num_op, rho_ss))

# ----------------------------
# Sweep over right bath temperature T_R
# ----------------------------
T_R_vals = np.linspace(0.01, 0.5, 40)
JR_ana_vals = []
JR_num_vals = []

# Fixed left bath stats
nL = bose(omega, T_L)
mL = bose(2 * omega, T_L)

for T_R in T_R_vals:
    nR = bose(omega, T_R)
    JR_ana_vals.append(analytic_JR(nL, mL, nR))
    JR_num_vals.append(numerical_JR(nL=nL, nR=nR, mL=mL))

# ----------------------------
# Plot results
# ----------------------------
fig, ax1 = plt.subplots(figsize=(9, 6))

# Plot heat currents

ax1.plot(T_R_vals, JR_ana_vals, '-', color='green', lw=2.5, label="Analytical")
ax1.plot(T_R_vals, JR_num_vals, 'o', color='red', label="Numerical", markersize=5)


# Axis labels
ax1.set_xlabel(r'$T_R$', fontsize=24)
ax1.set_ylabel(r'$\mathcal{J}_R$', fontsize=26)
ax1.legend(fontsize=14)
ax1.grid(True)

# Beautify plot
ax1.minorticks_on()
ax1.tick_params('both', top=True, right=True, direction='in',
                length=8, width=2.2, which='major', labelsize=16)
ax1.tick_params('both', top=True, right=True, direction='in',
                length=5, width=1.2, which='minor')
for side in ['top', 'bottom', 'left', 'right']:
    ax1.spines[side].set_linewidth(2.0)

ax1.yaxis.get_offset_text().set_fontsize(14)
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax1.yaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.savefig("fig3.eps", dpi=600, bbox_inches='tight', transparent=True)
plt.show()

