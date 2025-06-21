# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 15:18:10 2025

@author: Tahir Naseem
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from qutip import *

# === Constants ===
omega = 1.0
gamma_R = 0.5
gamma_L = 0.5
T_L_fixed = 2.0  # Updated temperature
kB = hbar = 1.0
fock_dim = 50
Gamma_list = [0.001, 0.1]  # Different Gamma_L values to simulate

# === Thermal functions ===
def n_th(T, omega=1.0):
    return 1.0 / (np.exp(omega / (kB * T)) - 1)

def n_th_2ph(T, omega=1.0):
    return 1.0 / (np.exp(2 * omega / (kB * T)) - 1)

# === Numerical heat current ===
def numerical_JR(T_R_val, T_L_val, Gamma_L_val):
    n_R = n_th(T_R_val)
    n_L = n_th(T_L_val)
    m_L = n_th_2ph(T_L_val)

    a = destroy(fock_dim)
    H = omega * a.dag() * a

    c_ops = [
        np.sqrt(gamma_R * (n_R + 1)) * a,
        np.sqrt(gamma_R * n_R) * a.dag(),
        np.sqrt(gamma_L * (n_L + 1)) * a,
        np.sqrt(gamma_L * n_L) * a.dag(),
        np.sqrt(Gamma_L_val * (m_L + 1)) * a**2,
        np.sqrt(Gamma_L_val * m_L) * (a.dag())**2
    ]

    rho_ss = steadystate(H, c_ops)
    n_ss = expect(a.dag() * a, rho_ss)
    return omega * gamma_R * (n_R - n_ss)

# === Rectification ===
def compute_rectification(J_plus_list, J_minus_list):
    R_list = []
    for Jp, Jm in zip(J_plus_list, J_minus_list):
        numerator = abs(Jp + Jm)
        denominator = max(abs(Jp), abs(Jm))
        R = numerator / denominator if denominator != 0 else 0
        R_list.append(R)
    return R_list

# === Temperature range for numerical evaluation ===
T_R_vals = np.linspace(0.01, 4.0, 100)

# === Plotting setup ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)

colors = ['green', 'red']
line_styles_forward = ['solid', 'dotted']
line_styles_reverse = ['dashed', 'dashdot']
line_styles_rect = ['solid', 'dashed']

for idx, Gamma_L in enumerate(Gamma_list):
    JR_forward = [numerical_JR(T_R, T_L_fixed, Gamma_L) for T_R in T_R_vals]
    JR_reverse = [numerical_JR(T_L_fixed, T_R, Gamma_L) for T_R in T_R_vals]
    R_vals = compute_rectification(JR_forward, JR_reverse)

    # Plot heat currents
    ax1.plot(T_R_vals, JR_forward,
             color=colors[idx],
             linestyle=line_styles_forward[idx],
             lw=2.5,
             label=fr'Forward $\Gamma_L={Gamma_L}$')

    ax1.plot(T_R_vals, JR_reverse,
             color=colors[idx],
             linestyle=line_styles_reverse[idx],
             lw=2.5,
             label=fr'Reverse $\Gamma_L={Gamma_L}$')

    # Plot rectification
    ax2.plot(T_R_vals, R_vals,
             color=colors[idx],
             linestyle=line_styles_rect[idx],
             lw=2.5,
             label=fr'$\Gamma_L={Gamma_L}$')

# === Axis labels and grid ===
ax1.set_ylabel(r'$\mathcal{J}_R$', fontsize=24)
ax1.legend(fontsize=13)
ax1.grid(True)

ax2.set_xlabel(r'$T$', fontsize=22)
ax2.set_ylabel(r'$\mathcal{R}$', fontsize=24)
ax2.legend(fontsize=13)
ax2.grid(True)

# === Formatting ===
for ax in [ax1, ax2]:
    ax.minorticks_on()
    ax.tick_params('both', top=True, right=True, direction='in',
                   length=8, width=2.2, which='major', labelsize=16)
    ax.tick_params('both', top=True, right=True, direction='in',
                   length=5, width=1.2, which='minor')
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_linewidth(2.0)
    ax.yaxis.get_offset_text().set_fontsize(14)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)

plt.tight_layout()
#plt.savefig("fig4.eps", dpi=600, bbox_inches='tight', transparent=True)
plt.show()
