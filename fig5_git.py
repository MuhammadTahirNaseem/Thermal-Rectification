# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 10:52:17 2025

@author: Tahir Naseem
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from matplotlib import ticker

# Constants
omega0 = 1.0
gamma_L = 0.2
gamma_R = 0.2
Gamma_L = 0.1  # Fixed value
Gamma_R_list = [0.001, 0.01]  # Right bath variation

kB = 1.0
n_max = 50
T_fixed = 2.0
T_values = np.linspace(0.01, 4.0, 50)

def n_th(T, omega=1.0):
    return 1.0 / (np.exp(omega / (kB * T)) - 1)

def n_th_2ph(T, omega=1.0):
    return 1.0 / (np.exp(2 * omega / (kB * T)) - 1)

def get_dissipator(L, rho):
    return L * rho * L.dag() - 0.5 * (L.dag() * L * rho + rho * L.dag() * L)

def compute_JR(TL, TR, Gamma_R_val):
    a = destroy(n_max)
    H = omega0 * a.dag() * a

    nL = n_th(TL)
    nR = n_th(TR)
    mL = n_th_2ph(TL)
    mR = n_th_2ph(TR)

    c_ops = [
        # One-photon
        np.sqrt(gamma_L * (nL + 1)) * a,
        np.sqrt(gamma_L * nL) * a.dag(),
        np.sqrt(gamma_R * (nR + 1)) * a,
        np.sqrt(gamma_R * nR) * a.dag(),

        # Two-photon
        np.sqrt(Gamma_L * (mL + 1)) * a**2,
        np.sqrt(Gamma_L * mL) * a.dag()**2,
        np.sqrt(Gamma_R_val * (mR + 1)) * a**2,
        np.sqrt(Gamma_R_val * mR) * a.dag()**2
    ]

    rho_ss = steadystate(H, c_ops)
    
    
    # L_emit = np.sqrt(gamma_R * (nR + 1)) * a
    # L_absorb = np.sqrt(gamma_R * nR) * a.dag()
    # J = expect(H, get_dissipator(L_emit, rho_ss) + get_dissipator(L_absorb, rho_ss))
    
    c_op_a = [gamma_R * (nR + 1) * lindblad_dissipator(a), gamma_R * nR * lindblad_dissipator(a.dag()),
              Gamma_R_val * (mR + 1) * lindblad_dissipator(a**2), Gamma_R_val * mR * lindblad_dissipator(a.dag()**2) ]
    L_a = liouvillian(H, c_op_a)
    rho_l = rho_ss
    Lovl = vector_to_operator(L_a * operator_to_vector(rho_l))  
    return np.real((Lovl * H).tr())

def compute_rectification(Jf, Jr):
    return np.abs(Jf + Jr) / max(np.abs(Jf), np.abs(Jr)) if max(np.abs(Jf), np.abs(Jr)) > 0 else 0

# Plotting setup
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
colors = ['green', 'red']
linestyles_fwd = ['solid', 'dotted']
linestyles_rev = ['dashed', 'dashdot']
linestyles_rect = ['solid', 'dashed']

for idx, Gamma_R in enumerate(Gamma_R_list):
    JR_fwd = [compute_JR(T_fixed, T, Gamma_R) for T in T_values]
    JR_rev = [compute_JR(T, T_fixed, Gamma_R) for T in T_values]
    R_vals = [compute_rectification(Jf, Jr) for Jf, Jr in zip(JR_fwd, JR_rev)]

    # Plot heat currents
    ax1.plot(T_values, JR_fwd,
             color=colors[idx],
             linestyle=linestyles_fwd[idx],
             lw=2.5,
             label=fr'Forward $\Gamma_R={Gamma_R}$')
    ax1.plot(T_values, JR_rev,
             color=colors[idx],
             linestyle=linestyles_rev[idx],
             lw=2.5,
             label=fr'Reverse $\Gamma_R={Gamma_R}$')

    # Plot rectification
    ax2.plot(T_values, R_vals,
             color=colors[idx],
             linestyle=linestyles_rect[idx],
             lw=2.5,
             label=fr'$\Gamma_R={Gamma_R}$')

# Axis labels and grid
ax1.set_ylabel(r'$\mathcal{J}_R$', fontsize=22)
ax1.legend(fontsize=13)
ax1.grid(True)

ax2.set_xlabel(r'$T$', fontsize=22)
ax2.set_ylabel(r'$\mathcal{R}$', fontsize=24)
ax2.legend(fontsize=13)
ax2.grid(True)

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
#plt.savefig("fig5.eps", dpi=600, bbox_inches='tight', transparent=True)
plt.show()
