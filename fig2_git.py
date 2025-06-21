# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 15:10:34 2025

@author: Tahir Naseem
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from matplotlib import ticker

# Constants
omega = 1.0 
Gamma_L = 0.1
Gamma_R_main = 0.001  # main case
Gamma_R_alt = 0.01    # second rectification curve
kB = 1
n_max = 50

# Temperature values
T_values = np.linspace(0.01, 4.0, 50)       # for heat current plot
T_dense = np.linspace(0.01, 4.0, 1000)      # for rectification plot
T_fixed = 2.0                               # fixed T_L or T_R

# Store heat currents
J_anal_forward, J_num_forward = [], []
J_anal_reverse, J_num_reverse = [], []

# Define dissipator
def get_dissipator(L, rho):
    return L * rho * L.dag() - 0.5 * (L.dag() * L * rho + rho * L.dag() * L)

# === Heat Current ===
for T in T_values:
    H = omega * num(n_max)

    # Forward
    m_L = 1 / (np.exp(2 * omega / (kB * T_fixed)) - 1)
    m_R = 1 / (np.exp(2 * omega / (kB * T)) - 1)

    c_ops_fwd = [
        np.sqrt(Gamma_L * (m_L + 1)) * destroy(n_max)**2,
        np.sqrt(Gamma_L * m_L) * create(n_max)**2,
        np.sqrt(Gamma_R_main * (m_R + 1)) * destroy(n_max)**2,
        np.sqrt(Gamma_R_main * m_R) * create(n_max)**2
    ]
    rho_ss_fwd = steadystate(H, c_ops_fwd)
    L_emit = np.sqrt(Gamma_R_main * (m_R + 1)) * destroy(n_max)**2
    L_absorb = np.sqrt(Gamma_R_main * m_R) * create(n_max)**2
    J_num_fwd = expect(H, get_dissipator(L_emit, rho_ss_fwd) + get_dissipator(L_absorb, rho_ss_fwd))
    J_anal_fwd = (4 * omega * Gamma_L * Gamma_R_main / (Gamma_L + Gamma_R_main)**2) * (
        (Gamma_L * (4 * m_L + 1) + Gamma_R_main * (4 * m_R + 1)) * (m_R - m_L)
    )

    J_anal_forward.append(np.real(J_anal_fwd))
    J_num_forward.append(np.real(J_num_fwd))

    # Reverse
    m_L_rev = 1 / (np.exp(2 * omega / (kB * T)) - 1)
    m_R_rev = 1 / (np.exp(2 * omega / (kB * T_fixed)) - 1)

    c_ops_rev = [
        np.sqrt(Gamma_L * (m_L_rev + 1)) * destroy(n_max)**2,
        np.sqrt(Gamma_L * m_L_rev) * create(n_max)**2,
        np.sqrt(Gamma_R_main * (m_R_rev + 1)) * destroy(n_max)**2,
        np.sqrt(Gamma_R_main * m_R_rev) * create(n_max)**2
    ]
    rho_ss_rev = steadystate(H, c_ops_rev)
    L_emit_rev = np.sqrt(Gamma_R_main * (m_R_rev + 1)) * destroy(n_max)**2
    L_absorb_rev = np.sqrt(Gamma_R_main * m_R_rev) * create(n_max)**2
    J_num_rev = expect(H, get_dissipator(L_emit_rev, rho_ss_rev) + get_dissipator(L_absorb_rev, rho_ss_rev))
    J_anal_rev = (4 * omega * Gamma_L * Gamma_R_main / (Gamma_L + Gamma_R_main)**2) * (
        (Gamma_L * (4 * m_L_rev + 1) + Gamma_R_main * (4 * m_R_rev + 1)) * (m_R_rev - m_L_rev)
    )

    J_anal_reverse.append(np.real(J_anal_rev))
    J_num_reverse.append(np.real(J_num_rev))

# === Rectification ===
R_dense_anal_main, R_dense_num = [], []
R_dense_anal_alt = []

for T in T_dense:
    try:
        H = omega * num(n_max)

        def mB(omega, T):
            return 1 / (np.exp(2 * omega / (kB * T)) - 1)

        m_L = mB(omega, T_fixed)
        m_R = mB(omega, T)
        m_L_rev = mB(omega, T)
        m_R_rev = mB(omega, T_fixed)

        # Numerical
        c_ops_fwd = [
            np.sqrt(Gamma_L * (m_L + 1)) * destroy(n_max)**2,
            np.sqrt(Gamma_L * m_L) * create(n_max)**2,
            np.sqrt(Gamma_R_main * (m_R + 1)) * destroy(n_max)**2,
            np.sqrt(Gamma_R_main * m_R) * create(n_max)**2
        ]
        rho_fwd = steadystate(H, c_ops_fwd)
        L_emit = np.sqrt(Gamma_R_main * (m_R + 1)) * destroy(n_max)**2
        L_absorb = np.sqrt(Gamma_R_main * m_R) * create(n_max)**2
        J_f = np.real(expect(H, get_dissipator(L_emit, rho_fwd) + get_dissipator(L_absorb, rho_fwd)))

        c_ops_rev = [
            np.sqrt(Gamma_L * (m_L_rev + 1)) * destroy(n_max)**2,
            np.sqrt(Gamma_L * m_L_rev) * create(n_max)**2,
            np.sqrt(Gamma_R_main * (m_R_rev + 1)) * destroy(n_max)**2,
            np.sqrt(Gamma_R_main * m_R_rev) * create(n_max)**2
        ]
        rho_rev = steadystate(H, c_ops_rev)
        L_emit_rev = np.sqrt(Gamma_R_main * (m_R_rev + 1)) * destroy(n_max)**2
        L_absorb_rev = np.sqrt(Gamma_R_main * m_R_rev) * create(n_max)**2
        J_r = np.real(expect(H, get_dissipator(L_emit_rev, rho_rev) + get_dissipator(L_absorb_rev, rho_rev)))

        R_num = np.abs(J_f + J_r) / max(np.abs(J_f), np.abs(J_r))
        R_dense_num.append(R_num)

        # Analytical (main)
        J_f_anal = (4 * omega * Gamma_L * Gamma_R_main / (Gamma_L + Gamma_R_main)**2) * (
            (Gamma_L * (4 * m_L + 1) + Gamma_R_main * (4 * m_R + 1)) * (m_R - m_L)
        )
        J_r_anal = (4 * omega * Gamma_L * Gamma_R_main / (Gamma_L + Gamma_R_main)**2) * (
            (Gamma_L * (4 * m_L_rev + 1) + Gamma_R_main * (4 * m_R_rev + 1)) * (m_R_rev - m_L_rev)
        )
        R_anal = np.abs(J_f_anal + J_r_anal) / max(np.abs(J_f_anal), np.abs(J_r_anal))
        R_dense_anal_main.append(R_anal)

        # Analytical (alt Gamma_R)
        J_f_alt = (4 * omega * Gamma_L * Gamma_R_alt / (Gamma_L + Gamma_R_alt)**2) * (
            (Gamma_L * (4 * m_L + 1) + Gamma_R_alt * (4 * m_R + 1)) * (m_R - m_L)
        )
        J_r_alt = (4 * omega * Gamma_L * Gamma_R_alt / (Gamma_L + Gamma_R_alt)**2) * (
            (Gamma_L * (4 * m_L_rev + 1) + Gamma_R_alt * (4 * m_R_rev + 1)) * (m_R_rev - m_L_rev)
        )
        R_alt = np.abs(J_f_alt + J_r_alt) / max(np.abs(J_f_alt), np.abs(J_r_alt))
        R_dense_anal_alt.append(R_alt)

    except Exception as e:
        print(f"Error at T = {T:.3f}: {e}")
        R_dense_anal_main.append(np.nan)
        R_dense_num.append(np.nan)
        R_dense_anal_alt.append(np.nan)

# === Plotting ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

# --- Heat current plot (top) ---
ax1.plot(T_values, J_anal_forward, '--', color='black', lw=2.5, label="Analytical forward (solid red)")
ax1.plot(T_values, J_num_forward, 'o', color='red', label="Numerical forward (red circles)", markersize=4)

ax1.plot(T_values, J_anal_reverse, '-', color='green', lw=2.5, label="Analytical reverse (solid green)")
ax1.plot(T_values, J_num_reverse, 'o', color='green', label="Numerical reverse (green circles)", markersize=4)

ax1.set_ylabel(r'$\mathcal{J}_{R}$', fontsize=22)
ax1.legend(fontsize=13)
ax1.grid(True)

# --- Rectification plot (bottom) ---
ax2.plot(T_dense, R_dense_anal_main, '-', color='blue', lw=2.5, label=r'$\mathcal{R}_{\mathrm{anal}}$, $\Gamma_R = 0.001$')
ax2.plot(T_dense, R_dense_num, '--', color='orange', lw=2, label=r'$\mathcal{R}_{\mathrm{num}}$, $\Gamma_R = 0.001$')
ax2.plot(T_dense, R_dense_anal_alt, '-.', color='black', lw=2.5, label=r'$\mathcal{R}_{\mathrm{anal}}$, $\Gamma_R = 0.01$')

ax2.set_xlabel(r'$T$', fontsize=22)
ax2.set_ylabel(r'$\mathcal{R}$', fontsize=22)
ax2.legend(fontsize=14)
ax2.grid(True)

# Beautify
for ax in [ax1, ax2]:
    ax.minorticks_on()
    ax.tick_params('both', top=True, right=True, direction='in',
                   length=8, width=2.8, which='major', labelsize=18)
    ax.tick_params('both', top=True, right=True, direction='in',
                   length=6, width=1.4, which='minor', labelsize=18)
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_linewidth(2.0)
    ax.yaxis.get_offset_text().set_fontsize(14)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)

plt.tight_layout()
#plt.savefig("fig2.eps", dpi=600, bbox_inches='tight', transparent=True)
plt.show()
