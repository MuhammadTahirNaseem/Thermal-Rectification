import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from matplotlib import ticker

# Constants
omega0 = 1.0
kB = 1.0
n_max = 50
T_fixed = 2.0
T_values = np.linspace(0.01, 4.0, 400)

def n_th(T, omega=1.0):
    return 1.0 / (np.exp(omega / (kB * T)) - 1)

def n_th_2ph(T, omega=1.0):
    return 1.0 / (np.exp(2 * omega / (kB * T)) - 1)

def n_th_3ph(T, omega=1.0):
    return 1.0 / (np.exp(3 * omega / (kB * T)) - 1)

def compute_JR(TL, TR, gamma_dict):
    a = destroy(n_max)
    H = omega0 * a.dag() * a

    # Thermal occupations
    nL = n_th(TL)
    nR = n_th(TR)
    mL = n_th_2ph(TL)
    mR = n_th_2ph(TR)
    tL = n_th_3ph(TL)
    tR = n_th_3ph(TR)

    # Collapse operators
    c_ops = []

    # 1-photon
    if gamma_dict['L1'] > 0:
        c_ops += [np.sqrt(gamma_dict['L1'] * (nL + 1)) * a,
                  np.sqrt(gamma_dict['L1'] * nL) * a.dag()]
    if gamma_dict['R1'] > 0:
        c_ops += [np.sqrt(gamma_dict['R1'] * (nR + 1)) * a,
                  np.sqrt(gamma_dict['R1'] * nR) * a.dag()]

    # 2-photon
    if gamma_dict['L2'] > 0:
        c_ops += [np.sqrt(gamma_dict['L2'] * (mL + 1)) * a**2,
                  np.sqrt(gamma_dict['L2'] * mL) * a.dag()**2]
    if gamma_dict['R2'] > 0:
        c_ops += [np.sqrt(gamma_dict['R2'] * (mR + 1)) * a**2,
                  np.sqrt(gamma_dict['R2'] * mR) * a.dag()**2]

    # 3-photon
    if gamma_dict['L3'] > 0:
        c_ops += [np.sqrt(gamma_dict['L3'] * (tL + 1)) * a**3,
                  np.sqrt(gamma_dict['L3'] * tL) * a.dag()**3]
    if gamma_dict['R3'] > 0:
        c_ops += [np.sqrt(gamma_dict['R3'] * (tR + 1)) * a**3,
                  np.sqrt(gamma_dict['R3'] * tR) * a.dag()**3]

    rho_ss = steadystate(H, c_ops)

    # Right bath dissipators only
    dissipators = []
    if gamma_dict['R1'] > 0:
        dissipators += [
            lindblad_dissipator(np.sqrt(gamma_dict['R1'] * (nR + 1)) * a),
            lindblad_dissipator(np.sqrt(gamma_dict['R1'] * nR) * a.dag())
        ]
    if gamma_dict['R2'] > 0:
        dissipators += [
            lindblad_dissipator(np.sqrt(gamma_dict['R2'] * (mR + 1)) * a**2),
            lindblad_dissipator(np.sqrt(gamma_dict['R2'] * mR) * a.dag()**2)
        ]
    if gamma_dict['R3'] > 0:
        dissipators += [
            lindblad_dissipator(np.sqrt(gamma_dict['R3'] * (tR + 1)) * a**3),
            lindblad_dissipator(np.sqrt(gamma_dict['R3'] * tR) * a.dag()**3)
        ]

    L = liouvillian(H, dissipators)
    Lovl = vector_to_operator(L * operator_to_vector(rho_ss))
    return np.real((Lovl * H).tr())

def compute_rectification(Jf, Jr):
    return np.abs(Jf + Jr) / max(np.abs(Jf), np.abs(Jr)) if max(np.abs(Jf), np.abs(Jr)) > 0 else 0

# Case (i): 3-photon only
gamma_3ph = {'L1': 0.0, 'R1': 0.0, 'L2': 0.0, 'R2': 0.0, 'L3': 0.1, 'R3': 0.01}
JR_fwd_3 = [compute_JR(T_fixed, T, gamma_3ph) for T in T_values]
JR_rev_3 = [compute_JR(T, T_fixed, gamma_3ph) for T in T_values]
R_3 = [compute_rectification(Jf, Jr) for Jf, Jr in zip(JR_fwd_3, JR_rev_3)]

# Case (ii): 2-photon only
gamma_2ph = {'L1': 0.0, 'R1': 0.0, 'L2': 0.1, 'R2': 0.01, 'L3': 0.0, 'R3': 0.0}
JR_fwd_2 = [compute_JR(T_fixed, T, gamma_2ph) for T in T_values]
JR_rev_2 = [compute_JR(T, T_fixed, gamma_2ph) for T in T_values]
R_2 = [compute_rectification(Jf, Jr) for Jf, Jr in zip(JR_fwd_2, JR_rev_2)]

# Case (iii): All processes
gamma_all = {'L1': 0.1, 'R1': 0.01, 'L2': 0.1, 'R2': 0.01, 'L3': 0.1, 'R3': 0.01}
JR_fwd_all = [compute_JR(T_fixed, T, gamma_all) for T in T_values]
JR_rev_all = [compute_JR(T, T_fixed, gamma_all) for T in T_values]
R_all = [compute_rectification(Jf, Jr) for Jf, Jr in zip(JR_fwd_all, JR_rev_all)]

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(T_values, R_3, color='blue', linestyle='solid', lw=2.5)     
ax.plot(T_values, R_2, color='red', linestyle='dashed', lw=2.5)    
ax.plot(T_values, R_all, color='green', linestyle='dashdot', lw=2.5)   

ax.set_xlabel(r'$T$', fontsize=22)
ax.set_ylabel(r'$\mathcal{R}$', fontsize=24)
ax.grid(True)

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
plt.savefig("fig6.eps", dpi=600, bbox_inches='tight', transparent=True)
plt.show()
