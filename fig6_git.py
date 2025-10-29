import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker  # for scientific tick formatter like in your reference
from qutip import *

# ============================================================
# Quantum thermal rectification via two-photon dissipation:
# Verify steady right-bath heat current for (1) even sector,
# (2) odd sector, and (3) mixed parity, by comparing:
#   - time-dependent numerical current J_R(t) from mesolve()
#   - closed-form analytic steady-state expressions
# ============================================================

# -----------------------------
# Global parameters (two-photon)
# -----------------------------
omega   = 1.0        # Harmonic oscillator frequency
Gamma_L = 0.1        # Two-photon rate of left bath
Gamma_R = 0.001      # Two-photon rate of right bath
kB      = 1.0        # Boltzmann constant (set to 1)
n_max   = 40         # Fock-space cutoff; increase if needed

# -----------------------------------------
# Temperatures (example "forward" bias case)
# -----------------------------------------
T_L = 2.0            # Left bath temperature (hotter)
T_R = 0.6            # Right bath temperature (colder)

# ----------------------------------------------
# Time grid long enough to approach steady state
# ----------------------------------------------
t_max = 20.0        # total evolution time
Nt    = 200         # number of time points
tlist = np.linspace(0.0, t_max, Nt)

# --------------------------
# Build oscillator operators
# --------------------------
a  = destroy(n_max)         # annihilation operator a
ad = a.dag()                # creation operator a^\dagger
H  = omega * ad * a         # system Hamiltonian H = ω a^\dagger a

def mbar(T):
    """
    Bose factor for TWO-PHOTON channels:
    n_B(2*omega, T) = 1 / (exp(2*omega / (k_B*T)) - 1)
    """
    return 1.0 / (np.exp(2*omega/(kB*T)) - 1.0)

# -------------------------------------------
# Bath Bose factors at frequency 2*omega
# -------------------------------------------
mL = mbar(T_L)
mR = mbar(T_R)

# -------------------------------------------------
# Two-photon Lindblad operators for each reservoir
#   L_emit  ~ sqrt(Gamma * (m + 1)) * a^2
#   L_absorb~ sqrt(Gamma * m)       * (a^\dagger)^2
# -------------------------------------------------
L_L_emit   = np.sqrt(Gamma_L*(mL+1.0)) * (a*a)
L_L_absorb = np.sqrt(Gamma_L*(mL     )) * (ad*ad)

L_R_emit   = np.sqrt(Gamma_R*(mR+1.0)) * (a*a)
L_R_absorb = np.sqrt(Gamma_R*(mR     )) * (ad*ad)

# ----------------------------------------
# Combined collapse-operator list for LME
# ----------------------------------------
c_ops = [L_L_emit, L_L_absorb, L_R_emit, L_R_absorb]

# -------------------------------------------------------
# Even/odd projectors (helpful diagnostics of sectoring)
# -------------------------------------------------------
Pe = sum([ket2dm(basis(n_max, 2*k))   for k in range(n_max//2)])     # even Fock subspace
Po = sum([ket2dm(basis(n_max, 2*k+1)) for k in range(n_max//2)])     # odd  Fock subspace

def dissipator(L, rho):
    """
    Lindblad dissipator acting on state rho: D[L] rho
    """
    return L * rho * L.dag() - 0.5*(L.dag()*L*rho + rho*L.dag()*L)

def JR_of_rho(rho):
    """
    Right-bath heat current for state rho:
      J_R(rho) = Tr[ H * ( D[L_R_emit](rho) + D[L_R_absorb](rho) ) ].
    """
    DR = dissipator(L_R_emit, rho) + dissipator(L_R_absorb, rho)
    return float(np.real((H * DR).tr()))

# -----------------------------------------
# Closed-form analytic steady-state currents
#   (derived for pure two-photon Liouvillian)
# -----------------------------------------
Sigma = Gamma_L + Gamma_R
A     = Gamma_L*mL + Gamma_R*mR

def JR_even_analytic():
    """
    Even sector:  J_R^even = (4 ω Γ_R / Σ^2) * (Σ mR - A) * (Σ + 4 A)
    """
    return (4*omega*Gamma_R/(Sigma**2)) * (Sigma*mR - A) * (Sigma + 4*A)

def JR_odd_analytic():
    """
    Odd sector:   J_R^odd  = (4 ω Γ_R / Σ^2) * (Σ mR - A) * (3 Σ + 4 A)
    """
    return (4*omega*Gamma_R/(Sigma**2)) * (Sigma*mR - A) * (3*Sigma + 4*A)

def JR_mixed_analytic(w_odd):
    """
    Mixed parity with weights w_e = 1 - w_odd, w_o = w_odd:
      J_R^mix = (1 - w_odd) * J_R^even + w_odd * J_R^odd
    """
    return (1.0 - w_odd) * JR_even_analytic() + w_odd * JR_odd_analytic()

# =========================
# (1) EVEN-SECTOR experiment
# =========================
# Initialize in |0><0| (pure even parity) so dynamics stay in even ladder.
rho0_even = ket2dm(basis(n_max, 0))

# Evolve master equation
res_even = mesolve(H, rho0_even, tlist, c_ops, e_ops=[])

# Compute time-dependent right-bath current J_R(t)
JR_t_even = np.array([JR_of_rho(r) for r in res_even.states])

# Sector weights at final time (should be ~1 for even, 0 for odd)
rho_ss_even = res_even.states[-1]
w_even_even = float(np.real((Pe * rho_ss_even).tr()))
w_odd_even  = float(np.real((Po * rho_ss_even).tr()))

# Analytic even-sector steady current
JR_even_an = JR_even_analytic()

print("=== Case 1: Even sector ===")
print(f"w_even ≈ {w_even_even:.6f}, w_odd ≈ {w_odd_even:.6f}")
print(f"J_R(t_final) numeric = {JR_t_even[-1]:.8e}")
print(f"J_R (even analytic)  = {JR_even_an:.8e}\n")

# =======================
# (2) ODD-SECTOR experiment
# =======================
# Initialize in |1><1| (pure odd parity) so dynamics stay in odd ladder.
rho0_odd = ket2dm(basis(n_max, 1))

# Evolve master equation
res_odd = mesolve(H, rho0_odd, tlist, c_ops, e_ops=[])

# Time-dependent J_R(t)
JR_t_odd = np.array([JR_of_rho(r) for r in res_odd.states])

# Sector weights at final time (should be ~1 for odd, 0 for even)
rho_ss_odd = res_odd.states[-1]
w_even_odd = float(np.real((Pe * rho_ss_odd).tr()))
w_odd_odd  = float(np.real((Po * rho_ss_odd).tr()))

# Analytic odd-sector steady current
JR_odd_an = JR_odd_analytic()

print("=== Case 2: Odd sector ===")
print(f"w_even ≈ {w_even_odd:.6f}, w_odd ≈ {w_odd_odd:.6f}")
print(f"J_R(t_final) numeric = {JR_t_odd[-1]:.8e}")
print(f"J_R (odd analytic)   = {JR_odd_an:.8e}\n")

# ==================================
# (3) MIXED-PARITY (even + odd) test
# ==================================
# Choose a target odd weight (steady weight equals initial weight because parity is conserved)
w_odd_target = 0.30
rho0_mix = (1.0 - w_odd_target) * ket2dm(basis(n_max, 0)) + w_odd_target * ket2dm(basis(n_max, 1))

# Evolve master equation
res_mix = mesolve(H, rho0_mix, tlist, c_ops, e_ops=[])

# Time-dependent J_R(t)
JR_t_mix = np.array([JR_of_rho(r) for r in res_mix.states])

# Confirm sector weights at final time (should equal initial weights)
rho_ss_mix = res_mix.states[-1]
w_even_mix = float(np.real((Pe * rho_ss_mix).tr()))
w_odd_mix  = float(np.real((Po * rho_ss_mix).tr()))

# Analytic mixed steady current
JR_mix_an = JR_mixed_analytic(w_odd_target)

print("=== Case 3: Mixed sector ===")
print(f"target w_odd = {w_odd_target:.6f}  -->  steady w_odd ≈ {w_odd_mix:.6f}, w_even ≈ {w_even_mix:.6f}")
print(f"J_R(t_final) numeric = {JR_t_mix[-1]:.8e}")
print(f"J_R (mixed analytic) = {JR_mix_an:.8e}\n")

# ======================================================
# Plotting (beautified using your reference styling)
#   - minor ticks on
#   - inward ticks (top/right also)
#   - thicker spines
#   - scientific y-axis formatter
#   - tight_layout + high-DPI EPS
# ======================================================

def beautify_axis(ax):
    """
    Apply your reference 'beautify' settings to an Axes object.
    """
    ax.minorticks_on()
    ax.tick_params('both', top=True, right=True, direction='in',
                   length=8, width=2.2, which='major', labelsize=16)
    ax.tick_params('both', top=True, right=True, direction='in',
                   length=5, width=1.2, which='minor')
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_linewidth(2.0)
    # scientific notation on y-axis
    ax.yaxis.get_offset_text().set_fontsize(14)
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(fmt)
    ax.grid(True, alpha=0.25)

# -----------------------
# Create 1x3 subplot grid
# -----------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)

# ---- Panel (1): Even ----
axes[0].plot(tlist, JR_t_even, '-', color='tab:blue', lw=2.5,
             label=r'$\mathcal{J}_R(t)$ (even init)')
axes[0].axhline(JR_even_an, ls='--', color='green', lw=2.5,
                label='Analytic (even)')
axes[0].set_xlabel('t', fontsize=22)
axes[0].set_ylabel(r'$\mathcal{J}_R(t)$', fontsize=22)
axes[0].legend(fontsize=12)
beautify_axis(axes[0])

# ---- Panel (2): Odd ----
axes[1].plot(tlist, JR_t_odd, '-', color='tab:orange', lw=2.5,
             label=r'$\mathcal{J}_R(t)$ (odd init)')
axes[1].axhline(JR_odd_an, ls='--', color='green', lw=2.5,
                label='Analytic (odd)')
axes[1].set_xlabel('t', fontsize=22)
axes[1].legend(fontsize=12)
beautify_axis(axes[1])

# ---- Panel (3): Mixed ----
axes[2].plot(tlist, JR_t_mix, '-', color='tab:purple', lw=2.5,
             label=rf'$\mathcal{{J}}_R(t)$ (mixed, $w_o={w_odd_target}$)')
axes[2].axhline(JR_mix_an, ls='--', color='green', lw=2.5,
                label='Analytic (mixed)')
axes[2].set_xlabel('t', fontsize=22)
axes[2].legend(fontsize=12)
beautify_axis(axes[2])

plt.tight_layout()

# Save high-quality EPS like your reference
plt.savefig("parity_JR.eps", dpi=600, bbox_inches='tight', transparent=True)
plt.show()
