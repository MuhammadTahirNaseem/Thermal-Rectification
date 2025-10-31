# -*- coding: utf-8 -*-
"""
Compare steady-state oscillator fidelity between:
  (1) Full TLS+oscillator master equation in the dressed (polaron) picture
  (2) Reduced oscillator-only master equation (after adiabatic elimination)
as a function of bath temperature T for a flat spectral density.

Requires: qutip, numpy, matplotlib
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from matplotlib import ticker

# ----------------------------
# Model parameters (edit here)
# ----------------------------
w0    = 1.0          # TLS frequency ω0
wa    = 0.1          # oscillator frequency ωa
alpha = 0.01         # dressing parameter α = g/ωa  (α << 1)
kappa = 0.01         # flat spectral density strength (overall coupling)
N     = 25           # oscillator truncation (increase if needed)
T_min, T_max, nT = 0.03, 2, 200   # temperature sweep (units of energy; k_B = 1)

# ----------------------------
# Helpers: operators & spectra
# ----------------------------
def nbar(freq, T):
    """Bose-Einstein occupation n̄(ω) with k_B = 1. Guard against tiny/zero T."""
    freq = float(abs(freq))
    if T <= 0.0:
        return 0.0
    x = freq / T
    if x > 700.0:   # avoid overflow
        return 0.0
    return 1.0 / (np.exp(x) - 1.0)

def G_flat(delta, kappa, T):
    """
    One-sided transform G(Δ) for a *flat spectral density* at temperature T:
      G(Δ>0) = κ [ n̄(Δ) + 1 ]
      G(Δ<0) = κ [ n̄(|Δ|) ]
      G(0)   = 0
    """
    if delta > 0:
        return kappa * (nbar(delta, T) + 1.0)
    elif delta < 0:
        return kappa * nbar(-delta, T)
    else:
        return 0.0

# Tensor-product operators (TLS ⊗ osc)
sm = qt.tensor(qt.sigmam(), qt.qeye(N))
sp = qt.tensor(qt.sigmap(), qt.qeye(N))
sz = qt.tensor(qt.sigmaz(), qt.qeye(N))
I2 = qt.tensor(qt.qeye(2), qt.qeye(N))

a  = qt.tensor(qt.qeye(2), qt.destroy(N))
ad = a.dag()

def tilde_a(alpha):
    """Dressed oscillator operator: ã = a - (α/2) σ_z."""
    return a - (alpha/2.0) * sz

def tilde_n(alpha):
    """ã†ã."""
    ta = tilde_a(alpha)
    return ta.dag() * ta

# ---------------------------------------------------------
# Full joint (TLS+oscillator) collapse operators (no H term)
# ---------------------------------------------------------
def full_joint_collapse_ops(T, params):
    """
    Build collapse operators for the full dressed interaction-picture master equation:
        sum_Ω rate(Ω) D[J_Ω]
    with J_Ω given explicitly in terms of ã, σ_±, and (1 - α^2/2 - α^2 ã†ã).
    """
    w0, wa, alpha, kappa = params
    ta  = tilde_a(alpha)
    tad = ta.dag()
    tn  = tilde_n(alpha)

    c_ops = []

    # Carrier (ω0): D[ (1 - α^2/2 - α^2 ã†ã) σ_- ] and conjugate with σ_+
    J0_down = ((1.0 - 0.5*alpha**2) * I2 - (alpha**2) * tn) * sm
    J0_up   = ((1.0 - 0.5*alpha**2) * I2 - (alpha**2) * tn) * sp

    g0_down = G_flat(+w0, kappa, T)
    g0_up   = G_flat(-w0, kappa, T)

    if g0_down > 0: c_ops.append(np.sqrt(g0_down) * J0_down)
    if g0_up   > 0: c_ops.append(np.sqrt(g0_up)   * J0_up)

    # One-phonon sidebands (ω0 ± ωa): D[ σ_- ã ] etc., rates carry α^2
    Jp_down = sm * ta             # ω0 + ωa block
    Jp_up   = sp * tad
    gp_down = (alpha**2) * G_flat(w0 + wa, kappa, T)
    gp_up   = (alpha**2) * G_flat(-w0 - wa, kappa, T)
    if gp_down > 0: c_ops.append(np.sqrt(gp_down) * Jp_down)
    if gp_up   > 0: c_ops.append(np.sqrt(gp_up)   * Jp_up)

    Jm_down = sm * tad            # ω0 - ωa block
    Jm_up   = sp * ta
    gm_down = (alpha**2) * G_flat(w0 - wa, kappa, T)
    gm_up   = (alpha**2) * G_flat(-w0 + wa, kappa, T)
    if gm_down > 0: c_ops.append(np.sqrt(gm_down) * Jm_down)
    if gm_up   > 0: c_ops.append(np.sqrt(gm_up)   * Jm_up)

    # Two-phonon harmonics (ω0 ± 2ωa): D[ σ_- ã^2 ] etc., rates carry α^4/4
    J2p_down = sm * (ta**2)       # ω0 + 2ωa
    J2p_up   = sp * (tad**2)
    g2p_down = (alpha**4 / 4.0) * G_flat(w0 + 2.0*wa, kappa, T)
    g2p_up   = (alpha**4 / 4.0) * G_flat(-w0 - 2.0*wa, kappa, T)
    if g2p_down > 0: c_ops.append(np.sqrt(g2p_down) * J2p_down)
    if g2p_up   > 0: c_ops.append(np.sqrt(g2p_up)   * J2p_up)

    J2m_down = sm * (tad**2)      # ω0 - 2ωa
    J2m_up   = sp * (ta**2)
    g2m_down = (alpha**4 / 4.0) * G_flat(w0 - 2.0*wa, kappa, T)
    g2m_up   = (alpha**4 / 4.0) * G_flat(-w0 + 2.0*wa, kappa, T)
    if g2m_down > 0: c_ops.append(np.sqrt(g2m_down) * J2m_down)
    if g2m_up   > 0: c_ops.append(np.sqrt(g2m_up)   * J2m_up)

    return c_ops

# ---------------------------------------------------------
# Reduced oscillator-only collapse operators (no H term)
# ---------------------------------------------------------
def reduced_osc_collapse_ops(T, params):
    """
    Build collapse operators for the reduced oscillator master equation:
        Γ↓ D[a] + Γ↑ D[a†] + κ↓ D[a^2] + κ↑ D[(a†)^2] + γφ D[n]
    where the rates are expressed through G_flat and TLS steady quantities.

    TLS populations are set via the σ_z expectation:
        <σ_z> = (G(-ω0) - G(+ω0)) / (G(+ω0) + G(-ω0))
        p_e   = (1 + <σ_z>)/2
        p_g   = (1 - <σ_z>)/2
    """
    w0, wa, alpha, kappa = params

    # Oscillator-only operators
    aN  = qt.destroy(N)
    adN = aN.dag()
    nN  = adN * aN

    # Carrier rates
    Gp0 = G_flat(+w0, kappa, T)  # emission (down)
    Gm0 = G_flat(-w0, kappa, T)  # absorption (up)

    den = Gp0 + Gm0
    sz_avg = (Gm0 - Gp0) / den if den > 0 else 0.0  # = p_e - p_g

    # Populations via <σ_z>
    pe = 0.5 * (1.0 + sz_avg)  # excited
    pg = 0.5 * (1.0 - sz_avg)  # ground

    # One-phonon (∝ α^2)
    Gam_down = (alpha**2) * (pe * G_flat(w0 + wa, kappa, T) + pg * G_flat(-w0 + wa, kappa, T))
    Gam_up   = (alpha**2) * (pe * G_flat(w0 - wa, kappa, T) + pg * G_flat(-w0 - wa, kappa, T))

    # Two-phonon (∝ α^4/4)
    Kap_down = (alpha**4 / 4.0) * (pe * G_flat(w0 + 2.0*wa, kappa, T) + pg * G_flat(-w0 + 2.0*wa, kappa, T))
    Kap_up   = (alpha**4 / 4.0) * (pe * G_flat(w0 - 2.0*wa, kappa, T) + pg * G_flat(-w0 - 2.0*wa, kappa, T))

    # Number-dephasing (∝ α^4)
    Gam_phi  = (alpha**4) * (pe * G_flat(w0, kappa, T) + pg * G_flat(-w0, kappa, T))

    c_ops = []
    if Gam_down > 0: c_ops.append(np.sqrt(Gam_down) * aN)
    if Gam_up   > 0: c_ops.append(np.sqrt(Gam_up)   * adN)
    if Kap_down > 0: c_ops.append(np.sqrt(Kap_down) * (aN**2))
    if Kap_up   > 0: c_ops.append(np.sqrt(Kap_up)   * (adN**2))
    if Gam_phi  > 0: c_ops.append(np.sqrt(Gam_phi)  * nN)

    return c_ops

# ----------------------------
# Steady states and fidelity
# ----------------------------
def joint_and_reduced_steady_states(T, params):
    """
    Compute:
      - joint steady state ρ_ss for the full equation (H=0, c_ops from full)
      - oscillator reduced state ρ_a = Tr_TLS ρ_ss
      - reduced oscillator steady state ρ_red_ss
    """
    # Full joint
    c_full = full_joint_collapse_ops(T, params)
    H_full = 0 * sz  # zero Hamiltonian (interaction picture, Lamb shift neglected)
    rho_ss = qt.steadystate(H_full, c_full)  # density matrix on TLS⊗osc

    # Oscillator marginal
    rho_a = rho_ss.ptrace(1)  # trace out TLS (subsystem order: [TLS, osc])

    # Reduced oscillator-only
    c_red = reduced_osc_collapse_ops(T, params)
    H_red = 0 * qt.qeye(N)
    rho_red = qt.steadystate(H_red, c_red)

    return rho_a, rho_red, rho_ss

def oscillator_fidelity(rho_a, rho_red):
    """Uhlmann fidelity between two oscillator density matrices."""
    return qt.metrics.fidelity(rho_a, rho_red)

# ----------------------------
# Sweep temperature & compare
# ----------------------------
params = (w0, wa, alpha, kappa)
T_grid = np.linspace(T_min, T_max, nT)

fids = []
n_full = []
n_red  = []

aN = qt.destroy(N)

for T in T_grid:
    rho_a, rho_red, rho_joint = joint_and_reduced_steady_states(T, params)
    fids.append(oscillator_fidelity(rho_a, rho_red))
    n_full.append(qt.expect(aN.dag()*aN, rho_a))
    n_red.append(qt.expect(aN.dag()*aN, rho_red))

# ----------------------------
# Plot & simple printouts
# ----------------------------
infid = np.clip(1.0 - np.asarray(fids), 1e-16, None)  # avoid log(0)

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(T_grid, infid, lw=2.5)
ax.set_yscale('log')  # log scale
ax.set_xlabel(r'$T$', fontsize=22)
ax.set_ylabel(r"$1-\mathcal{F}(\rho_a^{\rm full},\rho_a^{\rm red})$", fontsize=22)
ax.grid(False, which='both')

ax.minorticks_on()
ax.tick_params('both', top=True, right=True, direction='in',
               length=8, width=2.2, which='major', labelsize=16)
ax.tick_params('both', top=True, right=True, direction='in',
               length=5, width=1.2, which='minor')

for side in ['top', 'bottom', 'left', 'right']:
    ax.spines[side].set_linewidth(2.0)

# Log tick formatting
ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=6))
ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0,
                                             subs=np.arange(2, 10) * 0.1,
                                             numticks=12))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())

# optional: tighten y-limits
ax.set_ylim(infid.min()*0.8, infid.max()*1.2)

plt.tight_layout()
plt.savefig("fig7.eps", dpi=600, bbox_inches='tight', transparent=True)
plt.show()

# Optional: print a small table
print("\n T    Fidelity   <n>_full   <n>_red")
for T, F, nf, nr in zip(T_grid, fids, n_full, n_red):
    print(f"{T:5.3f}  {F:8.5f}   {nf:8.4f}  {nr:8.4f}")
