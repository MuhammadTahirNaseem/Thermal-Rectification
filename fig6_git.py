# Transient heat currents: J_R(t) analytic vs numerical
# Cases: (i) Linear one-photon, (ii) Two-photon low-T, (iii) Two-photon high-T
# Requirements: pip install qutip numpy matplotlib

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import qutip as qt

# ----------------------------- progress tracker -----------------------------
class Progress:
    def __init__(self, total_tasks: int):
        self.total = total_tasks
        self.done = 0
        self.t0 = time.time()
    def mark(self, label: str):
        self.done += 1
        now = time.time()
        elapsed = now - self.t0
        avg = elapsed / self.done
        remaining = max(0.0, (self.total - self.done) * avg)
        print(f"[{label}] {self.done}/{self.total} | elapsed {elapsed:6.2f}s | est. left {remaining:6.2f}s")

# ----------------------------- styling helper (reference look) -----------------------------
def beautify_axis(ax):
    ax.minorticks_on()
    ax.tick_params('both', top=True, right=True, direction='in',
                   length=8, width=2.2, which='major', labelsize=16)
    ax.tick_params('both', top=True, right=True, direction='in',
                   length=5, width=1.2, which='minor')
    for side in ['top', 'bottom', 'left', 'right']:
        ax.spines[side].set_linewidth(2.0)
    ax.yaxis.get_offset_text().set_fontsize(14)
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(fmt)
    ax.grid(True, alpha=0.25)

# =============================== (i) Linear one-photon ===============================
def simulate_linear(nbar_L, nbar_R, gamma_L, gamma_R, N=60, w=1.0, t_max=25.0, Nt=800):
    gamma = gamma_L + gamma_R
    # initial vacuum
    rho0 = qt.fock_dm(N, 0)
    a = qt.destroy(N)
    H = w * a.dag() * a
    c_ops = [
        np.sqrt(gamma_L * (nbar_L + 1.0)) * a,
        np.sqrt(gamma_L * nbar_L) * a.dag(),
        np.sqrt(gamma_R * (nbar_R + 1.0)) * a,
        np.sqrt(gamma_R * nbar_R) * a.dag(),
    ]
    tlist = np.linspace(0.0, t_max, Nt)
    # analytic
    n0 = 0.0
    n_ss = (gamma_L * nbar_L + gamma_R * nbar_R) / (gamma_L + gamma_R)
    n_an = n_ss + (n0 - n_ss) * np.exp(-gamma * tlist)
    JR_an = w * gamma_R * (nbar_R - n_an)
    # numerical
    res = qt.mesolve(H, rho0, tlist, c_ops=c_ops, e_ops=[a.dag() * a], progress_bar=None)
    n_num = np.array(res.expect[0])
    JR_num = w * gamma_R * (nbar_R - n_num)
    return tlist, JR_an, JR_num

# ========================= (ii) Two-photon, low-T (two-level analytics) =========================
def simulate_lowT(mL, mR, Gamma_L, Gamma_R, N=3, omega=1.0, t_max=30.0, Nt=201):
    a = qt.destroy(N); adag = a.dag()
    H = omega * (adag * a)
    # two-photon collapses
    c_ops = [
        np.sqrt(Gamma_L * (mL + 1.0)) * (a * a),
        np.sqrt(Gamma_L *  mL       ) * (adag * adag),
        np.sqrt(Gamma_R * (mR + 1.0)) * (a * a),
        np.sqrt(Gamma_R *  mR       ) * (adag * adag),
    ]
    rho0 = qt.fock_dm(N, 0)  # |0><0|
    tlist = np.linspace(0.0, t_max, Nt)
    # analytics in {|0>,|2>}
    P2_0 = 0.0
    k_up_L   = 2.0 * Gamma_L * mL
    k_down_L = 2.0 * Gamma_L * (mL + 1.0)
    k_up_R   = 2.0 * Gamma_R * mR
    k_down_R = 2.0 * Gamma_R * (mR + 1.0)
    kappa_L  = k_up_L + k_down_L
    kappa_R  = k_up_R + k_down_R
    kappa    = kappa_L + kappa_R
    k_up_tot = k_up_L + k_up_R
    P2_ss    = k_up_tot / kappa
    P2_an = P2_ss + (P2_0 - P2_ss) * np.exp(-kappa * tlist)
    JR_an = 2.0 * omega * (kappa_R * P2_an - k_up_R)
    # numerical JR_R(t)
    res = qt.mesolve(H, rho0, tlist, c_ops, [], progress_bar=None)
    A2dagA2 = (adag*adag) * (a*a)     # N(N-1)
    A2A2dag = (a*a) * (adag*adag)     # (N+1)(N+2)
    exp_A2dagA2 = np.array([qt.expect(A2dagA2, rho) for rho in res.states])
    exp_A2A2dag = np.array([qt.expect(A2A2dag, rho) for rho in res.states])
    JR_num = 2.0 * omega * (Gamma_R * (mR + 1.0) * exp_A2dagA2 - Gamma_R * mR * exp_A2A2dag)
    return tlist, JR_an, JR_num

# ========================= (iii) Two-photon, high-T (closure analytics) =========================
def simulate_highT(mL, mR, Gamma_L, Gamma_R, N, omega=1.0, t_max=30.0, Nt=120):
    # closure symbols
    Sigma_Gamma = Gamma_L + Gamma_R
    A           = Gamma_L * mL + Gamma_R * mR
    n_ss        = 2.0 * A / Sigma_Gamma
    lam         = 8.0 * A + 2.0 * Sigma_Gamma
    n0 = 0.0
    K0 = (n0 - n_ss) / (n0 + 0.5)
    tlist = np.linspace(0.0, t_max, Nt)
    def n_an(t):
        x = np.exp(-lam * t)
        return (n_ss + 0.5 * K0 * x) / (1.0 - K0 * x)
    def JR_from_n(n):
        # corrected polynomial
        return 2.0 * omega * Gamma_R * (2.0*n**2 + (1.0 - 4.0*mR)*n - 2.0*mR)
    JR_an = JR_from_n(n_an(tlist))
    # numerical
    a = qt.destroy(N); Nop = a.dag() * a
    a2 = a*a; adag2 = a.dag()*a.dag()
    H = omega * Nop
    c_ops = [
        np.sqrt(Gamma_L * (mL + 1.0)) * a2,
        np.sqrt(Gamma_L *  mL       ) * adag2,
        np.sqrt(Gamma_R * (mR + 1.0)) * a2,
        np.sqrt(Gamma_R *  mR       ) * adag2,
    ]
    res = qt.mesolve(H, qt.basis(N, 0), tlist, c_ops=c_ops,
                     e_ops=[adag2*a2, a2*adag2], progress_bar=None, options=qt.Options(nsteps=50000))
    exp_adag2_a2 = np.array(res.expect[0])  # ⟨N(N-1)⟩
    exp_a2_adag2 = np.array(res.expect[1])  # ⟨(N+1)(N+2)⟩
    JR_num = 2.0 * omega * Gamma_R * ((mR+1.0)*exp_adag2_a2 - mR*exp_a2_adag2)
    return tlist, JR_an, JR_num

# ================================== run ==================================
if __name__ == "__main__":
    # Fixed parameters (exactly as before)
    # Linear
    w        = 1.0
    N_lin    = 60
    gamma_L  = 0.30
    gamma_R  = 0.10
    def nbar(omega, T): return 1.0/(np.exp(omega/T)-1.0)
    T_L      = 5.5
    T_R      = 1.44269504089  # 1/ln(2)
    nbar_L   = nbar(w, T_L)
    nbar_R   = nbar(w, T_R)

    # Low-T two-photon
    omega_low   = 1.0
    Gamma_L_low = 0.08
    Gamma_R_low = 0.08
    mL_low      = 0.05
    mR_low      = 0.002
    N_low       = 3

    # High-T two-photon
    omega_hi   = 1.0
    Gamma_L_hi = 0.01
    Gamma_R_hi = 0.01
    mL_hi      = 10.0
    mR_hi      = 9.0
    N_hi       = 160

    prog = Progress(total_tasks=3)

    # Linear (forward only)
    t_lin, JR_an_lin, JR_num_lin = simulate_linear(
        nbar_L, nbar_R, gamma_L, gamma_R, N=N_lin, w=w, t_max=25.0, Nt=100
    )
    prog.mark("Linear")

    # Low-T (forward only)
    t_low, JR_an_low, JR_num_low = simulate_lowT(
        mL_low, mR_low, Gamma_L_low, Gamma_R_low, N=N_low, omega=omega_low, t_max=30.0, Nt=151
    )
    prog.mark("Low-T two-photon")

    # High-T (forward only)
    t_hi, JR_an_hi, JR_num_hi = simulate_highT(
        mL_hi, mR_hi, Gamma_L_hi, Gamma_R_hi, N=N_hi, omega=omega_hi, t_max=30.0, Nt=120
    )
    prog.mark("High-T two-photon")

    # ---------------- figure (three subplots; no titles) ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)

    # Linear
    axes[0].plot(t_lin, JR_an_lin,        '-', lw=2.2, label='analytic')
    axes[0].plot(t_lin, JR_num_lin,       ':', lw=2.2, label='numerical')
    axes[0].set_xlabel(r'$t$', fontsize=24)
    axes[0].set_ylabel(r'$\mathcal{J}_R(t)$', fontsize=22)
    axes[0].legend(fontsize=12)
    beautify_axis(axes[0])

    # Low-T
    axes[1].plot(t_low, JR_an_low,        '-', lw=2.2, label='analytic')
    axes[1].plot(t_low, JR_num_low,       ':', lw=2.2, label='numerical')
    axes[1].set_xlabel(r'$t$', fontsize=22)
    axes[1].legend(fontsize=12)
    beautify_axis(axes[1])

    # High-T
    axes[2].plot(t_hi, JR_an_hi,          '-', lw=2.2, label='analytic')
    axes[2].plot(t_hi, JR_num_hi,         ':', lw=2.2, label='numerical')
    axes[2].set_xlabel(r'$t$', fontsize=22)
    axes[2].legend(fontsize=12)
    beautify_axis(axes[2])

    plt.tight_layout()
    # Optional high-quality save:
    plt.savefig("JR_transients.eps", dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
