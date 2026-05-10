import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ------------------------------------------------------------------ paths
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_DIR     = r"C:\Program1\9 sem\task 3\new"
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from solvers import SubsonicCylinderSolver

# ================================================================ helpers
def compute_fields(solver):
    """Return u, v, V_mag, M_local, Cp in physical (x,y) space (full plane)."""
    Phi, rho  = solver.Phi, solver.rho
    Xi, Eta   = solver.Xi, solver.Eta
    dxi, deta = solver.dxi, solver.deta
    Minf, g   = solver.Minf, solver.gamma

    # Gradients of Phi in (xi,eta)
    dPhi_dxi  = np.zeros_like(Phi)
    dPhi_deta = np.zeros_like(Phi)

    dPhi_dxi [1:-1, :]  = (Phi[2:,  :] - Phi[:-2,  :]) / (2*dxi)
    dPhi_deta[:, 1:-1]  = (Phi[:,  2:] - Phi[:,  :-2]) / (2*deta)

    # Boundary gradients consistent with symmetry BCs
    dPhi_dxi [0,  :]  = (Phi[1, :] - Phi[0, :]) / dxi          # cylinder (no-flux => 0, but for density)
    dPhi_dxi [-1, :]  = (3*Phi[-1,:]-4*Phi[-2,:]+Phi[-3,:])/(2*dxi)
    dPhi_deta[:,  0]  = (Phi[:, 1] - Phi[:, 0]) / deta
    dPhi_deta[:, -1]  = (Phi[:,-2] - Phi[:,-1]) / (-deta)

    # Physical velocity
    u = np.exp(-Xi)*(np.cos(Eta)*dPhi_dxi - np.sin(Eta)*dPhi_deta)
    v = np.exp(-Xi)*(np.sin(Eta)*dPhi_dxi + np.cos(Eta)*dPhi_deta)
    V_mag = np.sqrt(u**2 + v**2)

    # Local speed of sound and Mach
    a2 = 1.0/Minf**2 + 0.5*(g-1)*(1.0 - V_mag**2)
    a2 = np.maximum(a2, 1e-6)
    M_local = V_mag / np.sqrt(a2)

    # Pressure coefficient: Cp = 2*(p - p∞)/(rho∞ U∞^2)
    # For isentropic: p/p∞ = rho^gamma  (since rho is normalised: rho∞=1)
    # Cp = 2/gamma/Minf^2 * (rho^gamma - 1) alternatively via Bernoulli:
    # Cp = 1 - V^2   (incompressible limit, exact for Minf->0)
    # Full compressible: Cp = (2/(gamma*Minf^2)) * (rho^gamma - 1)
    Cp = (2.0/(g*Minf**2)) * (rho**g - 1.0)

    # Build full (x,y) plane by mirroring eta: 0..pi -> 0..2pi
    def mirror(arr):
        top = arr
        bot = arr[:, ::-1]
        # avoid duplicating eta=0 row
        return np.concatenate((bot[:, :-1], top), axis=1)

    X   = np.exp(Xi) * np.cos(Eta)
    Y   = np.exp(Xi) * np.sin(Eta)

    # Full plane
    X_full   = mirror(X)
    Y_full   = np.concatenate((-Y[:, ::-1][:, :-1], Y), axis=1)
    Ml_full  = mirror(M_local)
    Cp_full  = mirror(Cp)

    # Stream function (for streamlines)
    rho_u = rho * dPhi_dxi
    Psi   = np.zeros_like(Phi)
    for j in range(1, solver.Neta + 1):
        Psi[:, j] = Psi[:, j-1] + 0.5*(rho_u[:, j]+rho_u[:, j-1])*deta
    Psi_full = np.concatenate((-Psi[:, ::-1][:, :-1], Psi), axis=1)

    return X_full, Y_full, Ml_full, Cp_full, Psi_full


def plot_comparison(sol_pnm, sol_slor, Minf):
    """Two-row figure: top row=M_local, bottom row=Cp.  Columns: ПНМ | SLOR"""
    X_n,  Y_n,  Ml_n,  Cp_n,  Psi_n  = compute_fields(sol_pnm)
    X_s,  Y_s,  Ml_s,  Cp_s,  Psi_s  = compute_fields(sol_slor)

    theta_cyl = np.linspace(0, 2*np.pi, 300)
    cx, cy    = np.cos(theta_cyl), np.sin(theta_cyl)

    xlim = (-3.5, 3.5)
    ylim = (-2.8,  2.8)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(fr"$M_\infty = {Minf}$: ПНМ (Newton) vs SLOR (строчная релаксация)", fontsize=13)

    titles_col = ["ПНМ (Newton)", "SLOR (строчная релаксация)"]
    data_rows  = [
        (Ml_n, Ml_s, X_n, Y_n, X_s, Y_s, Psi_n, Psi_s, r"Локальное число Маха $M_{local}$", 'magma'),
        (Cp_n, Cp_s, X_n, Y_n, X_s, Y_s, None,  None,  r"Коэффициент давления $C_p$",        'RdBu_r'),
    ]

    for row, (D1, D2, X1, Y1, X2, Y2, P1, P2, row_title, cmap) in enumerate(data_rows):
        vmin = min(D1.min(), D2.min())
        vmax = max(D1.max(), D2.max())
        for col, (D, X, Y, P, sol) in enumerate([(D1, X1, Y1, P1, sol_pnm),
                                                    (D2, X2, Y2, P2, sol_slor)]):
            ax = axes[row, col]
            cf = ax.contourf(X, Y, D, levels=40, cmap=cmap, vmin=vmin, vmax=vmax)

            if P is not None:
                levels_psi = np.linspace(P.min(), P.max(), 25)
                ax.contour(X, Y, P, levels=levels_psi, colors='k', linewidths=0.6, alpha=0.55)

            ax.fill(cx, cy, color='white', zorder=5)
            ax.plot(cx, cy, 'k-', lw=1.5, zorder=6)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect('equal')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(True, alpha=0.15)
            fig.colorbar(cf, ax=ax, pad=0.02, fraction=0.04)

            if row == 0:
                ax.set_title(titles_col[col], fontsize=11)

        # Row label on the left
        axes[row, 0].set_ylabel(f"{row_title}\ny", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = os.path.join(OUT_DIR, f"comparison_M{Minf}.png")
    plt.savefig(fname, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


def plot_convergence(hist_pnm, hist_slor, Minf):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(range(1, len(hist_pnm)+1),  hist_pnm,  'o-b', lw=2, ms=5, label="ПНМ (Newton)")
    ax.semilogy(range(1, len(hist_slor)+1), hist_slor, 's-r', lw=2, ms=4, label="SLOR")
    ax.set_xlabel("Номер итерации")
    ax.set_ylabel(r"$\max|\Delta\Phi|$")
    ax.set_title(fr"Сходимость методов, $M_\infty={Minf}$")
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fname = os.path.join(OUT_DIR, f"convergence_M{Minf}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


def plot_cp_surface(sol_pnm, sol_slor, Minf):
    """Cp distribution along the cylinder surface (xi=0)."""
    g = sol_pnm.gamma

    def surface_cp(sol):
        Phi = sol.Phi
        rho_s = sol.rho[0, :]
        dPhi_dxi_0  = (Phi[1, :] - Phi[0, :]) / sol.dxi
        dPhi_deta_0 = np.zeros(sol.Neta+1)
        dPhi_deta_0[1:-1] = (Phi[0, 2:] - Phi[0, :-2]) / (2*sol.deta)
        dPhi_deta_0[0]    = (Phi[0, 1] - Phi[0, 0]) / sol.deta
        dPhi_deta_0[-1]   = (Phi[0,-2] - Phi[0,-1]) / sol.deta
        # On cylinder e^{-xi}=1, so V^2 = (dPhi_dxi)^2 + (dPhi_deta)^2 (xi=0)
        V2 = dPhi_dxi_0**2 + dPhi_deta_0**2
        Cp_bern = 1.0 - V2  # incompressible-style; for compressible use rho
        Cp_comp = (2.0/(g*sol.Minf**2))*(rho_s**g - 1.0)
        return Cp_comp

    eta_arr = sol_pnm.eta * 180 / np.pi   # degrees

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eta_arr, surface_cp(sol_pnm),  'b-',  lw=2, label="ПНМ (Newton)")
    ax.plot(eta_arr, surface_cp(sol_slor), 'r--', lw=2, label="SLOR")
    ax.invert_yaxis()
    ax.set_xlabel(r"$\eta$ (°) вдоль поверхности цилиндра")
    ax.set_ylabel(r"$C_p$")
    ax.set_title(fr"Распределение давления на цилиндре, $M_\infty={Minf}$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fname = os.path.join(OUT_DIR, f"cp_surface_M{Minf}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")


# ================================================================ main
def run_case(Minf, Nxi=80, Neta=60):
    print(f"\n{'='*55}")
    print(f"  Minf = {Minf}")
    print(f"{'='*55}")

    # --- Newton FIM ---
    sol_n = SubsonicCylinderSolver(Nxi=Nxi, Neta=Neta, Minf=Minf)
    ok_n, hist_n = sol_n.solve_newton(max_iter=40, tol=1e-8)
    if not ok_n:
        print("[WARNING] Newton did not fully converge.")

    # --- SLOR ---
    sol_s = SubsonicCylinderSolver(Nxi=Nxi, Neta=Neta, Minf=Minf)
    ok_s, hist_s = sol_s.solve_slor(max_iter=3000, tol=1e-6, omega=1.5)
    if not ok_s:
        print("[WARNING] SLOR did not fully converge.")

    return sol_n, sol_s, hist_n, hist_s


if __name__ == "__main__":
    MACH_LIST = [0.1, 0.3, 0.4]

    for Minf in MACH_LIST:
        sol_n, sol_s, hist_n, hist_s = run_case(Minf, Nxi=80, Neta=60)
        plot_comparison(sol_n, sol_s, Minf)
        plot_convergence(hist_n, hist_s, Minf)
        plot_cp_surface (sol_n, sol_s, Minf)

    print("\nAll done. Results in:", OUT_DIR)
