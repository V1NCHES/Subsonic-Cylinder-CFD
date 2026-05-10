# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
Experiment 3 - Comparison of Newton FIM vs SLOR
Fixed Mach (0.1, 0.2, 0.3, 0.4, 0.5), fixed grid (80x60).

Plots:
  - M_local and Cp fields side-by-side for both methods
  - Convergence curves
  - Surface Cp and M_local profiles (both methods overlaid)
  - Summary table: iterations, final error

Output: C:\Program1\9 sem\task 3\new\comparison\
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---- paths ---------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLVER_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "result")
OUT_DIR    = os.path.join(SCRIPT_DIR, "comparison")
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, SOLVER_DIR)
from solvers import SubsonicCylinderSolver

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
})

# ---- field computation ---------------------------------------------------
def compute_fields(solver):
    Phi, rho  = solver.Phi, solver.rho
    Xi, Eta   = solver.Xi, solver.Eta
    dxi, deta = solver.dxi, solver.deta
    Minf, g   = solver.Minf, solver.gamma

    dPhi_dxi  = np.zeros_like(Phi)
    dPhi_deta = np.zeros_like(Phi)
    dPhi_dxi [1:-1, :]  = (Phi[2:,  :] - Phi[:-2,  :]) / (2*dxi)
    dPhi_deta[:, 1:-1]  = (Phi[:,  2:] - Phi[:,  :-2]) / (2*deta)
    dPhi_dxi [0,  :]    = (Phi[1, :] - Phi[0, :]) / dxi
    dPhi_dxi [-1, :]    = (3*Phi[-1,:]-4*Phi[-2,:]+Phi[-3,:]) / (2*dxi)
    dPhi_deta[:,  0]    = (Phi[:, 1] - Phi[:, 0]) / deta
    dPhi_deta[:, -1]    = (Phi[:,-2] - Phi[:,-1]) / (-deta)

    u     = np.exp(-Xi)*(np.cos(Eta)*dPhi_dxi - np.sin(Eta)*dPhi_deta)
    v     = np.exp(-Xi)*(np.sin(Eta)*dPhi_dxi + np.cos(Eta)*dPhi_deta)
    V_mag = np.sqrt(u**2 + v**2)

    a2      = 1.0/Minf**2 + 0.5*(g-1)*(1.0 - V_mag**2)
    a2      = np.maximum(a2, 1e-6)
    M_local = V_mag / np.sqrt(a2)
    Cp      = (2.0/(g*Minf**2)) * (rho**g - 1.0)

    X = np.exp(Xi) * np.cos(Eta)
    Y = np.exp(Xi) * np.sin(Eta)

    def mirror(arr):
        return np.concatenate((arr[:, ::-1][:, :-1], arr), axis=1)

    X_full  = mirror(X)
    Y_full  = np.concatenate((-Y[:, ::-1][:, :-1], Y), axis=1)
    Ml_full = mirror(M_local)
    Cp_full = mirror(Cp)
    return X_full, Y_full, Ml_full, Cp_full


def surface_cp(sol):
    g     = sol.gamma
    rho_s = sol.rho[0, :]
    return (2.0/(g*sol.Minf**2)) * (rho_s**g - 1.0)


def surface_mach(sol):
    Phi   = sol.Phi
    Eta0  = sol.eta
    g     = sol.gamma
    Minf  = sol.Minf
    dxi   = sol.dxi
    deta  = sol.deta

    dPdxi  = (Phi[1, :] - Phi[0, :]) / dxi
    dPdeta = np.zeros(sol.Neta + 1)
    dPdeta[1:-1] = (Phi[0, 2:] - Phi[0, :-2]) / (2*deta)
    dPdeta[0]    = (Phi[0, 1] - Phi[0, 0]) / deta
    dPdeta[-1]   = (Phi[0,-2] - Phi[0,-1]) / deta

    u = np.cos(Eta0)*dPdxi - np.sin(Eta0)*dPdeta
    v = np.sin(Eta0)*dPdxi + np.cos(Eta0)*dPdeta
    V = np.sqrt(u**2 + v**2)
    a2 = np.maximum(1.0/Minf**2 + 0.5*(g-1)*(1 - V**2), 1e-6)
    return V / np.sqrt(a2)


# ---- plot: field comparison (2×2) per Mach --------------------------------
def plot_field_comparison(r_pnm, r_slor, Minf):
    theta = np.linspace(0, 2*np.pi, 400)
    cx, cy = np.cos(theta), np.sin(theta)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        f"Newton FIM vs SLOR: flow fields, Minf={Minf:.1f}",
        fontsize=13, fontweight='bold'
    )

    row_data = [
        (r_pnm['Ml'], r_slor['Ml'],  r"$M_{\rm local}$", 'inferno',  False),
        (r_pnm['Cp'], r_slor['Cp'],  r"$C_p$",           'RdBu_r',   True),
    ]
    col_titles = ["Newton FIM", "SLOR"]

    for row, (D_n, D_s, rlabel, cmap, symmetric) in enumerate(row_data):
        if symmetric:
            vext = max(abs(D_n).max(), abs(D_s).max())
            vmin, vmax = -vext, vext
        else:
            vmin = min(D_n.min(), D_s.min())
            vmax = max(D_n.max(), D_s.max())
            vmin = max(vmin, 0)

        for col, (D, r) in enumerate([(D_n, r_pnm), (D_s, r_slor)]):
            ax = axes[row, col]
            X, Y = r['X'], r['Y']
            cf = ax.contourf(X, Y, D, levels=60, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.contour(X, Y, D, levels=16, colors='white' if row==0 else 'k',
                       linewidths=0.4, alpha=0.45)
            if row == 0:
                ax.fill(cx, cy, color='lightgray', zorder=5)
            else:
                ax.fill(cx, cy, color='white', zorder=5)
            ax.plot(cx, cy, 'k-', lw=1.5, zorder=6)
            ax.set_xlim(-3.5, 3.5); ax.set_ylim(-2.8, 2.8)
            ax.set_aspect('equal')
            ax.set_xlabel("x")
            ax.set_ylabel(f"{rlabel}\ny" if col == 0 else "y")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11)
            ax.grid(True, alpha=0.12)
            fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.02, label=rlabel)

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"field_comparison_M{Minf:.1f}.png")
    plt.savefig(fname, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# ---- plot: surface profiles comparison -----------------------------------
def plot_surface_comparison(r_pnm, r_slor, Minf):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Newton FIM vs SLOR: cylinder surface profiles, Minf={Minf:.1f}",
        fontsize=12, fontweight='bold'
    )
    eta_d = r_pnm['sol'].eta * 180 / np.pi

    axes[0].plot(eta_d, r_pnm['msurf'],  'b-',  lw=2.5, label="Newton FIM")
    axes[0].plot(eta_d, r_slor['msurf'], 'r--', lw=2.5, label="SLOR")
    axes[0].set_xlabel(r"$\eta$ (deg)")
    axes[0].set_ylabel(r"$M_{\rm local}$")
    axes[0].set_title(r"Mach M_local on surface")
    axes[0].legend(); axes[0].grid(True, alpha=0.3); axes[0].set_xlim(0, 180)

    axes[1].plot(eta_d, r_pnm['cpsurf'],  'b-',  lw=2.5, label="Newton FIM")
    axes[1].plot(eta_d, r_slor['cpsurf'], 'r--', lw=2.5, label="SLOR")
    axes[1].invert_yaxis()
    axes[1].set_xlabel(r"$\eta$ (deg)")
    axes[1].set_ylabel(r"$C_p$")
    axes[1].set_title(r"Pressure coefficient Cp")
    axes[1].legend(); axes[1].grid(True, alpha=0.3); axes[1].set_xlim(0, 180)

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"surface_comparison_M{Minf:.1f}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# ---- plot: convergence ---------------------------------------------------
def plot_convergence(r_pnm, r_slor, Minf):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(range(1, len(r_pnm['hist'])+1),  r_pnm['hist'],
                'b-o', lw=2, ms=6, label="Newton FIM")
    ax.semilogy(range(1, len(r_slor['hist'])+1), r_slor['hist'],
                'r-s', lw=2, ms=4, alpha=0.8, label="SLOR")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\max|\delta\Phi|$")
    ax.set_title(f"Newton FIM vs SLOR convergence, Minf={Minf:.1f}")
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    fname = os.path.join(OUT_DIR, f"convergence_M{Minf:.1f}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# ---- summary figure: convergence for all Machs ----------------------------
def plot_convergence_summary(all_pnm, all_slor, machs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Newton FIM and SLOR convergence for various Mach numbers", fontsize=13, fontweight='bold')
    cmap = plt.cm.plasma
    n = len(machs)

    for i, Minf in enumerate(machs):
        color = cmap(i / (n-1)) if n > 1 else cmap(0.5)
        h_n = all_pnm[Minf]['hist']
        h_s = all_slor[Minf]['hist']
        axes[0].semilogy(range(1, len(h_n)+1), h_n, '-o', color=color,
                         ms=5, lw=2, label=fr"$M_\infty={Minf:.1f}$")
        axes[1].semilogy(range(1, len(h_s)+1), h_s, '-', color=color,
                         lw=1.5, alpha=0.85, label=fr"$M_\infty={Minf:.1f}$")

    for ax, title in zip(axes, ["Newton FIM", "SLOR"]):
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\max|\delta\Phi|$")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, "convergence_all_methods.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# ---- summary table figure ------------------------------------------------
def plot_summary_table(summary_rows):
    """
    Table: Mach | PNM iters | SLOR iters | PNM final err | SLOR final err
    | max(Cp) diff | max(M) diff
    """
    col_labels = [
        r"$M_\infty$",
        "ПНМ итер.", "SLOR итер.",
        r"ПНМ $\|\delta\Phi\|_\infty$ послед.",
        r"SLOR $\|\delta\Phi\|_\infty$ послед.",
        r"Δ max $M_{\rm loc}$ (поверхн.)",
        r"Δ min $C_p$ (поверхн.)"
    ]

    cell_text = []
    for r in summary_rows:
        cell_text.append([
            f"{r['Minf']:.1f}",
            str(r['n_pnm']),
            str(r['n_slor']),
            f"{r['err_pnm']:.2e}",
            f"{r['err_slor']:.2e}",
            f"{r['dm']:.4f}",
            f"{r['dcp']:.4f}"
        ])

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis('off')
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    ax.set_title("Summary: Newton FIM vs SLOR", fontsize=13, pad=15, fontweight='bold')
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, "summary_table.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# ================================================================ main ====
if __name__ == "__main__":
    MACH_LIST  = [0.1, 0.2, 0.3, 0.4, 0.5]
    NXI, NETA  = 80, 60

    all_pnm  = {}
    all_slor = {}
    summary_rows = []

    for Minf in MACH_LIST:
        print(f"\n{'='*55}")
        print(f"  Comparison  Minf = {Minf}")
        print(f"{'='*55}")

        # --- Newton FIM ---
        sol_n = SubsonicCylinderSolver(Nxi=NXI, Neta=NETA, Minf=Minf)
        ok_n, hist_n = sol_n.solve_newton(max_iter=50, tol=1e-8)
        if not ok_n:
            print("  [WARNING] Newton did not converge")

        X_n, Y_n, Ml_n, Cp_n = compute_fields(sol_n)
        r_pnm = dict(
            sol=sol_n, hist=hist_n,
            X=X_n, Y=Y_n, Ml=Ml_n, Cp=Cp_n,
            msurf=surface_mach(sol_n), cpsurf=surface_cp(sol_n)
        )

        # --- SLOR ---
        sol_s = SubsonicCylinderSolver(Nxi=NXI, Neta=NETA, Minf=Minf)
        ok_s, hist_s = sol_s.solve_slor(max_iter=4000, tol=1e-6, omega=1.5)
        if not ok_s:
            print("  [WARNING] SLOR did not converge")

        X_s, Y_s, Ml_s, Cp_s = compute_fields(sol_s)
        r_slor = dict(
            sol=sol_s, hist=hist_s,
            X=X_s, Y=Y_s, Ml=Ml_s, Cp=Cp_s,
            msurf=surface_mach(sol_s), cpsurf=surface_cp(sol_s)
        )

        all_pnm[Minf]  = r_pnm
        all_slor[Minf] = r_slor

        # per-Mach plots
        plot_field_comparison(r_pnm, r_slor, Minf)
        plot_surface_comparison(r_pnm, r_slor, Minf)
        plot_convergence(r_pnm, r_slor, Minf)

        # stats for summary table
        dm  = abs(r_pnm['msurf'].max()  - r_slor['msurf'].max())
        dcp = abs(r_pnm['cpsurf'].min() - r_slor['cpsurf'].min())
        summary_rows.append(dict(
            Minf=Minf,
            n_pnm=len(hist_n),
            n_slor=len(hist_s),
            err_pnm=hist_n[-1],
            err_slor=hist_s[-1],
            dm=dm, dcp=dcp
        ))

    # combined plots
    plot_convergence_summary(all_pnm, all_slor, MACH_LIST)
    plot_summary_table(summary_rows)

    print(f"\nAll done. Results in: {OUT_DIR}")
