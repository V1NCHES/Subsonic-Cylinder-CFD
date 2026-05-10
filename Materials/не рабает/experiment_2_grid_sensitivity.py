# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
Experiment 2 - Grid sensitivity study (Newton FIM, fixed Minf)
Test grids: 30x20, 60x40, 80x60, 120x80.
Plots: M_local and Cp fields for each grid, plus overlay comparison on cylinder surface.
Output: C:\Program1\9 sem\task 3\new\experiments_1\
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- paths ---------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLVER_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "result")
OUT_DIR    = os.path.join(SCRIPT_DIR, "experiments_1")
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, SOLVER_DIR)
from solvers import SubsonicCylinderSolver

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
})

# ---- compute fields (same as experiment 1) --------------------------------
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


# ---- surface Cp along cylinder (xi=0) ------------------------------------
def surface_cp(sol):
    g   = sol.gamma
    rho_s = sol.rho[0, :]
    return (2.0/(g*sol.Minf**2)) * (rho_s**g - 1.0)


# ---- surface M_local along cylinder (xi=0) --------------------------------
def surface_mach(sol):
    Phi   = sol.Phi
    Xi0   = sol.Xi[0, :]
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

    # xi=0 => exp(-xi)=1
    u = np.cos(Eta0)*dPdxi - np.sin(Eta0)*dPdeta
    v = np.sin(Eta0)*dPdxi + np.cos(Eta0)*dPdeta
    V = np.sqrt(u**2 + v**2)
    a2 = np.maximum(1.0/Minf**2 + 0.5*(g-1)*(1 - V**2), 1e-6)
    return V / np.sqrt(a2)


# ---- 2D field comparison across grids -----------------------------------
def plot_grid_fields(all_results, Minf):
    """
    Rows: M_local | Cp
    Columns: each grid
    """
    n = len(all_results)
    fig, axes = plt.subplots(2, n, figsize=(5*n, 10))
    fig.suptitle(
        f"Grid sensitivity: Minf={Minf:.1f} (Newton FIM)",
        fontsize=14, fontweight='bold'
    )

    theta = np.linspace(0, 2*np.pi, 400)
    cx, cy = np.cos(theta), np.sin(theta)

    # global color limits
    ml_max = max(r['Ml'].max() for r in all_results)
    cp_ext = max(abs(r['Cp']).max() for r in all_results)

    for col, r in enumerate(all_results):
        lbl = f"{r['Nxi']}×{r['Neta']}"
        X, Y, Ml, Cp = r['X'], r['Y'], r['Ml'], r['Cp']

        # M_local
        ax = axes[0, col]
        cf = ax.contourf(X, Y, Ml, levels=50, cmap='inferno', vmin=0, vmax=ml_max)
        ax.fill(cx, cy, color='lightgray', zorder=5)
        ax.plot(cx, cy, 'k-', lw=1, zorder=6)
        ax.set_xlim(-3, 3); ax.set_ylim(-2.5, 2.5); ax.set_aspect('equal')
        ax.set_title(lbl)
        ax.set_xlabel("x")
        if col == 0: ax.set_ylabel(r"$M_{\rm local}$" + "\ny")
        fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.02)

        # Cp
        ax = axes[1, col]
        cf2 = ax.contourf(X, Y, Cp, levels=50, cmap='RdBu_r', vmin=-cp_ext, vmax=cp_ext)
        ax.fill(cx, cy, color='white', zorder=5)
        ax.plot(cx, cy, 'k-', lw=1, zorder=6)
        ax.set_xlim(-3, 3); ax.set_ylim(-2.5, 2.5); ax.set_aspect('equal')
        ax.set_xlabel("x")
        if col == 0: ax.set_ylabel(r"$C_p$" + "\ny")
        fig.colorbar(cf2, ax=ax, fraction=0.04, pad=0.02)

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"grid_fields_M{Minf:.1f}.png")
    plt.savefig(fname, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# ---- surface profiles comparison ----------------------------------------
def plot_surface_profiles(all_results, Minf):
    """
    Two sub-plots: M_local and Cp along cylinder surface vs η, overlaid for each grid.
    """
    cmap = plt.cm.viridis
    n    = len(all_results)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Surface profiles vs grid, Minf={Minf:.1f}",
        fontsize=13, fontweight='bold'
    )

    for i, r in enumerate(all_results):
        lbl   = f"{r['Nxi']}×{r['Neta']}"
        color = cmap(i / (n - 1)) if n > 1 else cmap(0.5)
        eta_d = r['sol'].eta * 180 / np.pi

        axes[0].plot(eta_d, r['msurf'], color=color, lw=2, label=lbl)
        axes[1].plot(eta_d, r['cpsurf'], color=color, lw=2, label=lbl)

    axes[0].set_xlabel(r"$\eta$ (deg)")
    axes[0].set_ylabel(r"$M_{\rm local}$")
    axes[0].set_title(r"Mach M_local on surface")

    axes[1].set_xlabel(r"$\eta$ (deg)")
    axes[1].set_ylabel(r"$C_p$")
    axes[1].set_title(r"Pressure coefficient Cp on surface")

    axes[1].invert_yaxis()
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"surface_profiles_M{Minf:.1f}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# ---- max-Mach vs grid size bar chart ------------------------------------
def plot_grid_summary(all_results_by_mach):
    """
    For multiple Mach numbers: plot peak M_local on surface vs grid resolution.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Grid summary: peak values on cylinder surface",
                 fontsize=13, fontweight='bold')

    cmap_m = plt.cm.cool
    machs  = list(all_results_by_mach.keys())
    nm     = len(machs)

    for im, Minf in enumerate(machs):
        all_results = all_results_by_mach[Minf]
        labels  = [f"{r['Nxi']}×{r['Neta']}" for r in all_results]
        peak_M  = [r['msurf'].max() for r in all_results]
        min_Cp  = [r['cpsurf'].min() for r in all_results]
        color   = cmap_m(im / (nm-1)) if nm > 1 else cmap_m(0.5)

        axes[0].plot(range(len(labels)), peak_M, '-o', color=color, lw=2,
                     ms=7, label=fr"$M_\infty={Minf:.1f}$")
        axes[1].plot(range(len(labels)), min_Cp, '-s', color=color, lw=2,
                     ms=7, label=fr"$M_\infty={Minf:.1f}$")

    for ax, ylabel, title in zip(
        axes,
        [r"$\max M_{\rm local}$", r"$\min C_p$"],
        ["Peak Mach on surface", "Min Cp on surface"]
    ):
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, fontsize=9)
        ax.set_xlabel("Grid (Nxi x Neta)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, "grid_summary.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# ================================================================ main ====
if __name__ == "__main__":
    # Fixed Mach numbers to study
    MACH_LIST  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # Grid configurations: (Nxi, Neta)
    GRIDS = [(30, 20), (60, 40), (80, 60), (120, 80)]

    all_results_by_mach = {}

    for Minf in MACH_LIST:
        print(f"\n{'='*55}")
        print(f"  Grid sensitivity: Minf = {Minf}")
        print(f"{'='*55}")
        all_results = []

        for (Nxi, Neta) in GRIDS:
            print(f"  Grid {Nxi}x{Neta} ...")
            sol = SubsonicCylinderSolver(Nxi=Nxi, Neta=Neta, Minf=Minf)
            ok, hist = sol.solve_newton(max_iter=50, tol=1e-8)
            if not ok:
                print(f"    [WARNING] did not converge")

            X, Y, Ml, Cp = compute_fields(sol)
            msurf  = surface_mach(sol)
            cpsurf = surface_cp(sol)
            all_results.append(dict(
                Nxi=Nxi, Neta=Neta, sol=sol,
                X=X, Y=Y, Ml=Ml, Cp=Cp,
                msurf=msurf, cpsurf=cpsurf, hist=hist
            ))

        all_results_by_mach[Minf] = all_results

        # 2D field grid
        plot_grid_fields(all_results, Minf)
        # Surface profiles
        plot_surface_profiles(all_results, Minf)

    # Summary across all mach and grids
    plot_grid_summary(all_results_by_mach)

    print(f"\nAll done. Results in: {OUT_DIR}")
