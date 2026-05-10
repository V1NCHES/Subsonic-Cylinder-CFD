# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
Experiment 1 - Newton FIM for Minf = 0.1, 0.3, 0.5, 0.7, 0.9
Saves: M_local field and Cp field in physical (x,y) plane.
Output: C:\Program1\9 sem\task 3\new\experiments\
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize

# ---- paths ---------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLVER_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "result")
OUT_DIR    = os.path.join(SCRIPT_DIR, "experiments")
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, SOLVER_DIR)
from solvers import SubsonicCylinderSolver

# ---- style ---------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 100,
})

# ---- helper: compute physical fields -------------------------------------
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
        # mirror: eta 0..pi -> full 0..2pi
        return np.concatenate((arr[:, ::-1][:, :-1], arr), axis=1)

    X_full  = mirror(X)
    Y_full  = np.concatenate((-Y[:, ::-1][:, :-1],  Y), axis=1)
    Ml_full = mirror(M_local)
    Cp_full = mirror(Cp)

    return X_full, Y_full, Ml_full, Cp_full


# ---- plot one case -------------------------------------------------------
def plot_case(sol, Minf):
    X, Y, Ml, Cp = compute_fields(sol)

    theta = np.linspace(0, 2*np.pi, 400)
    cx, cy = np.cos(theta), np.sin(theta)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        fr"ПНМ (Newton FIM), $M_\infty = {Minf:.1f}$",
        fontsize=14, fontweight='bold'
    )

    # --- M_local ---
    ax = axes[0]
    cf = ax.contourf(X, Y, Ml, levels=60, cmap='inferno')
    ax.contour (X, Y, Ml, levels=20, colors='white', linewidths=0.4, alpha=0.5)
    ax.fill(cx, cy, color='lightgray', zorder=5)
    ax.plot(cx, cy, 'k-', lw=1.5, zorder=6)
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-2.8, 2.8)
    ax.set_aspect('equal')
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(r"Локальное число Маха $M_{\rm local}$")
    ax.grid(True, alpha=0.15, color='white')
    cb = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.03)
    cb.set_label(r"$M_{\rm local}$")

    # --- Cp ---
    ax = axes[1]
    vext = np.max(np.abs(Cp))
    cf2 = ax.contourf(X, Y, Cp, levels=60, cmap='RdBu_r',
                       vmin=-vext, vmax=vext)
    ax.contour(X, Y, Cp, levels=20, colors='k', linewidths=0.4, alpha=0.5)
    ax.fill(cx, cy, color='white', zorder=5)
    ax.plot(cx, cy, 'k-', lw=1.5, zorder=6)
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-2.8, 2.8)
    ax.set_aspect('equal')
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(r"Коэффициент давления $C_p$")
    ax.grid(True, alpha=0.15)
    cb2 = fig.colorbar(cf2, ax=ax, shrink=0.85, pad=0.03)
    cb2.set_label(r"$C_p$")

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"newton_M{Minf:.1f}.png")
    plt.savefig(fname, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# ---- summary figure: all Mach side-by-side ------------------------------
def plot_summary(results):
    """One large figure: rows = M local / Cp; columns = each Mach number."""
    machs     = [r['Minf'] for r in results]
    n         = len(machs)

    fig, axes = plt.subplots(2, n, figsize=(5*n, 10))
    fig.suptitle("ПНМ: M_local и Cp при различных числах Маха", fontsize=14, fontweight='bold')

    theta = np.linspace(0, 2*np.pi, 400)
    cx, cy = np.cos(theta), np.sin(theta)

    for col, r in enumerate(results):
        X, Y, Ml, Cp = r['X'], r['Y'], r['Ml'], r['Cp']
        Minf = r['Minf']

        # Row 0: M_local
        ax = axes[0, col]
        cf = ax.contourf(X, Y, Ml, levels=50, cmap='inferno', vmin=0)
        ax.fill(cx, cy, color='lightgray', zorder=5)
        ax.plot(cx, cy, 'k-', lw=1, zorder=6)
        ax.set_xlim(-3, 3); ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.set_title(fr"$M_\infty={Minf:.1f}$")
        ax.set_xlabel("x")
        if col == 0: ax.set_ylabel(r"$M_{\rm local}$" + "\ny")
        fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.02)

        # Row 1: Cp
        ax = axes[1, col]
        vext = max(np.abs(Cp).max(), 0.5)
        cf2 = ax.contourf(X, Y, Cp, levels=50, cmap='RdBu_r', vmin=-vext, vmax=vext)
        ax.fill(cx, cy, color='white', zorder=5)
        ax.plot(cx, cy, 'k-', lw=1, zorder=6)
        ax.set_xlim(-3, 3); ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.set_xlabel("x")
        if col == 0: ax.set_ylabel(r"$C_p$" + "\ny")
        fig.colorbar(cf2, ax=ax, fraction=0.04, pad=0.02)

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, "summary_all_mach.png")
    plt.savefig(fname, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# ---- convergence history -------------------------------------------------
def plot_all_convergence(results):
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.plasma
    n = len(results)
    for i, r in enumerate(results):
        h = r['hist']
        color = cmap(i / (n - 1)) if n > 1 else cmap(0.5)
        ax.semilogy(range(1, len(h)+1), h, '-o', color=color,
                    lw=2, ms=5, label=fr"$M_\infty={r['Minf']:.1f}$")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\max|\delta\Phi|$")
    ax.set_title("Newton FIM convergence for various Mach numbers")
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    fname = os.path.join(OUT_DIR, "convergence_all.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# ================================================================ main ====
if __name__ == "__main__":
    MACH_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    NXI, NETA = 80, 60

    results = []
    for Minf in MACH_LIST:
        print(f"\n{'='*50}")
        print(f"  Newton FIM  Minf = {Minf}")
        print(f"{'='*50}")
        sol = SubsonicCylinderSolver(Nxi=NXI, Neta=NETA, Minf=Minf)
        ok, hist = sol.solve_newton(max_iter=50, tol=1e-8)
        if not ok:
            print(f"  [WARNING] Newton did not converge for M={Minf}")

        plot_case(sol, Minf)

        X, Y, Ml, Cp = compute_fields(sol)
        results.append(dict(Minf=Minf, X=X, Y=Y, Ml=Ml, Cp=Cp, hist=hist))

    plot_summary(results)
    plot_all_convergence(results)

    print(f"\nAll done. Results in: {OUT_DIR}")
