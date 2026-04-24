from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from solver import SubsonicCylinderSolver, save_npz
from multiprocessing import Pool

MACH_LIST = [0.1, 0.2, 0.3, 0.4]
R_INF = 5.0
GAMMA = 1.4

def plot_profile(eta: np.ndarray, values: np.ndarray, ylabel: str, title: str, save_path: Path):
    plt.figure(figsize=(10, 6))
    plt.plot(eta * 180 / np.pi, values, linewidth=2)
    plt.xlabel('Angle theta (degrees)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, 180)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_convergence(histories: Dict[str, List[float]], title: str, save_path: Path):
    plt.figure(figsize=(10, 6))
    for label, history in histories.items():
        plt.semilogy(history, label=label)
    plt.xlabel('Iteration')
    plt.ylabel('Max Correction |delta_phi|')
    plt.title(title)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_comparison(eta: np.ndarray, sip_vals: np.ndarray, slor_vals: np.ndarray, ylabel: str, title: str, save_path: Path):
    plt.figure(figsize=(10, 6))
    plt.plot(eta * 180 / np.pi, sip_vals, 'b-', label='PNM (SIP)', linewidth=2)
    plt.plot(eta * 180 / np.pi, slor_vals, 'r--', label='SLOR', linewidth=1.5)
    plt.xlabel('Angle theta (degrees)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(0, 180)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_mach(args):
    minf, n_xi, n_eta, base_dir = args
    print(f"Starting M={minf:.1f}...", flush=True)
    
    # 1. Run SIP (PNM)
    solver_sip = SubsonicCylinderSolver(n_xi=n_xi, n_eta=n_eta, minf=minf)
    res_sip = solver_sip.solve_sip(max_outer=2000, relax=0.8, outer_tol=1e-7, alpha=0.92)
    prof_sip = solver_sip.boundary_profiles()
    
    # 2. Run SLOR
    solver_slor = SubsonicCylinderSolver(n_xi=n_xi, n_eta=n_eta, minf=minf)
    res_slor = solver_slor.solve_slor(max_outer=5000, relax=0.8, outer_tol=1e-7, omega=1.7)
    prof_slor = solver_slor.boundary_profiles()
    
    # 3. Generate Plots
    plot_profile(prof_sip['eta'], prof_sip['mach'], 'M_local', 
                f'PNM Local Mach, M_inf={minf:.1f}', base_dir / f"pnm_mach_M{minf:.1f}.png")
    plot_profile(prof_sip['eta'], prof_sip['cp'], 'Cp', 
                f'PNM Cp, M_inf={minf:.1f}', base_dir / f"pnm_cp_M{minf:.1f}.png")
    plot_comparison(prof_sip['eta'], prof_sip['mach'], prof_slor['mach'], 'M_local',
                   f'PNM vs SLOR: Local Mach, M_inf={minf:.1f}', base_dir / f"comp_mach_M{minf:.1f}.png")
    plot_comparison(prof_sip['eta'], prof_sip['cp'], prof_slor['cp'], 'Cp',
                   f'PNM vs SLOR: Cp, M_inf={minf:.1f}', base_dir / f"comp_cp_M{minf:.1f}.png")
    
    mid = len(prof_sip['mach']) // 2
    result = {
        'M_inf': minf,
        'M_local_pnm': prof_sip['mach'][mid],
        'Cp_pnm': prof_sip['cp'][mid],
        'Iters_pnm': res_sip.outer_iterations,
        'Iters_slor': res_slor.outer_iterations,
        'Err_pnm': res_sip.outer_history[-1],
        'Err_slor': res_slor.outer_history[-1],
        'history_pnm': res_sip.outer_history,
        'history_slor': res_slor.outer_history
    }
    print(f"Finished M={minf:.1f}", flush=True)
    return result

def main():
    base_dir = Path("results")
    base_dir.mkdir(parents=True, exist_ok=True)
    n_xi, n_eta = 120, 90
    mach_list = [0.1, 0.2, 0.3, 0.4]
    
    tasks = [(m, n_xi, n_eta, base_dir) for m in mach_list]
    
    with Pool(processes=2) as pool:
        all_results = pool.map(run_mach, tasks)
    
    # Process results
    summary_data = []
    convergence_data = {}
    
    for r in all_results:
        m = r['M_inf']
        summary_data.append({k: v for k, v in r.items() if not k.startswith('history')})
        convergence_data[f'PNM M={m:.1f}'] = r['history_pnm']
        convergence_data[f'SLOR M={m:.1f}'] = r['history_slor']

    df = pd.DataFrame(summary_data)
    df.sort_values('M_inf', inplace=True)
    df.to_csv(base_dir / "full_summary.csv", index=False)
    
    plot_convergence(convergence_data, 'Convergence History: PNM vs SLOR', base_dir / "convergence_all.png")
    
    print("\nAll experiments completed.")
    print(df)

if __name__ == "__main__":
    main()
