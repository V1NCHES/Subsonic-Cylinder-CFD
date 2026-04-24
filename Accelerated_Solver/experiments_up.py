from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from solver_up import FastSubsonicCylinderSolver
import sys
import os

# Add root to path to find solver module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_all():
    base_dir = Path("solver_up/results")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    n_xi, n_eta = 120, 90
    machs = [0.1, 0.2, 0.3, 0.4]
    
    # ---------------------------------------------------------
    # Experiment 1: Mach Sweep (Comparison Fast SIP vs SLOR)
    # ---------------------------------------------------------
    results = []
    print("=== Starting Mach Sweep (Fast SIP vs SLOR) ===")
    
    for m in machs:
        # 1. Fast SIP
        solver = FastSubsonicCylinderSolver(n_xi=n_xi, n_eta=n_eta, minf=m)
        res_sip = solver.solve_sip_fast(max_outer=1000, inner_iters=20, alpha=0.92, relax=0.8)
        p_sip = solver.boundary_profiles()
        mid = len(p_sip['mach']) // 2
        
        # 2. SLOR (Standard)
        solver_slor = FastSubsonicCylinderSolver(n_xi=n_xi, n_eta=n_eta, minf=m)
        res_slor = solver_slor.solve_slor(max_outer=5000, omega=1.7, relax=0.8)
        
        results.append({
            'M_inf': m,
            'M_loc_SIP': p_sip['mach'][mid],
            'Cp_SIP': p_sip['cp'][mid],
            'Iters_FastSIP': res_sip.outer_iterations,
            'Iters_SLOR': res_slor.outer_iterations,
            'Speedup': res_slor.outer_iterations / res_sip.outer_iterations if res_sip.outer_iterations > 0 else 0
        })
        print(f"M={m}: FastSIP={res_sip.outer_iterations}, SLOR={res_slor.outer_iterations}, Speedup={results[-1]['Speedup']:.1f}x")

    df_sweep = pd.DataFrame(results)
    df_sweep.to_csv(base_dir / "summary_comparison.csv", index=False)
    
    # ---------------------------------------------------------
    # Experiment 2: Parameter Sensitivity (M=0.3)
    # ---------------------------------------------------------
    print("\n=== Starting Parameter Sensitivity (M=0.3) ===")
    m = 0.3
    params_results = []
    
    # Alpha study for Fast SIP
    for alpha in [0.92, 0.00]:
        solver = FastSubsonicCylinderSolver(n_xi=n_xi, n_eta=n_eta, minf=m)
        res = solver.solve_sip_fast(max_outer=1000, inner_iters=20, alpha=alpha, relax=0.8)
        prof = solver.boundary_profiles()
        mid = len(prof['mach']) // 2
        params_results.append({
            'Method': f'FastSIP (a={alpha})',
            'Iters': res.outer_iterations,
            'M_loc': prof['mach'][mid],
            'Cp': prof['cp'][mid],
            'Err': res.outer_history[-1]
        })
        print(f"FastSIP alpha={alpha}: Iters={res.outer_iterations}, Err={res.outer_history[-1]:.2e}")

    # Omega study for SLOR
    for omega in [1.70, 1.00]:
        solver = FastSubsonicCylinderSolver(n_xi=n_xi, n_eta=n_eta, minf=m)
        res = solver.solve_slor(max_outer=5000, omega=omega, relax=0.8)
        prof = solver.boundary_profiles()
        mid = len(prof['mach']) // 2
        params_results.append({
            'Method': f'SLOR (w={omega})',
            'Iters': res.outer_iterations,
            'M_loc': prof['mach'][mid],
            'Cp': prof['cp'][mid],
            'Err': res.outer_history[-1]
        })
        print(f"SLOR omega={omega}: Iters={res.outer_iterations}, Err={res.outer_history[-1]:.2e}")

    df_params = pd.DataFrame(params_results)
    df_params.to_csv(base_dir / "param_sensitivity.csv", index=False)
    
    print("\nAll experiments completed.")
    print("Comparison Table:")
    print(df_sweep.to_string())
    print("\nSensitivity Table:")
    print(df_params.to_string())

if __name__ == "__main__":
    run_all()
