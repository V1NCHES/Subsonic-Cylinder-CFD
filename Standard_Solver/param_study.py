from solver import SubsonicCylinderSolver
import pandas as pd
import numpy as np
from pathlib import Path

def run_param_study():
    m = 0.3
    n_xi, n_eta = 120, 90
    max_iters = 5000
    tol = 1e-6
    relax = 0.8
    
    configs = [
        ("SIP", 0.92, None),
        ("SIP", 0.00, None),
        ("SLOR", None, 1.70),
        ("SLOR", None, 1.00)
    ]
    
    results = []
    print(f"Starting parameter study for M={m}, grid={n_xi}x{n_eta}, max_iters={max_iters}")
    
    for method, alpha, omega in configs:
        solver = SubsonicCylinderSolver(n_xi=n_xi, n_eta=n_eta, minf=m)
        param_val = alpha if method == "SIP" else omega
        print(f"Running {method} with param={param_val}...")
        
        if method == "SIP":
            res = solver.solve_sip(max_outer=max_iters, alpha=alpha, relax=relax, outer_tol=tol)
        else:
            res = solver.solve_slor(max_outer=max_iters, omega=omega, relax=relax, outer_tol=tol)
            
        prof = solver.boundary_profiles()
        mid = len(prof['mach']) // 2
        
        results.append({
            'Method': method,
            'Parameter': param_val,
            'Iters': res.outer_iterations,
            'Residual': res.outer_history[-1],
            'M_local_max': prof['mach'][mid],
            'Cp_min': prof['cp'][mid],
            'Converged': res.converged
        })
        print(f"  Done. Iters: {res.outer_iterations}, Err: {res.outer_history[-1]:.3e}")

    df = pd.DataFrame(results)
    output_path = Path("results/param_sensitivity_M0.3.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("\nStudy completed. Results saved to results/param_sensitivity_M0.3.csv")
    print(df.to_string())

if __name__ == "__main__":
    run_param_study()
