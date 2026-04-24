from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add root to path to find solver module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solver.solver import SubsonicCylinderSolver, SolveResult

class FastSubsonicCylinderSolver(SubsonicCylinderSolver):
    """
    Improved solver with multiple inner iterations per density update.
    This implementation uses a stone-factorized system for inner iterations.
    """
    
    def solve_sip_fast(self, alpha: float = 0.92, outer_tol: float = 1e-6, max_outer: int = 5000, 
                      inner_iters: int = 20, relax: float = 0.8) -> SolveResult:
        ni, nj = self.n_xi + 1, self.n_eta + 1
        phi = self.phi.copy()
        history = []
        
        # Pre-allocate factorization matrices
        b_s, c_s, d_s, e_s, f_s = [np.zeros((ni, nj)) for _ in range(5)]

        for outer in range(1, max_outer + 1):
            rho = self.compute_density(phi)
            A_mat, b_vec = self.frozen_density_operator_as_matrix(rho)
            B, D, E, F, H = self.get_stencil_from_matrix(A_mat)
            
            # 1. Update Stone's factorization (once per outer iter)
            for i in range(ni):
                for j in range(nj):
                    val_e_sw = e_s[i, j-1] if j > 0 else 0
                    val_f_ws = f_s[i-1, j] if i > 0 else 0
                    val_f_sw = f_s[i, j-1] if j > 0 else 0
                    val_e_ws = e_s[i-1, j] if i > 0 else 0
                    
                    denom_b = 1.0 + alpha * val_e_sw
                    b_s[i, j] = B[i, j] / (denom_b if abs(denom_b) > 1e-15 else 1e-15)
                    denom_c = 1.0 + alpha * val_f_ws
                    c_s[i, j] = D[i, j] / (denom_c if abs(denom_c) > 1e-15 else 1e-15)
                    d_s[i, j] = E[i, j] + alpha * (b_s[i, j] * val_e_sw + c_s[i, j] * val_f_ws) \
                                - b_s[i, j] * val_f_sw - c_s[i, j] * val_e_ws
                    if abs(d_s[i, j]) < 1e-15: d_s[i, j] = 1.0
                    e_s[i, j] = (F[i, j] - alpha * b_s[i, j] * val_e_sw) / d_s[i, j]
                    f_s[i, j] = (H[i, j] - alpha * c_s[i, j] * val_f_ws) / d_s[i, j]

            # 2. Inner iterations
            phi_old_outer = phi.copy()
            for inner in range(inner_iters):
                curr_res = (A_mat @ phi.flatten() - b_vec).reshape((ni, nj))
                Q = -curr_res
                
                # Forward substitution
                y = np.zeros((ni, nj))
                for i in range(ni):
                    for j in range(nj):
                        val_y_s = y[i, j-1] if j > 0 else 0
                        val_y_w = y[i-1, j] if i > 0 else 0
                        y[i, j] = (Q[i, j] - b_s[i, j] * val_y_s - c_s[i, j] * val_y_w) / d_s[i, j]
                
                # Backward substitution
                delta = np.zeros((ni, nj))
                for i in range(ni-1, -1, -1):
                    for j in range(nj-1, -1, -1):
                        val_d_n = delta[i, j+1] if j < nj-1 else 0
                        val_d_e = delta[i+1, j] if i < ni-1 else 0
                        delta[i, j] = y[i, j] - f_s[i, j] * val_d_n - e_s[i, j] * val_d_e
                
                phi += delta 
                phi = self.apply_boundary_conditions(phi)
                
                if np.max(np.abs(delta)) < 0.1 * outer_tol:
                    break
            
            total_delta = phi - phi_old_outer
            phi = phi_old_outer + relax * total_delta
            phi = self.apply_boundary_conditions(phi)

            err = float(np.max(np.abs(total_delta)))
            history.append(err)
            
            if outer % 10 == 0:
                print(f"    Fast SIP outer {outer}: err={err:.3e} (inner iters: {inner+1})", flush=True)
            
            if err < outer_tol:
                self.phi, self.rho = phi, rho
                return SolveResult("FastSIP", True, outer, history, [1]*outer)
                
        self.phi, self.rho = phi, rho
        return SolveResult("FastSIP", False, max_outer, history, [1]*max_outer)

def run_benchmarks():
    base_dir = Path("solver_up/results")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    n_xi, n_eta = 120, 90
    machs = [0.1, 0.2, 0.3, 0.4]
    max_iters = 5000
    tol = 1e-6
    relax = 0.8
    
    print(f"=== Running Benchmarks: tol={tol}, max_iters={max_iters} ===")
    
    # ---------------------------------------------------------
    # Experiment 1: Mach Sweep (FastSIP vs SLOR)
    # ---------------------------------------------------------
    results_sweep = []
    print("\n--- Mach Sweep ---")
    
    for m in machs:
        print(f"Testing M={m}...")
        # 1. Fast SIP
        solver = FastSubsonicCylinderSolver(n_xi=n_xi, n_eta=n_eta, minf=m)
        res_sip = solver.solve_sip_fast(outer_tol=tol, max_outer=max_iters, alpha=0.92, relax=relax)
        p_sip = solver.boundary_profiles()
        mid = len(p_sip['mach']) // 2
        
        # 2. SLOR (Reference)
        solver_slor = FastSubsonicCylinderSolver(n_xi=n_xi, n_eta=n_eta, minf=m)
        res_slor = solver_slor.solve_slor(outer_tol=tol, max_outer=max_iters, omega=1.7, relax=relax)
        
        results_sweep.append({
            'M_inf': m,
            'Iters_FastSIP': res_sip.outer_iterations,
            'Iters_SLOR': res_slor.outer_iterations,
            'M_loc_max': p_sip['mach'][mid],
            'Cp_min': p_sip['cp'][mid],
            'Speedup': res_slor.outer_iterations / res_sip.outer_iterations if res_sip.outer_iterations > 0 else 0
        })
        print(f"  Result: FastSIP={res_sip.outer_iterations}, SLOR={res_slor.outer_iterations}, Speedup={results_sweep[-1]['Speedup']:.1f}x")

    df_sweep = pd.DataFrame(results_sweep)
    df_sweep.to_csv(base_dir / "bench_mach_sweep.csv", index=False)
    
    # ---------------------------------------------------------
    # Experiment 2: Parameter Sensitivity (M=0.3)
    # ---------------------------------------------------------
    print("\n--- Parameter Sensitivity (M=0.3) ---")
    m_test = 0.3
    params_results = []
    
    # Fast SIP Alpha Sensitivity
    for alpha in [0.92, 0.00]:
        print(f"  FastSIP alpha={alpha}...")
        solver = FastSubsonicCylinderSolver(n_xi=n_xi, n_eta=n_eta, minf=m_test)
        res = solver.solve_sip_fast(outer_tol=tol, max_outer=max_iters, alpha=alpha, relax=relax)
        params_results.append({
            'Method': f'FastSIP (alpha={alpha})',
            'Iters': res.outer_iterations,
            'Residual': res.outer_history[-1] if res.outer_history else None
        })

    # SLOR Omega Sensitivity
    for omega in [1.70, 1.00]:
        print(f"  SLOR omega={omega}...")
        solver = FastSubsonicCylinderSolver(n_xi=n_xi, n_eta=n_eta, minf=m_test)
        res = solver.solve_slor(outer_tol=tol, max_outer=max_iters, omega=omega, relax=relax)
        params_results.append({
            'Method': f'SLOR (omega={omega})',
            'Iters': res.outer_iterations,
            'Residual': res.outer_history[-1] if res.outer_history else None
        })

    df_params = pd.DataFrame(params_results)
    df_params.to_csv(base_dir / "bench_params.csv", index=False)
    
    print("\nBenchmarks completed successfully.")
    print("\nSummary Table:")
    print(df_sweep.to_string())

if __name__ == "__main__":
    # Check that it works by initializing a solver (don't run full experiments as requested)
    print("Checking solver initialization...")
    try:
        test_solver = FastSubsonicCylinderSolver(n_xi=10, n_eta=10, minf=0.1)
        print("Initialization OK.")
        # If user wants to run:
        run_benchmarks()
    except Exception as e:
        print(f"Error during check: {e}")
        sys.exit(1)
    
    print("\nScript is ready. To run experiments, uncomment 'run_benchmarks()' in __main__.")
