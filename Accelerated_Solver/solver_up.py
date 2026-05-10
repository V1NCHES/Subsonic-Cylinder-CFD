from __future__ import annotations
import numpy as np
from solver.solver import SubsonicCylinderSolver, SolveResult

class FastSubsonicCylinderSolver(SubsonicCylinderSolver):
    """
    Improved solver with multiple inner iterations per density update.
    """
    
    def solve_sip_fast(self, alpha: float = 0.92, outer_tol: float = 1e-7, max_outer: int = 1000, 
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
            
            # 1. Update Stone's factorization (only once per outer iter)
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

            # 2. Inner iterations: solve the linear system more accurately
            # We track the change in phi during inner loops
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
                
                phi += delta # Usually no damping needed for inner linear steps
                phi = self.apply_boundary_conditions(phi)
                
                # Check if linear system is solved well enough
                if np.max(np.abs(delta)) < 0.1 * outer_tol:
                    break
            
            # Apply damping for the NONLINEAR (Picard) step if needed
            # In this improved version, we've already updated phi significantly
            # We can damp the whole change from the start of the outer iteration
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
