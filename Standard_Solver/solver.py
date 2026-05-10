from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


@dataclass
class SolveResult:
    method: str
    converged: bool
    outer_iterations: int
    outer_history: List[float]
    inner_history: List[int]


class SubsonicCylinderSolver:
    """
    Nonlinear potential-flow solver for subsonic compressible flow around a unit cylinder.

    Unknown: phi(xi, eta) on rectangle
        xi in [0, ln(R_inf)], eta in [0, pi]
    after conformal map z = exp(xi + i eta).

    PDE (frozen-density Picard linearization):
        d/dxi (rho dphi/dxi) + d/deta (rho dphi/deta) = 0

    BCs:
        xi = 0      : dphi/dxi = 0               (impermeability)
        xi = xi_max : phi = exp(xi_max) cos eta  (far field)
        eta = 0, pi : dphi/deta = 0              (symmetry)
    """

    def __init__(
        self,
        n_xi: int = 120,
        n_eta: int = 90,
        minf: float = 0.3,
        gamma: float = 1.4,
        r_inf: float = 5.0,
    ) -> None:
        self.n_xi = int(n_xi)
        self.n_eta = int(n_eta)
        self.minf = float(minf)
        self.gamma = float(gamma)
        self.r_inf = float(r_inf)

        self.xi_max = np.log(self.r_inf)
        self.xi = np.linspace(0.0, self.xi_max, self.n_xi + 1)
        self.eta = np.linspace(0.0, np.pi, self.n_eta + 1)
        self.dxi = self.xi[1] - self.xi[0]
        self.deta = self.eta[1] - self.eta[0]

        self.Xi, self.Eta = np.meshgrid(self.xi, self.eta, indexing="ij")
        self.r = np.exp(self.Xi)
        self.X = self.r * np.cos(self.Eta)
        self.Y = self.r * np.sin(self.Eta)

        self.phi = self.initial_guess()
        self.rho = self.compute_density(self.phi)

    # ----------------------------- utilities -----------------------------
    def initial_guess(self) -> np.ndarray:
        return np.exp(self.Xi) * np.cos(self.Eta)

    def apply_boundary_conditions(self, phi: np.ndarray) -> np.ndarray:
        out = phi.copy()
        out[0, :] = out[1, :]                 # dphi/dxi = 0 at xi = 0
        out[-1, :] = np.exp(self.xi[-1]) * np.cos(self.eta)  # Dirichlet far field
        out[:, 0] = out[:, 1]                 # dphi/deta = 0 at eta = 0
        out[:, -1] = out[:, -2]               # dphi/deta = 0 at eta = pi
        # keep corners consistent with the Dirichlet side at the far field
        out[-1, 0] = np.exp(self.xi[-1])
        out[-1, -1] = -np.exp(self.xi[-1])
        return out

    def derivatives(self, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        phi_bc = self.apply_boundary_conditions(phi)
        dphi_dxi = np.zeros_like(phi_bc)
        dphi_deta = np.zeros_like(phi_bc)

        dphi_dxi[1:-1, :] = (phi_bc[2:, :] - phi_bc[:-2, :]) / (2.0 * self.dxi)
        dphi_deta[:, 1:-1] = (phi_bc[:, 2:] - phi_bc[:, :-2]) / (2.0 * self.deta)

        dphi_dxi[0, :] = 0.0
        dphi_dxi[-1, :] = (phi_bc[-1, :] - phi_bc[-2, :]) / self.dxi
        dphi_deta[:, 0] = 0.0
        dphi_deta[:, -1] = 0.0
        return dphi_dxi, dphi_deta

    def velocity_components(self, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dphi_dxi, dphi_deta = self.derivatives(phi)
        factor = np.exp(-self.Xi)
        u = factor * (np.cos(self.Eta) * dphi_dxi - np.sin(self.Eta) * dphi_deta)
        v = factor * (np.sin(self.Eta) * dphi_dxi + np.cos(self.Eta) * dphi_deta)
        q2 = u * u + v * v
        return u, v, q2

    def compute_density(self, phi: np.ndarray, use_biasing: bool = True) -> np.ndarray:
        u, v, q2 = self.velocity_components(phi)
        arg = 1.0 + 0.5 * (self.gamma - 1.0) * self.minf**2 * (1.0 - q2)
        arg = np.maximum(arg, 1.0e-8)
        rho = arg ** (1.0 / (self.gamma - 1.0))

        if not use_biasing or self.minf < 0.35:
            return rho

        # Density biasing for transonic stability (upwind biasing)
        mach = self.compute_local_mach(phi)
        mu = np.maximum(0.0, 1.0 - 1.0 / np.maximum(mach**2, 1e-8))
        rho_biased = rho.copy()
        # Bias in xi direction (outward from cylinder)
        for i in range(1, self.n_xi + 1):
            rho_biased[i, :] = rho[i, :] - mu[i, :] * (rho[i, :] - rho[i - 1, :])
        return rho_biased

    def compute_local_mach(self, phi: np.ndarray) -> np.ndarray:
        u, v, q2 = self.velocity_components(phi)
        vmag = np.sqrt(q2)
        a2 = 1.0 / self.minf**2 + 0.5 * (self.gamma - 1.0) * (1.0 - q2)
        a2 = np.maximum(a2, 1.0e-8)
        return vmag / np.sqrt(a2)

    def compute_cp(self, phi: np.ndarray, rho: np.ndarray | None = None) -> np.ndarray:
        if rho is None:
            rho = self.compute_density(phi, use_biasing=False)
        return (2.0 / (self.gamma * self.minf**2)) * (rho**self.gamma - 1.0)

    def residual(self, phi: np.ndarray, rho: np.ndarray | None = None) -> np.ndarray:
        phi_bc = self.apply_boundary_conditions(phi)
        if rho is None:
            rho = self.compute_density(phi_bc, use_biasing=False)
        res = np.zeros_like(phi_bc)
        dx2, de2 = self.dxi**2, self.deta**2

        # Midpoint densities
        rho_mid_xi = 0.5 * (rho[1:, :] + rho[:-1, :])
        rho_mid_eta = 0.5 * (rho[:, 1:] + rho[:, :-1])

        # Internal points (1 to N-1)
        rw = rho_mid_xi[:-1, 1:-1]
        re = rho_mid_xi[1:, 1:-1]
        rs = rho_mid_eta[1:-1, :-1]
        rn = rho_mid_eta[1:-1, 1:]

        phi_p = phi_bc[1:-1, 1:-1]
        res[1:-1, 1:-1] = (re * (phi_bc[2:, 1:-1] - phi_p) - rw * (phi_p - phi_bc[:-2, 1:-1])) / dx2 + \
                          (rn * (phi_bc[1:-1, 2:] - phi_p) - rs * (phi_p - phi_bc[1:-1, :-2])) / de2
        return res

    def get_stencil_coeffs(self, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ni, nj = self.n_xi + 1, self.n_eta + 1
        B, D, E, F, H = [np.zeros((ni, nj)) for _ in range(5)]
        Q = self.residual(self.phi, rho)
        dx2, de2 = self.dxi**2, self.deta**2

        # Internal points
        rho_mid_xi = 0.5 * (rho[1:, :] + rho[:-1, :])
        rho_mid_eta = 0.5 * (rho[:, 1:] + rho[:, :-1])
        
        rw = rho_mid_xi[:-1, 1:-1]
        re = rho_mid_xi[1:, 1:-1]
        rs = rho_mid_eta[1:-1, :-1]
        rn = rho_mid_eta[1:-1, 1:]

        B[1:-1, 1:-1] = -rs / de2
        D[1:-1, 1:-1] = -rw / dx2
        F[1:-1, 1:-1] = -re / dx2
        H[1:-1, 1:-1] = -rn / de2
        E[1:-1, 1:-1] = -(B[1:-1, 1:-1] + D[1:-1, 1:-1] + F[1:-1, 1:-1] + H[1:-1, 1:-1])

        # Boundary coefficients (matching residuals)
        E[0, :], F[0, :] = 1.0, -1.0           # xi=0 Neumann
        E[-1, :] = 1.0                        # xi=max Dirichlet
        E[1:-1, 0], H[1:-1, 0] = 1.0, -1.0    # eta=0 Neumann
        E[1:-1, -1], B[1:-1, -1] = 1.0, -1.0  # eta=pi Neumann

        return B, D, E, F, H, -Q

    def solve_direct(self, outer_tol: float = 1e-7, max_outer: int = 100, relax: float = 0.5) -> SolveResult:
        """Fast Picard iteration using direct sparse solver."""
        from scipy.sparse.linalg import spsolve
        phi = self.phi.copy()
        history = []
        ni, nj = self.n_xi + 1, self.n_eta + 1

        for outer in range(1, max_outer + 1):
            rho = self.compute_density(phi)
            A, b_vec = self.frozen_density_operator_as_matrix(rho)
            phi_new = spsolve(A, b_vec).reshape((ni, nj))
            
            delta = phi_new - phi
            phi += relax * delta
            phi = self.apply_boundary_conditions(phi)
            err = float(np.max(np.abs(delta)))
            history.append(err)
            if err < outer_tol:
                self.phi = phi
                self.rho = rho
                return SolveResult("Direct", True, outer, history, [1]*outer)
        
        self.phi = phi
        self.rho = rho
        return SolveResult("Direct", False, max_outer, history, [1]*max_outer)

    def frozen_density_operator_as_matrix(self, rho: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
        from scipy.sparse import csr_matrix
        ni, nj = self.n_xi + 1, self.n_eta + 1
        n = ni * nj
        dx2, de2 = self.dxi**2, self.deta**2
        phi_far = np.exp(self.xi[-1]) * np.cos(self.eta)

        b = np.zeros(n)
        
        # Internal points indices
        ii, jj = np.meshgrid(np.arange(1, ni-1), np.arange(1, nj-1), indexing='ij')
        k = ii * nj + jj
        
        re = 0.5 * (rho[2:, 1:-1] + rho[1:-1, 1:-1])
        rw = 0.5 * (rho[:-2, 1:-1] + rho[1:-1, 1:-1])
        rn = 0.5 * (rho[1:-1, 2:] + rho[1:-1, 1:-1])
        rs = 0.5 * (rho[1:-1, :-2] + rho[1:-1, 1:-1])

        # We will build CSR matrix manually for speed
        rows = []
        cols = []
        data = []

        # Internal points (5 entries per row)
        rows.append(k.flatten())
        cols.append(k.flatten())
        data.append((-(re + rw) / dx2 - (rn + rs) / de2).flatten())

        rows.append(k.flatten())
        cols.append((k + nj).flatten())
        data.append((re / dx2).flatten())

        rows.append(k.flatten())
        cols.append((k - nj).flatten())
        data.append((rw / dx2).flatten())

        rows.append(k.flatten())
        cols.append((k + 1).flatten())
        data.append((rn / de2).flatten())

        rows.append(k.flatten())
        cols.append((k - 1).flatten())
        data.append((rs / de2).flatten())

        # Boundaries
        # xi = 0
        k_b = np.arange(nj)
        rows.append(k_b); cols.append(k_b); data.append(np.ones(nj))
        rows.append(k_b); cols.append(k_b + nj); data.append(-np.ones(nj))
        
        # xi = ni-1
        k_b = (ni - 1) * nj + np.arange(nj)
        rows.append(k_b); cols.append(k_b); data.append(np.ones(nj))
        b[k_b] = phi_far

        # eta = 0 (excluding corners already handled by xi=0, xi=ni-1)
        k_b = np.arange(1, ni-1) * nj
        rows.append(k_b); cols.append(k_b); data.append(np.ones(ni-2))
        rows.append(k_b); cols.append(k_b + 1); data.append(-np.ones(ni-2))

        # eta = nj-1
        k_b = np.arange(1, ni-1) * nj + (nj - 1)
        rows.append(k_b); cols.append(k_b); data.append(np.ones(ni-2))
        rows.append(k_b); cols.append(k_b - 1); data.append(-np.ones(ni-2))

        rows_f = np.concatenate(rows)
        cols_f = np.concatenate(cols)
        data_f = np.concatenate(data)
        
        A = csr_matrix((data_f, (rows_f, cols_f)), shape=(n, n))
        return A, b

    def get_stencil_from_matrix(self, A: csr_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ni, nj = self.n_xi + 1, self.n_eta + 1
        B, D, E, F, H = [np.zeros((ni, nj)) for _ in range(5)]
        
        # A is ni*nj x ni*nj
        # Row k = i*nj + j
        # Col k-1: B (South), k-nj: D (West), k: E (Center), k+nj: F (East), k+1: H (North)
        
        for i in range(ni):
            for j in range(nj):
                k = i * nj + j
                row = A.getrow(k)
                for col_idx, val in zip(row.indices, row.data):
                    if col_idx == k - 1: B[i, j] = val
                    elif col_idx == k - nj: D[i, j] = val
                    elif col_idx == k: E[i, j] = val
                    elif col_idx == k + nj: F[i, j] = val
                    elif col_idx == k + 1: H[i, j] = val
        return B, D, E, F, H

    def solve_sip(self, alpha: float = 0.0, outer_tol: float = 1e-7, max_outer: int = 1000, relax: float = 0.1) -> SolveResult:
        ni, nj = self.n_xi + 1, self.n_eta + 1
        phi = self.phi.copy()
        history = []
        b_s, c_s, d_s, e_s, f_s = [np.zeros((ni, nj)) for _ in range(5)]

        for outer in range(1, max_outer + 1):
            rho = self.compute_density(phi)
            A_mat, b_vec = self.frozen_density_operator_as_matrix(rho)
            B, D, E, F, H = self.get_stencil_from_matrix(A_mat)
            
            # Stone's factorization
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

            curr_res = (A_mat @ phi.flatten() - b_vec).reshape((ni, nj))
            Q = -curr_res
            y = np.zeros((ni, nj))
            for i in range(ni):
                for j in range(nj):
                    val_y_s = y[i, j-1] if j > 0 else 0
                    val_y_w = y[i-1, j] if i > 0 else 0
                    y[i, j] = (Q[i, j] - b_s[i, j] * val_y_s - c_s[i, j] * val_y_w) / d_s[i, j]
            delta = np.zeros((ni, nj))
            for i in range(ni-1, -1, -1):
                for j in range(nj-1, -1, -1):
                    val_d_n = delta[i, j+1] if j < nj-1 else 0
                    val_d_e = delta[i+1, j] if i < ni-1 else 0
                    delta[i, j] = y[i, j] - f_s[i, j] * val_d_n - e_s[i, j] * val_d_e

            phi += relax * delta
            phi = self.apply_boundary_conditions(phi)
            err = float(np.max(np.abs(delta)))
            history.append(err)
            if outer % 50 == 0: print(f"    SIP iter {outer}: err={err:.3e}", flush=True)
            if err < outer_tol:
                self.phi, self.rho = phi, rho
                return SolveResult("SIP", True, outer, history, [1]*outer)
        self.phi, self.rho = phi, rho
        return SolveResult("SIP", False, max_outer, history, [1]*max_outer)

    def solve_slor(self, omega: float = 1.0, outer_tol: float = 1e-7, max_outer: int = 5000, relax: float = 0.1) -> SolveResult:
        ni, nj = self.n_xi + 1, self.n_eta + 1
        phi = self.phi.copy()
        history = []
        for outer in range(1, max_outer + 1):
            rho = self.compute_density(phi)
            A_mat, b_vec = self.frozen_density_operator_as_matrix(rho)
            B, D, E, F, H = self.get_stencil_from_matrix(A_mat)
            curr_res = (A_mat @ phi.flatten() - b_vec).reshape((ni, nj))
            delta = np.zeros((ni, nj))
            # Sweep along lines of constant j (ETA)
            for j in range(nj):
                A_tri, B_tri, C_tri = D[:, j], E[:, j], F[:, j]
                RHS = -curr_res[:, j]
                if j > 0: RHS -= B[:, j] * delta[:, j-1]
                if j < nj - 1: RHS -= H[:, j] * delta[:, j+1] # delta_next is 0 in first pass
                delta[:, j] = omega * self.solve_tridiagonal(A_tri, B_tri, C_tri, RHS)
            phi += relax * delta
            phi = self.apply_boundary_conditions(phi)
            err = float(np.max(np.abs(delta)))
            history.append(err)
            if outer % 100 == 0: print(f"    SLOR iter {outer}: err={err:.3e}", flush=True)
            if err < outer_tol:
                self.phi, self.rho = phi, rho
                return SolveResult("SLOR", True, outer, history, [1]*outer)
        self.phi, self.rho = phi, rho
        return SolveResult("SLOR", False, max_outer, history, [1]*max_outer)

    @staticmethod
    def solve_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
        n = len(d)
        cp, dp, x = np.zeros(n), np.zeros(n), np.zeros(n)
        if abs(b[0]) < 1e-18: b[0] = 1.0
        cp[0] = c[0] / b[0]
        dp[0] = d[0] / b[0]
        for i in range(1, n):
            denom = b[i] - a[i] * cp[i-1]
            if abs(denom) < 1e-18: denom = 1e-18
            cp[i] = c[i] / denom if i < n-1 else 0.0
            dp[i] = (d[i] - a[i] * dp[i-1]) / denom
        x[-1] = dp[-1]
        for i in range(n-2, -1, -1): x[i] = dp[i] - cp[i] * x[i+1]
        return x

    # -------------------------- postprocessing ---------------------------
    def boundary_profiles(self) -> Dict[str, np.ndarray]:
        phi = self.apply_boundary_conditions(self.phi)
        rho = self.compute_density(phi, use_biasing=False) # Use physical density for output
        mach = self.compute_local_mach(phi)
        cp = self.compute_cp(phi, rho)
        x = np.cos(self.eta)
        y = np.sin(self.eta)
        return {
            "eta": self.eta.copy(),
            "x": x,
            "y": y,
            "mach": mach[0, :].copy(),
            "cp": cp[0, :].copy(),
        }

    def full_fields(self) -> Dict[str, np.ndarray]:
        phi = self.apply_boundary_conditions(self.phi)
        rho = self.compute_density(phi, use_biasing=False)
        mach = self.compute_local_mach(phi)
        cp = self.compute_cp(phi, rho)
        return {
            "X": self.X.copy(),
            "Y": self.Y.copy(),
            "phi": phi.copy(),
            "rho": rho.copy(),
            "mach": mach.copy(),
            "cp": cp.copy(),
        }


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
