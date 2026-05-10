import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time

class SubsonicCylinderSolver:
    def __init__(self, Nxi=60, Neta=60, xi_max=3.0, Minf=0.3, gamma=1.4):
        self.Nxi  = Nxi
        self.Neta = Neta
        self.xi_max = xi_max
        self.Minf  = Minf
        self.gamma = gamma

        self.dxi  = xi_max / Nxi
        self.deta = np.pi  / Neta

        self.xi  = np.linspace(0, xi_max, Nxi + 1)
        self.eta = np.linspace(0, np.pi,  Neta + 1)
        self.Xi, self.Eta = np.meshgrid(self.xi, self.eta, indexing='ij')

        self.Phi = np.zeros((Nxi + 1, Neta + 1))
        self.rho = np.ones ((Nxi + 1, Neta + 1))

        # Incompressible starting field: phi = (e^xi + e^-xi) cos(eta)
        self.Phi_incomp = (np.exp(self.Xi) + np.exp(-self.Xi)) * np.cos(self.Eta)
        self.Phi = self.Phi_incomp.copy()

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------
    def _apply_bcs(self, Phi):
        """
        Right boundary (i = Nxi):  Dirichlet – fixed infinity values.
        All other boundaries:  5-point cross scheme with symmetry ghost nodes.
          Left  (i=0, cylinder) : Phi[-1,j] = Phi[+1,j]  (i-1 = i+1)
          Bottom (j=0)          : Phi[i,-1] = Phi[i,+1]  (j-1 = j+1)
          Top    (j=Neta)       : Phi[i,Neta+1] = Phi[i,Neta-1]  (j+1 = j-1)
        For the cross equation we expand:
          Left   : 2*rho*Phi[1,:]/dxi^2 side (effectively Phi[0] appears in interior cross with ghost)
          The ghost-node value is set explicitly before interior sweeps.
        """
        # Right boundary: Dirichlet
        Phi[self.Nxi, :] = np.exp(self.xi_max) * np.cos(self.eta)
        return Phi

    def _apply_symmetry_bcs(self, Phi):
        """Apply symmetry ghost-cell equalities used inside the stencil."""
        # Left boundary (cylinder): ghost cell i=-1 equals i=1
        # Phi[0, :] is an interior node for the cross stencil, not a ghost;
        # but we need to close the stencil: Phi[-1, j] = Phi[1, j] => handled in stencil explicitly
        # Bottom: ghost j=-1 = j=1
        # Top   : ghost j=Neta+1 = j=Neta-1
        # These are not stored; they are used inline in the solvers.
        pass

    # ------------------------------------------------------------------
    # Density
    # ------------------------------------------------------------------
    def _update_density(self, Phi):
        dPhi_dxi  = np.zeros_like(Phi)
        dPhi_deta = np.zeros_like(Phi)

        # Interior: central differences
        dPhi_dxi [1:-1, :]  = (Phi[2:,  :] - Phi[:-2,  :]) / (2*self.dxi)
        dPhi_deta[:, 1:-1]  = (Phi[:,  2:] - Phi[:,  :-2]) / (2*self.deta)

        # Left (cylinder) – symmetry: d/dxi uses Phi[1] and Phi[-1]=Phi[1]
        # => one-sided forward (first order) is equivalent to (Phi[1]-Phi[-1])/(2h) = 0 (no-flux)
        # but density gradient still needed from Phi values, use forward:
        dPhi_dxi[0, :]  = (Phi[1, :] - Phi[0, :]) / self.dxi  # 1st order, consistent with zero-flux BC
        dPhi_dxi[-1, :] = (3*Phi[-1, :] - 4*Phi[-2, :] + Phi[-3, :]) / (2*self.dxi)

        # Bottom / top symmetry
        dPhi_deta[:, 0]    = (Phi[:, 1] - Phi[:, 0]) / self.deta
        dPhi_deta[:, -1]   = (Phi[:, -2] - Phi[:, -1]) / (-self.deta)

        q2 = np.exp(-2*self.Xi) * (dPhi_dxi**2 + dPhi_deta**2)
        base = 1.0 + 0.5*(self.gamma-1)*self.Minf**2*(1.0 - q2)
        base = np.maximum(base, 1e-5)
        rho  = base**(1.0/(self.gamma-1))
        return rho, q2

    def update_density(self, Phi):
        return self._update_density(Phi)

    # ------------------------------------------------------------------
    # Helper: get half-point densities for node (i,j)
    # with symmetry BCs for boundary nodes
    # ------------------------------------------------------------------
    def _half_rho(self, rho, i, j):
        """
        Returns rho_{i+1/2,j}, rho_{i-1/2,j}, rho_{i,j+1/2}, rho_{i,j-1/2}
        using symmetry ghost cells on boundaries.
        """
        Nxi, Neta = self.Nxi, self.Neta

        # i neighbours with symmetry (i=0: ghost i=-1 = i=1)
        i_prev = i-1 if i > 0 else 1          # symmetry
        i_next = i+1 if i < Nxi else i-1      # shouldn't be needed for interior

        j_prev = j-1 if j > 0 else 1          # symmetry j=-1 = j=1
        j_next = j+1 if j < Neta else j-1     # symmetry j=Neta+1 = j=Neta-1

        r_ep = 0.5*(rho[i_next, j] + rho[i, j])
        r_em = 0.5*(rho[i_prev, j] + rho[i, j])
        r_np = 0.5*(rho[i, j_next] + rho[i, j])
        r_nm = 0.5*(rho[i, j_prev] + rho[i, j])
        return r_ep, r_em, r_np, r_nm

    # ------------------------------------------------------------------
    # SLOR – Successive Line Over-Relaxation
    # Rows (j = const) are swept, solving tridiagonal system in i for each row
    # ------------------------------------------------------------------
    def solve_slor(self, max_iter=3000, tol=1e-6, omega=1.5):
        """
        Successive Line Over-Relaxation (строчная релаксация по строкам).
        На каждой итерации для каждой строки j=const решается ДСЛА
        с трехдиагональной матрицей методом прогонки (алгоритм Томаса).
        Соседние строки j±1 берутся с предыдущей итерации.
        """
        print(f"Starting SLOR solver (Minf={self.Minf}, Grid={self.Nxi}x{self.Neta})")
        start = time.time()
        self.Phi = self.Phi_incomp.copy()
        self._apply_bcs(self.Phi)

        history = []

        for k in range(max_iter):
            Phi_old = self.Phi.copy()
            rho, _ = self._update_density(self.Phi)

            # Sweep over rows j = 0 .. Neta
            for j in range(0, self.Neta + 1):
                # Ghost column indices for symmetry BCs
                j_prev = j-1 if j > 0 else 1
                j_next = j+1 if j < self.Neta else self.Neta - 1

                # Build tridiagonal system for row j, unknowns i=0..Nxi
                n = self.Nxi + 1
                lo  = np.zeros(n)   # sub-diagonal  (i-1)
                di  = np.zeros(n)   # diagonal      (i)
                hi  = np.zeros(n)   # super-diagonal (i+1)
                rhs = np.zeros(n)

                for i in range(n):
                    # Right boundary: Dirichlet
                    if i == self.Nxi:
                        di[i]  = 1.0
                        rhs[i] = np.exp(self.xi_max) * np.cos(self.eta[j])
                        continue

                    # Left boundary (cylinder): symmetry i-1 = i+1
                    if i == 0:
                        i_next_i = 1
                        i_prev_i = 1  # ghost = i+1 (symmetry)
                    else:
                        i_prev_i = i - 1
                        i_next_i = i + 1 if i < self.Nxi else i - 1

                    r_ep, r_em, r_np, r_nm = self._half_rho(rho, i, j)
                    dx2 = self.dxi**2
                    dy2 = self.deta**2

                    # Coefficient from j-direction (known from previous iter):
                    rhs_j = (r_np * self.Phi[i, j_next] + r_nm * self.Phi[i, j_prev]) / dy2

                    # Diagonal system in i-direction:
                    # r_ep*(Phi[i+1]-Phi[i])/dx2 - r_em*(Phi[i]-Phi[i-1])/dx2
                    # + rhs_j - (r_np + r_nm)/dy2 * Phi[i] = 0
                    a_center = -(r_ep + r_em)/dx2 - (r_np + r_nm)/dy2

                    if i == 0:
                        # Symmetry: both i-1 and i+1 contribute to i+1 coefficient
                        hi[i]  =  (r_ep + r_em) / dx2   # combined coefficient for Phi[1]
                        di[i]  = a_center
                        rhs[i] = -rhs_j
                    else:
                        lo[i]  = r_em / dx2   # Phi[i-1]
                        hi[i]  = r_ep / dx2   # Phi[i+1]
                        di[i]  = a_center
                        rhs[i] = -rhs_j

                # Thomas algorithm (tridiagonal solve)
                Phi_line = self._thomas(lo, di, hi, rhs)

                # Apply SOR relaxation
                self.Phi[:, j] = (1 - omega)*self.Phi[:, j] + omega*Phi_line

            self._apply_bcs(self.Phi)

            diff = np.max(np.abs(self.Phi - Phi_old))
            history.append(diff)

            if diff < tol:
                print(f"SLOR converged in {k+1} iters, diff={diff:.2e}, time={time.time()-start:.2f}s")
                self.rho, _ = self._update_density(self.Phi)
                return True, history

            if k % 200 == 0:
                print(f"  SLOR iter {k:4d}, diff={diff:.2e}")

        print(f"SLOR did NOT converge (last diff={history[-1]:.2e})")
        self.rho, _ = self._update_density(self.Phi)
        return False, history

    # ------------------------------------------------------------------
    # Thomas algorithm (tridiagonal)
    # ------------------------------------------------------------------
    @staticmethod
    def _thomas(a, b, c, d):
        """Solve Ax=d where A is tridiagonal: a[i]*x[i-1]+b[i]*x[i]+c[i]*x[i+1]=d[i]"""
        n = len(d)
        c_ = np.zeros(n)
        d_ = np.zeros(n)
        x  = np.zeros(n)

        c_[0] = c[0] / b[0]
        d_[0] = d[0] / b[0]
        for i in range(1, n):
            m    = a[i] / (b[i] - a[i]*c_[i-1])
            c_[i] = c[i] / (b[i] - a[i]*c_[i-1])
            d_[i] = (d[i] - a[i]*d_[i-1]) / (b[i] - a[i]*c_[i-1])
        x[-1] = d_[-1]
        for i in range(n-2, -1, -1):
            x[i] = d_[i] - c_[i]*x[i+1]
        return x

    # ------------------------------------------------------------------
    # Newton FIM (Полностью неявный метод)
    # ------------------------------------------------------------------
    def solve_newton(self, max_iter=40, tol=1e-8):
        """Fully Implicit Method (ПНМ) via Newton linearisation."""
        print(f"Starting Newton FIM (Minf={self.Minf}, Grid={self.Nxi}x{self.Neta})")
        start = time.time()
        self.Phi = self.Phi_incomp.copy()
        self._apply_bcs(self.Phi)

        Ntot = (self.Nxi+1)*(self.Neta+1)
        idx  = lambda i, j: i*(self.Neta+1) + j

        history = []

        for k in range(max_iter):
            rho, _ = self._update_density(self.Phi)

            rows, cols, vals = [], [], []
            F = np.zeros(Ntot)

            for i in range(self.Nxi + 1):
                for j in range(self.Neta + 1):
                    row = idx(i, j)

                    # ---------- Right boundary: Dirichlet ----------
                    if i == self.Nxi:
                        val = np.exp(self.xi_max)*np.cos(self.eta[j])
                        F[row]  = self.Phi[i, j] - val
                        rows.append(row); cols.append(row); vals.append(1.0)
                        continue

                    # Ghost indices for symmetry BCs
                    i_prev = i-1 if i > 0 else 1          # left: ghost=+1
                    i_next = i+1 if i < self.Nxi else i-1

                    j_prev = j-1 if j > 0 else 1          # bottom: ghost=+1
                    j_next = j+1 if j < self.Neta else self.Neta-1  # top: ghost=Neta-1

                    r_ep, r_em, r_np, r_nm = self._half_rho(rho, i, j)

                    dx2 = self.dxi**2
                    dy2 = self.deta**2

                    flux_e = r_ep*(self.Phi[i_next, j] - self.Phi[i, j]) / dx2
                    flux_w = r_em*(self.Phi[i, j]      - self.Phi[i_prev, j]) / dx2
                    flux_n = r_np*(self.Phi[i, j_next] - self.Phi[i, j]) / dy2
                    flux_s = r_nm*(self.Phi[i, j]      - self.Phi[i, j_prev]) / dy2

                    F[row] = flux_e - flux_w + flux_n - flux_s

                    # Picard Jacobian (freeze density)
                    dFdPhiC = -(r_ep + r_em)/dx2 - (r_np + r_nm)/dy2
                    rows.append(row); cols.append(row); vals.append(dFdPhiC)

                    # i+1 / i-1
                    rows.append(row); cols.append(idx(i_next, j)); vals.append( r_ep/dx2)
                    # For i=0 symmetry: i_prev==i_next, so we must ADD coefficient
                    rows.append(row); cols.append(idx(i_prev, j)); vals.append(-r_em/dx2)
                    # j+1 / j-1
                    rows.append(row); cols.append(idx(i, j_next)); vals.append( r_np/dy2)
                    rows.append(row); cols.append(idx(i, j_prev)); vals.append(-r_nm/dy2)

            J = sp.csr_matrix((vals, (rows, cols)), shape=(Ntot, Ntot))
            dPhi_vec = spla.spsolve(J, -F)
            delta_Phi = dPhi_vec.reshape((self.Nxi+1, self.Neta+1))
            self.Phi += delta_Phi
            self._apply_bcs(self.Phi)

            max_d  = np.max(np.abs(delta_Phi))
            norm_F = np.linalg.norm(F)
            history.append(max_d)
            print(f"  Iter {k:2d}: max|dPhi|={max_d:.2e}, ||F||={norm_F:.2e}")

            if max_d < tol:
                print(f"Newton FIM converged in {k+1} iters. Time={time.time()-start:.2f}s")
                self.rho, _ = self._update_density(self.Phi)
                return True, history

        print("Newton FIM did NOT converge.")
        self.rho, _ = self._update_density(self.Phi)
        return False, history
