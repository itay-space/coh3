import numpy as np
from scipy.special import spherical_jn, spherical_yn
from scipy.integrate import solve_ivp

class Scatter:
    """
    A class encapsulating basic scattering computations
    with 'per-object' parameters (masses, energies, quantum numbers),
    while keeping certain physical constants (cso, amu, hbarc) at the class level.
    """

    # Class-level constants (physical constants)
    cso   = 2.04553
    amu   = 931.5
    hbarc = 197.3

    def __init__(self, m1_amu, m2_amu, Ein_lab, j, l, s , V0, r0, a0, Vso0, rso0, aso0, A):
        """
        Parameters
        ----------
        m1_amu : float
            Mass of projectile in amu.
        m2_amu : float
            Mass of target in amu.
        Ein_lab : float
            Incident (laboratory frame) energy in MeV.
        j, l, s : float
            Total, orbital, and spin angular momenta (quantum numbers).
        """
        self.m1_amu  = m1_amu
        self.m2_amu  = m2_amu
        self.Ein_lab = Ein_lab
        self.j       = j
        self.l       = l
        self.s       = s
        self.V0 = V0
        self.r0 = r0
        self.a0 = a0
        self.Vso0 = Vso0
        self.rso0 = rso0
        self.aso0 = aso0
        self.A = A
        self.so_term = self.j*(self.j+1) - self.l*(self.l+1) - self.s*(self.s+1)

        # Compute channel params for this object
        self.c1, self.Ec = self._channel_params()
        self.kc = np.sqrt(self.c1 * self.Ec)
        # Compute R0
        self.R0 = self.r0 * (self.A**(1.0/3.0))
        self.Rso0 = self.rso0 * (self.A**(1.0/3.0))

        # Compute matchRad
        self.matchRad = self.R0 + self.a0 * np.log(1e6 * self.V0 / self.Ec)
        self.beta_matchRad = self.beta(self.matchRad)
        self.U_matchRad = self.Ufunc(self.matchRad)
        self.T = 1-abs(self.U_matchRad)**2

    def _channel_params(self):
        """
        Compute c1 and Ec, stored as instance attributes.
          c1 = 2 * mu(MeV) / (hbar*c)^2
          Ec = center-of-mass energy in MeV
        """
        mu_amu = (self.m1_amu * self.m2_amu) / (self.m1_amu + self.m2_amu)  # reduced mass in amu
        c1 = 2.0 * (mu_amu * Scatter.amu) / (Scatter.hbarc**2)
        Ec = self.Ein_lab * (self.m1_amu / (self.m1_amu + self.m2_amu))
        return c1, Ec

    # -----------------------------------------------------------------------
    #  Spherical Hankel/Bessel and related functions (use self.l as needed).
    # -----------------------------------------------------------------------
    def spherical_hankel1(self, n, x):
        """Spherical Hankel function of the first kind."""
        return spherical_jn(n, x) + 1j * spherical_yn(n, x)

    def spherical_hankel2(self, n, x):
        """Spherical Hankel function of the second kind."""
        return spherical_jn(n, x) - 1j * spherical_yn(n, x)

    def spherical_hankel1_derivative(self, n, x):
        """Derivative of spherical Hankel function of the first kind."""
        return (spherical_jn(n, x, derivative=True)
                + 1j * spherical_yn(n, x, derivative=True))

    def spherical_hankel2_derivative(self, n, x):
        """Derivative of spherical Hankel function of the second kind."""
        return (spherical_jn(n, x, derivative=True)
                - 1j * spherical_yn(n, x, derivative=True))

    def Ifunc(self, r):
        """
        Incoming wave function I(r) = -i * [kc*r * h2_l(kc*r)].
        Uses self.l as the angular momentum.
        """
        return -1j * self.kc * r * self.spherical_hankel2(self.l, self.kc * r)

    def Ofunc(self, r):
        """
        Outgoing wave function O(r) = i * [kc*r * h1_l(kc*r)].
        Uses self.l as the angular momentum.
        """
        return 1j * self.kc * r * self.spherical_hankel1(self.l, self.kc * r)

    def Ifunc_derivative(self, r):
        """
        Derivative d/dr of Ifunc(r, self.kc).
        """
        h2  = self.spherical_hankel2(self.l, self.kc * r)
        dh2 = self.spherical_hankel2_derivative(self.l, self.kc * r)
        return -1j * self.kc * (h2 + self.kc*r * dh2)

    def Ofunc_derivative(self, r):
        """
        Derivative d/dr of Ofunc(r).
        """
        h1  = self.spherical_hankel1(self.l, self.kc * r)
        dh1 = self.spherical_hankel1_derivative(self.l, self.kc * r)
        return 1j * self.kc * (h1 + self.kc*r * dh1)

    # -----------------------------------------------------------------------
    #              Potential Functions (e.g., Woods-Saxon, etc.)
    @staticmethod
    def woods_saxon_potential(R, a, r):
        """Woods-Saxon form factor: 1 / [1 + exp((r-R)/a)]"""
        return 1.0 / (1.0 + np.exp((r - R) / a))

    @staticmethod
    def thomas_potential(R, a, r):
        """
        Thomas form factor (derivative of Woods-Saxon).
        """
        r = np.asarray(r, dtype=float)
        out = np.zeros_like(r)
        mask = (r != 0)
        y = np.exp((r[mask] - R) / a)
        z = 1.0 + y
        out[mask] = y / (r[mask]*a * z*z)
        return out

    def Vr(self, r):
        """
        Central real WS factor: V(r) = Woods-Saxon(R0, a0).
        R0 = self.r0 * A^(1/3).
        """
        return self.woods_saxon_potential(self.R0, self.a0, r)

    def Vso(self, r):
        """
        Spin-orbit WS factor: derivative (Thomas).
        Rso0 = self.rso0 * A^(1/3).
        """
        
        return self.thomas_potential(self.Rso0, self.aso0, r)

    def total_potential(self, r):
        """
        Total potential:
          V(r) = V0 * Vr(r)
               + Vso0 * cso * Vso(r) * [ j(j+1) - l(l+1) - s(s+1) ].

        Uses self.V0, self.r0, self.a0, etc. and the precomputed self.so_term.
        """
        return (
            self.V0 * self.Vr(r)
            + self.Vso0 * Scatter.cso * self.Vso(r) * self.so_term
        )
    



    def schrodinger_eq(self, r, u):
        """
        Schrödinger equation in radial form:
        
        u''(r) = c1 * u(r) * (-Ec + l(l+1)/(c1*r^2) - V_total(r)).
        """
        return self.c1 * u * (-self.Ec + (self.l * (self.l + 1)) / (self.c1 * r**2) - self.total_potential(r))

    def differential_equation(self, r, y):
        """
        Converts the second-order Schrödinger equation into a first-order system
        suitable for numerical integration.
        """
        u, up = y  # y = [u, u']
        return [up, self.schrodinger_eq(r, u)]

    def solve_schrodinger(self):
        """
        Solve the radial Schrödinger equation using `solve_ivp` with the `Radau` method.
        """
        start = 0.001
        end = self.matchRad * 1.1
        meshN = int(self.matchRad / start)
        r_vals = np.linspace(start, end, meshN)

        # Initial conditions: u(0) = 0, u'(0) = 1
        y0 = [0, 1]

        sol = solve_ivp(
            self.differential_equation,
            [start, end],
            y0,
            t_eval=r_vals,
            method="Radau"
        )
        return sol

    def beta(self, r):
        """
        Compute β(r) = r * u'(r) / u(r), using interpolated solution.
        """
        sol = self.solve_schrodinger()
        
        # Ensure r is an array to use np.interp
        r = np.asarray(r, dtype=float)
        
        u_interp = np.interp(r, sol.t, sol.y[0])
        up_interp = np.interp(r, sol.t, sol.y[1])
        
        # Avoid division by zero
        return np.where(u_interp != 0, r * up_interp / u_interp, np.nan)
    
    def Ufunc(self, r):
        """Computes U function."""
        numerator = self.beta(r) * self.Ifunc(r) - r * self.Ifunc_derivative(r)
        denominator = self.beta(r) * self.Ofunc(r) - r * self.Ofunc_derivative(r)
        return numerator / denominator