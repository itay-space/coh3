import numpy as np
from scipy.special import spherical_jn, spherical_yn
from scipy.integrate import solve_ivp
import pandas as pd

import os
import re
import ast


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
        self.compare_dir = ""
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
    
    def u(self,r):
        sol = self.solve_schrodinger()
        r = np.asarray(r, dtype=float)
        return np.interp(r, sol.t, sol.y[0])

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
    



    def load_merge_and_pivot_data(self,L_values=None, s_values=None, j_values=None):
        """
        Reads optical potential and wavefunction files from a directory, merges them on the common
        radial coordinate 'r' and quantum numbers ('l', 's', 'j'), and pivots the merged DataFrame
        so that each unique r is a single row. The columns become a MultiIndex with the variable names
        (e.g., 'Real', 'Imag', 'Re_psi', 'Im_psi', 'Abs_psi') and the corresponding quantum numbers.

        Optical potential files should be named as:
        opticalpotential_l_{l}_pspin_{s}_xj_{j}.dat
        and contain two columns:
        - r (float)
        - complex potential as a string (e.g., "(real, imag)")

        Wavefunction files should be named as:
        intwavefunction_l_{l}_pspin_{s}_xj_{j}.dat
        and contain four columns:
        - r (float)
        - Re(ψ)
        - Im(ψ)
        - |ψ|
        
        Optional filters (L_values, s_values, j_values) can be provided.

        Returns:
        final_pivot: pd.DataFrame where the index is r and columns are a MultiIndex with
                    levels [variable, l, s, j]. For instance, ('Real', 1.0, 0.5, 1.5) will hold
                    the optical potential's real part for that quantum number combination.
        """
        data_dir = self.compare_dir
        # Patterns to extract quantum numbers from filenames
        opt_pattern = re.compile(r"^opticalpotential_l_(?P<l>\S+)_pspin_(?P<s>\S+)_xj_(?P<j>\S+)\.dat$")
        wfn_pattern = re.compile(r"^intwavefunction_l_(?P<l>\S+)_pspin_(?P<s>\S+)_xj_(?P<j>\S+)\.dat$")
        
        all_files = os.listdir(data_dir)
        optical_list = []
        wavefunction_list = []
        
        # Process optical potential files
        opt_files = [f for f in all_files if opt_pattern.match(f)]
        for filename in opt_files:
            m = opt_pattern.match(filename)
            if not m:
                continue
            l_str, s_str, j_str = m.group("l"), m.group("s"), m.group("j")
            try:
                l_val = float(l_str)
                s_val = float(s_str)
                j_val = float(j_str)
            except ValueError:
                continue
            
            # Apply optional filters
            if L_values is not None and l_val not in L_values:
                continue
            if s_values is not None and s_val not in s_values:
                continue
            if j_values is not None and j_val not in j_values:
                continue
            
            path = os.path.join(data_dir, filename)
            try:
                # Expect two columns: r and a string for the complex potential
                df = pd.read_csv(path, delim_whitespace=True, header=None)
            except Exception as e:
                print(f"Cannot read {path}: {e}")
                continue
            
            if df.shape[1] < 2:
                print(f"Skipping {filename}: not 2 columns.")
                continue
            
            try:
                df[0] = df[0].astype(float)
            except Exception as e:
                print(f"Error converting r to float in {filename}: {e}")
                continue
            
            try:
                # Parse the complex potential from string to a complex number
                df[1] = df[1].apply(lambda x: complex(*ast.literal_eval(x)))
            except Exception as e:
                print(f"Error parsing complex number in {filename}: {e}")
                continue
            
            # Extract real and imaginary parts; rename first column as 'r'
            df["Real"] = df[1].apply(lambda c: c.real)
            df["Imag"] = df[1].apply(lambda c: c.imag)
            df.rename(columns={0: "r"}, inplace=True)
            
            # Add quantum number columns
            df["l"] = l_val
            df["s"] = s_val
            df["j"] = j_val
            
            # Keep only needed columns
            optical_list.append(df[["r", "Real", "Imag", "l", "s", "j"]])
        
        # Process wavefunction files
        wfn_files = [f for f in all_files if wfn_pattern.match(f)]
        for filename in wfn_files:
            m = wfn_pattern.match(filename)
            if not m:
                continue
            l_str, s_str, j_str = m.group("l"), m.group("s"), m.group("j")
            try:
                l_val = float(l_str)
                s_val = float(s_str)
                j_val = float(j_str)
            except ValueError:
                continue
            
            # Apply optional filters
            if L_values is not None and l_val not in L_values:
                continue
            if s_values is not None and s_val not in s_values:
                continue
            if j_values is not None and j_val not in j_values:
                continue
            
            path = os.path.join(data_dir, filename)
            try:
                data = np.loadtxt(path)
            except Exception as e:
                print(f"Cannot read {path}: {e}")
                continue
            
            if data.ndim == 1 or data.shape[1] < 4:
                print(f"Skipping {filename}: not 4 columns.")
                continue
            
            # Expect four columns: r, Re_psi, Im_psi, Abs_psi
            r, Re_psi, Im_psi, Abs_psi = data.T
            
            df_wfn = pd.DataFrame({
                "r": r,
                "Re_psi": Re_psi,
                "Im_psi": Im_psi,
                "Abs_psi": Abs_psi,
                "l": l_val,
                "s": s_val,
                "j": j_val
            })
            wavefunction_list.append(df_wfn)
        
        # Concatenate the dataframes from individual files
        optical_df = pd.concat(optical_list, ignore_index=True) if optical_list else pd.DataFrame()
        wavefunction_df = pd.concat(wavefunction_list, ignore_index=True) if wavefunction_list else pd.DataFrame()
        
        # Merge the two DataFrames on common keys: r, l, s, j
        merged_df = pd.merge(optical_df, wavefunction_df, on=["r", "l", "s", "j"], how="outer", 
                            suffixes=("_opt", "_wfn"))
        
        # Pivot the optical potential variables
        optical_pivot = merged_df.pivot_table(
            index="r",
            columns=["l", "s", "j"],
            values=["Real", "Imag"],
            aggfunc="first"
        )
        
        # Pivot the wavefunction variables
        wfn_pivot = merged_df.pivot_table(
            index="r",
            columns=["l", "s", "j"],
            values=["Re_psi", "Im_psi", "Abs_psi"],
            aggfunc="first"
        )
        
        # Join the two pivoted DataFrames (they share the same index r)
        final_pivot = optical_pivot.join(wfn_pivot)
        
        # Optionally, sort the index
        final_pivot.sort_index(inplace=True)
        
        return final_pivot

    def file_wfn_and_pot(self):
        return self.load_merge_and_pivot_data(L_values=[self.l], s_values=[self.s], j_values=[self.j])
    def u_and_v_df(self ,r):  
        return pd.DataFrame({"u_real": np.real(self.u(r)), "u_imag": np.imag(self.u(r)), "tp_real": np.real(-self.c1 * self.total_potential(r)), "tp_imag": np.imag(-self.c1 * self.total_potential(r))} , index = r)

    def get_functions_for_quantum_numbers(self,l_val, s_val, j_val):
       
        idx = pd.IndexSlice
        pivot_df = self.load_merge_and_pivot_data()
        result = pivot_df.loc[:, idx[:, l_val, s_val, j_val]]
        result.columns = result.columns.droplevel([1, 2, 3])
        return result