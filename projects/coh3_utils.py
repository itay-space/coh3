import os
import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, spherical_yn
cso = 2.04553
amu=931.5
hbarc=197.3
def channel_params(m1_amu, m2_amu, Ein_lab):
    """
    Compute c1 and Ec for two-body scattering.

    m1_amu, m2_amu: masses in amu
    Ein_lab: incident energy in MeV (lab frame)
    u: MeV per amu (default 931.5)
    hbarc: MeV*fm (default 197.3)

    Returns
    -------
    c1: float
        2 * reduced_mass(MeV) / (hbar*c)^2 [MeV/fm^2]
    Ec: float
        center-of-mass energy in MeV for the incident particle [MeV]
    """
    # Reduced mass in amu
    mu_amu = m1_amu * m2_amu / (m1_amu + m2_amu)
    # c1 and Ec
    c1 = 2 * (mu_amu * amu) / (hbarc**2)
    Ec = Ein_lab * (m1_amu / (m1_amu + m2_amu))
    return c1, Ec

def spherical_hankel1(n, x):
    """Computes the spherical Hankel function of the first kind."""
    return spherical_jn(n, x) + 1j * spherical_yn(n, x)

def spherical_hankel2(n, x):
    """Computes the spherical Hankel function of the second kind."""
    return spherical_jn(n, x) - 1j * spherical_yn(n, x)

def spherical_hankel1_derivative(n, x):
    """Computes the derivative of the spherical Hankel function of the first kind."""
    return spherical_jn(n, x, derivative=True) + 1j * spherical_yn(n, x, derivative=True)

def spherical_hankel2_derivative(n, x):
    """Computes the derivative of the spherical Hankel function of the second kind."""
    return spherical_jn(n, x, derivative=True) - 1j * spherical_yn(n, x, derivative=True)

def Ifunc(r, kc, l):
    """Computes I function."""
    return -1j * kc * r * spherical_hankel2(l, kc * r)

def Ofunc(r, kc, l):
    """Computes O function."""
    return 1j * kc * r * spherical_hankel1(l, kc * r)

def Ifunc_derivative(r, kc, l):
    """Computes the derivative of I function."""
    hankel2_deriv = spherical_hankel2_derivative(l, kc * r)
    return -1j * kc * (spherical_hankel2(l, kc * r) + kc * r * hankel2_deriv)

def Ofunc_derivative(r, kc, l):
    """Computes the derivative of O function."""
    hankel1_deriv = spherical_hankel1_derivative(l, kc * r)
    return 1j * kc * (spherical_hankel1(l, kc * r) + kc * r * hankel1_deriv)

def woods_saxon_potential(R, a, r):
    return 1/ (1 + np.exp((r - R) / a))
# Define the potential function for Fe-56
def Thomas_potential(R, a, r):
    r = np.asarray(r, dtype=float)
    out = np.zeros_like(r)
    mask = (r != 0)
    y = np.exp((r[mask] - R) / a)
    z = 1 + y
    out[mask] = y / (r[mask] * a * z * z)
    return out

def Vr(r, A,r0 , a0):
    R0 = r0 * A**(1/3)
    return woods_saxon_potential(R0, a0, r)

def Vso(r,A,rso0 , aso0 ):
    Rso0 = rso0 * A**(1/3)
    return Thomas_potential(Rso0, aso0, r) 

def TotalV(r,j,l,s, V0 ,r0,a0  , Vso0 , rso0 , aso0 ,A):
    so_term = (j*(j+1)  - l*(l+1) - s*(s+1)) ## why not 1/2
    #print(so_term)
    return V0 * Vr(r,r0,a0,A) + Vso0 * cso * Vso(r,rso0,aso0,A) * so_term

def plot_optical_and_wavefunctions(
    data_dir,TotalV,
    L_values=None,
    pspin_values=None,
    xj_values=None
):
    """
    Reads:
      1) opticalpotential_l_{l}_pspin_{pspin}_xj_{xj}.dat
         - two columns: radius (float), "(real, imag)" as string
         -> Plots file's Real/Imag parts and compares with TotalV(r, j, l, s).

      2) intwavefunction_l_{l}_pspin_{pspin}_xj_{xj}.dat
         - four columns: R, Re(ψ), Im(ψ), |ψ|
         -> Plots |ψ| from file and compares with solve_schrodinger(j, l, s).

    Both 'solve_schrodinger' and 'TotalV' must be defined or imported elsewhere.
    """

    

    # Regex patterns
    opt_pattern = re.compile(r"^opticalpotential_l_(?P<l>\S+)_pspin_(?P<pspin>\S+)_xj_(?P<xj>\S+)\.dat$")
    wfn_pattern = re.compile(r"^intwavefunction_l_(?P<l>\S+)_pspin_(?P<pspin>\S+)_xj_(?P<xj>\S+)\.dat$")

    all_files = os.listdir(data_dir)

    # ---------------------------------------
    # 1) Optical Potentials + TotalV
    # ---------------------------------------
    opt_files = [f for f in all_files if opt_pattern.match(f)]
    if opt_files:
        plt.figure(figsize=(10, 6))
        for filename in opt_files:
            m = opt_pattern.match(filename)
            if not m:
                continue

            l_str, s_str, j_str = m.group("l"), m.group("pspin"), m.group("xj")
            try:
                l_val  = float(l_str)
                s_val  = float(s_str)  # spin
                j_val  = float(j_str)  # total ang. momentum
            except ValueError:
                continue

            # User filters
            if L_values is not None and l_val not in L_values:
                continue
            if pspin_values is not None and s_val not in pspin_values:
                continue
            if xj_values is not None and j_val not in xj_values:
                continue

            path = os.path.join(data_dir, filename)
            try:
                df = pd.read_csv(path, delim_whitespace=True, header=None)
            except Exception as e:
                print(f"Cannot read {path}: {e}")
                continue
            if df.shape[1] < 2:
                print(f"Skipping {filename}: not 2 columns.")
                continue

            # Parse radius, complex potential
            df[0] = df[0].astype(float)
            try:
                df[1] = df[1].apply(lambda x: complex(*ast.literal_eval(x)))
            except Exception as e:
                print(f"Error parsing complex in {filename}: {e}")
                continue

            df["Real_file"] = df[1].apply(lambda c: c.real)
            df["Imag_file"] = df[1].apply(lambda c: c.imag)
            df.set_index(0, inplace=True)
            radii = df.index.values

            # Compare with your TotalV(r, j, l, s):
            #  (Must be defined or imported in your code.)
            V_calc = [TotalV(r, j_val, l_val, s_val) for r in radii]
            real_calc = [v.real for v in V_calc]
            imag_calc = [v.imag for v in V_calc]

            lbl = f"j={j_val}, l={l_val}, s={s_val}"
            plt.plot(radii, df["Real_file"], label=f"{lbl}, file Re")
            plt.plot(radii, df["Imag_file"], label=f"{lbl}, file Im")
            plt.plot(radii, real_calc, "--", label=f"{lbl}, TotalV Re")
            plt.plot(radii, imag_calc, "--", label=f"{lbl}, TotalV Im")

        plt.title("Optical Potential vs. TotalV")
        plt.xlabel("Radius")
        plt.ylabel("Potential")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
    else:
        print(f"No opticalpotential files found in {data_dir}")

    # ---------------------------------------
    # 2) Wavefunctions + solve_schrodinger
    # ---------------------------------------
    wfn_files = [f for f in all_files if wfn_pattern.match(f)]
    if wfn_files:
        plt.figure(figsize=(10, 6))
        for filename in wfn_files:
            m = wfn_pattern.match(filename)
            if not m:
                continue

            l_str, s_str, j_str = m.group("l"), m.group("pspin"), m.group("xj")
            try:
                l_val = float(l_str)
                s_val = float(s_str)
                j_val = float(j_str)
            except ValueError:
                continue

            # Filters
            if L_values is not None and l_val not in L_values:
                continue
            if pspin_values is not None and s_val not in pspin_values:
                continue
            if xj_values is not None and j_val not in xj_values:
                continue

            path = os.path.join(data_dir, filename)
            try:
                data = np.loadtxt(path)
            except Exception as e:
                print(f"Cannot read {path}: {e}")
                continue
            if data.shape[1] < 4:
                print(f"Skipping {filename}: not 4 columns.")
                continue

            R, Re_psi, Im_psi, Abs_psi = data.T
            norm_f = np.max(Abs_psi) or 1.0
            lbl_wfn = f"File j={j_val}, l={l_val}, s={s_val}"
            plt.plot(R, Abs_psi / norm_f, label=lbl_wfn, alpha=0.7)

            # Compare with your solver:
            #  (Must be defined or imported in your code.)
            try:
                sol = solve_schrodinger(j=j_val, l=l_val, s=s_val)
                wfn_ode = sol.y[0]
                r_ode   = sol.t
                norm_ode = np.max(np.abs(wfn_ode)) or 1.0
                plt.plot(r_ode, np.abs(wfn_ode)/norm_ode, "--",
                         label=f"Solver j={j_val}, l={l_val}, s={s_val}",
                         alpha=0.7)
            except Exception as ex:
                print(f"Error solve_schrodinger(j={j_val},l={l_val},s={s_val}): {ex}")

        plt.title("Wavefunction vs. Schrödinger Solver")
        plt.xlabel("Radius")
        plt.ylabel("|ψ| (normalized)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
    else:
        print(f"No wavefunction files found in {data_dir}")

    # Show both figures
    plt.show()
