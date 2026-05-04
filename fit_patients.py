"""
Per-patient fitting of PK/PD model to the 16-patient mCRPC adaptive therapy dataset.

Strategy (Option 1):
    - EC50 fixed at population level (15 ng/mL)
    - Fitted per patient: S0, R0, alpha_max
    - All PK parameters fixed at Stuyckens 2014 population means
    - delta (PSA production rate) derived from initial PSA and initial cell
      fractions so that simulated PSA(0) = observed PSA(0) exactly

PK approximation for fitting:
    PSA observations are every 4-6 weeks; the PK half-life is 12 h.
    Drug concentration reaches quasi-steady state (C_ss_avg) within ~3 days
    of starting treatment and falls to ~0 within ~3 days of stopping.
    For fitting, C is approximated as:
        ON  : C = C_ss_avg = dose / (ke * V) * 1e3  [ng/mL]
        OFF : C = 0
    This collapses PK to a constant effective kill rate within each interval,
    which is sufficient given monthly observation resolution.

    C_ss_avg = 1000 mg / (1.386 day^-1 * 26840 L) * 1000 = 26.9 ng/mL
    alpha_eff = alpha_max * C_ss / (EC50 + C_ss) = alpha_max * 0.643  (EC50=15)

Objective:
    Sum of squared log-ratio errors: sum( (log PSA_sim - log PSA_obs)^2 )
    Log scale is appropriate because PSA spans orders of magnitude.

Reference:
    Brady-Nicholls & Enderling (2022); Stuyckens et al. (2014)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import openpyxl
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams.update({'font.family': 'Arial', 'font.size': 12})


# ── Fixed parameters ───────────────────────────────────────────────────────────

P_BASE = {
    'rho_S':   0.027,    # day^-1   sensitive cell proliferation (placeholder)
    'rho_R':   0.003,    # day^-1   resistant cell proliferation (placeholder)
    'K':       1.0,      # carrying capacity (normalized)
    'gamma':   0.08,     # day^-1   PSA clearance
    # PK — Stuyckens et al. 2014, mCRPC population means, 1000 mg QD fasted
    'ka':      39.6,     # day^-1   (not used in QSS fitting; retained for reference)
    'ke':      1.386,    # day^-1   elimination rate constant (from t½ = 12 h)
    'V':       26840.0,  # L        apparent V/F
    'dose_mg': 1000.0,   # mg       abiraterone acetate QD fasted
    # PD — EC50 fixed; alpha_max fitted per patient
    'EC50':    15.0,     # ng/mL    fixed (~0.5 * C_ss_avg)
    'n_hill':  1.0,      # Hill coefficient (fixed)
}

# Quasi-steady-state concentration (derived from PK parameters)
C_SS = P_BASE['dose_mg'] / (P_BASE['ke'] * P_BASE['V']) * 1e3  # ≈ 26.9 ng/mL


# ── Data loading ───────────────────────────────────────────────────────────────

def load_adaptive_patients(filepath):
    """
    Load the adaptive therapy arm from TrialPatientData.xlsx.

    Returns
    -------
    dict: {patient_id: {'days': array, 'psa': array, 'abi': array}}
    """
    wb = openpyxl.load_workbook(filepath)
    ws = wb['Adaptive']

    row2 = [ws.cell(2, c).value for c in range(1, ws.max_column + 1)]
    row1 = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
    day_col_indices = [i for i, v in enumerate(row2) if v == 'Days']

    patients = {}
    for col_idx in day_col_indices:
        pid = row1[col_idx]
        days, psa, abi = [], [], []
        for r in range(3, ws.max_row + 1):
            d   = ws.cell(r, col_idx + 1).value
            p_v = ws.cell(r, col_idx + 2).value
            a   = ws.cell(r, col_idx + 3).value
            if d is None:
                break
            days.append(float(d))
            psa.append(float(p_v))
            abi.append(int(a))
        patients[pid] = {
            'days': np.array(days),
            'psa':  np.array(psa),
            'abi':  np.array(abi),
        }
    return patients


# ── ODE system (state: [S, R, P]) ─────────────────────────────────────────────

def odes(t, y, p, C, delta):
    """
    Parameters
    ----------
    C     : float  drug concentration during this interval (ng/mL)
    delta : float  PSA production rate (derived from initial conditions)
    """
    S, R, P = y

    C_pos = max(C, 0.0)
    alpha = (p['alpha_max'] * C_pos**p['n_hill']
             / (p['EC50']**p['n_hill'] + C_pos**p['n_hill']))

    N  = S + R
    dS = p['rho_S'] * S * (1 - N / p['K']) - alpha * S
    dR = p['rho_R'] * R * (1 - N / p['K'])
    dP = delta * (S + R) - p['gamma'] * P

    return [dS, dR, dP]


# ── Single-patient simulation ──────────────────────────────────────────────────

def simulate_patient(p, days_obs, abi_obs, S0, R0, alpha_max, psa0):
    """
    Simulate model for one patient using the QSS PK approximation.

    Integrates between consecutive observation timepoints; C is set to
    C_ss_avg (26.9 ng/mL) when Abi=1, and 0 when Abi=0.

    Returns
    -------
    psa_sim : array of simulated PSA at days_obs (length = len(days_obs))
    """
    p = p.copy()
    p['alpha_max'] = alpha_max

    # Derive delta so PSA(0) = psa0 at quasi-steady state
    delta = psa0 * p['gamma'] / (S0 + R0)

    y = np.array([S0, R0, psa0])
    psa_sim = [psa0]

    for i in range(len(days_obs) - 1):
        C = C_SS if abi_obs[i] == 1 else 0.0

        sol = solve_ivp(
            fun=lambda t, y: odes(t, y, p, C, delta),
            t_span=(days_obs[i], days_obs[i + 1]),
            y0=y,
            method='RK45',
            rtol=1e-7,
            atol=1e-10,
        )
        y = sol.y[:, -1].copy()
        y[0] = max(y[0], 1e-9)   # keep S non-negative
        y[1] = max(y[1], 1e-9)   # keep R non-negative

        psa_sim.append(float(y[2]))

    return np.array(psa_sim)


# ── Objective function ─────────────────────────────────────────────────────────

def objective(params, days_obs, psa_obs, abi_obs, p_base):
    S0, R0, alpha_max, gamma = params

    if S0 + R0 >= p_base['K'] or S0 <= 0 or R0 <= 0 or alpha_max <= 0 or gamma <= 0:
        return 1e6

    p = p_base.copy()
    p['gamma'] = gamma

    try:
        psa_sim = simulate_patient(
            p, days_obs, abi_obs, S0, R0, alpha_max, psa_obs[0]
        )
        psa_sim = np.maximum(psa_sim, 1e-6)
        return float(np.sum((np.log(psa_sim) - np.log(psa_obs)) ** 2))
    except Exception:
        return 1e6


# ── Per-patient fitting ────────────────────────────────────────────────────────

def fit_patient(pid, data, p_base):
    """
    Fit S0, R0, alpha_max for one patient using differential evolution.

    Returns
    -------
    dict with fitted parameters and simulation output
    """
    days_obs = data['days']
    psa_obs  = data['psa']
    abi_obs  = data['abi']

    # gamma (PSA clearance) is also fitted per patient.
    # With gamma fixed at 0.08 day^-1 (t½ ≈ 8.7 days), PSA cannot physically
    # drop as fast as observed in some patients (e.g. 6.06 → 0.03 ng/mL in
    # 27 days). Brady et al. fit their equivalent parameter (phi) at the
    # population level from 70 patients, giving a different value.
    # Bounds: gamma ∈ [0.05, 1.0] day^-1 (t½ between 16.6 h and 14 days).
    bounds = [
        (0.01, 0.94),   # S0
        (0.001, 0.48),  # R0        (S0 + R0 < 1 enforced in objective)
        (0.001, 0.50),  # alpha_max
        (0.05,  1.00),  # gamma     PSA clearance rate (day^-1)
    ]

    result = differential_evolution(
        objective,
        bounds=bounds,
        args=(days_obs, psa_obs, abi_obs, p_base),
        seed=42,
        maxiter=300,
        popsize=15,
        tol=1e-4,
        mutation=(0.5, 1.5),
        recombination=0.9,
        polish=False,
        workers=1,
    )

    S0, R0, alpha_max, gamma = result.x
    p_fit = p_base.copy()
    p_fit['gamma'] = gamma
    psa_sim = simulate_patient(
        p_fit, days_obs, abi_obs, S0, R0, alpha_max, psa_obs[0]
    )

    return {
        'pid':       pid,
        'S0':        S0,
        'R0':        R0,
        'alpha_max': alpha_max,
        'gamma':     gamma,
        'days':      days_obs,
        'psa_obs':   psa_obs,
        'psa_sim':   psa_sim,
        'abi':       abi_obs,
        'fun':       result.fun,
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_fits(fits, ncols=4):
    n     = len(fits)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = axes.flatten()

    for ax, fit in zip(axes, fits):
        t_yr = fit['days'] / 365.0

        # Shade treatment-ON periods
        for i in range(len(fit['days']) - 1):
            if fit['abi'][i] == 1:
                ax.axvspan(fit['days'][i] / 365, fit['days'][i + 1] / 365,
                           color='steelblue', alpha=0.12, lw=0)

        ax.semilogy(t_yr, fit['psa_obs'], 'o', color='darkorange', ms=4,
                    label='Observed')
        ax.semilogy(t_yr, fit['psa_sim'], '-', color='steelblue', lw=1.5,
                    label='Fitted')
        ax.set_title(f"{fit['pid']}\n"
                     f"S0={fit['S0']:.2f}, R0={fit['R0']:.3f}, "
                     f"α_max={fit['alpha_max']:.3f}",
                     fontsize=9)
        ax.set_xlabel('Time (years)', fontsize=8)
        ax.set_ylabel('PSA (ng/mL)', fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    for ax in axes[n:]:
        ax.set_visible(False)

    axes[0].legend(frameon=False, fontsize=8)
    plt.suptitle('Per-patient PK/PD fits — adaptive therapy arm\n'
                 f'Fixed: EC50={P_BASE["EC50"]} ng/mL, C_ss={C_SS:.1f} ng/mL',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    return fig


def print_summary(fits):
    print(f"\n{'Patient':<10} {'S0':>6} {'R0':>7} {'alpha_max':>10} "
          f"{'alpha_eff':>10} {'SSE_log':>8}")
    print('-' * 58)
    alpha_eff_factor = C_SS / (P_BASE['EC50'] + C_SS)
    for f in fits:
        alpha_eff = f['alpha_max'] * alpha_eff_factor
        print(f"{f['pid']:<10} {f['S0']:>6.3f} {f['R0']:>7.4f} "
              f"{f['alpha_max']:>10.4f} {alpha_eff:>10.4f} {f['fun']:>8.3f}")
    print(f"\nC_ss_avg = {C_SS:.1f} ng/mL  |  "
          f"alpha_eff = alpha_max x {alpha_eff_factor:.3f}  (EC50={P_BASE['EC50']} ng/mL)")


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    DATA_PATH = ('/Users/80031987/Desktop/Basanta_Marusyk_Lab_2026/'
                 'PKPD_AdaptiveTherapy/TrialPatientData.xlsx')

    print("Loading patient data...")
    patients = load_adaptive_patients(DATA_PATH)
    print(f"  {len(patients)} patients: {list(patients.keys())}\n")

    fits = []
    for i, (pid, data) in enumerate(patients.items()):
        print(f"[{i+1:2d}/{len(patients)}] Fitting {pid}...", end=' ', flush=True)
        fit = fit_patient(pid, data, P_BASE)
        fits.append(fit)
        print(f"S0={fit['S0']:.3f}, R0={fit['R0']:.4f}, "
              f"alpha_max={fit['alpha_max']:.4f}, SSE={fit['fun']:.3f}")

    print_summary(fits)

    fig = plot_fits(fits)
    plt.tight_layout()
    plt.savefig('/Users/80031987/Desktop/Basanta_Marusyk_Lab_2026/'
                'PKPD_AdaptiveTherapy/patient_fits.pdf',
                dpi=300, bbox_inches='tight')
    print("\nFigure saved to patient_fits.pdf")
    plt.show()
