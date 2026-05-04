"""
PK/PD extension of Brady-Nicholls & Enderling (2022) adaptive therapy model for mCRPC.

Original model: binary treatment switch driving cell kill rate alpha.
Extension: 1-compartment PK (depot -> central) + Emax PD: alpha(C) = alpha_max * C^n / (EC50^n + C^n)

State vector: [D, C, S, R, P]
    D  : drug amount in depot       (mg)
    C  : drug concentration, central (ng/mL)
    S  : sensitive cell fraction     (normalized to K)
    R  : resistant cell fraction     (normalized to K)
    P  : PSA                         (ng/mL)

Units note:
    Dose added to D in mg. Conversion to C:
        dC/dt = ka * (D/V_L) * unit_factor - ke * C
    With V in L and D in mg: D/V is mg/L = ug/mL = 1000 ng/mL.
    Set unit_factor = 1000 to get C in ng/mL, or work in ug/mL by adjusting EC50.
    Here we keep everything in consistent scaled units (see params below).
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.family': 'Arial', 'font.size': 14})


# ── Parameters ────────────────────────────────────────────────────────────────

p = {
    # --- Tumor dynamics (Brady-Nicholls & Enderling 2022 Table 1 typical values) ---
    'rho_S':    0.027,   # day^-1   sensitive cell net proliferation rate
    'rho_R':    0.003,   # day^-1   resistant cell net proliferation rate
                         #          must be << rho_S for competitive suppression to hold across cycles
                         #          placeholder — will be fit per-patient from the 16-patient dataset
    'K':        1.0,     # carrying capacity (cells normalized; S+R in [0,1])
    'delta_S':  1.0,     # PSA production rate per unit sensitive cell mass
    'delta_R':  1.0,     # PSA production rate per unit resistant cell mass
    'gamma':    0.08,    # day^-1   PSA clearance rate

    # --- PK: 1-compartment with first-order absorption ---
    # Population-mean values from Stuyckens et al. 2014 (Clin Pharmacokinet),
    # mCRPC patient subgroup, 1000 mg QD fasted + prednisone 5 mg BID.
    # Stuyckens used a 2-compartment Erlang transit model; values below are
    # adapted for a 1-compartment first-order approximation:
    #
    #   ka  : Stuyckens Erlang absorption rate = 1.65 h^-1 = 39.6 day^-1.
    #         Used directly as first-order ka (conservative simplification).
    #   ke  : Derived from terminal t½ ≈ 12 h (clinical label) rather than
    #         CL/V_central (which gives the faster distribution-phase t½ of
    #         ~2.5 h and is inappropriate for a 1-compartment model).
    #         ke = ln(2) / 0.5 day = 1.386 day^-1.
    #   V   : Apparent V/F = (CL/F) / ke = 37200 L/day / 1.386 day^-1
    #         = 26840 L.  Works with apparent CL/F throughout (F not needed
    #         separately because dose enters depot and F is implicit in CL/F).
    #
    # Resulting steady-state average concentration:
    #   C_ss_avg = dose / (CL/F * tau) * 1e3
    #            = 1000 mg / (37200 L/day * 1 day) * 1000 ng/mL per mg/L
    #            ≈ 27 ng/mL   (consistent with reported fasted AUC ~645 ng·h/mL)
    #
    # Units: D in mg, C in ng/mL, V in L
    # dC/dt = ka * (D/V) * 1e3 - ke * C   [1 mg/L = 1000 ng/mL]
    'ka':       39.6,    # day^-1   absorption rate constant (Stuyckens 2014)
    'ke':       1.386,   # day^-1   elimination rate constant, from t½ = 12 h
    'V':        26840.0, # L        apparent V/F = (CL/F)/ke (Stuyckens 2014)

    # --- PD: Emax (Hill) model ---
    # C_ss_avg ≈ 27 ng/mL with abiraterone population-mean PK above.
    # EC50 must be on the same scale for meaningful drug effect.
    # No published EC50 for abiraterone cell-kill is available; placeholder
    # set to ~0.5 * C_ss so the drug operates near half-maximal effect at
    # steady state. Requires calibration to tumour response data.
    'alpha_max': 0.040,  # day^-1   maximum drug-induced kill rate (placeholder)
    'EC50':      15.0,   # ng/mL    placeholder: ~0.5 * C_ss_avg ≈ 27/2 ng/mL
    'n_hill':    1.0,    # Hill coefficient (n=1: hyperbolic; n>1: sigmoidal)

    # --- Dosing ---
    'dose_mg':   1000.0, # mg       abiraterone acetate, standard SOC dose
    'interval':  1.0,    # days     QD (once daily)

    # --- Adaptive therapy PSA thresholds ---
    # Treatment ON  when PSA >= frac_on  * PSA_0
    # Treatment OFF when PSA <= frac_off * PSA_0
    'frac_on':   1.0,    # start treatment when PSA returns to baseline
    'frac_off':  0.5,    # stop treatment when PSA halved from baseline
}


# ── ODE system ────────────────────────────────────────────────────────────────

def odes(t, y, p):
    D, C, S, R, P = y

    # PK
    dD = -p['ka'] * D
    # Unit conversion: D in mg, V in L -> D/V in mg/L = 1000 ng/mL
    dC = p['ka'] * (D / p['V']) * 1e3 - p['ke'] * C

    # PD: Emax kill rate from current concentration
    C_pos = max(C, 0.0)   # guard against tiny negative values from integrator
    alpha = (p['alpha_max'] * C_pos**p['n_hill']
             / (p['EC50']**p['n_hill'] + C_pos**p['n_hill']))

    # Tumor (logistic growth + drug kill on sensitive cells only)
    N = S + R
    dS = p['rho_S'] * S * (1 - N / p['K']) - alpha * S
    dR = p['rho_R'] * R * (1 - N / p['K'])

    # PSA (produced by both populations, cleared at rate gamma)
    dP = p['delta_S'] * S + p['delta_R'] * R - p['gamma'] * P

    return [dD, dC, dS, dR, dP]


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(p, t_end=1825.0, S0=0.45, R0=0.05):
    """
    Simulate adaptive therapy with PK/PD over t_end days.

    Parameters
    ----------
    p      : parameter dict
    t_end  : float  simulation duration (days)
    S0, R0 : float  initial sensitive / resistant cell fractions (sum < 1)

    Returns
    -------
    t      : (n,)   time array (days)
    Y      : (5, n) state array rows = [D, C, S, R, P]
    treat  : list of (t_on, t_off) treatment intervals
    """
    # Initial PSA at approximate quasi-steady state
    P0 = (p['delta_S'] * S0 + p['delta_R'] * R0) / p['gamma']

    PSA_thresh_on  = p['frac_on']  * P0
    PSA_thresh_off = p['frac_off'] * P0

    y = np.array([0.0, 0.0, S0, R0, P0])   # D, C, S, R, P

    # Dose event times across full simulation
    dose_times = np.arange(0.0, t_end, p['interval'])

    t_all, Y_all = [], []
    treatment_intervals = []   # list of (t_on, t_off) pairs

    treating   = False
    t_on_start = None

    for i, t_now in enumerate(dose_times):
        t_next = dose_times[i + 1] if i + 1 < len(dose_times) else t_end

        # ── Adaptive switching (evaluated at each dose time) ──────────────────
        PSA = y[4]
        if not treating and PSA >= PSA_thresh_on:
            treating   = True
            t_on_start = t_now
        elif treating and PSA <= PSA_thresh_off:
            treating = False
            treatment_intervals.append((t_on_start, t_now))
            t_on_start = None

        # ── Administer dose (bolus to depot) ──────────────────────────────────
        if treating:
            y[0] += p['dose_mg']

        # ── Integrate to next event ───────────────────────────────────────────
        sol = solve_ivp(
            fun=lambda t, y: odes(t, y, p),
            t_span=(t_now, t_next),
            y0=y,
            method='RK45',
            max_step=0.05,          # 1.2 h max step — resolves Cmax after each dose
            dense_output=False,
            rtol=1e-6,
            atol=1e-9,
        )

        t_all.append(sol.t)
        Y_all.append(sol.y)
        y = sol.y[:, -1].copy()

    # Close any open treatment interval at end of simulation
    if treating and t_on_start is not None:
        treatment_intervals.append((t_on_start, t_end))

    return (np.concatenate(t_all),
            np.hstack(Y_all),
            treatment_intervals)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(t, Y, treatment_intervals, p):
    D, C, S, R, P = Y

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    t_yr = t / 365.0

    # Shade treatment-ON periods on all panels
    for ax in axes:
        for (t_on, t_off) in treatment_intervals:
            ax.axvspan(t_on / 365, t_off / 365,
                       color='steelblue', alpha=0.12, lw=0)

    # Panel 1: Drug concentration
    axes[0].plot(t_yr, C, color='steelblue', lw=1.5)
    axes[0].set_ylabel('Concentration (ng/mL)')
    axes[0].set_title('Drug concentration — central compartment')
    axes[0].spines[['top', 'right']].set_visible(False)

    # Panel 2: Cell populations
    axes[1].plot(t_yr, S, color='steelblue', lw=2, label='Sensitive (S)')
    axes[1].plot(t_yr, R, color='firebrick',  lw=2, label='Resistant (R)')
    axes[1].plot(t_yr, S + R, color='dimgray', lw=1.5, ls='--', label='Total (S+R)')
    axes[1].set_ylabel('Cell fraction')
    axes[1].set_title('Tumor cell populations')
    axes[1].legend(frameon=False)
    axes[1].spines[['top', 'right']].set_visible(False)

    # Panel 3: PSA with threshold lines
    axes[2].plot(t_yr, P, color='darkorange', lw=2, label='PSA')
    P0 = P[0]
    axes[2].axhline(p['frac_on']  * P0, color='firebrick', ls='--', lw=1,
                    label=f'Treat ON threshold ({p["frac_on"]:.1f} x PSA_0)')
    axes[2].axhline(p['frac_off'] * P0, color='green', ls='--', lw=1,
                    label=f'Treat OFF threshold ({p["frac_off"]:.1f} x PSA_0)')
    axes[2].set_ylabel('PSA (ng/mL)')
    axes[2].set_xlabel('Time (years)')
    axes[2].set_title('PSA dynamics (adaptive therapy trigger)')
    axes[2].legend(frameon=False)
    axes[2].spines[['top', 'right']].set_visible(False)

    # Alpha(C) at each time point — annotate on panel 0
    C_pos = np.maximum(C, 0)
    alpha_t = (p['alpha_max'] * C_pos**p['n_hill']
               / (p['EC50']**p['n_hill'] + C_pos**p['n_hill']))
    ax_alpha = axes[0].twinx()
    ax_alpha.plot(t_yr, alpha_t, color='gray', lw=1, alpha=0.6, ls=':')
    ax_alpha.set_ylabel('alpha(C) — day$^{-1}$', color='gray', fontsize=11)
    ax_alpha.tick_params(axis='y', colors='gray')
    ax_alpha.spines[['top']].set_visible(False)

    plt.tight_layout()
    return fig


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t, Y, treat_intervals = simulate(p, t_end=1825, S0=0.45, R0=0.05)

    print(f"Simulation complete: {t[-1]:.0f} days ({t[-1]/365:.1f} years)")
    print(f"Treatment intervals: {len(treat_intervals)}")
    for i, (on, off) in enumerate(treat_intervals):
        print(f"  Cycle {i+1}: day {on:.0f} – {off:.0f}  ({(off-on):.0f} days ON)")

    D, C, S, R, P = Y
    print(f"\nFinal state:")
    print(f"  C     = {C[-1]:.2f} ng/mL")
    print(f"  S     = {S[-1]:.4f}")
    print(f"  R     = {R[-1]:.4f}")
    print(f"  PSA   = {P[-1]:.2f} ng/mL")

    fig = plot_results(t, Y, treat_intervals, p)
    plt.show()
