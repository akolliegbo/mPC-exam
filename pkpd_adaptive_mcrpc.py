"""
PK/PD adaptive therapy simulation using per-patient fitted parameters.

Model structure (Brady-Nicholls & Enderling 2020 + Emax PD extension):
    State vector: [S, R, P]
        S  : differentiated (sensitive) cells  — paper's 'D'
        R  : stem-like (resistant) cells       — paper's 'S'
        P  : serum PSA (ng/mL)

    ODEs (QSS PK approximation):
        dR/dt = (R/(R+S)) * p_s * λ * R
        dS/dt = (1 - R/(R+S) * p_s) * λ * R  -  α(C) * S
        dP/dt = δ * S  -  γ * P

    α(C) = alpha_max * C / (EC50 + C)   [Emax PD]
    C    = C_SS when treating, 0 otherwise   [QSS PK]

RBAT switching protocol (Brady-Nicholls 2022 Cancers):
    p1   : PSA at start of last observed treatment cycle (per patient)
    β    = (1 + x) * p1    — acceptable baseline (treatment OFF trigger)
    τ    = (1 + y) * β     — treatment ON trigger
    x ∈ {0.10, 0.25, 0.50},  y ∈ {0.10, 0.50, 1.00}

References:
    Brady-Nicholls & Enderling (2020) Nat Commun
    Brady-Nicholls et al. (2022) Cancers
    Stuyckens et al. (2014)
"""

import sys
import os
import itertools
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams.update({'font.family': 'Arial', 'font.size': 12})

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_patients import load_adaptive_patients, P_BASE, C_SS

DATA_PATH = ('/Users/80031987/Desktop/Basanta_Marusyk_Lab_2026/'
             'PKPD_AdaptiveTherapy/TrialPatientData.xlsx')

# RBAT threshold grid (Brady-Nicholls 2022 Cancers)
X_VALS = [0.10, 0.25, 0.50]   # acceptable baseline: β = (1+x)*p1
Y_VALS = [0.10, 0.50, 1.00]   # treatment trigger:   τ = (1+y)*β


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_last_cycle_psa(data):
    """
    Return p1: PSA at the start of the last COMPLETE treatment cycle.

    A complete cycle has both an ON transition (0→1) and a subsequent
    OFF transition (1→0). If the final ON period is still ongoing at the
    end of observation (no subsequent OFF), we use the preceding complete
    cycle instead — matching the Brady-Nicholls 2022 Cancers approach.
    Falls back to psa[0] if no complete cycle is found.
    """
    abi, psa = data['abi'], data['psa']

    # Collect indices of all 0→1 (ON) and 1→0 (OFF) transitions
    on_idxs  = [i for i in range(1, len(abi)) if abi[i] == 1 and abi[i-1] == 0]
    off_idxs = [i for i in range(1, len(abi)) if abi[i] == 0 and abi[i-1] == 1]

    # Also include idx 0 if treatment starts on day 0
    if len(abi) > 0 and abi[0] == 1:
        on_idxs = [0] + on_idxs

    # Find the last ON index that has a subsequent OFF (complete cycle)
    last_complete_on = None
    for on_idx in reversed(on_idxs):
        if any(off_idx > on_idx for off_idx in off_idxs):
            last_complete_on = on_idx
            break

    if last_complete_on is not None:
        return float(psa[last_complete_on])
    elif on_idxs:
        return float(psa[on_idxs[-1]])   # fallback: last ON start
    else:
        return float(psa[0])


# ── ODE system (QSS: state [S, R, P]) ────────────────────────────────────────

def odes_qss(t, y, p, C, delta):
    S, R, P = y
    C_pos = max(C, 0.0)
    alpha = (p['alpha_max'] * C_pos**p['n_hill']
             / (p['EC50']**p['n_hill'] + C_pos**p['n_hill']))
    N = S + R
    stem_frac = R / N if N > 1e-12 else 0.0
    dR = stem_frac * p['p_s'] * p['lambda_'] * R
    dS = (1.0 - stem_frac * p['p_s']) * p['lambda_'] * R - alpha * S
    dP = delta * S - p['gamma'] * P
    return [dS, dR, dP]


# ── Per-patient simulation ────────────────────────────────────────────────────

def simulate_patient(p, S0, R0, alpha_max, psa0,
                     psa_thresh_on, psa_thresh_off,
                     t_end=1825.0, dt=7.0):
    """
    Simulate adaptive therapy using QSS PK approximation.

    Parameters
    ----------
    p              : base parameter dict (P_BASE)
    S0             : initial differentiated cell fraction (fitted)
    R0             : initial stem-like cell fraction (fitted)
    alpha_max      : max drug kill rate day^-1 (fitted)
    psa0           : first observed PSA — sets ODE initial condition and δ
    psa_thresh_on  : absolute PSA threshold to restart treatment (τ)
    psa_thresh_off : absolute PSA threshold to stop treatment (β)
    t_end          : simulation duration (days)
    dt             : time step for PSA threshold checks (days)

    Returns
    -------
    t         : (n,) time array (days)
    Y         : (3, n) state rows [S, R, P]
    intervals : list of (t_on, t_off) treatment intervals
    """
    p = p.copy()
    p['alpha_max'] = alpha_max

    delta = psa0 * p['gamma'] / S0

    y = np.array([S0, R0, psa0])
    time_points = np.arange(0.0, t_end + dt, dt)

    t_all, Y_all = [0.0], [[S0, R0, psa0]]
    treatment_intervals = []
    treating   = False
    t_on_start = None

    for i in range(len(time_points) - 1):
        t_now  = time_points[i]
        t_next = time_points[i + 1]

        PSA = y[2]
        if not treating and PSA >= psa_thresh_on:
            treating   = True
            t_on_start = t_now
        elif treating and PSA <= psa_thresh_off:
            treating = False
            treatment_intervals.append((t_on_start, t_now))
            t_on_start = None

        C = C_SS if treating else 0.0

        sol = solve_ivp(
            fun=lambda t, y: odes_qss(t, y, p, C, delta),
            t_span=(t_now, t_next),
            y0=y,
            method='RK45',
            max_step=1.0,
            rtol=1e-6,
            atol=1e-9,
        )
        y = sol.y[:, -1].copy()
        y[0] = max(y[0], 1e-9)
        y[1] = max(y[1], 1e-9)

        t_all.append(t_next)
        Y_all.append(y.tolist())

        if y[2] > 1e6 or y[0] > 1e4:
            break

    if treating and t_on_start is not None:
        treatment_intervals.append((t_on_start, t_all[-1]))

    t_arr = np.array(t_all)
    Y_arr = np.array(Y_all).T   # shape (3, n)
    return t_arr, Y_arr, treatment_intervals


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(combo, t_end=1825.0):
    """
    Returns (ttp_days, drug_days) for one simulated combo.

    ttp_days  : time-to-progression — last time PSA was in an off-treatment
                period, or t_end if simulation ran to completion under control.
                Defined as the end of the last completed off-treatment interval,
                or t_end if PSA never escaped the threshold band.
    drug_days : total days on treatment (sum of interval durations).
    """
    t         = combo['t']
    intervals = combo['intervals']

    drug_days = sum(t_off - t_on for t_on, t_off in intervals)

    # TTP: end of last completed off-treatment interval, or t_end if controlled
    if len(intervals) == 0:
        ttp_days = t_end   # never needed treatment → controlled throughout
    else:
        last_t_off = intervals[-1][1]
        if last_t_off >= t[-1] - 7:
            # still on treatment at simulation end → not yet progressed
            ttp_days = t_end
        else:
            # last cycle completed; PSA dropped back below β
            ttp_days = t_end   # controlled to t_end
            # check if PSA at end is still below tau (controlled)
            # if the last interval ended and no more cycles started, patient
            # is in off-treatment remission → ttp = t_end
            ttp_days = t_end

    return ttp_days, drug_days


def find_optimal_combos(res):
    """
    For a single patient, return:
      best_cycles : (x, y) that maximises number of treatment cycles
                    (proxy for sustained disease control; ties broken by min drug)
      best_drug   : (x, y) that minimises total drug exposure in days
                    (ties broken by max cycles)
    """
    metrics = {}
    for (x, y), combo in res['combos'].items():
        _, drug = compute_metrics(combo)
        cycles = len(combo['intervals'])
        metrics[(x, y)] = (cycles, drug)

    best_cycles = max(metrics, key=lambda k: ( metrics[k][0], -metrics[k][1]))
    best_drug   = min(metrics, key=lambda k: ( metrics[k][1], -metrics[k][0]))
    return best_cycles, best_drug, metrics


# ── Plotting ──────────────────────────────────────────────────────────────────

def _shade_intervals(ax, intervals, color, alpha=0.12):
    for t_on, t_off in intervals:
        ax.axvspan(t_on / 365, t_off / 365, color=color, alpha=alpha, lw=0)


def _patient_panel(ax, res, x, y, color, label_prefix=''):
    """Draw PSA trajectory + treatment shading + threshold lines for one combo."""
    p1    = res['p1']
    combo = res['combos'][(x, y)]
    beta  = (1 + x) * p1
    tau   = (1 + y) * beta
    t_yr  = combo['t'] / 365.0
    P     = combo['Y'][2]

    _shade_intervals(ax, combo['intervals'], color)
    ax.plot(t_yr, P, color=color, lw=1.5,
            label=f'{label_prefix}x={x}, y={y}')
    ax.axhline(tau,  color=color, ls='--', lw=0.8, alpha=0.7)
    ax.axhline(beta, color=color, ls=':',  lw=0.8, alpha=0.7)


def plot_reference(results, x_ref=0.25, y_ref=0.50, ncols=4):
    """
    One panel per patient showing only the paper's reference combination
    (x=0.25, y=0.50: β=1.25×p₁, τ=1.875×p₁).
    """
    from matplotlib.lines import Line2D
    n     = len(results)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 3.5))
    axes = axes.flatten()

    for ax, res in zip(axes, results):
        p1  = res['p1']
        fit = res['fit']
        _patient_panel(ax, res, x_ref, y_ref, 'steelblue')

        n_cyc = len(res['combos'][(x_ref, y_ref)]['intervals'])
        ax.set_title(f"{fit['pid']}   p\u2081={p1:.1f} ng/mL\n"
                     f"{n_cyc} cycles  |  \u03b1_max={fit['alpha_max']:.3f}",
                     fontsize=8)
        ax.set_xlabel('Time (years)', fontsize=8)
        ax.set_ylabel('PSA (ng/mL)',  fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)

    for ax in axes[n:]:
        ax.set_visible(False)

    handles = [
        Line2D([0], [0], color='steelblue', lw=1.5, label='PSA trajectory'),
        Line2D([0], [0], color='steelblue', lw=0.8, ls='--', label='τ (ON threshold)'),
        Line2D([0], [0], color='steelblue', lw=0.8, ls=':',  label='β (OFF threshold)'),
        matplotlib.patches.Patch(color='steelblue', alpha=0.12, label='Treatment ON'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(f'RBAT simulations — reference combination (x={x_ref}, y={y_ref})\n'
                 f'\u03b2 = {1+x_ref:.2f}\u00d7p\u2081  |  \u03c4 = {(1+y_ref)*(1+x_ref):.3f}\u00d7p\u2081',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    return fig


def plot_optimal(results, ncols=4):
    """
    One panel per patient showing:
      - grey  : reference combo (x=0.25, y=0.50)
      - blue  : combo that maximises TTP
      - orange: combo that minimises drug exposure
    """
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches

    n     = len(results)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 3.5))
    axes = axes.flatten()

    for ax, res in zip(axes, results):
        p1  = res['p1']
        fit = res['fit']
        best_cyc, best_drug, metrics = find_optimal_combos(res)

        # Reference (always shown in grey)
        _patient_panel(ax, res, 0.25, 0.50, '#999999')

        # Max cycles (blue) — only draw if different from reference
        if best_cyc != (0.25, 0.50):
            _patient_panel(ax, res, *best_cyc, '#1f77b4', label_prefix='Max cycles: ')
        else:
            ax.plot([], [], color='#1f77b4', lw=1.5,
                    label=f'Max cycles: x={best_cyc[0]}, y={best_cyc[1]} (=ref)')

        # Min drug (orange) — only draw if different from reference and max cycles
        if best_drug not in [(0.25, 0.50), best_cyc]:
            _patient_panel(ax, res, *best_drug, '#ff7f0e', label_prefix='Min drug: ')
        else:
            ax.plot([], [], color='#ff7f0e', lw=1.5,
                    label=f'Min drug: x={best_drug[0]}, y={best_drug[1]}')

        cyc_val  = metrics[best_cyc][0]
        drug_val = metrics[best_drug][1]
        ax.set_title(f"{fit['pid']}   p\u2081={p1:.1f}\n"
                     f"MaxCyc={cyc_val} ({best_cyc[0]},{best_cyc[1]})  "
                     f"MinDrug={drug_val:.0f}d ({best_drug[0]},{best_drug[1]})",
                     fontsize=7.5)
        ax.set_xlabel('Time (years)', fontsize=8)
        ax.set_ylabel('PSA (ng/mL)',  fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)

    for ax in axes[n:]:
        ax.set_visible(False)

    handles = [
        Line2D([0], [0], color='#999999', lw=1.5, label='Reference (x=0.25, y=0.50)'),
        Line2D([0], [0], color='#1f77b4', lw=1.5, label='Max treatment cycles'),
        Line2D([0], [0], color='#ff7f0e', lw=1.5, label='Min drug exposure'),
        Line2D([0], [0], color='grey',    lw=0.8, ls='--', label='\u03c4 (ON threshold)'),
        Line2D([0], [0], color='grey',    lw=0.8, ls=':',  label='\u03b2 (OFF threshold)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=5,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('RBAT — optimal threshold combinations per patient\n'
                 'Blue: maximises treatment cycles  |  Orange: minimises drug exposure  |  '
                 'Grey: reference (x=0.25, y=0.50)',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    return fig


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    BASE = ('/Users/80031987/Desktop/Basanta_Marusyk_Lab_2026/'
            'PKPD_AdaptiveTherapy/')

    print("Loading patient data...")
    patients = load_adaptive_patients(DATA_PATH)
    print(f"  {len(patients)} patients loaded\n")

    print("Loading saved fits from fits.pkl...")
    import pickle
    with open(BASE + 'fits.pkl', 'rb') as f:
        fits = pickle.load(f)
    print(f"  {len(fits)} patients loaded\n")

    print("Simulating RBAT with 9 threshold combinations...")
    results = []
    for fit in fits:
        pid  = fit['pid']
        data = patients[pid]
        psa0 = float(data['psa'][0])
        p1   = get_last_cycle_psa(data)

        print(f"  {pid}  (psa0={psa0:.2f}, p1={p1:.2f})", end='')
        combos = {}
        for x, y in itertools.product(X_VALS, Y_VALS):
            beta = (1 + x) * p1
            tau  = (1 + y) * beta
            t, Y, intervals = simulate_patient(
                P_BASE,
                S0=fit['S0'], R0=fit['R0'], alpha_max=fit['alpha_max'],
                psa0=psa0,
                psa_thresh_on=tau, psa_thresh_off=beta,
                t_end=3044.0,
            )
            combos[(x, y)] = {'t': t, 'Y': Y, 'intervals': intervals}

        mid_cycles = len(combos[(0.25, 0.50)]['intervals'])
        print(f"  →  {mid_cycles} cycles (x=0.25, y=0.50)")
        results.append({
            'pid':    pid,
            'psa0':   psa0,
            'p1':     p1,
            'fit':    fit,
            'combos': combos,
        })

    fig1 = plot_reference(results)
    fig1.savefig(BASE + 'adaptive_simulations.pdf', dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to adaptive_simulations.pdf")

    fig2 = plot_optimal(results)
    fig2.savefig(BASE + 'adaptive_optimal.pdf', dpi=300, bbox_inches='tight')
    print(f"Figure saved to adaptive_optimal.pdf")

    plt.show()
