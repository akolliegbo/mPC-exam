"""
sensitivity analysis on the brady-nicholls 2021 model
params: ps (stem cell self renewal), alpha (cytotoxicity),
        rho (psa production), phi (psa decay)

what we're doing: perturb each param by +/- 10% and see how much
the psa trajectory changes. the paper claims rho and phi can be
uniform across patients (not patient-specific) -- we're showing WHY
that makes sense by showing they're less sensitive than ps and alpha
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.io as sio
import warnings
warnings.filterwarnings('ignore')

# color palette -- clean scientific but not boring
COLORS = {
    'ps':    '#E63946',   # red: stem cell renewal -- most sensitive
    'alpha': '#F4A261',   # orange: cytotoxicity
    'rho':   '#457B9D',   # blue: psa production rate
    'phi':   '#2A9D8F',   # teal: psa decay rate
    'base':  '#1D1D2E',   # near-black for baseline
    'data':  '#888888',   # gray for actual data points
    'bg':    '#FAFAFA',   # off-white background
    'grid':  '#E8E8E8',   # light grid lines
}

PARAM_LABELS = {
    'ps':    r'$p_s$ (self-renewal)',
    'alpha': r'$\alpha$ (cytotoxicity)',
    'rho':   r'$\rho$ (PSA production)',
    'phi':   r'$\varphi$ (PSA decay)',
}

# the ode model from brady-nicholls eq 1:
# dS/dt = (S/(S+D)) * ps * lambda * S   <-- stem cells, resistant to treatment
# dD/dt = (1 - (S/(S+D))*ps) * lambda * S - alpha * Tx * D  <-- non-stem, killed by tx
# dP/dt = rho * D - phi * P   <-- psa tracks non-stem cells
LAM = np.log(2)   # stem cell division rate once per day (from paper)
PERTURB = 0.10    # 10% perturbation


def run_model(t_data, tx_data, ps, alpha, rho, phi, psa0, D0, S0):
    """
    run the ode and return psa at each measurement timepoint
    t_data: days at each measurement
    tx_data: 1=treatment on, 0=off at each timepoint
    returns: array of psa values matching t_data
    """
    y = [S0, D0, psa0]
    P_out = [psa0]

    for i in range(len(t_data) - 1):
        t0 = float(t_data[i])
        t1 = float(t_data[i + 1])
        Tx = float(tx_data[i])

        def ode(t, y, Tx=Tx):
            # clip negatives to avoid blowup
            S, D, P = [max(x, 0.0) for x in y]
            total = max(S + D, 1e-12)
            frac = S / total
            dS = frac * ps * LAM * S
            dD = (1 - frac * ps) * LAM * S - alpha * Tx * D
            dP = rho * D - phi * P
            return [dS, dD, dP]

        sol = solve_ivp(ode, (t0, t1), y, max_step=0.5, rtol=1e-5, atol=1e-8)
        if sol.success:
            y = [max(x, 1e-15) for x in sol.y[:, -1]]
        P_out.append(y[2])

    return np.array(P_out)


def get_initial_conditions(psa0, phi=0.08, rho=0.06):
    """
    set up initial conditions from patient psa baseline
    at quasi steady state: P = rho*D/phi so D0 = psa0*phi/rho
    S0 is small (stem cells are rare at treatment start -- biologically realistic)
    """
    D0 = psa0 * phi / rho
    S0 = 0.01 * D0  # 1% stem cell fraction initially
    return D0, S0


def compute_sensitivity(t_data, tx_data, psa_data, ps, alpha, rho, phi):
    """
    compute normalized sensitivity index (NSI) for each param
    NSI = (delta_output/output) / (delta_param/param)
    NSI > 1 means nonlinear amplification of param changes
    NSI ~ 1 means linear proportional relationship
    NSI < 1 means output is less sensitive than param change
    """
    psa0 = psa_data[0]
    D0, S0 = get_initial_conditions(psa0, phi, rho)

    # baseline run with unperturbed params
    P_base = run_model(t_data, tx_data, ps, alpha, rho, phi, psa0, D0, S0)

    sensitivities = {}
    P_perturbed = {}

    for param_name in ['ps', 'alpha', 'rho', 'phi']:
        # set up perturbed param sets
        params_up = dict(ps=ps, alpha=alpha, rho=rho, phi=phi)
        params_dn = dict(ps=ps, alpha=alpha, rho=rho, phi=phi)
        params_up[param_name] *= (1 + PERTURB)
        params_dn[param_name] *= (1 - PERTURB)

        P_up = run_model(t_data, tx_data,
                         params_up['ps'], params_up['alpha'],
                         params_up['rho'], params_up['phi'],
                         psa0, D0, S0)
        P_dn = run_model(t_data, tx_data,
                         params_dn['ps'], params_dn['alpha'],
                         params_dn['rho'], params_dn['phi'],
                         psa0, D0, S0)

        # mean relative change normalized to param perturbation size
        rel_up = np.abs(P_up - P_base) / (np.abs(P_base) + 0.001)
        rel_dn = np.abs(P_dn - P_base) / (np.abs(P_base) + 0.001)
        nsi = ((np.mean(rel_up) + np.mean(rel_dn)) / 2) / PERTURB

        sensitivities[param_name] = nsi
        P_perturbed[param_name] = {'up': P_up, 'dn': P_dn, 'base': P_base}

    return sensitivities, P_perturbed, P_base


# load data
print("loading data...")
trial = sio.loadmat('../data/TrialPatientData.mat')
patients = sorted([k for k in trial.keys() if k.startswith('P')])
print(f"found {len(patients)} patients")

# baseline uniform params (rho and phi are uniform across patients per the paper)
# ps and alpha are derived analytically from typical treatment response dynamics
# phi=0.08 gives psa half-life of ~9 days (biologically plausible range)
# alpha=0.74 = lambda*(1-ps) + k_off where k_off from observed psa drop rate
BASE_PS    = 0.10
BASE_ALPHA = 0.74
BASE_RHO   = 0.06
BASE_PHI   = 0.08


# =============================================================
# VIZ 1: single patient deep dive (P1001)
# 2x2 grid, one panel per param, shows perturbation band over time
# =============================================================

print("running single patient analysis...")

pt_key = 'P1001'
p = trial[pt_key]
t_data, psa_data, tx_data = p[:, 0], p[:, 1], p[:, 2]

sensitivities_1pt, P_perturbed_1pt, P_base_1pt = compute_sensitivity(
    t_data, tx_data, psa_data,
    BASE_PS, BASE_ALPHA, BASE_RHO, BASE_PHI
)

fig1, axes = plt.subplots(2, 2, figsize=(14, 9))
fig1.patch.set_facecolor(COLORS['bg'])
fig1.suptitle(
    f'Sensitivity Analysis: Patient {pt_key} -- ±10% Perturbation of Each Parameter',
    fontsize=13, fontweight='bold', color=COLORS['base'], y=0.99
)

for param_name, ax in zip(['ps', 'alpha', 'rho', 'phi'], axes.flat):
    ax.set_facecolor(COLORS['bg'])

    # shade treatment-on periods so you can see when tx is active
    for i in range(len(t_data) - 1):
        if tx_data[i] == 1:
            ax.axvspan(t_data[i], t_data[i + 1], alpha=0.07, color='gray', linewidth=0)

    P_up = P_perturbed_1pt[param_name]['up']
    P_dn = P_perturbed_1pt[param_name]['dn']

    # fill between up/dn -- the sensitivity envelope
    ax.fill_between(t_data, P_dn, P_up,
                    alpha=0.22, color=COLORS[param_name], label='±10% band')

    ax.plot(t_data, P_base_1pt, '-', color=COLORS['base'],
            linewidth=2.0, label='baseline', zorder=3)
    ax.plot(t_data, P_up, '--', color=COLORS[param_name],
            linewidth=1.5, alpha=0.9, label='+10%')
    ax.plot(t_data, P_dn, ':', color=COLORS[param_name],
            linewidth=1.5, alpha=0.9, label='-10%')
    ax.scatter(t_data, psa_data, s=22, color=COLORS['data'],
               zorder=5, alpha=0.75, label='patient data')

    nsi = sensitivities_1pt[param_name]
    ax.set_title(f'{PARAM_LABELS[param_name]}   |   NSI = {nsi:.2f}',
                 fontsize=11, color=COLORS['base'], pad=8)
    ax.set_xlabel('time (days)', fontsize=9, color='#555')
    ax.set_ylabel('PSA (ng/mL)', fontsize=9, color='#555')
    ax.legend(fontsize=8, framealpha=0.7, loc='upper left')
    ax.grid(True, color=COLORS['grid'], linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8, colors='#555')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('../outputs/viz1_single_patient.png',
            dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
print("  saved viz1_single_patient.png")
plt.close()


# =============================================================
# VIZ 2: all patients -- NSI as connected lines
# thin gray = each patient, thick colored dots + line = mean
# =============================================================

print("running all patients sensitivity...")

all_sensitivities = {p: [] for p in ['ps', 'alpha', 'rho', 'phi']}
patient_nsi_matrix = []
valid_patients = []

for pt_key in patients:
    p = trial[pt_key]
    t_d, psa_d, tx_d = p[:, 0], p[:, 1], p[:, 2]
    if len(t_d) < 5:
        print(f"  skipping {pt_key}")
        continue

    sens, _, _ = compute_sensitivity(
        t_d, tx_d, psa_d, BASE_PS, BASE_ALPHA, BASE_RHO, BASE_PHI
    )
    row = [sens[pn] for pn in ['ps', 'alpha', 'rho', 'phi']]
    patient_nsi_matrix.append(row)
    valid_patients.append(pt_key)
    for pn in ['ps', 'alpha', 'rho', 'phi']:
        all_sensitivities[pn].append(sens[pn])
    print(f"  {pt_key}: ps={sens['ps']:.2f}, alpha={sens['alpha']:.2f}, rho={sens['rho']:.2f}, phi={sens['phi']:.2f}")

patient_nsi_matrix = np.array(patient_nsi_matrix)
param_names_ordered = ['ps', 'alpha', 'rho', 'phi']
x_positions = np.arange(4)

fig2, ax = plt.subplots(figsize=(12, 6))
fig2.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg'])

# one thin line per patient
for i in range(len(patient_nsi_matrix)):
    ax.plot(x_positions, patient_nsi_matrix[i], '-o',
            color='#999999', linewidth=0.8, markersize=3, alpha=0.35, zorder=1)

mean_nsi = np.mean(patient_nsi_matrix, axis=0)
std_nsi = np.std(patient_nsi_matrix, axis=0)

# error band around mean
ax.fill_between(x_positions, mean_nsi - std_nsi, mean_nsi + std_nsi,
                alpha=0.12, color='#333333')

# thick mean line
ax.plot(x_positions, mean_nsi, '-', color=COLORS['base'],
        linewidth=2.5, zorder=4, label='mean across patients')

# colored dots for mean at each param
for xi, pn in enumerate(param_names_ordered):
    ax.plot(xi, mean_nsi[xi], 'o', color=COLORS[pn], markersize=13, zorder=5)
    ax.annotate(f'{mean_nsi[xi]:.2f}',
                xy=(xi, mean_nsi[xi]),
                xytext=(xi + 0.08, mean_nsi[xi] + std_nsi[xi] * 0.6),
                fontsize=10, fontweight='bold', color=COLORS[pn])

ax.set_xticks(x_positions)
ax.set_xticklabels([PARAM_LABELS[p] for p in param_names_ordered], fontsize=12)
ax.set_ylabel('normalized sensitivity index (NSI)', fontsize=11, color='#555')
ax.set_title(
    f'All {len(valid_patients)} Patients: NSI Per Parameter\n'
    'gray lines = individual patients  |  thick line = mean ± std',
    fontsize=13, fontweight='bold', color=COLORS['base']
)
ax.legend(fontsize=10, framealpha=0.7)
ax.grid(True, color=COLORS['grid'], linewidth=0.5, axis='y')
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(colors='#555')

plt.tight_layout()
plt.savefig('../outputs/viz2_all_patients_nsi.png',
            dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
print("  saved viz2_all_patients_nsi.png")
plt.close()


# =============================================================
# VIZ 3: bar chart summary -- the clean key result slide
# mean NSI bars with individual patient dots overlaid
# =============================================================

fig3, ax = plt.subplots(figsize=(9, 6))
fig3.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg'])

bar_colors = [COLORS[p] for p in param_names_ordered]
bars = ax.bar(x_positions, mean_nsi,
              color=bar_colors, alpha=0.82,
              edgecolor='white', linewidth=1.5, width=0.55, zorder=2)

ax.errorbar(x_positions, mean_nsi, yerr=std_nsi,
            fmt='none', color=COLORS['base'],
            capsize=6, capthick=2, linewidth=2, zorder=3)

# jitter individual patient dots over bars
np.random.seed(42)
for j, pn in enumerate(param_names_ordered):
    vals = all_sensitivities[pn]
    jitter = np.random.uniform(-0.12, 0.12, size=len(vals))
    ax.scatter(j + jitter, vals,
               color='white', edgecolors=COLORS[pn],
               s=38, linewidths=1.8, zorder=5, alpha=0.9)

# label bars with mean values
for bar, mean_val, std_val in zip(bars, mean_nsi, std_nsi):
    ax.text(bar.get_x() + bar.get_width() / 2,
            mean_val + std_val + 0.15,
            f'{mean_val:.2f}',
            ha='center', va='bottom',
            fontsize=12, fontweight='bold', color=COLORS['base'])

ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.5)
ax.text(3.45, 1.05, 'NSI = 1\n(linear)', fontsize=8, color='gray')

ax.set_xticks(x_positions)
ax.set_xticklabels([PARAM_LABELS[p] for p in param_names_ordered], fontsize=12)
ax.set_ylabel('mean NSI (±10% perturbation)', fontsize=11, color='#555')
ax.set_title(
    'Parameter Sensitivity: Why $\\rho$ and $\\varphi$ Can Be Uniform\n'
    'mean NSI across all patients  |  dots = individual patients',
    fontsize=13, fontweight='bold', color=COLORS['base']
)
ax.grid(True, color=COLORS['grid'], linewidth=0.5, axis='y', zorder=0)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(colors='#555')
ax.set_ylim(0, mean_nsi.max() * 1.35)

plt.tight_layout()
plt.savefig('../outputs/viz3_summary_bars.png',
            dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
print("  saved viz3_summary_bars.png")
plt.close()


# =============================================================
# VIZ 4: trajectory grid -- all patients, baseline vs +10% ps vs +10% rho
# shows visually that ps has bigger effect on curve shape than rho
# =============================================================

print("running trajectory comparison grid...")

n_pts = len(valid_patients)
ncols = 4
nrows = (n_pts + ncols - 1) // ncols

fig4, axes4 = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
fig4.patch.set_facecolor(COLORS['bg'])
fig4.suptitle(
    'PSA Trajectories: Baseline vs +10% Perturbation (all patients)\n'
    'red dashed = +10% $p_s$ (sensitive)   |   blue dotted = +10% $\\rho$ (insensitive)   |   gray dots = data',
    fontsize=12, fontweight='bold', color=COLORS['base'], y=1.01
)

for idx, pt_key in enumerate(valid_patients):
    row_i, col_i = divmod(idx, ncols)
    if nrows > 1:
        ax = axes4[row_i, col_i]
    else:
        ax = axes4[col_i]
    ax.set_facecolor(COLORS['bg'])

    p = trial[pt_key]
    t_d, psa_d, tx_d = p[:, 0], p[:, 1], p[:, 2]
    psa0 = psa_d[0]
    D0, S0 = get_initial_conditions(psa0)

    P_b   = run_model(t_d, tx_d, BASE_PS, BASE_ALPHA, BASE_RHO, BASE_PHI, psa0, D0, S0)
    P_ps  = run_model(t_d, tx_d, BASE_PS * 1.1, BASE_ALPHA, BASE_RHO, BASE_PHI, psa0, D0, S0)
    P_rho = run_model(t_d, tx_d, BASE_PS, BASE_ALPHA, BASE_RHO * 1.1, BASE_PHI, psa0, D0, S0)

    # treatment shading
    for i in range(len(t_d) - 1):
        if tx_d[i] == 1:
            ax.axvspan(t_d[i], t_d[i + 1], alpha=0.06, color='gray', linewidth=0)

    ax.plot(t_d, P_b,   '-',  color=COLORS['base'], linewidth=1.6, label='baseline')
    ax.plot(t_d, P_ps,  '--', color=COLORS['ps'],   linewidth=1.3, alpha=0.9, label='+10% ps')
    ax.plot(t_d, P_rho, ':',  color=COLORS['rho'],  linewidth=1.3, alpha=0.9, label='+10% rho')
    ax.scatter(t_d, psa_d, s=10, color=COLORS['data'], zorder=5, alpha=0.65)

    ax.set_title(pt_key, fontsize=9, color=COLORS['base'], pad=3)
    ax.tick_params(labelsize=7, colors='#555')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, color=COLORS['grid'], linewidth=0.4)
    if col_i == 0:
        ax.set_ylabel('PSA (ng/mL)', fontsize=8, color='#555')
    if row_i == nrows - 1:
        ax.set_xlabel('days', fontsize=8, color='#555')

# legend in last panel
handles = [
    plt.Line2D([0], [0], color=COLORS['base'], linewidth=2, label='baseline'),
    plt.Line2D([0], [0], color=COLORS['ps'],   linewidth=1.5, linestyle='--', label='+10% ps'),
    plt.Line2D([0], [0], color=COLORS['rho'],  linewidth=1.5, linestyle=':',  label='+10% rho'),
    plt.Line2D([0], [0], color=COLORS['data'], linewidth=0, marker='o', markersize=5, label='data'),
]
if nrows > 1:
    axes4.flat[-1].legend(handles=handles, fontsize=8, framealpha=0.8)
else:
    axes4[-1].legend(handles=handles, fontsize=8, framealpha=0.8)

# hide unused panels
for idx in range(len(valid_patients), nrows * ncols):
    row_i, col_i = divmod(idx, ncols)
    if nrows > 1:
        axes4[row_i, col_i].set_visible(False)
    else:
        axes4[col_i].set_visible(False)

plt.tight_layout()
plt.savefig('../outputs/viz4_all_trajectories.png',
            dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
print("  saved viz4_all_trajectories.png")
plt.close()


# print summary table
print("\n" + "=" * 55)
print("FINAL RESULTS SUMMARY")
print("=" * 55)
print(f"{'param':<8} {'mean NSI':>10} {'std':>8}  {'conclusion'}")
print("-" * 55)
for pn in param_names_ordered:
    vals = all_sensitivities[pn]
    m, s = np.mean(vals), np.std(vals)
    conclusion = "high: patient-specific" if m > 2.0 else "low: uniform ok"
    print(f"{pn:<8} {m:>10.3f} {s:>8.3f}  {conclusion}")
print("=" * 55)
print("\nall 4 figures saved!")
