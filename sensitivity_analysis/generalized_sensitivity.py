"""
generalized sensitivity + identifiability analysis
works on ANY model your groupmates built -- you just need to swap in:
  1. the model function (run_model)
  2. the baseline parameter dict (BASE_PARAMS)
  3. the patient data loading section

the script does three things:
  A. sensitivity analysis   -- how much does output change when each param changes 10%?
  B. identifiability        -- which params are correlated (and therefore hard to tell apart)?
  C. cross-model comparison -- does the ranking of param importance hold across models?
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
import scipy.io as sio
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# SECTION 1: SWAP THIS OUT FOR EACH MODEL
# your groupmates give you their model function and their params
# the only rule: run_model must take (t_data, tx_data, params_dict)
# and return an array of output values (psa, tumor size, whatever)
# ================================================================

def run_model(t_data, tx_data, params):
    """
    THIS IS THE BRADY-NICHOLLS MODEL -- swap for your groupmates' models
    
    params dict keys:
      ps     -- stem cell self renewal probability (0 to 1)
      alpha  -- cytotoxicity (how fast treatment kills non-stem cells)
      rho    -- psa production rate
      phi    -- psa decay rate

    to swap in a different model, just rewrite this function
    keeping the same signature: (t_data, tx_data, params) -> array
    """
    ps    = params['ps']
    alpha = params['alpha']
    rho   = params['rho']
    phi   = params['phi']
    lam   = np.log(2)  # stem cell division rate, fixed from paper

    # initial conditions from first psa value
    psa0 = params.get('psa0', 6.06)
    D0   = psa0 * phi / rho
    S0   = 0.01 * D0

    y = [S0, D0, psa0]
    P_out = [psa0]

    for i in range(len(t_data) - 1):
        t0 = float(t_data[i])
        t1 = float(t_data[i + 1])
        Tx = float(tx_data[i])

        def ode(t, y, Tx=Tx):
            S, D, P = [max(x, 0.0) for x in y]
            total = max(S + D, 1e-12)
            frac  = S / total
            dS = frac * ps * lam * S
            dD = (1 - frac * ps) * lam * S - alpha * Tx * D
            dP = rho * D - phi * P
            return [dS, dD, dP]

        sol = solve_ivp(ode, (t0, t1), y, max_step=0.5, rtol=1e-5, atol=1e-8)
        if sol.success:
            y = [max(x, 1e-15) for x in sol.y[:, -1]]
        P_out.append(y[2])

    return np.array(P_out)


# baseline params for THIS model
# swap these out when running on your groupmates' models
BASE_PARAMS = {
    'ps':    0.10,   # stem cell self renewal
    'alpha': 0.74,   # cytotoxicity
    'rho':   0.06,   # psa production rate
    'phi':   0.08,   # psa decay rate
}

# human readable labels for each param -- add yours here
PARAM_LABELS = {
    'ps':    'ps (self-renewal)',
    'alpha': 'alpha (cytotoxicity)',
    'rho':   'rho (PSA production)',
    'phi':   'phi (PSA decay)',
    # if your groupmates have different params, add them like:
    # 'beta':  'beta (whatever it means)',
    # 'k':     'k (growth rate)',
}

# model name -- change this when running on each model
MODEL_NAME = 'Brady-Nicholls 2021'


# ================================================================
# SECTION 2: LOAD PATIENT DATA
# ================================================================

print("loading data...")
trial    = sio.loadmat('../data/TrialPatientData.mat')
patients = sorted([k for k in trial.keys() if k.startswith('P')])
print(f"loaded {len(patients)} patients")

# collect patient time series into a list of dicts
patient_data = []
for pt_key in patients:
    p = trial[pt_key]
    if len(p) < 5:
        continue
    patient_data.append({
        'id':   pt_key,
        't':    p[:, 0],   # time in days
        'psa':  p[:, 1],   # absolute psa
        'tx':   p[:, 2],   # treatment on/off
    })

print(f"using {len(patient_data)} patients with >= 5 timepoints")


# ================================================================
# SECTION 3: CORE ANALYSIS FUNCTIONS
# these are model-agnostic -- they work on any run_model function
# ================================================================

PERTURB = 0.10  # 10% perturbation size

def compute_nsi(t_data, tx_data, params, param_name):
    """
    compute normalized sensitivity index for ONE param on ONE patient
    
    NSI = mean(|output_perturbed - output_base| / output_base) / perturbation_fraction
    
    NSI > 1  : nonlinear amplification -- small param change -> big output change
    NSI ~ 1  : linear proportional relationship
    NSI < 1  : output less sensitive than param change (robust to this param)
    """
    # inject psa0 into params for initial condition
    params_with_psa = dict(params, psa0=t_data[0] if 'psa0' not in params else params['psa0'])
    params_with_psa['psa0'] = psa_data[0] if 'psa0' not in params else params['psa0']

    P_base = run_model(t_data, tx_data, params_with_psa)

    # perturb up
    p_up = dict(params_with_psa)
    p_up[param_name] *= (1 + PERTURB)
    P_up = run_model(t_data, tx_data, p_up)

    # perturb down
    p_dn = dict(params_with_psa)
    p_dn[param_name] *= (1 - PERTURB)
    P_dn = run_model(t_data, tx_data, p_dn)

    rel_up = np.abs(P_up - P_base) / (np.abs(P_base) + 0.001)
    rel_dn = np.abs(P_dn - P_base) / (np.abs(P_base) + 0.001)
    nsi = ((np.mean(rel_up) + np.mean(rel_dn)) / 2) / PERTURB

    return nsi, P_base, P_up, P_dn


def compute_all_nsi(patient_data, params):
    """
    run NSI for every param on every patient
    returns: dict of {param_name: [nsi_patient1, nsi_patient2, ...]}
    """
    param_names = [k for k in params.keys() if k != 'psa0']
    results = {pn: [] for pn in param_names}

    for pt in patient_data:
        params_pt = dict(params, psa0=pt['psa'][0])
        for pn in param_names:
            nsi, _, _, _ = compute_nsi(pt['t'], pt['tx'], params_pt, pn)
            results[pn].append(nsi)
            
    return results


def compute_identifiability(patient_data, params, n_samples=200):
    """
    identifiability analysis via parameter correlation
    
    idea: if two params produce similar CHANGES in output when perturbed,
    you can't tell them apart from data -- they're correlated / not identifiable
    
    method: sample random perturbations of all params, compute output fingerprint
    (the full psa trajectory) for each, then correlate the fingerprints
    if two params produce correlated fingerprints -> poor identifiability between them
    
    returns: correlation matrix between params
    """
    param_names = [k for k in params.keys() if k != 'psa0']
    
    # use first patient for identifiability (computationally cheaper)
    pt = patient_data[0]
    params_pt = dict(params, psa0=pt['psa'][0])
    
    # for each param, collect output deltas from many random perturbations
    # delta = output_perturbed - output_base  (the "fingerprint" of each param)
    P_base = run_model(pt['t'], pt['tx'], params_pt)
    
    fingerprints = {}
    for pn in param_names:
        deltas = []
        for _ in range(n_samples):
            scale = np.random.uniform(0.5, 1.5)  # random ±50% perturbation
            p_rand = dict(params_pt)
            p_rand[pn] *= scale
            P_rand = run_model(pt['t'], pt['tx'], p_rand)
            deltas.append(P_rand - P_base)
        fingerprints[pn] = np.array(deltas)  # shape: (n_samples, n_timepoints)
    
    # correlate the fingerprints between param pairs
    # high correlation = params have similar effect on output = hard to distinguish
    n_params = len(param_names)
    corr_matrix = np.zeros((n_params, n_params))
    
    for i, pn_i in enumerate(param_names):
        for j, pn_j in enumerate(param_names):
            # flatten fingerprints and correlate
            fi = fingerprints[pn_i].flatten()
            fj = fingerprints[pn_j].flatten()
            if np.std(fi) < 1e-10 or np.std(fj) < 1e-10:
                corr_matrix[i, j] = 0.0
            else:
                corr_matrix[i, j] = np.corrcoef(fi, fj)[0, 1]
    
    return corr_matrix, param_names


# ================================================================
# SECTION 4: RUN THE ANALYSIS
# ================================================================

print("\nrunning sensitivity analysis...")
nsi_results = compute_all_nsi(patient_data, BASE_PARAMS)

param_names = list(nsi_results.keys())
mean_nsi    = np.array([np.mean(nsi_results[pn]) for pn in param_names])
std_nsi     = np.array([np.std(nsi_results[pn]) for pn in param_names])

print("\nrunning identifiability analysis (this takes ~30 seconds)...")
corr_matrix, id_param_names = compute_identifiability(patient_data, BASE_PARAMS)

# print summary
print(f"\n{'='*55}")
print(f"RESULTS: {MODEL_NAME}")
print(f"{'='*55}")
print(f"{'param':<12} {'mean NSI':>10} {'std':>8}  conclusion")
print("-"*55)
for pn, m, s in zip(param_names, mean_nsi, std_nsi):
    label = PARAM_LABELS.get(pn, pn)
    conc  = "HIGH -- patient specific" if m > 2.0 else "low -- uniform ok"
    print(f"{pn:<12} {m:>10.3f} {s:>8.3f}  {conc}")
print(f"{'='*55}")


# ================================================================
# SECTION 5: PLOTS
# ================================================================

COLORS_LIST = ['#E63946', '#F4A261', '#457B9D', '#2A9D8F',
               '#8338EC', '#FB5607', '#3A86FF', '#06D6A0']
BG = '#FAFAFA'
BASE_COLOR = '#1D1D2E'
GRID_COLOR = '#E8E8E8'

param_colors = {pn: COLORS_LIST[i % len(COLORS_LIST)] for i, pn in enumerate(param_names)}

# --- plot A: NSI bar chart with individual patient dots ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

x = np.arange(len(param_names))
bars = ax.bar(x, mean_nsi,
              color=[param_colors[pn] for pn in param_names],
              alpha=0.82, edgecolor='white', linewidth=1.5,
              width=0.55, zorder=2)
ax.errorbar(x, mean_nsi, yerr=std_nsi,
            fmt='none', color=BASE_COLOR, capsize=6, capthick=2,
            linewidth=2, zorder=3)

np.random.seed(42)
for j, pn in enumerate(param_names):
    vals   = nsi_results[pn]
    jitter = np.random.uniform(-0.12, 0.12, size=len(vals))
    ax.scatter(j + jitter, vals,
               color='white', edgecolors=param_colors[pn],
               s=38, linewidths=1.8, zorder=5, alpha=0.9)

for bar, m, s in zip(bars, mean_nsi, std_nsi):
    ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.1,
            f'{m:.2f}', ha='center', fontsize=11,
            fontweight='bold', color=BASE_COLOR)

ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.2, alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels([PARAM_LABELS.get(pn, pn) for pn in param_names], fontsize=11)
ax.set_ylabel('mean NSI (±10% perturbation)', fontsize=11, color='#555')
ax.set_title(f'{MODEL_NAME}: Parameter Sensitivity\nmean NSI -- dots = individual patients',
             fontsize=13, fontweight='bold', color=BASE_COLOR)
ax.grid(True, color=GRID_COLOR, linewidth=0.5, axis='y', zorder=0)
ax.spines[['top','right']].set_visible(False)
ax.tick_params(colors='#555')
ax.set_ylim(0, mean_nsi.max() * 1.4)

plt.tight_layout()
plt.savefig('../outputs/sensitivity_nsi.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
print("saved sensitivity_nsi.png")
plt.close()


# --- plot B: identifiability heatmap ---
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

labels = [PARAM_LABELS.get(pn, pn) for pn in id_param_names]
im = ax.imshow(np.abs(corr_matrix), cmap='RdYlGn_r', vmin=0, vmax=1,
               aspect='auto')
plt.colorbar(im, ax=ax, label='|correlation| (higher = harder to identify separately)')

ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
ax.set_yticklabels(labels, fontsize=10)

# annotate each cell with the correlation value
for i in range(len(id_param_names)):
    for j in range(len(id_param_names)):
        ax.text(j, i, f'{corr_matrix[i,j]:.2f}',
                ha='center', va='center',
                fontsize=10, color='black', fontweight='bold')

ax.set_title(f'{MODEL_NAME}: Identifiability\nhigh correlation = params look alike to the model',
             fontsize=12, fontweight='bold', color=BASE_COLOR)
plt.tight_layout()
plt.savefig('../outputs/identifiability_heatmap.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
print("saved identifiability_heatmap.png")
plt.close()


# --- plot C: single patient trajectory showing all param perturbations ---
pt = patient_data[0]
params_pt = dict(BASE_PARAMS, psa0=pt['psa'][0])
P_base = run_model(pt['t'], pt['tx'], params_pt)

fig, axes = plt.subplots(1, len(param_names), figsize=(5*len(param_names), 5), sharey=True)
fig.patch.set_facecolor(BG)
fig.suptitle(f'{MODEL_NAME}: PSA Trajectory Sensitivity -- Patient {pt["id"]}\n'
             'gray band = ±10% perturbation range',
             fontsize=13, fontweight='bold', color=BASE_COLOR)

if len(param_names) == 1:
    axes = [axes]

for ax, pn in zip(axes, param_names):
    ax.set_facecolor(BG)
    _, _, P_up, P_dn = compute_nsi(pt['t'], pt['tx'], params_pt, pn)
    nsi = np.mean([v for v in nsi_results[pn]])

    for i in range(len(pt['t'])-1):
        if pt['tx'][i] == 1:
            ax.axvspan(pt['t'][i], pt['t'][i+1], alpha=0.07, color='gray')

    ax.fill_between(pt['t'], P_dn, P_up,
                    alpha=0.25, color=param_colors[pn])
    ax.plot(pt['t'], P_base, '-', color=BASE_COLOR, linewidth=2, label='baseline')
    ax.plot(pt['t'], P_up,  '--', color=param_colors[pn], linewidth=1.5, label='+10%')
    ax.plot(pt['t'], P_dn,  ':',  color=param_colors[pn], linewidth=1.5, label='-10%')
    ax.scatter(pt['t'], pt['psa'], s=18, color='#888', zorder=5, alpha=0.7)

    ax.set_title(f'{PARAM_LABELS.get(pn, pn)}\nNSI = {np.mean(nsi_results[pn]):.2f}',
                 fontsize=10, color=BASE_COLOR)
    ax.set_xlabel('days', fontsize=9, color='#555')
    ax.grid(True, color=GRID_COLOR, linewidth=0.4)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=8, colors='#555')
    ax.legend(fontsize=8)

axes[0].set_ylabel('PSA (ng/mL)', fontsize=10, color='#555')
plt.tight_layout()
plt.savefig('../outputs/sensitivity_trajectories.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
print("saved sensitivity_trajectories.png")
plt.close()


# ================================================================
# SECTION 6: CROSS-MODEL COMPARISON HELPER
# run this script once per model, save the nsi dicts, then call this
# to compare rankings across models
# ================================================================

def compare_models(model_results_dict):
    """
    model_results_dict = {
        'Model A': {'ps': [...], 'alpha': [...], ...},
        'Model B': {'param1': [...], 'param2': [...], ...},
    }
    plots NSI rankings side by side for each model
    biological conclusion: if ranking is preserved across models,
    the finding is ROBUST (not model-dependent)
    """
    n_models = len(model_results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5), sharey=False)
    fig.patch.set_facecolor(BG)
    if n_models == 1:
        axes = [axes]

    all_means = {}
    for ax, (model_name, nsi_dict) in zip(axes, model_results_dict.items()):
        ax.set_facecolor(BG)
        pns   = list(nsi_dict.keys())
        means = [np.mean(nsi_dict[pn]) for pn in pns]
        stds  = [np.std(nsi_dict[pn]) for pn in pns]
        all_means[model_name] = dict(zip(pns, means))

        colors = [COLORS_LIST[i % len(COLORS_LIST)] for i in range(len(pns))]
        ax.bar(range(len(pns)), means, color=colors, alpha=0.82,
               edgecolor='white', linewidth=1.5, width=0.55)
        ax.errorbar(range(len(pns)), means, yerr=stds,
                    fmt='none', color=BASE_COLOR, capsize=5, capthick=1.5)
        ax.set_xticks(range(len(pns)))
        ax.set_xticklabels([PARAM_LABELS.get(pn, pn) for pn in pns],
                           rotation=20, ha='right', fontsize=9)
        ax.set_title(model_name, fontsize=12, fontweight='bold', color=BASE_COLOR)
        ax.set_ylabel('mean NSI', fontsize=10)
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, axis='y')
        ax.spines[['top','right']].set_visible(False)
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    fig.suptitle('Cross-Model Comparison: Do Parameter Rankings Hold?',
                 fontsize=13, fontweight='bold', color=BASE_COLOR)
    plt.tight_layout()
    plt.savefig('../outputs/cross_model_comparison.png',
                dpi=150, bbox_inches='tight', facecolor=BG)
    print("saved cross_model_comparison.png")
    plt.close()
    return all_means

# example of how to call compare_models once you have all three models' results:
# 
#   results_A = compute_all_nsi(patient_data, BASE_PARAMS_A)  # your model
#   results_B = compute_all_nsi(patient_data, BASE_PARAMS_B)  # groupmate 1
#   results_C = compute_all_nsi(patient_data, BASE_PARAMS_C)  # groupmate 2
#   compare_models({'Brady-Nicholls': results_A, 'Model B': results_B, 'Model C': results_C})

print("\ndone! to compare across models: call compare_models() with all three results dicts")
print("to run on a different model: swap out run_model() and BASE_PARAMS at the top")
