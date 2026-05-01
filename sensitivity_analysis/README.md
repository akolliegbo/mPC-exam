# sensitivity + identifiability analysis
### IMO2 Spring 2026 — Exam 3

---

## scripts

### `model_template.py`
the only file that changes between models. contains three clearly labeled sections for groupmates to fill in. send this to each groupmate and ask them to complete their section.

### `generalized_sensitivity.py`
the full analysis pipeline. does not change between models — just swap in a completed `model_template.py` and run it. produces:
- NSI bar chart across all patients (`sensitivity_nsi.png`)
- identifiability heatmap (`identifiability_heatmap.png`)
- per-patient PSA trajectory comparison (`sensitivity_trajectories.png`)
- cross-model comparison chart when you call `compare_models()` at the bottom (`cross_model_comparison.png`)

---

## from each of y'all

i need exactly four things. can you fill in these three functions in `model_template.py` and send it back?

### 1. `BASE_PARAMS`: parameter names and baseline values
a plain python dict. keys are whatever y'all named parameters, values are the calibrated baseline estimates.

```python
# pk/pd example
BASE_PARAMS = {
    'ke':    0.12,   # elimination rate constant (day^-1)
    'kd':    0.85,   # drug effect rate on tumor (day^-1)
    'EC50':  2.40,   # drug concentration at half-max effect (ng/mL)
    'gamma': 0.30,   # PSA production scaling
}

# for example
BASE_PARAMS = {
    'r':     0.18,   # tumor growth rate (day^-1)
    'K':     500.0,  # carrying capacity (arbitrary units)
    'd':     1.10,   # treatment-driven death rate (day^-1)
    'kappa': 0.04,   # PSA per unit tumor volume
}
```

### 2. `ode(t, y, params, Tx)`: the right-hand side of the ODE
a function that takes the current state `y`, the params dict, and the treatment indicator `Tx` (1=on, 0=off), and returns `dy/dt` as a list. `t` is passed by scipy but usually unused.

```python
# pk/pd example
def ode(t, y, params, Tx):
    C, P = [max(x, 0.0) for x in y]          # drug concentration, PSA
    ke    = params['ke']
    kd    = params['kd']
    EC50  = params['EC50']
    gamma = params['gamma']
    effect = (kd * C * Tx) / (EC50 + C)       # hill-type drug effect
    dC = -ke * C * Tx                          # drug clears when on treatment
    dP = gamma - effect * P                    # PSA suppressed by drug effect
    return [dC, dP]

# tumor burden example
def ode(t, y, params, Tx):
    V, P = [max(x, 0.0) for x in y]          # tumor volume, PSA
    r     = params['r']
    K     = params['K']
    d     = params['d']
    kappa = params['kappa']
    dV = r * V * (1 - V/K) - d * Tx * V      # logistic growth minus treatment
    dP = kappa * V - 0.08 * P                 # PSA tracks tumor volume
    return [dV, dP]
```

### 3. state variable order + which index is the observable
 what is in `y[0]`, `y[1]`, etc., and which one is PSA (or equivalent observable). this determines the one line to change in `run_model()`:

```python
out.append(y[0])   # change 0 to whichever index is PSA in their state vector
```

### 4. `get_initial_conditions(params, output0)` — how to initialize state from patient data
`output0` is the first observed PSA value for a patient. return the starting state vector `y0`.

```python
# pk/pd example
def get_initial_conditions(params, output0):
    C0 = 0.0          # no drug in system at t=0 (pre-treatment)
    P0 = output0      # PSA starts at observed baseline
    return [C0, P0]

# tumor burden example
def get_initial_conditions(params, output0):
    V0 = output0 / params['kappa']   # infer tumor volume from PSA
    P0 = output0
    return [V0, P0]
```

---

## how to run

1. get completed `model_template.py` files for each model
2. in `generalized_sensitivity.py`, replace the four sections at the top with the content (labeled "FILL IN 1/2/3")
3. run `generalized_sensitivity.py` — all plots save automatically
4. to compare all three models, uncomment and call `compare_models()` at the bottom with all three result dicts

---

## what the analysis actually computes

- **sensitivity (NSI):** perturbs each parameter by ±10%, measures mean relative change in PSA trajectory, normalizes by perturbation size. NSI >> 1 = patient-specific. NSI ≈ 1 = safe to treat as uniform across patients.
- **identifiability:** samples each parameter 200 times over a ±50% range, records PSA trajectory change fingerprints, correlates fingerprints between parameter pairs. high correlation = parameters cannot be independently estimated from PSA data alone.
- **cross-model comparison:** runs identical analysis on all three models, plots NSI rankings side by side. consistent ranking = biologically robust finding. inconsistent = model-dependent, needs richer data.
