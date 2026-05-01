"""
generalized sensitivity + identifiability template
================================================
your groupmates need to fill in THREE things:
  1. BASE_PARAMS       -- their parameter names and baseline values
  2. get_initial_conditions()  -- how to set up y0 from params
  3. ode()             -- the right-hand side of their ODE system

everything below the "DO NOT TOUCH" line stays identical across all models
"""

import numpy as np
from scipy.integrate import solve_ivp

# ================================================================
# FILL IN 1: parameter names and baseline values
# just a dict -- add/remove keys as needed for your model
# ================================================================

BASE_PARAMS = {
    'ps':    0.10,   # stem cell self-renewal probability
    'alpha': 0.74,   # cytotoxicity rate
    'rho':   0.06,   # PSA production rate
    'phi':   0.08,   # PSA decay rate
}

# human readable labels for plots -- match keys above
PARAM_LABELS = {
    'ps':    'ps (self-renewal)',
    'alpha': 'alpha (cytotoxicity)',
    'rho':   'rho (PSA production)',
    'phi':   'phi (PSA decay)',
}

MODEL_NAME = 'Brady-Nicholls 2021'


# ================================================================
# FILL IN 2: initial conditions
# given params and the first observed output value (psa0),
# return the initial state vector y0 that your ODE expects
# ================================================================

def get_initial_conditions(params, output0):
    """
    params:  the BASE_PARAMS dict for this model
    output0: the first observed PSA value for this patient (scalar)
    returns: list or array -- the initial state vector [y1_0, y2_0, ...]
             must match the order your ode() function expects
    """
    # brady-nicholls example:
    # at quasi-steady state, P = rho*D/phi, so D0 = P0*phi/rho
    # S0 is small (stem cells are rare at treatment start)
    D0 = output0 * params['phi'] / params['rho']
    S0 = 0.01 * D0
    P0 = output0
    return [S0, D0, P0]

    # --- groupmate example (swap the above for theirs): ---
    # maybe their model just has tumor volume V and PSA P:
    #   V0 = output0 / params['kappa']   # kappa = PSA per unit volume
    #   P0 = output0
    #   return [V0, P0]
    #
    # or maybe they have 5 state variables:
    #   return [params['S0'], params['R0'], params['D0'], output0, 0.0]


# ================================================================
# FILL IN 3: the ODE right-hand side
# takes current state y and returns dy/dt
# Tx is treatment on (1) or off (0) at this moment
# ================================================================

def ode(t, y, params, Tx):
    """
    t:      current time (scipy passes this, often unused)
    y:      current state vector -- same order as get_initial_conditions
    params: the full params dict
    Tx:     treatment indicator, 1 = on, 0 = off
    returns: list of derivatives [dy1/dt, dy2/dt, ...]
    """
    # brady-nicholls example:
    ps    = params['ps']
    alpha = params['alpha']
    rho   = params['rho']
    phi   = params['phi']
    lam   = np.log(2)

    S, D, P = [max(x, 0.0) for x in y]
    total = max(S + D, 1e-12)
    frac  = S / total

    dS = frac * ps * lam * S
    dD = (1 - frac * ps) * lam * S - alpha * Tx * D
    dP = rho * D - phi * P
    return [dS, dD, dP]

    # --- groupmate example (swap the above for theirs): ---
    # maybe their model is a simple logistic tumor + PSA:
    #   V, P = [max(x, 0.0) for x in y]
    #   dV = params['r'] * V * (1 - V/params['K']) - params['d'] * Tx * V
    #   dP = params['kappa'] * V - params['delta'] * P
    #   return [dV, dP]


# ================================================================
# DO NOT TOUCH BELOW THIS LINE
# run_model() uses your ode() and get_initial_conditions() above
# and is called by all the analysis functions
# ================================================================

def run_model(t_data, tx_data, params):
    """
    integrates the ODE piecewise across measurement intervals
    returns array of model output (first observable, e.g. PSA) at each timepoint
    
    t_data:  array of measurement times in days
    tx_data: array of treatment on/off at each timepoint (1=on, 0=off)
    params:  dict -- must include 'psa0' key (injected by compute_nsi per patient)
    """
    output0 = params['psa0']
    y = get_initial_conditions(params, output0)
    out = [output0]

    for i in range(len(t_data) - 1):
        t0 = float(t_data[i])
        t1 = float(t_data[i + 1])
        Tx = float(tx_data[i])

        # wrap ode() to match scipy's expected signature f(t, y)
        def rhs(t, y, Tx=Tx, params=params):
            return ode(t, y, params, Tx)

        sol = solve_ivp(rhs, (t0, t1), y, max_step=0.5, rtol=1e-5, atol=1e-8)

        if sol.success:
            y = [max(x, 1e-15) for x in sol.y[:, -1]]

        out.append(y[0])  # <-- index 0 assumes PSA/observable is first state var
                          #     change this index if your observable is elsewhere
                          #     e.g. y[2] if PSA is the third state variable

    return np.array(out)
