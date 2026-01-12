# PILE-BWT
#
# This module is the implementation of the PILE-BWT method
# The most important part is the definition of the constraints function

#################################################################################
# Import third party libraries
import numpy as np
import scipy.optimize as opt
from functools import partial
#################################################################################

#################################################################################
# Define all inequality that define the space of weights according to PILE-BWT
from constraints import constraints_func

#################################################################################

# Minimization problem, objective is the auxiliary variable z
def objective(x, var=-1):
    return x[var]  # z is the last variable in the array

def slsqp_run(objective_func, x0, bounds, num_criteria, dict_data, z_star):
    result = opt.minimize(
    objective_func,
    x0,
    method='SLSQP',
    constraints=[
        {'type': 'ineq', 'fun': constraints_func, 'args': (dict_data, z_star,)},
        {'type': 'eq', 'fun': lambda x: np.sum(x[:num_criteria]) - 1}
    ],
    bounds=bounds,
    options={'maxiter': 1000, 'ftol': 1e-3}
    )
    return result

def trust_constr_run(objective_func, x0, bounds, num_criteria, dict_data, z_star):
    nonlinear_constraints = [
        opt.NonlinearConstraint(
            lambda x: constraints_func(x, dict_data, z_star),
            lb=0,
            ub=np.inf,
        ),
        opt.NonlinearConstraint(
            lambda x: np.sum(x[:num_criteria]) - 1,
            lb=0,
            ub=0,
        ),
    ]

    result_trust = opt.minimize(
        objective_func,
        x0,
        method='trust-constr',
        constraints=nonlinear_constraints,
        bounds=bounds,
        options={'maxiter': int(2e3), 'gtol': 1e-3, 'xtol': 1e-6}
    )
    return result_trust  
  
def cobyqa_run(objective_func, x0, bounds, num_criteria, dict_data, z_star):
    # COBYQA in SciPy requires bounds as a list of tuples (min, max)
    result_cobyqa = opt.minimize(
        objective_func,
        x0,
        method='COBYQA',
        bounds=bounds,
        constraints=[
            {'type': 'ineq', 'fun': constraints_func, 'args': (dict_data, z_star)},
            {'type': 'eq', 'fun': lambda x: np.sum(x[:num_criteria]) - 1}
        ],
        options={'maxiter': 2000}
    )
    return result_cobyqa

def bwt(dict_data, var=-1, z_star=None, type='min'):
    # Count the number of groups and criteria
    num_criteria = sum(len(group_data['criteria']) for group_data in dict_data.values())

    # Initial guess: for all w_i + z (auxiliary variable)
    # We ensure it starts within bounds
    x0 = np.ones(num_criteria + 1) / (num_criteria)
    if z_star is not None:
        x0 = bwt(dict_data, var=-1, z_star=None, type='min')["solver_result"].x
        print("\033[93mRe-initializing x0 from known optimal solution without z_star constraint...\033[0m")

    # Bounds: 
    # Positive weights: w_i >= 0, 
    # Positive auxiliary variable z >= 0
    bounds = [(0.001, 1) for _ in range(num_criteria)] + [(0, None)]
    # print("Bounds:", bounds)

    # Solve optimization problem
    print("Running optimization...")
    objective_with_var = partial(objective, var=var)
    # Scipy optimize only minimizes functions so for maximization we negate the objective
    if type == 'max':
        def neg_objective(x):
            return -objective_with_var(x)
        objective_func = neg_objective
    else:
        objective_func = objective_with_var

    # Try SLSQP first
    # if it fails catastrophically we try trust-constr as a fallback
    if z_star is None:
        try:
            result = slsqp_run(objective_func, x0, bounds, num_criteria, dict_data, z_star)
        except Exception as e:
            print(f"SLSQP raised exception: {e}")
            class dummy_result:
                success = False
                message = str(e)
            result = dummy_result()
            pass
    
    # If SLSQP fails or does not converge, try trust-constr
    if z_star is not None or not result.success:
        if z_star is None:
            print(f"\033[93mSLSQP failed: {getattr(result, 'message', 'No message provided')}\033[0m")
            print(f"\033[93mTrying 'trust-constr' as fallback...\033[0m")
        else:
            print(f"\033[93mRe-running with 'trust-constr' due to z_star constraint...\033[0m")
        try:
            result = trust_constr_run(objective_func, x0, bounds, num_criteria, dict_data, z_star)
        except Exception as e:
            print(f"\033[91mtrust-constr fallback raised exception: {e}\033[0m")
            pass

    if not result.success:
        print(f"\033[93mtrust-constr failed (message: {getattr(result, 'message', None)})\033[0m")
        print("\033[93mTrying 'COBYQA' (SciPy) as fallback...\033[0m")
        try:
            result = cobyqa_run(objective_func, x0, bounds, num_criteria, dict_data, z_star)
            if result.success:
                print("\033[92mCOBYQA succeeded.\033[0m")
        except Exception as e:
            print(f"\033[91mCOBYQA fallback raised exception: {e}\033[0m")
            pass

    if not result.success :
        print(f"\033[93mCOBYQA failed (message: {getattr(result, 'message', None)})\033[0m")
        print("\033[93mFallback to known optimal solution...\033[0m")
        result = bwt(dict_data, var=-1, z_star=None, type='min')["solver_result"]


    # Extract weights for criteria and groups
    weights = result.x[:num_criteria]
    weights_dict = {}
    current_index = 0
    for group_name, group_data in dict_data.items():
        group_criteria = list(group_data['criteria'].keys())
        weights_dict[group_name] = {
            crit: weights[current_index + i] for i, crit in enumerate(group_criteria)
        }
        current_index += len(group_criteria)
    # Add auxiliary variable z
    z = result.x[-1]

    return {
        'solver_result': result,
        'criteria_weights': weights_dict,
        'z': z
    }