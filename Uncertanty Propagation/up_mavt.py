import math
import numpy as np

class Measurement:
    def __init__(self, value, uncertainty):
        if uncertainty < 0:
            raise ValueError("Uncertainty must be non-negative")
        self.value = value
        self.uncertainty = uncertainty

    # String representation
    def __str__(self):
        return f"{self.value:.6g} ± {self.uncertainty:.6g}"

    # Convert to Measurement if needed
    def _convert_to_measurement(self, other):
        if isinstance(other, (int, float)):
            return Measurement(other, 0)
        if not isinstance(other, Measurement):
            raise TypeError("Unsupported operand type")
        return other

    # Addition operator
    def __add__(self, other):
        other = self._convert_to_measurement(other)
        return Measurement(
            self.value + other.value,
            (self.uncertainty**2 + other.uncertainty**2)**0.5
        )

    # Subtraction operator
    def __sub__(self, other):
        other = self._convert_to_measurement(other)
        return Measurement(
            self.value - other.value,
            (self.uncertainty**2 + other.uncertainty**2)**0.5
        )

    # Multiplication operator
    def __mul__(self, other):
        other = self._convert_to_measurement(other)
        result_value = self.value * other.value
        result_uncertainty = abs(result_value * ((self.uncertainty / self.value)**2 + (other.uncertainty / other.value)**2)**0.5)
        return Measurement(result_value, result_uncertainty)

    # Division operator
    def __truediv__(self, other):
        other = self._convert_to_measurement(other)
        if other.value == 0:
            raise ZeroDivisionError("Division by zero encountered in Measurement")
        result_value = self.value / other.value
        if self.value == 0:  # Handle special case where numerator is 0
            result_uncertainty = self.uncertainty # Uncertainty in result is 0
        else:
            result_uncertainty = abs(result_value * ((self.uncertainty / self.value)**2 + (other.uncertainty / other.value)**2)**0.5)
        return Measurement(result_value, result_uncertainty)

    # Unary negation
    def __neg__(self):
        return Measurement(-self.value, self.uncertainty)

    # Power operator
    def __pow__(self, other):
        if isinstance(other, (int, float)):
            result_value = self.value ** other
            result_uncertainty = abs(other * (self.value ** (other - 1)) * self.uncertainty)
            return Measurement(result_value, result_uncertainty)
        raise TypeError("Power operation only supports numeric exponents")

    # Square root method
    def sqrt(self):
        if self.value < 0:
            raise ValueError("Cannot compute the square root of a negative number")
        result_value = math.sqrt(self.value)
        result_uncertainty = result_value/2 * (self.uncertainty / self.value)
        return Measurement(result_value, result_uncertainty)

    # Reverse operations
    __radd__ = __add__
    __rsub__ = lambda self, other: Measurement(other, 0) - self
    __rmul__ = __mul__
    __rtruediv__ = lambda self, other: Measurement(other, 0) / self
    
### TESTS
import math
from typing import Callable, List

def rss(func: Callable, *measurements: List['Measurement']) -> 'Measurement':
    """
    Compute the result and uncertainty of a multivariable function 'func' using finite differences.
    Parameters:
        func: Callable
            Function whose uncertainty is to be propagated.
        measurements: List[Measurement]
            Measurement objects (values and uncertainties).
    Returns:
        Measurement: Resulting value and propagated uncertainty.
    Raises:
        ValueError: If measurements is empty or func is not callable.
    """
    if not measurements:
        raise ValueError("At least one measurement must be provided.")
    if not callable(func):
        raise TypeError("func must be callable.")

    nominal_values = [m.value for m in measurements]
    value = func(*nominal_values)

    squared_uncertainty = 0
    for i, m in enumerate(measurements):
        epsilon = max(1e-6 * abs(m.value), 1e-8)
        perturbed_values_plus = nominal_values.copy()
        perturbed_values_minus = nominal_values.copy()
        perturbed_values_plus[i] += epsilon
        perturbed_values_minus[i] -= epsilon
        try:
            df_dxi = (func(*perturbed_values_plus) - func(*perturbed_values_minus)) / (2 * epsilon)
        except Exception as e:
            raise ValueError(f"Failed to compute partial derivative for input {i}: {e}")
        squared_uncertainty += (df_dxi * m.uncertainty) ** 2

    total_uncertainty = math.sqrt(squared_uncertainty)
    return Measurement(value, total_uncertainty)


def rss_rel(func: Callable, *measurements: List['Measurement']) -> 'Measurement':
    """
    Compute the result and uncertainty of a multivariable function 'func' using finite differences.
    Parameters:
        func: Callable
            Function whose uncertainty is to be propagated.
        measurements: List[Measurement]
            Measurement objects (values and uncertainties).
    Returns:
        Measurement: Resulting value and propagated uncertainty.
    Raises:
        ValueError: If measurements is empty or func is not callable.
    """
    if not measurements:
        raise ValueError("At least one measurement must be provided.")
    if not callable(func):
        raise TypeError("func must be callable.")

    nominal_values = [m.value for m in measurements]
    value = func(*nominal_values)

    squared_uncertainty = 0
    for i, m in enumerate(measurements):
        epsilon = max(1e-6 * abs(m.value), 1e-8)
        perturbed_values_plus = nominal_values.copy()
        perturbed_values_minus = nominal_values.copy()
        perturbed_values_plus[i] += epsilon
        perturbed_values_minus[i] -= epsilon
        try:
            df_dxi = (func(*perturbed_values_plus) - func(*perturbed_values_minus)) / (2 * epsilon)
        except Exception as e:
            raise ValueError(f"Failed to compute partial derivative for input {i}: {e}")
        squared_uncertainty += (df_dxi * m.uncertainty/m.value) ** 2

    total_uncertainty = math.sqrt(squared_uncertainty*value)
    return Measurement(value, total_uncertainty)

def parse_distribution_as_normal(data):
    if isinstance(data, dict):
        data_values = list(data.values())[0]
        if list(data.keys())[0] == "Normal":
            mu = data_values[0]
            sigma = data_values[1]
        elif list(data.keys())[0] == "Uniform":
            a = data_values[0]
            b = data_values[1]
            mu = (a + b) / 2
            sigma = (b - a) / math.sqrt(12)
        elif list(data.keys())[0] == "Discrete":
            values = data_values[0]
            mu = np.mean(values)
            sigma = np.std(values)
        elif list(data.keys())[0] == "Triangular":
            a = data_values[0]
            b = data_values[1]
            c = data_values[2]
            mu = b
            sigma = np.std([a, b, c])
        else:
            raise ValueError(f"Unsupported distribution type in dictionary.{data.keys()}")
    else:
        # If it is just one exact value
        mu = data
        sigma = 0
    return mu, sigma

def mc_rss_mavt(alternatives, vf_list, weights, mc_simulations=1000):
    results = []
    for i in range(len(alternatives)):
        weighted_value = []
        temp_weights = []
        temp_values = []    
        for j in range(len(vf_list)):
            # Extract the data to evaluate
            data = alternatives[i][j]
            mu, sigma = parse_distribution_as_normal(data)
            
            sampled_values = []
            for sim in range(mc_simulations):  # MC Sampling
                sample = np.random.normal(mu, sigma)
                for vf in vf_list[j]:
                    sampled_values.append(vf(sample))
            mean_v = np.mean(sampled_values)
            std_v = np.std(sampled_values)
            value = Measurement(mean_v, std_v)

            mean_w = np.mean([weights[j][0], weights[j][1]])
            std_w = np.std([weights[j][0], weights[j][1]])
            weight = Measurement(mean_w, std_w)

            temp_values.append(value)
            temp_weights.append(weight)
        new_weights = [w/sum(temp_weights) for w in temp_weights]
        for k in range(len(temp_values)):
            weighted_value.append(temp_values[k] * new_weights[k])
        total_weighted_value = sum(weighted_value)
        results.append(total_weighted_value)
    return results


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

def weighted_sum(intermediate_results):
    total = 0
    for weight, value in intermediate_results:
        total += weight * value
    return total

def geometric_mean(intermediate_results):
    product = 1
    for weight, value in intermediate_results:
        product *= value ** weight
    return product

def harmonic_mean(intermediate_results):
    denom = 0
    for weight, value in intermediate_results:
        if value != 0:
            denom += weight / value
        else:
            return 0
    return 1 / denom

def minimum(intermediate_results):
    min_value = min(value for weight, value in intermediate_results)
    return min_value

def maximum(intermediate_results):
    max_value = max(value for weight, value in intermediate_results)
    return max_value

# Value sampling function
def sample_to_values_normal(input_data, value_function):
    mu, sigma = parse_distribution_as_normal(input_data)
    sample = np.random.normal(mu, sigma)
    value = value_function(sample)
    return value

def sample_to_values(data, value_function):
    if isinstance(data, dict):
        data_values = list(data.values())[0]
        if list(data.keys())[0] == "Normal":
            mu = data_values[0]
            sigma = data_values[1]
            sample = np.random.normal(mu, sigma)
            
        elif list(data.keys())[0] == "Uniform":
            a = data_values[0]
            b = data_values[1]
            sample = np.random.uniform(a, b)

        elif list(data.keys())[0] == "Discrete":
            values = data_values[0]
            sample = np.random.choice(values)

        elif list(data.keys())[0] == "Triangular":
            a = data_values[0]
            b = data_values[1]
            c = data_values[2]
            sample = np.random.triangular(a, b, c)

        else:
            raise ValueError(f"Unsupported distribution type in dictionary.{data.keys()}")
    else:
        sample = data
    
    value = float(value_function(sample))
    return value

def mc_simulation(alternatives, vf_list, posterior_samples_list, aggregation_method, sim_runs=1000, bin_size=100, strict=True, strict_d=True, weight_fixed=-1):
    """
    Perform a series of Monte Carlo simulations.

    Parameters:
    - weight_fixed: If >= 0, the number of weight sets to sample (each used for `sim_runs` internal runs).
                   If < 0, use a single random weight set for all runs.
    """
    all_results = []  # This will store results for each weight set
    all_relative_errors = []

    # Determine the number of weight sets to use
    if weight_fixed < 0:
        weight_iterations = 1
    else:
        weight_iterations = weight_fixed

    for t in range(weight_iterations):
        # Step 0: Select ONE set of weights for this entire weight iteration
        random_expert_idx = np.random.randint(0, len(posterior_samples_list))
        posterior_samples = posterior_samples_list[random_expert_idx]
        idx = np.random.randint(0, posterior_samples.shape[0])
        sampled_weights_all = posterior_samples[idx, :]
        sampled_weights = sampled_weights_all[:len(vf_list)]  # Extract criteria weights

        # Store results for this weight set
        results = []

        # Step 1: Perform `sim_runs` with the fixed weights
        for _ in range(sim_runs):
            temp_results = []
            for a, alt in enumerate(alternatives):
                intermediate_results = []
                for c, criterion_data in enumerate(alt):
                    if strict:
                        # Use the same expert's value function (if vf_list is per expert)
                        # Note: Your current vf_list structure is per criterion, not per expert
                        v_f = np.random.choice(vf_list[c])  # Randomly select from available VFs for this criterion
                    else:
                        # Randomly select a value function for this criterion
                        v_f = np.random.choice(vf_list[c])

                    # Sample data
                    if strict_d:
                        sampled_value = sample_to_values(criterion_data, v_f)
                    else:
                        sampled_value = sample_to_values_normal(criterion_data, v_f)

                    weight = sampled_weights[c]
                    intermediate_results.append([weight, sampled_value])

                alternative_value = aggregation_method(intermediate_results)
                temp_results.append(alternative_value)
            results.append(temp_results)

        # Convert to numpy array and compute relative errors for this weight set
        results_array = np.array(results)  # Shape: (sim_runs, num_alternatives)
        num_alternatives = results_array.shape[1]
        relative_errors = []

        for i in range(num_alternatives):
            res = results_array[:, i]
            MC_avg = np.mean(res)
            if len(res) > 1:  # Avoid division by zero for single runs
                MC_std = np.sqrt(1/(len(res)*(len(res)-1)) * np.sum((res - MC_avg)**2))
            else:
                MC_std = 0
            relative_errors.append(MC_std/MC_avg if MC_avg != 0 else 0)

        # Store results for this weight set
        all_results.append(results_array)
        all_relative_errors.append(relative_errors)

    # Return all results and errors (list of arrays, one per weight set)
    return all_results, all_relative_errors