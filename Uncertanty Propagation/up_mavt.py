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

def mc_rss_mavt(alternatives, vf_list, weights, mc_simulations=1000):
    results = []
    for i in range(len(alternatives)):
        weighted_value = []
        temp_weights = []
        temp_values = []    
        for j in range(len(vf_list)):
            # Extract the data to evaluate
            data = alternatives[i][j]
            if isinstance(data, tuple):
                # If it is a Gaussian distributed value
                mu = data[0]
                sigma = data[1]

            elif isinstance(data, list):
                # If it is a discrete set of values
                mu = np.mean(data)
                sigma = np.std(data)

            elif isinstance(data, dict):
                # If it is an uniformly distributed value
                a = list(data.keys())[0]
                b = list(data.values())[0]
                mu = (a + b) / 2
                sigma = (b - a) / math.sqrt(12)

            else:
                # If it is just one exact value
                mu = data
                sigma = 0
            
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
def sample_to_values(input_data, value_function):
    if isinstance(input_data, tuple):
        mean, std_dev = input_data
        sample = norm.rvs(loc=mean, scale=std_dev)
        value = value_function(sample)
        return value

    elif isinstance(input_data, list):
        # Discrete set
        unique_values, counts = np.unique(input_data, return_counts=True)
        probabilities = counts / len(input_data)
        chosen_value = np.random.choice(unique_values, p=probabilities)
        value = value_function(chosen_value)
        return value
    
    elif isinstance(input_data, dict):
        # Uniform distribution
        start = float(list(input_data.keys())[0])
        end = float(list(input_data.values())[0])
        sample = np.random.uniform(start, end)
        value = value_function(sample)
        return value
    else:
        # Exact value
        value = value_function(input_data)
        return value
    
def mc_simulation(alternatives, vf_list, weights, aggregation_method, sim_runs=1000, bin_size=100, strict=True):   
    # Compute overall distribution of weights by combining all uniform distributions
    weight_distributions = []
    for set_elicited_weights in weights:
        weight_array = np.zeros(bin_size)
        for weight_ranges in set_elicited_weights:
            minimum = weight_ranges[0]
            maximum = weight_ranges[1]
            # Find indices in the bins
            start_idx = int((minimum / 1.0) * bin_size)
            end_idx = int((maximum / 1.0) * bin_size)
            weight_array[start_idx:end_idx] += 1 / (maximum - minimum +1)
        weight_distributions.append(weight_array/np.sum(weight_array))

    results = []
    for _ in range(sim_runs):
        temp_results = []
        for a, alt in enumerate(alternatives):
            # Sample weights for each criterion
            sampled_weights = []
            for c, criterion_data in enumerate(alt):
                if strict:
                    weight_distribition_idx = np.random.randint(0, len(weights[c])) # Choose a random index
                    weight = np.random.uniform(weights[weight_distribition_idx][0], weights[weight_distribition_idx][1])
                    sampled_weights.append(weight)
                else:
                    weight_bins = np.linspace(0, 1, bin_size)
                    probs = weight_distributions[c] / np.sum(weight_distributions[c])
                    chosen_idx = np.random.choice(len(weight_bins), p=probs)
                    sampled_weights.append(weight_bins[chosen_idx])
            # Normalize weights to sum to 1
            total_weight = sum(sampled_weights) 
            sampled_weights = [w / total_weight for w in sampled_weights]

            # Compute weighted values for each criterion
            intermediate_results = []
            for c, criterion_data in enumerate(alt):
                if strict:
                    v_f = np.random.choice(vf_list[c]) # Value function sampling
                    value = sample_to_values(criterion_data, v_f) # Data sampling
                    weight = sampled_weights[c]
                    intermediate_results.append([weight ,value])
                else:
                    value_function_results = []
                    for v_f in vf_list[c]:
                        value = sample_to_values(criterion_data, v_f) # Data sampling
                        value_function_results.append(value)
                    # Construct a distribution over the values obtained from different value functions
                    mean_value = np.mean(value_function_results)
                    std_value = np.std(value_function_results)
                    sampled_value = np.random.normal(loc=mean_value, scale=std_value)
                    weight = sampled_weights[c]
                    intermediate_results.append([weight ,sampled_value])

            # Aggregation
            alternative_value = aggregation_method(intermediate_results)
            temp_results.append(alternative_value)

        results.append(temp_results)   

    results = np.array(results)  # Shape: (sim_runs, num_alternatives)
    num_alternatives = results.shape[1]
    relative_errors = []
    for i in range(num_alternatives):
        res = results[:,i]
        MC_avg = np.mean(res)
        MC_std = np.sqrt(1/(len(res)*(len(res)-1)) * np.sum((res - MC_avg)**2))
        relative_errors.append(MC_std/MC_avg)
    
    return results, relative_errors