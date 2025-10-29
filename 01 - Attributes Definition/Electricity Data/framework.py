import math
import warnings

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

