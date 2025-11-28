# aggregation_methods.py
#
# This module defines various aggregation models used in MAVT
#
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