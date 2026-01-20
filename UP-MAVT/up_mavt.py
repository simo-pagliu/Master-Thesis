# up-mavt.py
#
# this module containes the basic functions of the UP-MAVT framework

#################################################################################
# Import third party libraries
import numpy as np
#################################################################################

#################################################################################
# Import internal modules
from pile_bwt import weight_sampler
#################################################################################

#################################################################################
# Sampling of data from their distributions
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
        elif list(data.keys())[0] == "Trapezoidal":
            a, b, c, d = data_values[0], data_values[1], data_values[2], data_values[3]
            u = np.random.uniform(0, 1)
            k = d - a + c - b
            if u < (b - a) / k:
                sample = a + np.sqrt(u * (b - a) * k)
            elif u < (d - c) / k + (b - a) / k:
                sample = b + (u - (b - a) / k) * k / 2
            else:
                sample = d - np.sqrt((1 - u) * (d - c) * k)

        elif list(data.keys())[0] == "Special_1":
            a_possible = data_values[0]
            x_low = data_values[1]
            x_high = data_values[2]
            x = np.random.uniform(x_low, x_high)
            a = np.random.choice(a_possible)
            prob_0 = a * (1-x)
            prob_1 = a * x + (1-a)*(1-x)
            prob_2 = (1-a)*x
            sample = np.random.choice([0,1,2], p=[prob_0, prob_1, prob_2])

        elif list(data.keys())[0] == "QI_dist":
            # Qualitative Indicator distribution with confidence levels
            # Format: [(value1, confidence1), (value2, confidence2), ...]

            # Step 1: Randomly select one opinion
            selected_idx = np.random.choice(len(data_values))
            value, confidence = data_values[selected_idx]

            # Step 2: Map confidence to percentage error and compute range
            # Example mapping (customize as needed):
            if confidence >= 4:
                error_percent = 5   # 5% error for very high confidence
            elif confidence >= 3:
                error_percent = 10  # 10% error for high confidence
            elif confidence >= 2:
                error_percent = 20  # 20% error for medium confidence
            elif confidence >= 1:
                error_percent = 30  # 30% error for low confidence
            else:
                error_percent = 50  # 50% error for very low confidence

            dist_range = value * (error_percent / 100)
            a = max(value - dist_range, 0)  # Clamp to non-negative
            b = value + dist_range

            # Step 3: Sample from the uniform distribution
            sample = np.random.uniform(a, b)
   
        else:
            raise ValueError(f"Unsupported distribution type in dictionary.{data.keys()}")
    else:
        try:
            sample = float(data)  # Ensure the data is numeric
        except ValueError:
            raise TypeError(f"Invalid data type for sampling: {data}")
    # Normalize sampled values to numeric scalars (handle numpy scalar or numeric strings)
    if isinstance(sample, str):
        sample = float(sample)
    if isinstance(sample, np.generic):
        sample = float(sample)

    value = float(value_function(sample))
    return value

#################################################################################

#################################################################################
def evaluation_func(elicitation_idx, alternatives, opinion_weights, vf_list, conf_list, list_of_weight_space_points, dict_data_list, aggregation_method, strict, crit_index, random_weight_analysis):
    run_results = []
    if random_weight_analysis:
        sampled_weights = np.random.dirichlet(np.ones(len(crit_index)))
    else:
        # Select the weight space for this elicitation
        weight_space_points = list_of_weight_space_points[elicitation_idx]
        # Load the dict_data for this elicitation
        dict_data = dict_data_list[elicitation_idx]
        # For each expert we choose a random set of weights
        sampled_weights = weight_sampler(dict_data, weight_space_points)
        # Normalization to ensure weights sum to 1 (should be already the case)
        sampled_weights = sampled_weights / np.sum(sampled_weights)
    # For each alternative we compute its value
    for a, alt in enumerate(alternatives):
        intermediate_results = []
        # For each criterion
        # iterate over criterion name + data so we can align with vf_list by name
        for c_idx, (crit_name, criterion_data) in enumerate(alt.items()):
            # determine the index into vf_list for this criterion
            idx = crit_index.get(crit_name, None)

            # choose the value function: strict -> same expert index, otherwise confidence-weighted
            if strict:
                v_f = vf_list[idx][elicitation_idx]
            else:
                # In non-strict mode, weight VF selection by confidence level
                # Confidence levels range from 0-4, with 4 being most probable
                # Add baseline of 1 to ensure even confidence 0 has non-zero probability
                confidences = conf_list[idx]
                conf_array = np.array(confidences) + 1.0
                conf_weights = conf_array / np.sum(conf_array)
                v_f = np.random.choice(vf_list[idx], p=conf_weights)

            # compute the value of the criterion by sampling from its distribution
            sampled_value = sample_to_values(criterion_data, v_f)
            # Get the weight for this criterion from the sampled weight set
            weight = sampled_weights[idx]
            # save this pair of weight and value
            intermediate_results.append([weight, sampled_value])
        # Once all criterion have been evaluated we aggregate to get the overall alternative value
        # print(intermediate_results[2][1])
        alternative_value = aggregation_method(intermediate_results)
        # And save the alternative value to go then to the next alternative
        run_results.append(alternative_value)
    # Return the list of alternative values for this evaluation
    return run_results
# Montecarlo generator function
def mc_simulation(alternatives, opinion_weights, vf_list, conf_list, list_of_weight_space_points, dict_data_list, aggregation_method, sim_runs, strict, crit_index, random_weight_analysis):
    # weight_list = np.array(weight_list)
    possible_idxs = [i for i in range(len(list_of_weight_space_points))]
    for mc_run in range(sim_runs):
        # For each montecarlo run we iterate over all experts
        if strict:
            for elicitation_idx in range(len(list_of_weight_space_points)):
                run_results = evaluation_func(elicitation_idx, alternatives, None, vf_list, conf_list, list_of_weight_space_points, dict_data_list, aggregation_method, strict, crit_index, random_weight_analysis)
                yield elicitation_idx, run_results
        else:
            elicitation_idx = np.random.choice(possible_idxs, p=opinion_weights)
            run_results = evaluation_func(elicitation_idx, alternatives, opinion_weights, vf_list, conf_list, list_of_weight_space_points, dict_data_list, aggregation_method, strict, crit_index, random_weight_analysis)
            yield run_results
    return