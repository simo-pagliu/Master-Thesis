import numpy as np

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
        try:
            sample = float(data)  # Ensure the data is numeric
        except ValueError:
            raise TypeError(f"Invalid data type for sampling: {data}")
    value = float(value_function(sample))
    # if value < 0 or value > 1:
        # print(f"Sampled value: {sample} from data: {data}, value is {value}")
    return value

def mc_simulation(alternatives, vf_list, weight_list, aggregation_method, sim_runs, strict, crit_index):
    # weight_list = np.array(weight_list)
    for mc_run in range(sim_runs):
        # For each montecarlo run we iterate over all experts
        for elicitation_idx in range(len(weight_list)):
            run_results = []
            # For each expert we choose a random set of weights
            random_weight_idx = np.random.randint(0, len(weight_list[elicitation_idx]))
            sampled_weights = weight_list[elicitation_idx][random_weight_idx]
            # Normalization to ensure weights sum to 1 (produced by space_sampling so can have a little deviation)
            sampled_weights = sampled_weights / np.sum(sampled_weights)
            # For each alternative we compute its value
            for a, alt in enumerate(alternatives):
                intermediate_results = []
                # For each criterion
                # iterate over criterion name + data so we can align with vf_list by name
                for c_idx, (crit_name, criterion_data) in enumerate(alt.items()):
                    # determine the index into vf_list for this criterion
                    idx = crit_index.get(crit_name, None)

                    # choose the value function: strict -> same expert index, otherwise random
                    if strict:
                        v_f = vf_list[idx][elicitation_idx]
                    else:
                        v_f = np.random.choice(vf_list[idx])

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
            # Now we save the result from this run
            # So each run is consistent with the set of weight,
            # this means that we can later compute the ranking of the alternatives per run
            yield run_results
    return