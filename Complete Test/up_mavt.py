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
    return value

def mc_simulation(alternatives, vf_list, weight_list, aggregation_method, sim_runs=1000, batches=100, strict=True, weight_fixed=-1):
    # Determine the number of weight sets to use
    weight_iterations = 1 if weight_fixed < 0 else weight_fixed
    batch_size = sim_runs // batches

    for t in range(weight_iterations):
        weight_list = np.array(weight_list)
        elicitation_idx = np.random.randint(0, weight_list.shape[0])
        random_weight_idx = np.random.randint(0, weight_list.shape[1])  # 0 to 2857
        sampled_weights = weight_list[elicitation_idx, random_weight_idx, :]
        results = []
        relative_errors = []

        for b in range(batches):
            batch_results = []
            for _ in range(batch_size):
                temp_results = []
                for a, alt in enumerate(alternatives):
                    intermediate_results = []
                    for c, criterion_data in enumerate(alt.values()):
                        if strict:
                            v_f = vf_list[c][elicitation_idx]
                        else:
                            v_f = np.random.choice(vf_list[c])
                        sampled_value = sample_to_values(criterion_data, v_f)
                        weight = sampled_weights[c]
                        intermediate_results.append([weight, sampled_value])
                    alternative_value = aggregation_method(intermediate_results)
                    temp_results.append(alternative_value)
                batch_results.append(temp_results)

            # Convert to numpy array
            batch_results_array = np.array(batch_results)  # Shape: (batch_size, num_alternatives)
            results.append(batch_results_array)

            # Compute relative errors for this batch
            num_alternatives = batch_results_array.shape[1]
            batch_relative_errors = []
            for i in range(num_alternatives):
                res = batch_results_array[:, i]
                MC_avg = np.mean(res)
                MC_std = np.sqrt(1/(len(res)*(len(res)-1)) * np.sum((res - MC_avg)**2)) if len(res) > 1 else 0
                batch_relative_errors.append(MC_std/MC_avg if MC_avg != 0 else 0)
            relative_errors.append(batch_relative_errors)

            # Yield after each batch
            yield batch_results_array, batch_relative_errors

        # After all batches, yield all results for this weight set
        all_results = np.vstack(results)  # Shape: (sim_runs, num_alternatives)
        all_relative_errors = np.mean(relative_errors, axis=0)  # Average relative errors
        yield all_results, all_relative_errors
