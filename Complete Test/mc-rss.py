import numpy as np
from measurments import Measurement
from up_mavt import parse_distribution_as_normal

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