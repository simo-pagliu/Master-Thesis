def constraints_func(x, dict_data, z_star=None, eps=0.001):
    cons = []
    group_indices = {}
    current_index = 0
    # Map group and criterion names to their global indices in x
    for group_name, group_data in dict_data.items():
        criteria_in_group = list(group_data['criteria'].keys())
        group_indices[group_name] = {
            'start': current_index,
            'end': current_index + len(criteria_in_group),
            'criteria': criteria_in_group
        }
        current_index += len(criteria_in_group)

    # INTRA-GROUP CONSTRAINTS
    for group_name, group_data in dict_data.items():
        criteria_in_group = group_indices[group_name]['criteria']
        w_start = group_indices[group_name]['start']
        w_end = group_indices[group_name]['end']
        w = x[w_start:w_end]  # Weights for this group
        z = x[-1]             # z is the last element
        for crit, comparisons in group_data['criteria'].items():
            i = criteria_in_group.index(crit)
            # INTRA-GROUP BEST COMPARISONS
            for other_crit, value in comparisons['best_comparisons'].items():
                j = criteria_in_group.index(other_crit)
                v_f = comparisons['value_function']
                v_f_val = max(v_f(value), eps)
                cons.append(z - abs(w[i] / (w[j] + eps) - 1.0 / v_f_val))
            # INTRA-GROUP WORST COMPARISONS
            for other_crit, value in comparisons['worst_comparisons'].items():
                j = criteria_in_group.index(other_crit)
                v_f_other = group_data['criteria'][other_crit]['value_function']
                v_f_other_val = max(v_f_other(value), eps)
                cons.append(z - abs(1.0 / v_f_other_val - (w[j] + eps) / (w[i] + eps)))

    # INTER-GROUP CONSTRAINTS
    intraB = {}
    intraW = {}
    for group_data in dict_data.values():
        intraB.update(group_data['intraB'])
        intraW.update(group_data['intraW'])

    def add_comparison_constraint(comparison, comparison_type, cons, x, group_indices, dict_data, eps=0.001):
        ref_crit = comparison['reference']
        other_crit = comparison['other']
        def resolve_group_and_global_index(crit):
            for group_name, group_data in dict_data.items():
                if crit in group_data['criteria']:
                    local_index = group_indices[group_name]['criteria'].index(crit)
                    return group_name, group_indices[group_name]['start'] + local_index

        ref_group, ref_global_index = resolve_group_and_global_index(ref_crit)
        other_group, other_global_index = resolve_group_and_global_index(other_crit)
        v_f = dict_data[ref_group]['criteria'][ref_crit]['value_function']
        v_f_other = dict_data[other_group]['criteria'][other_crit]['value_function']
        v_f_val = max(v_f(comparison['value']), eps)
        v_f_other_val = max(v_f_other(comparison['value']), eps)
        if comparison_type == "best":
            cons.append(x[-1] - abs((x[ref_global_index] + eps) / (x[other_global_index] + eps) - (1.0 / v_f_val)))
        else:  # worst
            cons.append(x[-1] - abs(1.0 / v_f_other_val - (x[other_global_index] + eps) / (x[ref_global_index] + eps)))

    for comparison in intraB.values():
        add_comparison_constraint(comparison, comparison["type"], cons, x, group_indices, dict_data, eps)
    for comparison in intraW.values():
        add_comparison_constraint(comparison, comparison["type"], cons, x, group_indices, dict_data, eps)

    if z_star is not None:
        cons.append(z_star - x[-1])

    return cons