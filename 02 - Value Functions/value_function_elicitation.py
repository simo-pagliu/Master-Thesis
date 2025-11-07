# Temporary input - leave it as is for now
indicator_name = "attribute"
indicator_range = [0,57]

# Elicitation process
gaussian_input = input(f"Does the value of {indicator_name} present a maximum or a minimum within his range? (yes/no) ")
is_gaussian = False
if gaussian_input.lower() == 'yes':
    is_gaussian = True

if is_gaussian:
    direction = input(f"Is the value function of {indicator_name} concave or convex? (concave/convex) ")
    if direction.lower() == 'concave':
        function_type = "maximum"
        limit_type = "lowest"
        action_type = "increase"
    else:
        function_type = "minimum"
        limit_type = "highest"
        action_type = "decrease"
    peak_value = input(f"At which point does the value of {indicator_name} reach its {function_type}? ")
    value_R = input(f"At which point does the value of {indicator_name} reach its {limit_type} value on the right? ")
    value_L = input(f"At which point does the value of {indicator_name} reach its {limit_type} value on the left? ")
    x_05_R = input(f"Which value of {indicator_name} makes it indifferent to {action_type} from {value_R} to X and from X to {peak_value}? ")
    x_05_L = input(f"Which value of {indicator_name} makes it indifferent to {action_type} from {value_L} to X and from X to {peak_value}? ")

    # define the function basic shape
    # Piecewise function with three segments
    # first segment: from min(indicator_range) to value_L (constant at 0 if gaussian is type maximum, constant at 1 if minimum)
    # second segment: from value_R to max(indicator_range) (constant at 0 if gaussian is type maximum, constant at 1 if minimum)
    # third segment: a function, for now unknown that goes through the points (value_L,0 or 1), (x_05_L,0.5), (peak_value,1 or 0), (x_05_R,0.5), (value_R,0 or 1)
    # Plot the segments and the points
    # ask the user to choose between different function types for the third segment: polynomial, gaussian, linear segments

else:
    peak_value = input(f"At which point does the value of {indicator_name} reach its peak? ")
    low_value = input(f"At which point does the value of {indicator_name} reach its lowest value? ")
    critical_point = input(f"Is there any critical point between {low_value} and {peak_value} where value of {indicator_name} changes significantly? (yes/no) ")
    # Start with the mid-value splitting process
    x_05 = input(f"Which value of {indicator_name} makes it indifferent to increase from {low_value} to X and from X to {peak_value}? ")
    x_025 = input(f"Which value of {indicator_name} makes it indifferent to increase from {low_value} to X and from X to {x_05}? ")
    x_075 = input(f"Which value of {indicator_name} makes it indifferent to increase from {x_05} to X and from X to {peak_value}? ")
    # define the function basic shape
    # Piecewise function with three segments
    # first segment: from min(indicator_range) to low_value (constant at 0)
    # second segment: from peak_value to max(indicator_range) (constant at 1)
    # third segment: a function, for now unknown that goes through the points (low_value,0), (x_025,0.25), (x_05,0.5), (x_075,0.75), (peak_value,1)
    # Plot the segments and the points
    # ask the user to choose between different function types for the third segment: polynomial, linear segments, sigmoid