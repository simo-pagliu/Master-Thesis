import numpy as np
import matplotlib.pyplot as plt

# Temporary input - leave it as is for now
indicator_name = "attribute"
indicator_range = [0, 57]

print(f"Elicitation process for '{indicator_name}' (range: {indicator_range[0]} to {indicator_range[1]})\n")

# Step 1: Check if the value function is Gaussian (has a max/min within the range)
gaussian_input = input(f"Does the value of {indicator_name} present a maximum or a minimum within its range? (y/n): ").strip().lower()
is_gaussian = gaussian_input == 'y'

# Variables to store points and the lambda function
points = []
value_function = None
function_type_choice = None
function_str = ""

if is_gaussian:
    # Step 2: Determine if the function is concave or convex
    direction = input(f"Is the value function of {indicator_name} concave (c) or convex (x)? (c/x): ").strip().lower()
    if direction == 'c':
        function_type = "maximum"
        limit_type = "lowest"
        action_type = "increase"
        y_peak = 1.0
        y_boundary = 0.0
    else:
        function_type = "minimum"
        limit_type = "highest"
        action_type = "decrease"
        y_peak = 0.0
        y_boundary = 1.0

    # Step 3: Elicit peak and boundary values
    peak_value = float(input(f"At which point does the value of {indicator_name} reach its {function_type}? "))
    value_R = float(input(f"At which point does the value of {indicator_name} reach its {limit_type} value on the right? "))
    value_L = float(input(f"At which point does the value of {indicator_name} reach its {limit_type} value on the left? "))

    # Step 4: Elicit indifference points
    x_05_R = float(input(f"Which value of {indicator_name} makes it indifferent to {action_type} from {value_R} to X and from X to {peak_value}? "))
    x_05_L = float(input(f"Which value of {indicator_name} makes it indifferent to {action_type} from {value_L} to X and from X to {peak_value}? "))

    # Define points for the piecewise function
    points = [
        (indicator_range[0], y_boundary),  # Left boundary (constant)
        (value_L, y_boundary),
        (x_05_L, 0.5),
        (peak_value, y_peak),
        (x_05_R, 0.5),
        (value_R, y_boundary),
        (indicator_range[1], y_boundary)  # Right boundary (constant)
    ]

    # Plot the points and segments
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, 'o-', label=f"{function_type} (Gaussian-like)")
    plt.title(f"Value Function for {indicator_name}")
    plt.xlabel(indicator_name)
    plt.ylabel("Value")
    plt.grid(True)
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.axhline(y=1, color='gray', linestyle='--')
    plt.xticks(x_vals, labels=[f"{x:.1f}" for x in x_vals])
    plt.yticks([0, 0.5, 1])
    plt.show()

    # Step 5: Ask for the function type for the third segment
    print("\nChoose the function type for the third segment:")
    print("1. Polynomial (quadratic)")
    print("2. Gaussian")
    print("3. Linear segments")
    function_type_choice = input("Enter your choice (1/2/3): ").strip()

    # Define the lambda function and its string representation
    if function_type_choice == '1':
        # Fit a quadratic polynomial
        x_fit = [p[0] for p in points[1:6]]
        y_fit = [p[1] for p in points[1:6]]
        coefficients = np.polyfit(x_fit, y_fit, 2)
        poly = np.poly1d(coefficients)
        value_function = lambda x: (
            y_boundary if x <= value_L else
            y_boundary if x >= value_R else
            poly(x)
        )
        function_str = (
            f"lambda x: {y_boundary} if x <= {value_L} else "
            f"{y_boundary} if x >= {value_R} else "
            f"({coefficients[0]} * x**2 + {coefficients[1]} * x + {coefficients[2]})"
        )
    elif function_type_choice == '2':
        # Gaussian-like function
        amplitude = y_peak - y_boundary
        sigma = (peak_value - value_L) / 3
        value_function = lambda x: (
            y_boundary if x <= value_L else
            y_boundary if x >= value_R else
            y_boundary + amplitude * np.exp(-((x - peak_value) ** 2) / (2 * sigma ** 2))
        )
        function_str = (
            f"lambda x: {y_boundary} if x <= {value_L} else "
            f"{y_boundary} if x >= {value_R} else "
            f"{y_boundary} + {amplitude} * np.exp(-((x - {peak_value}) ** 2) / (2 * {sigma} ** 2))"
        )
    elif function_type_choice == '3':
        # Linear segments
        value_function = lambda x: (
            y_boundary if x <= value_L else
            y_boundary if x >= value_R else
            np.interp(x, [value_L, x_05_L, peak_value, x_05_R, value_R],
                      [y_boundary, 0.5, y_peak, 0.5, y_boundary])
        )
        function_str = (
            f"lambda x: {y_boundary} if x <= {value_L} else "
            f"{y_boundary} if x >= {value_R} else "
            f"np.interp(x, [{value_L}, {x_05_L}, {peak_value}, {x_05_R}, {value_R}], "
            f"[{y_boundary}, 0.5, {y_peak}, 0.5, {y_boundary}])"
        )

else:
    # Step 2: Elicit peak and lowest values for non-Gaussian function
    peak_value = float(input(f"At which point does the value of {indicator_name} reach its peak? "))
    low_value = float(input(f"At which point does the value of {indicator_name} reach its lowest value? "))

    # Step 3: Check for critical points
    critical_point = input(f"Is there any critical point between {low_value} and {peak_value} where the value changes significantly? (y/n): ").strip().lower()

    # Step 4: Elicit indifference points
    x_05 = float(input(f"Which value of {indicator_name} makes it indifferent to increase from {low_value} to X and from X to {peak_value}? "))
    x_025 = float(input(f"Which value of {indicator_name} makes it indifferent to increase from {low_value} to X and from X to {x_05}? "))
    x_075 = float(input(f"Which value of {indicator_name} makes it indifferent to increase from {x_05} to X and from X to {peak_value}? "))

    # Define points for the piecewise function
    points = [
        (indicator_range[0], 0.0),  # Left boundary (constant at 0)
        (low_value, 0.0),
        (x_025, 0.25),
        (x_05, 0.5),
        (x_075, 0.75),
        (peak_value, 1.0),
        (indicator_range[1], 1.0)  # Right boundary (constant at 1)
    ]

    # Plot the points and segments
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, 'o-', label="Non-Gaussian")
    plt.title(f"Value Function for {indicator_name}")
    plt.xlabel(indicator_name)
    plt.ylabel("Value")
    plt.grid(True)
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.axhline(y=1, color='gray', linestyle='--')
    plt.xticks(x_vals, labels=[f"{x:.1f}" for x in x_vals])
    plt.yticks([0, 0.25, 0.5, 0.75, 1])
    plt.show()

    # Step 5: Ask for the function type for the third segment
    print("\nChoose the function type for the third segment:")
    print("1. Polynomial (cubic)")
    print("2. Linear segments")
    print("3. Sigmoid")
    function_type_choice = input("Enter your choice (1/2/3): ").strip()

    # Define the lambda function and its string representation
    if function_type_choice == '1':
        # Fit a cubic polynomial
        x_fit = [p[0] for p in points[1:6]]
        y_fit = [p[1] for p in points[1:6]]
        coefficients = np.polyfit(x_fit, y_fit, 3)
        poly = np.poly1d(coefficients)
        value_function = lambda x: (
            0.0 if x <= low_value else
            1.0 if x >= peak_value else
            poly(x)
        )
        function_str = (
            f"lambda x: 0.0 if x <= {low_value} else "
            f"1.0 if x >= {peak_value} else "
            f"({coefficients[0]} * x**3 + {coefficients[1]} * x**2 + {coefficients[2]} * x + {coefficients[3]})"
        )
    elif function_type_choice == '2':
        # Linear segments
        value_function = lambda x: (
            0.0 if x <= low_value else
            1.0 if x >= peak_value else
            np.interp(x, [low_value, x_025, x_05, x_075, peak_value],
                      [0.0, 0.25, 0.5, 0.75, 1.0])
        )
        function_str = (
            f"lambda x: 0.0 if x <= {low_value} else "
            f"1.0 if x >= {peak_value} else "
            f"np.interp(x, [{low_value}, {x_025}, {x_05}, {x_075}, {peak_value}], "
            f"[0.0, 0.25, 0.5, 0.75, 1.0])"
        )
    elif function_type_choice == '3':
        # Sigmoid function
        x_mid = (low_value + peak_value) / 2
        scale = (peak_value - low_value) / 6
        value_function = lambda x: (
            0.0 if x <= low_value else
            1.0 if x >= peak_value else
            1 / (1 + np.exp(-(x - x_mid) / scale))
        )
        function_str = (
            f"lambda x: 0.0 if x <= {low_value} else "
            f"1.0 if x >= {peak_value} else "
            f"1 / (1 + np.exp(-(x - {x_mid}) / {scale}))"
        )

# Print the points and the lambda function as a string
print("\nPoints for the value function:")
for x, y in points:
    print(f"({x}, {y})")

print(f"\nLambda function for the value function:")
print(function_str)

# Test the function by plotting it
x_test = np.linspace(indicator_range[0], indicator_range[1], 100)
y_test = [value_function(x) for x in x_test]

plt.figure(figsize=(10, 6))
plt.plot(x_test, y_test, '-', label="Fitted Function")
plt.title(f"Fitted Value Function for {indicator_name}")
plt.xlabel(indicator_name)
plt.ylabel("Value")
plt.grid(True)
plt.axhline(y=0, color='gray', linestyle='--')
plt.axhline(y=1, color='gray', linestyle='--')
plt.show()
