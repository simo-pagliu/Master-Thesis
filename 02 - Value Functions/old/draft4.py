import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_csv():
    """Load the CSV file with attributes"""
    while True:
        try:
            file_path = input("Enter path to CSV file: ")
            df = pd.read_csv(file_path)
            required_columns = ["name", "group", "min", "max", "unit"]
            if not all(col in df.columns for col in required_columns):
                print(f"CSV must contain columns: {', '.join(required_columns)}")
                continue
            return df
        except Exception as e:
            print(f"Error loading file: {e}")

def ask_gaussian():
    """Ask if the value function has a max/min within the range (default is normal function)"""
    while True:
        response = input("Is the value function a normal increasing/decreasing function (not Gaussian)? (y/n): ").lower()
        if response in ['y', 'n', '']:
            return response != 'y'  # Default to False (normal function) if empty or 'n'
        print("Please enter 'y' or 'n' (or just press Enter for normal function)")

def ask_direction():
    """Ask if the function is concave or convex"""
    while True:
        response = input("Is the value function concave or convex? (c/x): ").lower()
        if response in ['c', 'x']:
            return response
        print("Please enter 'c' for concave or 'x' for convex")

def ask_peak_boundary(attr_range, is_gaussian):
    """Ask for peak value and optionally boundary values"""
    while True:
        try:
            peak = float(input(f"Most important value in the range from {attr_range[0]} to {attr_range[1]}: "))

            if is_gaussian:
                skip_boundaries = input("Do you want to skip boundary values? (y/n): ").lower()
                if skip_boundaries == 'y':
                    return peak, None, None

            value_R = float(input("After which value does the importance stagnate? "))
            value_L = float(input("Below which value does the importance stagnate? "))
            return peak, value_R, value_L
        except ValueError:
            if is_gaussian and 'skip_boundaries' in locals() and skip_boundaries == 'y':
                return peak, None, None
            print("Please enter valid numbers")

def ask_indifference_05(low_value, peak_value):
    """Ask for 0.5 indifference point with option for linear increase"""
    while True:
        response = input(f"Which value makes it indifferent to increase from {low_value} to X and from X to {peak_value}? (Enter L for linear increase): ").lower()

        if response == 'l':
            return 'linear'

        try:
            x_05 = float(response)
            return x_05
        except ValueError:
            print("Please enter a valid number or 'L' for linear increase")

def ask_skip_025_075():
    """Ask if user wants to add 0.25 and 0.75 points"""
    while True:
        response = input("Do you want to add 0.25 and 0.75 indifference points? (y/n): ").lower()
        if response in ['y', 'n', '']:
            return response != 'y'  # Default to skip if empty or 'n'
        print("Please enter 'y' or 'n' (or just press Enter to skip)")

def ask_indifference_025_075(low_value, x_05, peak_value):
    """Ask for 0.25 and 0.75 indifference points"""
    while True:
        try:
            x_025 = float(input(f"Which value makes it indifferent to increase from {low_value} to X and from X to {x_05}? "))
            x_075 = float(input(f"Which value makes it indifferent to increase from {x_05} to X and from X to {peak_value}? "))
            return x_025, x_075
        except ValueError:
            print("Please enter valid numbers")

def plot_points(points, attr_name, attr_range):
    """Plot the collected points"""
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, 'o-')
    plt.title(f"Value Function for {attr_name}")
    plt.xlabel(attr_name)
    plt.ylabel("Value")
    plt.xlim(attr_range[0], attr_range[1])
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    plt.xticks(x_vals)
    plt.yticks([0, 0.25, 0.5, 0.75, 1])
    plt.show()

def process_attribute(attr, attr_index, total_attrs):
    """Process a single attribute"""
    print(f"\nProcessing attribute {attr_index + 1}/{total_attrs}: {attr['name']}")
    print(f"Range: {attr['min']} to {attr['max']}")

    # Initialize variables
    is_gaussian = ask_gaussian()
    points = []
    attr_range = [attr['min'], attr['max']]

    if is_gaussian:
        direction = ask_direction()
        y_peak = 1.0 if direction == 'c' else 0.0
        y_boundary = 0.0 if direction == 'c' else 1.0

        peak_value, value_R, value_L = ask_peak_boundary(attr_range, is_gaussian)

        # Set default boundaries if skipped
        if value_R is None:
            value_R = attr_range[1]
        if value_L is None:
            value_L = attr_range[0]

        points = [
            (attr_range[0], y_boundary),
            (value_L, y_boundary),
            (peak_value, y_peak),
            (value_R, y_boundary),
            (attr_range[1], y_boundary)
        ]

        # Add 0.5 points
        print("\nFor the 0.5 indifference point:")
        x_05_L = ask_indifference_05(value_L, peak_value)
        x_05_R = ask_indifference_05(peak_value, value_R)

        # If linear, skip 0.25/0.75 points
        if x_05_L == 'linear' or x_05_R == 'linear':
            # Use midpoint for linear
            if x_05_L == 'linear':
                x_05_L = (value_L + peak_value) / 2
            if x_05_R == 'linear':
                x_05_R = (peak_value + value_R) / 2

            points.insert(2, (x_05_L, 0.5))
            points.insert(4, (x_05_R, 0.5))
            return points  # Skip 0.25/0.75 points

        points.insert(2, (x_05_L, 0.5))
        points.insert(4, (x_05_R, 0.5))

        # Ask about 0.25/0.75 points
        skip = ask_skip_025_075()
        if not skip:
            x_025, x_075 = ask_indifference_025_075(value_L, x_05_L, peak_value)
            points.insert(2, (x_025, 0.25))
            points.insert(5, (x_075, 0.75))
    else:
        # Normal function (not Gaussian)
        peak_value = float(input(f"Peak value in the range from {attr_range[0]} to {attr_range[1]}: "))
        low_value = float(input(f"Lowest value in the range from {attr_range[0]} to {attr_range[1]}: "))

        points = [
            (attr_range[0], 0.0),
            (low_value, 0.0),
            (peak_value, 1.0),
            (attr_range[1], 1.0)
        ]

        # Add 0.5 point
        print("\nFor the 0.5 indifference point:")
        x_05 = ask_indifference_05(low_value, peak_value)

        if x_05 == 'linear':
            # Use midpoint for linear
            x_05 = (low_value + peak_value) / 2
            points.insert(2, (x_05, 0.5))
            return points  # Skip 0.25/0.75 points

        points.insert(2, (x_05, 0.5))

        # Ask about 0.25/0.75 points
        skip = ask_skip_025_075()
        if not skip:
            x_025, x_075 = ask_indifference_025_075(low_value, x_05, peak_value)
            points.insert(2, (x_025, 0.25))
            points.insert(4, (x_075, 0.75))

    return points

def main():
    # Load the CSV
    df = load_csv()

    # Process each attribute
    results = []
    for i, attr in enumerate(df.to_dict('records')):
        points = process_attribute(attr, i, len(df))
        results.append({
            'name': attr['name'],
            'points': points
        })

        # Show the plot
        plot_points(points, attr['name'], [attr['min'], attr['max']])

        # Confirm points
        print("\nCollected points:")
        for i, (x, y) in enumerate(points):
            print(f"Point {i+1}: ({x}, {y})")

        while True:
            confirm = input("Are these points correct? (y/n): ").lower()
            if confirm == 'y':
                break
            elif confirm == 'n':
                print("Let's start over for this attribute")
                points = process_attribute(attr, i, len(df))
                results[-1]['points'] = points
                break
            else:
                print("Please enter 'y' or 'n'")

    # Save results
    for result in results:
        df.loc[df['name'] == result['name'], 'points'] = str(result['points'])

    save_path = input("Enter path to save the results: ")
    df.to_csv(save_path, index=False)
    print("Results saved successfully!")

if __name__ == "__main__":
    main()
