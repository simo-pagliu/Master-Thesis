import numpy as np
from typing import List, Callable, Dict, Any

def genetic_algorithm(
    num_criteria: int,
    constraints_func: Callable[[np.ndarray, Dict[str, Any]], np.ndarray],
    dict_data: Dict[str, Any],
    known_max_error: float,
    pop_size: int = 100,
    max_generations: int = 200,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.9,
    tolerance: float = 1e-6,
) -> List[np.ndarray]:
    """
    Genetic Algorithm to find all x_b satisfying:
    - sum(x_b) = 1.0
    - all(constraints_func(x_b) <= 0)
    - error_b <= known_max_error

    Args:
        num_criteria: Length of x_b.
        constraints_func: Function returning constraint values.
        dict_data: Data dictionary for constraints_func.
        known_max_error: Maximum allowed error.
        pop_size: Population size.
        max_generations: Maximum number of generations.
        mutation_rate: Probability of mutation.
        crossover_rate: Probability of crossover.
        tolerance: Tolerance for sum(x_b) == 1.0.

    Returns:
        List of solutions (x_b) that satisfy all conditions.
    """

    def initialize_population(size: int) -> List[np.ndarray]:
        """Initialize population with random x_b (sum to 1)."""
        pop = []
        for _ in range(size):
            x = np.random.rand(num_criteria)
            x /= x.sum()  # Normalize to sum to 1
            pop.append(x)
        return pop

    def evaluate_fitness(x_b: np.ndarray) -> float:
        """Evaluate fitness (lower is better)."""
        x_temp = np.concatenate([x_b, [0]])
        constraint_value = constraints_func(x_temp, dict_data)
        error_b = min(max([abs(cv) for cv in constraint_value]), 10)
        # condition_1 = all(cv <= 0 for cv in constraint_value)
        # condition_2 = abs(sum(x_b) - 1.0) <= tolerance
        # condition_3 = error_b <= known_max_error

        return 1.0 / (1.0 + error_b)  # Reward valid solutions


    def select_parents(population: List[np.ndarray], fitnesses: List[float]) -> List[np.ndarray]:
        """Select parents using tournament selection."""
        parents = []
        for _ in range(2):
            candidates = np.random.choice(len(population), size=3, replace=False)
            winner = max(candidates, key=lambda i: fitnesses[i])
            parents.append(population[winner])
        return parents

    def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Uniform crossover."""
        if np.random.rand() > crossover_rate:
            return parent1.copy()
        mask = np.random.rand(num_criteria) < 0.5
        child = parent1.copy()
        child[mask] = parent2[mask]
        child /= child.sum()  # Ensure sum to 1
        return child

    def mutate(x_b: np.ndarray) -> np.ndarray:
        """Gaussian mutation."""
        if np.random.rand() < mutation_rate:
            x_b += np.random.normal(0, 0.1, num_criteria)
            x_b = np.clip(x_b, 0, None)  # Ensure non-negative
            x_b /= x_b.sum()  # Renormalize
        return x_b

    # Initialize population
    population = initialize_population(pop_size)
    best_solutions = []

    for generation in range(max_generations):
        # Evaluate fitness
        fitnesses = [evaluate_fitness(x) for x in population]

        # Check for valid solutions
        for x, fit in zip(population, fitnesses):
            x_temp = np.concatenate([x, [0]])
            constraint_value = constraints_func(x_temp, dict_data)
            error_b = min(max([abs(cv) for cv in constraint_value]), 10)
            condition_1 = all(cv <= 0 for cv in constraint_value)
            condition_2 = abs(sum(x) - 1.0) <= tolerance
            condition_3 = error_b <= known_max_error
            if condition_1 and condition_2 and condition_3:
                print(f"Found valid solution in generation: {x}, error: {error_b}")
                best_solutions.append(x)

        if best_solutions:
            break  # Early exit if solutions found

        # Create next generation
        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = select_parents(population, fitnesses)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            new_population.extend([mutate(child1), mutate(child2)])

        population = new_population
    # In anycase print the solution with the lowest error found
    if not best_solutions:
        fitnesses = [evaluate_fitness(x) for x in population]
        best_index = np.argmax(fitnesses)
        best_solution = population[best_index]
        x_temp = np.concatenate([best_solution, [0]])
        constraint_value = constraints_func(x_temp, dict_data)
        error_b = min(max([abs(cv) for cv in constraint_value]), 10)
        print(f"No valid solution found. Best solution: {best_solution}, error: {error_b}")

    return best_solutions