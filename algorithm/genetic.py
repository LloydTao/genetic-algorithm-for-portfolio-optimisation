import numpy as np
import random

from algorithm.stocks import portfolio_sharpe


def init_population(size, num_weights):
    """Generate an initial population of portfolio weightings.

    Args:
        size (int): The number of weightings to produce.
        num_weights (int): The number of weights within a weighting.

    Returns:
        list: A list of portfolio weightings.
    """
    # Initialise a list to store weightings
    population = []
    for i in range(size):
        # Create a random series of weightings
        weighting = np.random.uniform(low=0, high=100, size=num_weights)

        # Normalise the weightings so that they sum to 1
        weighting /= weighting.sum()
        population.append(weighting)

    return population


def fitness(prices, population):
    """Calculate the fitness of each weighting across a population.

    Args:
        prices (dataframe): The historical daily price data of the assets.
        population (list): A list of weightings to receive a fitness score each.

    Returns:
        list, list: The fitnesses and the population, each ordered by fitness.
    """
    # Initialise a list to store scores
    scores = []
    for weighting in population:
        # Use Sharpe index of weighting as fitness score
        scores.append(portfolio_sharpe(prices, weighting))

    scores = np.array(scores)
    population = np.array(population)

    # Get population indices as highest-to-lowest fitness scores
    inds = np.argsort(scores)

    # Return the scores and the population, both ordered by fitness
    return list(scores[inds][::-1]), list(population[inds, :][::-1])


def selection(scored_population, mating_pool_size):
    """Select the parents to be used for creating the next generation.

    Args:
        scored_population (list): A list of weightings in order of fitness.
        mating_pool_size (int): The number of parents to grab.

    Returns:
        list: A list of weightings to represent the selected parents.
    """
    # Cut the list off at the number of mates specified
    return scored_population[:mating_pool_size]


def crossover(selected_population, population_size, num_weights):
    """Generate the next generation from selected parents.

    Args:
        selected_population (list): A list of parents represented as weightings.
        population_size (int): The number of children to generate.
        num_weights (int): The number of weights within a weighting.

    Returns:
        list: The next generation as a list of children.
    """
    # Initialise a list to store the next generation
    population = []
    for i in range(population_size):
        # Select two random parents from the mating pool
        male = random.choice(selected_population)
        female = random.choice(selected_population)

        # Exchange information based on probability
        child = np.zeros(num_weights)
        p = np.random.rand(num_weights)
        for i in range(len(p)):
            if p[i] < 0.5:
                child[i] = male[i]
            else:
                child[i] = female[i]

        # Normalise the weightings and add the child
        child /= child.sum()
        population.append(child)

    return population


def mutation(population, mutation_rate, num_weights):
    """Mutate a given percentage of the population.

    Args:
        population (list): The current generation, represented as weightings.
        mutation_rate (float): The probability of a mutation (i.e. `0.1` gives a 10% rate).
        num_weights (int): The number of weights within a weighting.

    Returns:
        list: The now-mutated generation as a list of children.
    """
    # Initialise a list to store the mutated generation
    mutated_population = []
    for weighting in population:
        # Initialise a child to store new weights
        mutated_child = np.zeros(num_weights)

        # Iterate through the weighting and possibly multiply
        for i in range(len(weighting)):
            weight = weighting[i]
            if random.random() < mutation_rate:
                weight = weight * 1.1
            mutated_child[i] = weight

        # Normalise the weightings and add the child
        mutated_child /= mutated_child.sum()
        mutated_population.append(mutated_child)

    return mutated_population
