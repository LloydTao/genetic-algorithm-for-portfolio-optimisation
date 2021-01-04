import numpy as np
import pandas as pd

from algorithm.genetic import crossover, fitness, init_population, mutation, selection

# Set data parameters
folder = "data/"
extension = ".csv"

# Define set of assets
assets = [
    "AAPL",  # Apple Inc. (AAPL)
    "ADBE",  # Adobe Inc. (ADBE)
    "AMZN",  # Amazon.com, Inc. (AMZN)
    "FB",  # Facebook, Inc. (FB)
    "GOOG",  # Alphabet Inc. (GOOG)
    "MSFT",  # Microsoft Corporation (MSFT)
    "NFLX",  # Netflix, Inc. (NFLX)
    "NVDA",  # NVIDIA Corporation (NVDA)
    "PYPL",  # PayPal Holdings, Inc. (PYPL)
    "TSLA",  # Tesla, Inc. (TSLA)
]

# Build data frame of daily asset prices
prices = pd.DataFrame(columns=assets)
for i in range(len(assets)):
    # Get file
    asset = assets[i]
    file = folder + asset + extension
    # Read file into column
    df = pd.read_csv(file, usecols=["Close"])
    prices[asset] = df["Close"]

# Genetic algorithm parameters
num_weights = len(assets)
population_size = 60
mating_pool_size = 10
generations = 20
mutation_rate = 0.1

# Initalise final values
scored_population = []
scores = []

# Initialise genetic algorithm
population = init_population(population_size, num_weights)
for i in range(generations):

    # Get population and scores in order of fitness score
    scores, scored_population = fitness(prices, population)

    # Select parents to create next generation
    selected_population = selection(scored_population, mating_pool_size)

    # Crossover parents through uniform information exchange to produce children
    crossed_population = crossover(selected_population, population_size, num_weights)

    # Mutate children to search more solution space and avoid local maxima
    population = mutation(crossed_population, mutation_rate, num_weights)

    # Log current generation's stats
    print("Generation: ", i)
    print("Best weighting: ", ["%.2f" % i for i in scored_population[0]])
    print("Fitness: ", scores[0])
    print()


print("Final weightings: ")
for i in range(num_weights):
    print(assets[i], str(round(scored_population[0][i] * 100)) + "%")
print("Sharpe index: ", scores[0])
print()
