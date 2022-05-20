import pandas as pd
import numpy as np
import random
from joblib import load
import os

glucose_model = load(os.path.join(os.path.dirname(__file__), 'glucose_model.mdl'))
sbp_model = load(os.path.join(os.path.dirname(__file__), 'sbp_model.mdl'))
dbp_model = load(os.path.join(os.path.dirname(__file__), 'dbp_model.mdl'))

drug_index = [1, 2, 3, 7, 8, 9, 11, 12, 15, 16]
def remove_drugs(population):
    for i in range(population.shape[0]):
        while population[i, drug_index].sum() > 3:
            population[i, random.choice(drug_index)] = 0

def select_parents(population, fitness, n_parents):
    parents = np.empty((n_parents, population.shape[1]))
    for parent in range(n_parents):
        min_fitness_idx = np.where(fitness == np.min(fitness))
        min_fitness_idx = min_fitness_idx[0][0]
        parents[parent, :] = population[min_fitness_idx, :]
        fitness[min_fitness_idx] = 99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = np.uint8(offspring_size[1] / 2)
    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover, alpha=1):
    for idx in range(offspring_crossover.shape[0]):
        random_value = np.random.uniform(0, 1.0)
        if random_value < alpha:
            random_feature = np.random.randint(0, offspring_crossover.shape[1])
            offspring_crossover[idx, random_feature] = np.abs(offspring_crossover[idx, random_feature] - 1)
    return offspring_crossover

def target_function(X, population):
    X[X.columns[11:]] = population
    return np.abs(glucose_model.predict(X)) + np.abs(dbp_model.predict(X)) + np.abs(sbp_model.predict(X))

def init_population(feature_num):
    return np.random.randint(0, 2, size=(100, feature_num))

def get_treatment(X, mode='random'):
    if mode == 'random':
        population = init_population(18)
        remove_drugs(population)
    else:
        population = np.array([X.values[11:]] * 100).shape
    X = pd.DataFrame([X] * 100)
    n_generations = 30
    n_parents = 20
    total_best = []
    total_best_fitness = 999999
    for generation in range(n_generations):
        fitness = target_function(X, population)
        min_fitness_idx = np.where(fitness == np.min(fitness))
        min_fitness_idx = min_fitness_idx[0][0]
        if fitness[min_fitness_idx] < total_best_fitness:
            total_best_fitness = fitness[min_fitness_idx]
            total_best = population[min_fitness_idx, :].copy()
        parents = select_parents(population, fitness, n_parents)
        offspring_crossover = crossover(parents, (100 - n_parents, 18))
        remove_drugs(offspring_crossover)
        offspring_mutation = mutation(offspring_crossover)
        remove_drugs(offspring_mutation)
        population[:n_parents, :] = parents
        population[n_parents:, :] = offspring_mutation
    best_result = total_best.copy()
    return best_result