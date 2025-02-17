import random
import time
import numpy as np
from matplotlib import pyplot as plt

def main():

    #GridSearch Parameters
    populationSize = [200,600,1000]
    crossoverRate = [0.6, 0.7, 0.8, 0.9]
    mutationRate = [0.4,0.5,0.6,0.8]
    generations = 100
    elitism = 2
    drawPlot =  False
    iterations = 5

    gridSearch("TSP Files/berlin25.txt", populationSize, crossoverRate, mutationRate, generations, elitism, drawPlot, iterations)

    #geneticAlgorithm("TSP Files/pr1002.tsp", 3000, 1500, 0.9, 0.3,2,  True )



def gridSearch(file, populationSize, crossoverRate, mutationRate, generations, elitism, drawPlot, iterations):

    """
        Iterates through given list of parameters and evaluates the average fitness across given iterations.
        Prints out sorted list of the best parameters combinations and their average fitness.

    :param file:
    :param populationSize:
    :param crossoverRate:
    :param mutationRate:
    :param generations:
    :param elitism:
    :param drawPlot:
    :param iterations:
    """


    results = []

    for size in populationSize:
        for crossover in crossoverRate:
            for mutation in mutationRate:
                print(f"Population: {size}, Crossover: {crossover}, Mutation: {mutation}")

                fitnessScores = []
                for _ in range(iterations):
                    bestScore = geneticAlgorithm(file, size, generations, crossover, mutation, elitism, drawPlot)
                    fitnessScores.append(bestScore)

                avgFitness = sum(fitnessScores) / len(fitnessScores)
                results.append((size, crossover, mutation, avgFitness))

    results.sort(key=lambda x: x[3], reverse=True)

    print("Results:")
    for result in results:
        print(f"Population: {result[0]}, Crossover: {result[1]}, Mutation: {result[2]} -> Average Fitness: {result[3]:.4f}")



def geneticAlgorithm(file, populationSize, generations, crossoverRate, mutationRate, elitism, drawPlot):

    """
        Implements selection, crossover and mutation operations to create, explore and calculate fitness of solutions.
        Repeats this over generations aiming to improve fitness each time
        Prints our best fitness produced after each generation
        Plots fitness over generations

    :param file:
    :param populationSize:
    :param generations:
    :param crossoverRate:
    :param mutationRate:
    :param elitism:
    :param drawPlot:
    """
    generationFitness = []
    startTime = time.time()

    dimension, coordinates = readFile(file)

    population = initialPopulation(dimension, populationSize)

    for generation in range(generations):
        fitness = [calculateFitness(coordinates, route) for route in population]
        population = [x for _, x in sorted(zip(fitness, population), key=lambda x: x[0])]

        bestFitness = calculateFitness(coordinates, population[0])
        generationFitness.append(bestFitness)

        print(f"Generation {generation + 1} - Best Fitness: {bestFitness}")

        newPopulation = population[:elitism]

        for _ in range((populationSize - elitism) // 2):

            #parent1 = tournamentSelection(population, fitness, tournamentSize)
            #parent2 = tournamentSelection(population, fitness, tournamentSize)
            parent1, parent2 = random.sample(population[:10], 2) #Select 2 parents from top 10

            if random.random() < crossoverRate:
                #child1, child2 = partiallyMappedCrossover(parent1, parent2)  # Can switch with PMX
                child1, child2 = orderCrossover(parent1, parent2)

                #if random.random() < mutationRate:
                    #child1 = swapMutation(child1)
                    #child2 = swapMutation(child2)

                if random.random() < mutationRate:
                    child1 = inversionMutation(child1)
                    child2 = inversionMutation(child2)

                newPopulation.extend([child1, child2])

        population = newPopulation

        endTime = time.time()
        elapsedTime = endTime - startTime

    bestRoute = population[0]
    bestFitness = calculateFitness(coordinates, bestRoute)

    print("Final Best Route:", bestRoute)
    print("Final Best Fitness:", bestFitness)
    print(f"Total Execution Time: {elapsedTime:.2f} seconds")

    if drawPlot == True:
        plt.plot(range(1, generations + 1), generationFitness, marker='o', linestyle='-')
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.title("Fitness Progression Over Generations")
        plt.grid(True)
        plt.show()

    return bestFitness


def readFile(file):

    """
        Takes in TSP File and parses its data
    :param file:
    :return: dimension, fitness
    """
    infile = open(file, 'r')

    name = infile.readline().strip().split()[1]
    fileType = infile.readline().strip().split()[1]
    comment = infile.readline().strip().split()[1]
    dimension = int(infile.readline().strip().split()[1])
    edgeWeightType = infile.readline().strip().split()[1]

    coordinates = []
    isNode = False

    for line in infile:
        line = line.strip()
        if line.startswith("NODE_COORD_SECTION"):
            isNode = True
            continue
        if line.startswith("EOF"):
            break
        if isNode:
            node = line.split()
            if len(node) >= 3:
                x, y = float(node[1]), float(node[2])
                coordinates.append([x, y])

    infile.close()

    print("Name:", name)
    print("File Type:", fileType)
    print("Comment:", comment)
    print("Dimension:", dimension)
    print("Edge Weight Type:", edgeWeightType)
    print("Coordinates:", coordinates)

    return dimension, coordinates

def initialPopulation(totalLocations, populationSize):

    """
        Generates initial population of given size.
    :param totalLocations:
    :param populationSize:
    :return: randomly initialized population
    """
    randomRoutes = []
    for _ in range(populationSize):
        route = list(range(totalLocations))
        random.shuffle(route)
        randomRoutes.append(route)
    return randomRoutes


def calculateFitness(locations,route):

    """
        Calculates fitness of given route.
    :param locations:
    :param route:
    :return: fitness
    """
    fitness = 0
    for i in range(len(route)):
        fitness += distance(locations[route[i]], locations[route[(i+1) % len(route)]])
    return fitness

def distance(location1, location2):
    """
        Calculates distance between two locations.
    :param location1:
    :param location2:
    :return: distance
    """
    return np.sqrt((location1[0] - location2[0])**2 + (location1[1] - location2[1])**2)


def tournamentSelection(population, fitness, size):
    """
        Selects a parent from the population using tournament selection.

        Tournament selection works by selecting a subset of individuals
        from the population, comparing their fitness values, and returning the
        individual with the best fitness.

    :param population:
    :param fitness:
    :param size:
    :return: Best route based on fitness value
    """
    tournament = random.sample(list(zip(population, fitness)), size)
    tournament.sort(key=lambda x: x[1])
    return tournament[0][0]


def orderCrossover(parent1, parent2):

    """
        Creates two offspring from two given parents using order crossover.
        Order crossover randomly selects a segment from one parent and crosses it with the other parent

    :param parent1:
    :param parent2:
    :return: child1, child2
    """
    size = len(parent1)
    child1, child2 = [-1] * size, [-1] * size

    start, end = sorted(random.sample(range(size), 2))

    child1[start:end+1] = parent1[start:end+1]
    child2[start:end+1] = parent2[start:end+1]

    def fill(child, parent):
        idx = (end + 1) % size
        for gene in parent:
            if gene not in child:
                child[idx] = gene
                idx = (idx + 1) % size

    fill(child1, parent2)
    fill(child2, parent1)

    return child1, child2


def partiallyMappedCrossover(parent1, parent2):

    """
        Creates two offspring using partially mapped crossover.
        It crosses segments of the two parents while maintaining relative order


    :param parent1:
    :param parent2:
    :return: child1, child2
    """
    point1 = random.randrange(0, len(parent1) - 1)
    point2 = random.randrange(point1 + 1, len(parent1))

    child1, child2 = [-1] * len(parent1), [-1] * len(parent1)

    child1[point1:point2] = parent1[point1:point2]
    child2[point1:point2] = parent2[point1:point2]

    def fill(child, parent):
        available_genes = set(parent) - set(child)  # Get unused genes

        for i in range(len(child)):
            if child[i] == -1:
                if available_genes:
                    child[i] = available_genes.pop()  # Pick a unique gene safely
                else:
                    print("Error: No available genes left!")  # Debugging line

    fill(child1, parent2)
    fill(child2, parent1)

    return child1, child2



def swapMutation(parent):

    """
        Applies swap mutation to a parent
        Selected two random points in the solution and swaps them
    :param parent:
    :return: mutated parent
    """
    mutation = parent[:]
    point1, point2 = random.sample(range(len(mutation)), 2)

    temp = mutation[point1]
    mutation[point1] = mutation[point2]
    mutation[point2] = temp

    return mutation

def inversionMutation(parent):

    """
        Applies inversion mutation to a parent
        Selected two random points in the solution and inverses the segment between them
    :param parent:
    :return: mutated parent
    """
    mutation = parent[:]
    point1, point2 = random.sample(range(len(mutation)), 2)

    mutation[point1:point2 + 1] = reversed(mutation[point1:point2 + 1])

    return mutation

if __name__ == "__main__":
    main()