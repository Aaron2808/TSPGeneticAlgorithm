How It Works
The algorithm follows these steps:

1.Read Input File: Parses a TSPLIB file to extract city coordinates.

2.Initialize Population: Generates a population of random routes.

3.Evaluate Fitness: Calculates the total distance for each route.

4.Selection: Uses elitism for selection

5.Crossover: Can use Partially Mapper Crossover or Order Crossover to create new solutions.

5.Mutation: Applies inversion or swap mutation to introduce variability.

6.Elitism: Retains the best solutions from each generation.

7.Repeat for Generations: Runs until the maximum number of generations is reached.

8.Results Visualization: Displays the fitness progression over generations using a plot.


Running the Algorithm
To run the genetic algorithm with a specific TSP file, use:

geneticAlgorithm("TSP Files/berlin25.tsp", populationSize:1000, generations:200, crossoverRate:0.8, mutationRate:0.6, elitism: 2)

Parameters:
1) file: Path to the TSP instance file.
2) populationSize: Number of solutions in the population.
3) generations: Number of generations to run the algorithm.
4) crossoverRate: Probability of crossover.
5) mutationRate: Probability of mutation.
6) elitism: Number of best solutions carried to the next generation.
7) drawPlot: True or False, if you want to plot the fitness over generations.

Output
The program prints:
1) Best fitness values per generation.
2) Final optimized route and distance.
3) A plot of the fitness progression over generations.

Example Output
Generation 1 - Best Fitness: 12345.67
Generation 2 - Best Fitness: 11234.56
...
Final Best Route: [0, 3, 2, 5, 1, ...]
Final Best Fitness: 7876.54
Total Execution Time: 45.2 seconds



Grid Search for Hyperparameter Tuning
To perform a grid search for optimal hyperparameters, use:

gridSearch(populationSize, crossoverRate, mutationRate, generations, elitism)

Configurable Parameters:
Modify these lists in the script to adjust the search space:

1) populationSize = [700, 1000, 1200]
2) crossoverRate = [0.5, 0.8, 0.9]
3) mutationRate = [0.2, 0.5, 0.8]
4) generations = 100
5) elitism = 2
6) drawPlot = false (Disables plotting during gridSearch)

Notes
1. Ensure that your TSP file follows the standard TSPLIB format.
2. The default crossover method is PMX (Partially Mapped Crossover). Order Crossover can be used instead by modifying the function call.
3. The script includes both inversion and swap mutation; modify the implementation to switch between them.