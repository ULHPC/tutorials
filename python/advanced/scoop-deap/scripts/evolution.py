import sys
import numpy
import random
import timeit
import json
import collections
import os
from deap.algorithms import *
from deap import base
from deap import creator
from deap import tools
from deap import benchmarks
from deap import algorithms
from deap import cma

# Create new type dynalically
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Create a toolbox and overload existing functions
toolbox = base.Toolbox()
toolbox.register("evaluate", benchmarks.rastrigin)

def tree():
    ''' 
        Recursive dictionnary with defaultdict 
    '''
    return collections.defaultdict(tree)

def main(N,out_sol_dict):
    '''
        Procedure setting up all the necessary parameters and components for 
        CMAES evolution

        Parameters:
        -----------
        N: Dimension of the problem (number of variables)
        out_sol_dict: Dictionnary to store the results

    '''
    # CMAES strategy
    strategy = cma.Strategy(centroid=[5.0]*N, sigma=5.0, lambda_=20*N)
    # Register the generation and update procedure for the algorithm workflow
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    # Create a set containing the best individual recorded
    hof = tools.HallOfFame(1)
    # Create a statistical object and tell it what you want to monitor
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    # Start the generation and update the population of solutions:w
    algorithms.eaGenerateUpdate(toolbox, ngen=250, stats=stats, halloffame=hof)
    # Get best solution and save it
    best_sol=tools.selBest(hof,1)[0]
    out_sol_dict["solution"]=list(best_sol)
    out_sol_dict["fit"]=float(best_sol.fitness.values[0])

if __name__ == "__main__":
    # Check number of parameters
    assert len(sys.argv)==2, "Please enter the dimension of the problem"
    solutions=tree()
    # Evaluate the running time
    t=timeit.timeit("main({0},solutions)".format(sys.argv[1]),setup="from __main__ import main,solutions",number=1)
    solutions['time']=t
    solutions['cores']=int(os.environ["SLURM_NTASKS"])
    solutions['dimensions']=int(sys.argv[1])
    # Save to json file
    with open('solutions_c{0}_n{1}.json'.format(sys.argv[1],os.environ["SLURM_NTASKS"]), 'w') as json_file:
        json.dump(solutions, json_file,indent=True)
