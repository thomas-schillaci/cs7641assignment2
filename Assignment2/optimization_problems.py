import time
from copy import deepcopy

import mlrose
import matplotlib.pyplot as plot
import numpy as np


def display_cm(matrix, title, xlabel, ylabel, xparamval, yparamval):
    plot.style.use('default')
    fig, ax = plot.subplots()
    im = ax.imshow(matrix, interpolation='nearest', cmap='magma_r')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(matrix.shape[1]),
           yticks=np.arange(matrix.shape[0]),
           xticklabels=xparamval, yticklabels=yparamval,
           title=title,
           ylabel=ylabel,
           xlabel=xlabel)

    plot.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = 'd'
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j,
                    i,
                    format(int(matrix[i, j]), fmt),
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > thresh else "black"
                    )
    fig.tight_layout()

    np.set_printoptions(precision=2)

    plot.show()


def saw(state):
    center = int(len(state) / 2)

    if state[center] != 1:
        return 0

    fitness = 1

    for j in range(center):
        i = center - j - 1
        if state[i] == j % 2:
            fitness += 1
        else:
            break

    for i in range(1, len(state) - center):
        if state[center + i] == (i + 1) % 2:
            fitness += 1
        else:
            break

    return fitness


def optimize_ga(problem, trials=10):
    pop_sizes = range(100, 701, 100)
    mutation_rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    fitness_values = [[] for _ in range(len(mutation_rates))]

    for i, mutation_rate in enumerate(mutation_rates):
        for pop in pop_sizes:
            samples = []
            time_samples = []
            for _ in range(trials):
                start = time.time()
                _, fitness_value, _ = mlrose.genetic_alg(problem, pop_size=pop, mutation_prob=mutation_rate)
                samples.append(fitness_value)
                time_samples.append(time.time() - start)
                samples.append(fitness_value)
            fitness_values[i].append(np.mean(samples))

    display_cm(
        np.array(fitness_values),
        f'Genetic algorithm tuning on {repr(problem.fitness_fn).split(".")[-1].split(" ")[0]}, length={problem.length}',
        'Population size',
        'Mutation rate',
        pop_sizes,
        mutation_rates
    )


def optimize_sa(problem, trials=50):
    init_temps = np.linspace(1.0, 10.0, 10)
    decay_rates = np.linspace(0.1, 0.99, 10)

    for i in range(len(decay_rates)):
        decay_rates[i] = int(decay_rates[i] * 100) / 100

    fitness_values = [[] for _ in range(len(decay_rates))]

    for i, decay_rate in enumerate(decay_rates):
        for init_temp in init_temps:
            samples = []
            for _ in range(trials):
                decay = mlrose.GeomDecay(init_temp=init_temp, decay=decay_rate)
                _, fitness_value, _ = mlrose.simulated_annealing(problem, decay)
                samples.append(fitness_value)
            fitness_values[i].append(np.mean(samples))

    display_cm(
        np.array(fitness_values),
        f'Simulated annealing tuning on '
        f'{repr(problem.fitness_fn).split(".")[-1].split(" ")[0]}, length={problem.length}',
        'Initial temperature',
        'Decay rate',
        init_temps,
        decay_rates
    )

    fitness_values = [[] for _ in range(len(decay_rates))]

    for i, decay_rate in enumerate(decay_rates):
        for init_temp in init_temps:
            samples = []
            for _ in range(trials):
                decay = mlrose.ArithDecay(init_temp=init_temp, decay=decay_rate)
                _, fitness_value, _ = mlrose.simulated_annealing(problem, decay)
                samples.append(fitness_value)
            fitness_values[i].append(np.mean(samples))

    display_cm(
        np.array(fitness_values),
        f'Simulated annealing tuning on '
        f'{repr(problem.fitness_fn).split(".")[-1].split(" ")[0]}, length={problem.length}',
        'Initial temperature',
        'Decay rate',
        init_temps,
        decay_rates
    )

    init_temps = np.linspace(1.0, 10.0, 10)
    exp_consts = np.linspace(0.0001, 0.003, 10)

    for i in range(len(exp_consts)):
        exp_consts[i] = int(exp_consts[i] * 10000) / 10000

    fitness_values = [[] for _ in range(len(exp_consts))]

    for i, exp_const in enumerate(exp_consts):
        for init_temp in init_temps:
            samples = []
            time_samples = []
            for _ in range(trials):
                decay = mlrose.ExpDecay(init_temp=init_temp, exp_const=exp_const)
                start = time.time()
                _, fitness_value, _ = mlrose.simulated_annealing(problem, decay)
                time_samples.append(time.time() - start)
                samples.append(fitness_value)
            fitness_values[i].append(np.mean(samples))

    display_cm(
        np.array(fitness_values),
        f'Simulated annealing tuning on '
        f'{repr(problem.fitness_fn).split(".")[-1].split(" ")[0]}, length={problem.length}',
        'Initial temperature',
        'Exp const',
        init_temps,
        exp_consts
    )


def optimize_rhc(problem, trials=100):
    fitness_values = []
    fitness_values_std = []
    times = []
    times_std = []

    restarts = range(0, 3001, 500)
    for restart in restarts:
        samples = []
        time_samples = []
        for _ in range(trials):
            start = time.time()
            _, fitness_value, _ = mlrose.random_hill_climb(problem, restarts=restart)
            time_samples.append(time.time() - start)
            samples.append(fitness_value)
        fitness_values.append(np.mean(samples))
        fitness_values_std.append(np.std(samples))
        times.append(np.mean(time_samples))
        times_std.append(np.std(time_samples))

    fitness_values = np.array(fitness_values)
    fitness_values_std = np.array(fitness_values_std)
    times = np.array(times)
    times_std = np.array(times_std)

    with plot.style.context('seaborn-darkgrid'):
        fig, ax1 = plot.subplots()
        plot.title(f'Influence of the number of restarts on RHC')
        ax1.set_xlabel('Number of restarts')
        ax1.set_ylabel('Fitness')
        ax1.tick_params(axis='y')

        ax1.fill_between(restarts,
                         fitness_values + fitness_values_std / 2,
                         fitness_values - fitness_values_std / 2,
                         alpha=0.5
                         )

        ax1.plot(restarts, fitness_values, 'o-')

        with plot.style.context('default'):
            ax2 = ax1.twinx()

            ax2.set_ylabel('Computing time (s)')
            ax2.fill_between(restarts,
                             times + times_std / 2,
                             times - times_std / 2,
                             alpha=0.5,
                             color='darkorange'
                             )

            ax2.plot(restarts, times, color='darkorange')
            ax2.tick_params(axis='y')

        fig.tight_layout()

        plot.show()


def optimize_mimic(problem, trials=10):
    keep_pcts = np.linspace(0.00001, 0.1, 5)
    pop_sizes = range(100, 501, 100)

    fitness_values = [[] for _ in range(len(keep_pcts))]

    for i in range(len(keep_pcts)):
        keep_pcts[i] = int(keep_pcts[i] * 100000) / 100000

    for pop_size in pop_sizes:
        for i, keep_pct in enumerate(keep_pcts):
            samples = []
            for _ in range(trials):
                _, fitness_value, _ = mlrose.mimic(problem, keep_pct=keep_pct, pop_size=pop_size)
                samples.append(fitness_value)
            fitness_values[i].append(np.mean(samples))

    fitness_values = np.array(fitness_values)
    display_cm(
        np.array(fitness_values),
        f'MIMIC tuning on '
        f'{repr(problem.fitness_fn).split(".")[-1].split(" ")[0]}, length={problem.length}',
        'Population size',
        'Keeping percentage',
        pop_sizes,
        keep_pcts
    )


def base_test_algorithms(fitness, lengths):
    return np.array([
        base_test_algorithm(fitness, mlrose.genetic_alg, lengths),
        base_test_algorithm(fitness, mlrose.simulated_annealing, lengths),
        base_test_algorithm(fitness, mlrose.random_hill_climb, lengths),
        base_test_algorithm(fitness, mlrose.mimic, lengths)
    ])


def base_test_algorithm(fitness, algorithm, lengths, trials=10):
    best_fitnesses = []
    best_fitnesses_std = []
    exec_times = []
    exec_times_std = []

    fitness_name = repr(fitness).split('.')[-1].split(' ')[0]

    for length in lengths:

        if fitness_name == 'Knapsack':
            fitness.values = fitness.values[:length]
            fitness.weights = fitness.weights[:length]

        problem = mlrose.DiscreteOpt(length=length, fitness_fn=fitness, maximize=True, max_val=2)

        samples = []
        time_samples = []
        for _ in range(trials):
            start = time.time()
            _, fitness_value, _ = algorithm(problem)
            time_samples.append(time.time() - start)
            samples.append(fitness_value)

        best_fitnesses.append(np.mean(samples))
        best_fitnesses_std.append(np.std(samples))
        exec_times.append(np.mean(time_samples))
        exec_times_std.append(np.std(time_samples))

    best_fitnesses = np.array(best_fitnesses)
    best_fitnesses_std = np.array(best_fitnesses_std)
    exec_times = np.array(exec_times)
    exec_times_std = np.array(exec_times_std)

    return best_fitnesses, best_fitnesses_std, exec_times, exec_times_std


def batches_plot(batches, lengths, fitness_name, theoretical_best_fitness):
    values_batches = batches[:, 0]
    values_std_batches = batches[:, 1]
    times_batches = batches[:, 2]
    times_std_batches = batches[:, 3]

    if fitness_name == 'CustomFitness':
        fitness_name = 'Saw'

    plot.style.use('seaborn-darkgrid')
    plot.title(f'Performance of each algorithm on {fitness_name}')
    plot.xlabel('Problem length')
    plot.ylabel('Fitness')

    for values, values_std in zip(values_batches, values_std_batches):
        plot.fill_between(lengths, values + values_std / 2, values - values_std / 2, alpha=0.5)
        plot.plot(lengths, values, 'o-')

    if theoretical_best_fitness is not None:
        y = []
        for x in lengths:
            y.append(theoretical_best_fitness(x))
        plot.plot(lengths, y, color='black', linestyle='dashed')

    legends = ['Genetic algorithm', 'Simulated annealing', 'Randomized hill climbing', 'MIMIC']
    if theoretical_best_fitness is not None:
        legends.append('Theoretical best fitness')

    plot.legend(legends, loc='upper left')
    plot.show()

    plot.style.use('seaborn-darkgrid')
    plot.title(f'Execution time of each algorithm on {fitness_name}')
    plot.xlabel('Problem length')
    plot.ylabel('Execution time (s)')

    for times, times_std in zip(times_batches, times_std_batches):
        plot.fill_between(lengths, times + times_std / 2, times - times_std / 2, alpha=0.5)
        plot.plot(lengths, times, 'o-')

    plot.legend(['Genetic algorithm', 'Simulated annealing', 'Randomized hill climbing', 'MIMIC'], loc='upper left')
    plot.show()


def base_test(fitness, lengths=range(20, 81, 10), theoretical_best_fitness=None):
    fitness_copy = deepcopy(fitness)
    fitness_name = repr(fitness).split('.')[-1].split(' ')[0]
    batches_plot(base_test_algorithms(fitness_copy, lengths), lengths, fitness_name, theoretical_best_fitness)


def final_test(problem, ga, sa, rhc, mimic, trials=50):
    ga_samples = []
    sa_samples = []
    rhc_samples = []
    mimic_samples = []
    ga_time_samples = []
    sa_time_samples = []
    rhc_time_samples = []
    mimic_time_samples = []

    for _ in range(trials):
        start = time.time()
        _, ga_fitness, _ = mlrose.genetic_alg(problem, pop_size=ga[0], mutation_prob=ga[1])
        ga_time_samples.append(time.time() - start)
        start = time.time()
        _, sa_fitness, _ = mlrose.simulated_annealing(problem, sa)
        sa_time_samples.append(time.time() - start)
        start = time.time()
        _, rhc_fitness, _ = mlrose.random_hill_climb(problem, rhc)
        rhc_time_samples.append(time.time() - start)
        start = time.time()
        _, mimic_fitness, _ = mlrose.mimic(problem, pop_size=mimic[0], keep_pct=mimic[1])
        mimic_time_samples.append(time.time() - start)

        ga_samples.append(ga_fitness)
        sa_samples.append(sa_fitness)
        rhc_samples.append(rhc_fitness)
        mimic_samples.append(mimic_fitness)

    fitness_name = repr(problem.fitness_fn).split('.')[-1].split(' ')[0]
    if fitness_name == 'CustomFitness':
        fitness_name = 'Saw'
    print(f'Final results on {fitness_name}')
    print()
    print(f'GA max: {np.max(ga_samples)}')
    print(f'SA max: {np.max(sa_samples)}')
    print(f'RHC max: {np.max(rhc_samples)}')
    print(f'MIMIC max: {np.max(mimic_samples)}')
    print()
    print(f'GA mean: {np.mean(ga_samples)}')
    print(f'SA mean: {np.mean(sa_samples)}')
    print(f'RHC mean: {np.mean(rhc_samples)}')
    print(f'MIMIC mean: {np.mean(mimic_samples)}')
    print()
    print(f'GA mean execution time: {np.mean(ga_time_samples)}')
    print(f'SA mean execution time: {np.mean(sa_time_samples)}')
    print(f'RHC mean execution time: {np.mean(rhc_time_samples)}')
    print(f'MIMIC mean execution time: {np.mean(mimic_time_samples)}')


# FOUR PEAKS

fourpeaks = mlrose.FourPeaks()
fourpeaks_problem = mlrose.DiscreteOpt(length=60, maximize=True, max_val=2, fitness_fn=fourpeaks)

base_test(fourpeaks, theoretical_best_fitness=lambda x: 2 * x - 0.1 * x - 1)  # 73s
optimize_ga(fourpeaks_problem)  # 768s
optimize_sa(fourpeaks_problem)  # 532s
optimize_rhc(fourpeaks_problem)  # 822s
optimize_mimic(fourpeaks_problem)  # 2464s
final_test(fourpeaks_problem, [700, 0.4], mlrose.ExpDecay(8, 0.00001), 5000, [500, 0.022])  # 1191s

# SAW
saw_fitness = mlrose.CustomFitness(saw, problem_type='discrete')
saw_problem = mlrose.DiscreteOpt(length=700, maximize=True, max_val=2, fitness_fn=saw_fitness)

base_test(saw_fitness, theoretical_best_fitness=lambda x: x, lengths=range(200, 701, 100))  # 2476s
optimize_ga(saw_problem)  # 7255s
optimize_sa(saw_problem)  # 8730s
optimize_rhc(saw_problem)  # 103s
optimize_mimic(saw_problem)  # 2192s
final_test(saw_problem, [600, 0.001], mlrose.ExpDecay(5, 0.0007), 1500, [500, 0.075])  # 14808

# FLIP FLOP

flipflop = mlrose.FlipFlop()
flipflop_problem = mlrose.DiscreteOpt(length=60, maximize=True, max_val=2, fitness_fn=flipflop)

base_test(flipflop, theoretical_best_fitness=lambda x: x - 1, lengths=range(50, 111, 10))  # 364s
optimize_ga(flipflop_problem)  # 1782s
optimize_sa(flipflop_problem)  # 4173s
optimize_rhc(flipflop_problem)  # 3839s
optimize_mimic(flipflop_problem)  # 5750s
final_test(flipflop_problem, [100, 0.1], mlrose.ExpDecay(1, 0.003), 1500, [500, 0.1])  # 1065s
