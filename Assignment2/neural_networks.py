import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from scipy import stats
from keras import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plot
from keras.utils import to_categorical
import mlrose
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(n_samples=12000):
    dataset1 = pd.read_csv('../wine-reviews/winemag-data-130k-v2.csv')[:int(n_samples / 2)]
    dataset2 = pd.read_csv('../wine-reviews/winemag-data_first150k.csv')[:int(n_samples / 2)]

    x = pd.concat([dataset1, dataset2], ignore_index=True, sort=False)

    del dataset1
    del dataset2

    x = x[x['country'] != '']
    x = x[x['province'] != '']
    x = x[x['variety'] != '']
    x = x[x['description'] != '']

    x = x[['country', 'price', 'province', 'variety', 'description', 'points']]
    x = x.dropna()
    y = x[['points']]
    x = x[['country', 'price', 'province', 'variety', 'description']]

    x = pd.get_dummies(x, columns=['country', 'province', 'variety'])
    x['description'] = x.apply(lambda s: len(s['description']), axis=1)
    x['description'] = (x['description'] - x['description'].min()) / (x['description'].max() - x['description'].min())

    y['points'] = pd.cut(y['points'], 5, labels=[k for k in range(5)])

    y = to_categorical(y)

    data = train_test_split(x, y, random_state=0)

    del x
    del y

    return data


def get_model(input_dim, output_dim):
    model = Sequential()

    model.add(Dense(10, input_dim=input_dim, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model


def learning_curve(lengths=range(100, 15101, 1000)):
    x_train, x_test, y_train, y_test = load_data(n_samples=10000)
    learning_curve = []
    learning_curve_std = []
    trials = 10
    for length in lengths:
        samples = []
        for _ in range(trials):
            model = get_model(len(x_train.keys()), len(y_train[0]))
            model.fit(x_train[:length], y_train[:length], epochs=20, batch_size=50, verbose=False)
            samples.append(model.evaluate(x_train[:length], y_train[:length], verbose=False)[1])
        learning_curve.append(np.mean(samples))
        learning_curve_std.append(np.std(samples))

    learning_curve = np.array(learning_curve)
    learning_curve_std = np.array(learning_curve_std)

    plot.style.use('seaborn-darkgrid')
    plot.title('Learning curve using the adam optimizer')
    plot.ylabel('Categorical accuracy')
    plot.xlabel('Training set size')
    plot.fill_between(lengths, learning_curve + learning_curve_std / 2, learning_curve - learning_curve_std / 2,
                      alpha=0.5)
    plot.plot(lengths, learning_curve, 'o-')
    plot.show()


def flatten_weights(weights):
    weights = np.array(weights)
    res = []
    i = 0
    for layer in [0, 2, 4]:
        for supindex in range(weights[layer].shape[0]):
            for index in range(weights[layer][supindex].size):
                res.append(weights[layer][supindex][index])
                i += 1

    for layer in [1, 3, 5]:
        for index in range(len(weights[layer])):
            res.append(weights[layer][index])
            i += 1

    return np.array(res)


def update_model(model, state):
    weights = np.array(model.get_weights())
    i = 0
    for layer in [0, 2, 4]:
        for supindex in range(weights[layer].shape[0]):
            for index in range(weights[layer][supindex].size):
                weights[layer][supindex][index] = state[i]
                i += 1

    for layer in [1, 3, 5]:
        for index in range(len(weights[layer])):
            weights[layer][index] = state[i]
            i += 1
    model.set_weights(weights)


def weights_distribution():
    x_train, x_test, y_train, y_test = load_data()

    model = get_model(len(x_train.keys()), len(y_train[0]))
    model.fit(x_train, y_train, epochs=20, batch_size=50)
    weights = flatten_weights(np.array(model.get_weights()))

    x = np.linspace(-1, 1, 1000)
    gk = stats.gaussian_kde(weights)
    y = gk.evaluate(x)

    plot.title('Weights distribution')
    plot.xlabel('Weight')
    plot.ylabel('Distribution')
    plot.plot(x, y)
    plot.show()


def propagate(state, model, x_train, y_train):
    update_model(model, state)
    return model.evaluate(x_train, y_train, verbose=False)[0]


def base_test(algorithm, kwargs=None):
    x_train, x_test, y_train, y_test = load_data()

    model = get_model(len(x_train.keys()), len(y_train[0]))

    fitness_kwargs = {'model': model, 'x_train': x_train, 'y_train': y_train}
    fitness = mlrose.CustomFitness(propagate, **fitness_kwargs)
    problem = mlrose.ContinuousOpt(model.count_params(), fitness, maximize=False, min_val=-0.5, max_val=0.5, step=0.01)

    if kwargs is None:
        best_state, best_fitness, history = algorithm(problem, curve=True)
    else:
        best_state, best_fitness, history = algorithm(problem, curve=True, **kwargs)

    algorithm_name = repr(algorithm).split(' ')[1]
    np.save(f'nn_{algorithm_name}_history', history)
    update_model(model, best_state)
    np.save(f'nn_{algorithm_name}_weights', np.array(model.get_weights()))

    print(f'{algorithm_name}: {model.evaluate(x_test, y_test, verbose=False)[1] * 100}% of accuracy')

    plot.style.use('seaborn-darkgrid')
    plot.title(f'Training history for {algorithm_name}')
    plot.ylabel('Categorical crossentropy')
    plot.xlabel('Epoch')
    plot.plot(history)
    plot.show()


def plot_histories():
    ga = np.load('nn_genetic_alg_history.npy.bkp')
    sa = np.load('nn_simulated_annealing_history.npy.bkp')
    rhc = np.load('nn_random_hill_climb_history.npy.bkp')

    plot.style.use('seaborn-darkgrid')
    plot.title(f'Training history for every algorithm')
    plot.ylabel('Categorical crossentropy')
    plot.xlabel('Epoch')
    plot.semilogx(ga)
    plot.semilogx(sa)
    plot.semilogx(rhc)
    plot.legend(['Genetic algorithm', 'Simulated annealing', 'Randomized hill climbing'], loc='lower right')
    plot.show()


learning_curve()
weights_distribution()
base_test(mlrose.genetic_alg)
base_test(mlrose.simulated_annealing, {'schedule': mlrose.ExpDecay(exp_const=0.1), 'max_iters': 1000})
base_test(mlrose.random_hill_climb, {'restarts': 30})
plot_histories()
