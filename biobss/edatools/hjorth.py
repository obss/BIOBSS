import numpy as np


def activity(signal):
    activity = 0
    s_mean = signal.mean()
    for s in signal:
        activity += np.power(s-s_mean, 2)
    activity = np.log10(activity)
    return activity


def mobility(signal):
    f_derivative = np.gradient(signal, edge_order=1)
    mobility = np.square(np.var(f_derivative)/np.var(signal))
    return mobility


def complexity(signal):

    _mobility = mobility(signal)
    f_derivative = np.gradient(signal, edge_order=1)
    complexity = mobility(f_derivative)/mobility
    # mobility=np.log10(mobility)
    # complexity=np.log10(complexity)
    return complexity, _mobility
