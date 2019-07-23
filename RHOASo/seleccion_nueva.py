# -*- coding: utf-8 -*-

'''
    RHOASo
	Copyright (C) 2018 Ángel Luis Muñoz Castañeda, David Escudero García,
    Noemí De Castro García, Miguel Carriegos Vieira
	
	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.
	
	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.
	
	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
import numpy as np
import operator
import itertools
import random
from functools import reduce
import sklearn.model_selection


def product(numbers):
    return reduce(operator.__mul__, numbers)


def sign(n):
    return abs(n) / n if n else 0


def func(n):
    return n if n else 1


def ad_sign(n):
    return 1 if n > 0 else 0


def process_hyperparameters(hyperparameters, params_lengths):
    '''
    Creates a dict of parameters from the raw values and its lengths

    Arguments
    ----------------------------
    hyperparameters: tuple shape(m,)
        Contains the current values of parameters
    param_lengths: list
        List of tuples with the names of the params and its length given that a
        scalar is length one and an iterable its length.

    Returns
    ------------------------
    param_dict: dict
        Its keys are the parameter values and its values the corresponding
        values
    '''
    current = 0
    param_dict = {}
    for name, length in params_lengths:
        if length == 1:
            param_dict[name] = hyperparameters[current]
        elif length > 1:
            param_dict[name] = hyperparameters[current:current + length]
        else:
            raise ValueError("length of hyperparameter must be greater than 1")
        current += length
    return param_dict


def get_accuracy(target_model, data, labels, params_lengths, hyperparameters):
    '''
    Computes cross validated accuracy for certain parameter values

    Arguments
    ----------------------------
    target_model: class with the estimator interface of of sklearn.
    data: numpy.ndarray shape(n, k)
        The data on which to train/test the model
    labels: numpy.ndarray shape(n, )
        Labels corresponding to data
    param_lengths: list
        List of tuples with the names of the params and its length given that a
        scalar is length one and an iterable its length.
    hyperparameters: tuple shape(m,)
        Contains the current values of parameters

    Returns
    ------------------------
    accuracy: float
        Mean of accuracies obtained through 10-fold cross validation
    '''
    params = process_hyperparameters(hyperparameters, params_lengths)
    results = sklearn.model_selection.cross_validate(
            target_model(**params), data, labels, scoring='accuracy', cv=10)
    return np.mean(results['test_score'])


def sum_tuples(*tuples):
    '''
    Computes the elementwise sum of several tuples (or iterables really)

    Arguments
    -----------------------------
    tuples: the iterables to sum

    Returns
    ----------------------------
    sum: tuple
        Its elements are the sum of the corresponding elements of each iterable
        in tuples
    '''
    return tuple(map(lambda *elems: reduce(operator.__add__, elems), *tuples))


def get_shifts(n_dims, n=1):
    '''
    Computes the shifts (increments) with n on each position

    Arguments
    --------------------------
    n_dims: int
        Number of elements of the shift
    n: int
        This value or 0 will be the elements of each shift.

    Returns
    --------------------------
    shifts: itertools.product
        Iterable which contains all possible tuples of shape(n_dims, )
        whose elements are n or 0
    '''
    shift_elements = [0, n]
    shifts = itertools.product(shift_elements, repeat=n_dims)
    return shifts


def is_valid(point, max_params_values):
    '''
    Determines whether a point is valid or not: all values are less or equal
    than their maximum.

    Returns an iterable of valid points from another iterable

    Arguments
    --------------------------
    point: tuple shape(m, )
        Contains values of parameters
    max_params_values: list shape(m, )
        The maximum values of parameters.

    Returns
    ---------------------
    valid: bool
        whether point is valid or not
    '''
    return reduce(
        operator.and_, (x <= y for x, y in zip(point, max_params_values))
    )


def select_valid(points, max_params_values):
    '''
    Returns an iterable of valid points from another iterable

    Arguments
    --------------------------
    points: iterable
        Its elements are tuples of the same shape(m, ) with values
        corresponding to parameters.
    max_params_values: list shape(m, )
        The maximum values of parameters.

    Returns
    ---------------------
    valid: iterable
        Valid elements from points
    '''
    return (point for point in points if is_valid(point, max_params_values))


def iter_sample_fast(iterable, samplesize):
    '''
    Stolen from DzinX at
    https://stackoverflow.com/questions/12581437/python-random-sample-with-a-generator-iterable-iterator

    Random sampling of a non sequence iterable

    Arguments
    --------------------------
    iterable: iterable
        iterable to sample from
    samplesize: int
         (approximate) Number of elements to sample from iterable

    Returns
    -------------------
    results: list shape(samplesize, )
        May or may not actually have exactly samplesize elements, but should be
        around that.
    '''
    results = []
    iterator = iter(iterable)
    # Fill in the first samplesize elements:
    try:
        for _ in range(samplesize):
            results.append(next(iterator))
    except StopIteration:
        return results
    random.shuffle(results)  # Randomize their positions
    for i, v in enumerate(iterator, samplesize):
        r = random.randint(0, i)
        if r < samplesize:
            results[r] = v  # at a decreasing rate, replace random items
    return results


def get_new_points(current_point, max_params_values, random=False, n=1):
    '''
    Returns an iterable of valid neighboring points at distance n

        Arguments
    ---------------------------
    current_point: tuple shape(m,)
        Contains the current values of parameters
    max_params_values: list shape(m, )
        The maximum values of parameters.
    random: bool
        If True, only select a random sample of neighbors from all candidates.
        The size of the sample is equal to the dimensionality of the point.
    n: int
        Distance of neighboring points

    Returns
    ----------------------
    validos: list
        Each element is tuple of shape(m, ) which is a valid neighboring
        point to current_point
    '''
    shifts = get_shifts(len(current_point), n)
    candidate_points = (sum_tuples(current_point, shift) for shift in shifts)
    validos = select_valid(candidate_points, max_params_values)
    if random:
        return iter_sample_fast(validos, len(current_point))
    return validos


def get_value(ac, current_point, new_point, option1):
    '''
    Computes the value of current_point relative to new_point

    Arguments
    -------------------------
    ac: dict
        Its keys are tuples whose values are the values of the parameters
        tried and its values the accuracy obtained for those parameter
        values.
    current_point: tuple shape(m, )
        Contains the current values of parameters
    new_point: tuple shape(m, )
        Contains values of parameters.
    option1: int in {0, 1}
        Option for controlling the computation

    Returns
    --------------------------
    value: float
        The value of current_point relative to new_point
    '''
    if current_point == new_point:
        return 1.0
    d = ac[new_point] - ac[current_point]
    if option1 == 0:
        value = d
    if option1 == 1:
        d = func(d)
        alpha = ad_sign(d) - sign(d) * abs(d)
        if alpha != 0:
            value = alpha ** (-sign(d))
        else:
            value = 1
    return value


def compute_stb(ac, current_point, vals, option1, option2):
    '''
    Computes the stabilizer value of a point given the computed list of values.

    Arguments
    -------------------------
    ac: dict
        Its keys are tuples whose values are the values of the parameters
        tried and its values the accuracy obtained for those parameter
        values.
    current_point: tuple
        Contains the current values of parameters
    vals: list
        contains the values of neighboring points relative to current_point
    option1: int in {0, 1}
        Option for controlling the computation of the stabilizer
    option2: int in {0, 1}
        Option for controlling the computation of the stabilizer

    Returns
    -------------------------
    stb: float
        stabilizer value of the point.
    '''
    if option1 == 0:
        tot = sum(vals)
    if option1 == 1:
        tot = product(vals)
    if option2 == 0:
        stb = tot * ac[current_point] * max(current_point)
    if option2 == 1:
        stb = tot * max(current_point) + ac[current_point]
    return stb


def check_accuracy(point, ac, target_model, data, labels, params_lengths):
    '''
    Checks whether the parameter values given by point have been evaluated.
    If not, evaluate it and add it to ac.

    Arguments
    ---------------------------
    point: tuple
        Contains the values of parameters to be evaluated
    ac: dict
        Its keys are tuples whose values are the values of the parameters
        tried and its values the accuracy obtained for those parameter
        values.
    target_model: class with the estimator interface of of sklearn.
    data: numpy.ndarray shape(n, k)
        The data on which to train/test the model
    labels: numpy.ndarray shape(n, )
        Labels corresponding to data
    param_lengths: list
        List of tuples with the names of the params and its length given that a
        scalar is length one and an iterable its length.

    Returns
    ------------------------
    None
    '''
    try:
        ac[point]
    except KeyError:
        ac[point] = get_accuracy(target_model, data, labels, params_lengths,
                                 point)


def get_stb(target_model, current_point, ac, data, labels, option1, option2,
            params_lengths, params_max_values, random):
    '''
    Computes the stabilizer value for a single point

    Arguments
    ---------------------------
    target_model: class with the estimator interface of of sklearn.
    current_point: tuple shape(m, )
        Contains the current values of parameters
    ac: dict
        Its keys are tuples whose values are the values of the parameters
        tried and its values the accuracy obtained for those parameter
        values.
    data: numpy.ndarray shape(n, k)
        The data on which to train/test the model
    labels: numpy.ndarray shape(n, )
        Labels corresponding to data
    option1: int in {0, 1}
        Option for controlling the computation of the stabilizer
    option2: int in {0, 1}
        Option for controlling the computation of the stabilizer
    param_lengths: list
        List of tuples with the names of the params and its length given that a
        scalar is length one and an iterable its length.
    params_max_values: list shape(m, )
        The maximum values of parameters.
    random: bool
        If True, only select a random sample of neighbors from all candidates.
        The size of the sample is equal to the dimensionality of the point.

    Returns
    -------------------------
    stb: float
        stabilizer value of current_point
    '''
    vals = []
    points = get_new_points(current_point, params_max_values, random=random)
    for new_point in points:
        check_accuracy(new_point, ac, target_model, data, labels,
                       params_lengths)
        value = get_value(ac, current_point, new_point, option1)
        vals.append(value)

    stb = compute_stb(ac, current_point, vals, option1, option2)
    return stb


def get_best_neighbour(target_model, current_point, ac, data, labels, option1,
                       option2, params_lengths, params_max_values, random):
    '''
    Computes the best neighbor following the criteria of the stabilizer

    Arguments
    ---------------------------
    target_model: class with the estimator interface of of sklearn.
    current_point: tuple shape(m, )
        Contains the current values of parameters
    ac: dict
        Its keys are tuples whose values are the values of the parameters
        tried and its values the accuracy obtained for those parameter
        values.
    data: numpy.ndarray shape(n, k)
        The data on which to train/test the model
    labels: numpy.ndarray shape(n, )
        Labels corresponding to data
    option1: int in {0, 1}
        Option for controlling the computation of the stabilizer
    option2: int in {0, 1}
        Option for controlling the computation of the stabilizer
    param_lengths: list
        List of tuples with the names of the params and its length given that a
        scalar is length one and an iterable its length.
    params_max_values: list shape(m, )
        The maximum values of parameters.
    random: bool
        If True, only select a random sample of neighbors from all candidates.
        The size of the sample is equal to the dimensionality of the point.

    Returns
    --------------------------
    maximum: Contains the point (a tuple) with the best stabilizer value
        and said value in that order
    '''
    stbs = {}
    points = get_new_points(current_point, params_max_values, random=random)
    for new_point in points:
        check_accuracy(new_point, ac, target_model, data, labels,
                       params_lengths)
        stb = get_stb(target_model, current_point, ac, data, labels, option1,
                      option2, params_lengths, params_max_values, random)
        stbs[new_point] = stb

    maximum = max(stbs.items(), key=lambda x: x[-1])
    return maximum


def last_phase(target_model, current_point, ac, data, labels, params_lengths,
               params_max_values, random, n=2):
    '''
    Selects neighbors at a distance of n and returns the maximum

    Arguments
    ---------------------------
    target_model: class with the estimator interface of of sklearn.
    current_point: tuple shape(m, )
        Contains the current values of parameters
    ac: dict
        Its keys are tuples whose values are the values of the parameters
        tried and its values the accuracy obtained for those parameter
        values.
    data: numpy.ndarray shape(n, k)
        The data on which to train/test the model
    labels: numpy.ndarray shape(n, )
        Labels corresponding to data
    param_lengths: list
        List of tuples with the names of the params and its length given that a
        scalar is length one and an iterable its length.
    params_max_values: list shape(m, )
        The maximum values of parameters.
    random: bool
        If True, only select a random sample of neighbors from all candidates.
        The size of the sample is equal to the dimensionality of the point.
    n: int
        Reach of the final accuracy evaluation of neighboring points

    Returns
    ---------------------------------
    maximum: tuple
        Contains the point (a tuple) with the best accuracy and said accuracy
        in that order
    '''
    accs = {}
    points = get_new_points(current_point, params_max_values, random=random,
                            n=n)
    for new_point in points:
        check_accuracy(new_point, ac, target_model, data, labels,
                       params_lengths)
        accs[new_point] = ac[new_point]

    maximum = max(accs.items(), key=lambda x: x[-1])
    return maximum


def extract_params_lengths(max_pars):
    '''
    Returns a list of tuples with the names of the params and its length
    given that a scalar is length one and iterable its length.

    Arguments
    -----------------------
    max_pars: dict
        Dictionary whose keys are strings with the names of parameters of
        target_model and whose values are the maximum values allowed for that
        parameter

    Returns
    -----------------------
    param_lengths: list
        List of tuples with the names of the params and its length given that a
        scalar is length one and an iterable its length.
    '''
    params = []
    for key, val in max_pars.items():
        param_length = 1  # whether param is scalar or iterable
        try:
            iter(val)
            param_length = len(val)
        except TypeError:
            pass
        finally:
            params.append((key, param_length))
    return params


def extract_max_params_values(max_pars):
    '''
    Extract the maximum values of parameters un list form

    Arguments
    -------------------------
    max_pars: dict
        Dictionary whose keys are strings with the names of parameters of
        target_model and whose values are the maximum values allowed for that
        parameter

    Returns
    -------------------------
    max_values: list
        The maximum values of parameters. The order depends on the internal
        ordering of dictionary keys.
    '''
    max_values_params = []
    for key, val in max_pars.items():
        try:
            iter(val)
            max_values_params.extend(val)
        except TypeError:
            max_values_params.append(val)
    return max_values_params


def selection_n(target_model, data, labels, max_pars, n=2, option1=0,
                option2=0, random=False):
    '''
    Optimizes hyperparameters.

    Arguments
    ----------------------------
    target_model: class with the estimator interface of of sklearn.
    data: numpy.ndarray shape(n, k)
        The data on which to train/test the model
    labels: numpy.ndarray shape(n, )
        Labels corresponding to data
    max_pars: dict
        Dictionary whose keys are strings with the names of parameters of
        target_model and whose values are the maximum values allowed for that
        parameter
    n: int
        Reach of the final accuracy evaluation of neighboring points
    option1: int in {0, 1}
        Option for controlling the computation of the stabilizer
    option2: int in {0, 1}
        Option for controlling the computation of the stabilizer
    random: bool
        If True, only select a random sample of neighbors from all candidates.
        The size of the sample is equal to the dimensionality of the point.

    Returns
    ------------------------------
    results: tuple
        It contains the following elements:
        selected_params: dict
            Its keys are the name of the parameter and the values the selected
            value for that parameter.
        accuracy: float in [0, 1]
            Accuracy obtained using selected_params
        computed: dict
            Its keys are tuples whose values are the values of the parameters
            tried and its values the accuracy obtained for those parameter
            values.
        num_evals: int
            The number of points for which accuracy has been computed. Equal to
            len(computed)
    '''
    ac = {}

    params_lengths = extract_params_lengths(max_pars)
    params_max_values = extract_max_params_values(max_pars)

    current_point = tuple([1 for _ in range(len(params_max_values))])

    while True:
        check_accuracy(current_point, ac, target_model, data, labels,
                       params_lengths)
        stb = get_stb(target_model, current_point, ac, data, labels, option1,
                      option2, params_lengths, params_max_values, random)
        neighbour_point, neighbour_stb = get_best_neighbour(
                target_model, current_point, ac, data, labels, option1,
                option2, params_lengths, params_max_values, random
        )
        if neighbour_stb > stb:
            current_point = neighbour_point
        else:
            break
    final_point, accuracy = last_phase(
            target_model, current_point, ac, data, labels, params_lengths,
            params_max_values, random, n=n
    )
    num_evals = len(ac)
    return (process_hyperparameters(final_point, params_lengths), accuracy, ac,
            num_evals)

