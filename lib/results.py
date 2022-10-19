import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


def differences(array):
    dif = []
    for i in range(len(array)):
        if i != 0:
            dif.append(array[i] - array[i-1])
    return dif


def remove_nan(array):
    array = np.asarray(array)
    nan_array = np.isnan(array)
    not_nan_array = ~ nan_array
    array = array[not_nan_array]
    return array


def get_results(array, ks_test=True):
    kt = stats.kurtosis(array)
    st = np.std(array)
    sk = stats.skew(array)
    mn = np.mean(array)

    if ks_test:
        for i in range(len(array)):
            array[i] = (array[i] - mn) / st
        _, ks = stats.kstest(array, 'norm')
        return kt+3, st, sk, mn, ks
    return kt + 3, st, sk, mn


def evaluate_series(dnn, array):
    kt, st, sk, mn, ks = get_results(differences(array), True)
    loop = dnn.recognize_pattern(array.tolist())
    if ks >= 0.05 and loop is False:
        reg = LinearRegression().fit(array[:len(array) - 1].reshape((-1, 1)), array[1:].reshape((-1, 1)))
        return 1 - float(reg.coef_[0][0])
    else:
        return None
