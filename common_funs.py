import numpy as np
from basic_functions.basic_funs import *

def collect_stats(data, trial_num):    
    if len(data) != trial_num:
        raise ValueError('Incorrect data structure!')
    sample_num = trial_num - np.count_nonzero(np.isnan(data), axis = 0)
    sample_avg = np.nanmean(data, axis = 0)
    sample_std = np.nanstd(data, axis = 0, ddof = 1)
    sample_se = np.divide(sample_std, np.sqrt(sample_num))
    return sample_avg, sample_se
        
def cointoss(coin_num = 1, trial_num = 1, bias = 0.5):
    return np.random.choice(2, size = (trial_num, coin_num), p = [1 - bias, bias])

def labelcount(cat_num, labels, weights = None):
    count, scale = np.zeros(cat_num), np.zeros(cat_num)
    if weights is None:
        weights = np.zeros_like(labels)
    for label, weight in zip(labels, weights):
        count[label] += 1
        scale[label] += weight
    return count, scale

def evenness(probs):
    item_num = len(probs)
    p = np.reshape(probs, (-1, item_num))
    return entropy(p)[0] / np.log(item_num)
    
def verbose_print(verbose, message):
    if verbose:
        print(message)

def randomise_abit(probs, noise):
    p = np.asarray(probs).astype(float)
    p[p < 0] = 0
    p[p > 1] = 1
    if noise < 0:
        e = 0
    elif noise > 1:
        e = 1
    else:
        e = noise
    q = p * (1 - 2 * e) + e
    return q

def ruin_abit(ternary_states, rate):
    x = np.asarray(ternary_states)
    p = np.full_like(x, 1 - rate, dtype = float)
    y = np.multiply(x, collapse(p)) # improve by mask
    return y

def binary_entropy(probs):
    p = np.asarray(probs)
    q = np.zeros_like(p)
    mask = np.logical_and(p > 0, p < 1)
    pm = p[mask]
    xlogx = lambda x : np.multiply(x, np.log(x))
    q[mask] = -(xlogx(pm) + xlogx(1 - pm))
    return q

def remove0vectors(data):
    m = np.asarray(data)
    return m[~np.all(m == 0, axis = 1)]

def biter(binary):
    x = np.asarray(binary, dtype = float)
    y = x * 2 - 1
    return y

def terbi(ternary):
    x = np.asarray(ternary, dtype = float)
    y = (x + 1) / 2
    return y