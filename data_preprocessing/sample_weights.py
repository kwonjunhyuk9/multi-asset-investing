import numpy as np
import pandas as pd


def count_concurrent_events(closeIdx, t1, molecule):
    # Count concurrent events over the molecule range.
    # Include events still open at the end of the sample.
    t1 = t1.fillna(closeIdx[-1])  # Unclosed events still affect other weights.
    t1 = t1[t1 >= molecule[0]]  # Events ending at or after molecule[0].
    t1 = t1.loc[:t1[molecule].max()]  # Events starting before the last relevant end time.

    # Count how many events span each bar.
    iloc = closeIdx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=closeIdx[iloc[0]:iloc[1] + 1])

    for tIn, tOut in t1.items():
        count.loc[tIn:tOut] += 1.

    return count.loc[molecule[0]:t1[molecule].max()]


def compute_average_uniqueness_weights(t1, numCoEvents, molecule):
    # Derive average uniqueness over each event's lifespan.
    wght = pd.Series(index=molecule)

    for tIn, tOut in t1.loc[wght.index].items():
        wght.loc[tIn] = (1. / numCoEvents.loc[tIn:tOut]).mean()

    return wght


def build_indicator_matrix(barIx, t1):
    # Build the indicator matrix.
    indM = pd.DataFrame(0, index=barIx, columns=range(t1.shape[0]))
    for i, (t0, t1_) in enumerate(t1.iteritems()):
        indM.loc[t0:t1_, i] = 1.
    return indM


def compute_average_uniqueness(indM):
    # Compute average uniqueness from the indicator matrix.
    c = indM.sum(axis=1)  # Concurrency.
    u = indM.div(c, axis=0)  # Uniqueness.
    avgU = u[u > 0].mean()  # Average uniqueness.
    return avgU


def sequential_bootstrap(indM, sLength=None):
    # Generate a sample via sequential bootstrap.
    if sLength is None:
        sLength = indM.shape[1]
    phi = []
    while len(phi) < sLength:
        avgU = pd.Series()
        for i in indM:
            indM_ = indM[phi + [i]]  # Reduce indM.
            avgU.loc[i] = compute_average_uniqueness(indM_).iloc[-1]
        prob = avgU / avgU.sum()  # Draw probabilities.
        phi += [np.random.choice(indM.columns, p=prob)]
    return phi


def generate_random_t1(numObs, numBars, maxH):
    # Generate a random t1 series.
    t1 = pd.Series()
    for i in range(numObs):
        ix = np.random.randint(0, numBars)
        val = ix + np.random.randint(1, maxH)
        t1.loc[ix] = val
    return t1.sort_index()


def run_monte_carlo_trial(numObs, numBars, maxH):
    # Run one Monte Carlo trial.
    t1 = generate_random_t1(numObs, numBars, maxH)
    barIx = range(t1.max() + 1)
    indM = build_indicator_matrix(barIx, t1)

    phi = np.random.choice(indM.columns, size=indM.shape[1])
    stdU = compute_average_uniqueness(indM[phi]).mean()

    phi = sequential_bootstrap(indM)
    seqU = compute_average_uniqueness(indM[phi]).mean()

    return {'stdU': stdU, 'seqU': seqU}


def build_monte_carlo_jobs(numObs=10, numBars=100, maxH=5, numIters=1E6, numThreads=24):
    # Build Monte Carlo jobs.
    jobs = []
    for i in range(int(numIters)):
        job = {'func': run_monte_carlo_trial, 'numObs': numObs, 'numBars': numBars, 'maxH': maxH}
        jobs.append(job)


def compute_sample_weights(t1, numCoEvents, close, molecule):
    # Derive sample weights by return attribution.
    ret = np.log(close).diff()  # Log returns are additive.
    wght = pd.Series(index=molecule)

    for tIn, tOut in t1.loc[wght.index].items():
        wght.loc[tIn] = (ret.loc[tIn:tOut] / numCoEvents.loc[tIn:tOut]).sum()

    return wght.abs()


def apply_time_decay(tW, clfLastW=1.):
    # Apply piecewise-linear decay to observed uniqueness.
    # The newest observation gets weight 1.
    # The oldest observation gets weight clfLastW.
    clfW = tW.sort_index().cumsum()

    if clfLastW >= 0:
        slope = (1. - clfLastW) / clfW.iloc[-1]
    else:
        slope = 1. / ((clfLastW + 1) * clfW.iloc[-1])

    const = 1. - slope * clfW.iloc[-1]
    clfW = const + slope * clfW
    clfW[clfW < 0] = 0

    print(const, slope)
    return clfW
