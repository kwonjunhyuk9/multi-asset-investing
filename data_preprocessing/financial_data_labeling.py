import numpy as np
import pandas as pd


def get_daily_volatility(close, span0=100):
    # Compute daily volatility reindexed to close.
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # Daily returns.
    df0 = df0.ewm(span=span0).std()
    return df0


def apply_profit_taking_stop_loss_on_t1(close, events, ptSl, molecule):
    # Apply stop-loss and profit-taking before the vertical barrier.
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)

    if ptSl[0] > 0:
        pt = ptSl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events.index)  # NaNs.

    if ptSl[1] > 0:
        sl = -ptSl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events.index)  # NaNs.

    for loc, t1 in events_['t1'].fillna(close.index[-1]).items():
        df0 = close[loc:t1]  # Path prices.
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']  # Path returns.
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()  # Earliest stop-loss hit.
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()  # Earliest profit-taking hit.

    return out


def get_events(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    # Filter targets by the minimum return threshold.
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt > minRet]

    # Default to no vertical barrier when t1 is not provided.
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=tEvents)

    # Build the event frame and apply horizontal barriers.
    if side is None:
        side_, ptSl_ = pd.Series(1., index=trgt.index), [ptSl[0], ptSl[0]]
    else:
        side_, ptSl_ = side.loc[trgt.index], ptSl[:2]

    events = pd.concat(
        {'t1': t1, 'trgt': trgt, 'side': side_},
        axis=1
    ).dropna(subset=['trgt'])

    df0 = mpPandasObj(
        func=apply_profit_taking_stop_loss_on_t1,
        pdObj=('molecule', events.index),
        numThreads=numThreads,
        close=close,
        events=events,
        ptSl=ptSl_
    )

    events['t1'] = df0.dropna(how='all').min(axis=1)  # pd.min ignores NaNs.

    if side is None:
        events = events.drop('side', axis=1)

    return events


def get_bins(events, close):
    # Compute labels from realized returns, with optional meta-labeling.
    # Prices must cover both event starts and event end times.
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')

    # Build the return and label output.
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1

    if 'side' in events_:
        out['ret'] *= events_['side']  # Meta-labeling.

    out['bin'] = np.sign(out['ret'])

    if 'side' in events_:
        out.loc[out['ret'] <= 0, 'bin'] = 0  # Meta-labeling.

    return out


def drop_labels(events, minPct=.05):
    # Drop labels with insufficient support.
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > minPct or df0.shape[0] < 3:
            break
        print('dropped label', df0.idxmin(), df0.min())
        events = events[events['bin'] != df0.idxmin()]
    return events
