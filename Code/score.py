import numpy as np
import pandas as pd

def calc_score_slow(predictions, targets):
    max_submission = len(predictions.loc[predictions.index[0]])
    scores = []
    idcgs = {i: idcg(i) for i in range(1, max_submission+1)}
    for id in targets.index:
        target = targets.loc[id]['country']
        preds = predictions.loc[id]
        dcg = 0
        for k, pred in enumerate(preds['country']):
            if k > max_submission:
                raise Exception('The entry %s has more destinationes than required.' % id)
            rel = 1 if pred == target else 0
            dcg += float((pow(2, rel) - 1)) / np.log2((k+1)+1)

        scores.append(dcg / idcgs[k])

    return np.mean(scores)


def calc_score_med(predictions, targets):
    """ Merging predictions and targets for faster processing. """
    aggregated_data = pd.DataFrame(index=predictions.index)
    aggregated_data['predictions'] = predictions['country']

    aggregated_data = aggregated_data.join(targets)
    aggregated_data = aggregated_data.rename(columns={'country':'targets'})

    """ Applying transformations. """
    # Matching the predictions with the target.
    groups = aggregated_data.groupby(aggregated_data.index)
    match = lambda row: 1 if row['predictions'] == row['targets'] else 0
    group_match = lambda group: group.apply(match, axis=1)
    match_countries = groups.apply(group_match)
    match_countries.index = aggregated_data.index
    aggregated_data['match'] = match_countries

    # Computing the coefficients it implies.
    aggregated_data['rank'] = range(5) * len(targets.index) # Creates a repetition of 0, 1, 2, 3...
    calc_coef = lambda row: float((pow(2, row['match']) - 1)) / np.log2((row['rank']+1)+1)
    coefs = aggregated_data.apply(calc_coef, axis=1)
    coefs.index = aggregated_data.index
    aggregated_data['coefs'] = coefs

    # Summing the coefs over groups
    groups = aggregated_data['coefs'].groupby(aggregated_data.index)
    coefs_per_id = groups.agg(np.sum)

    return coefs_per_id.mean()


def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k=5, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def calc_score(preds, truth, n_modes=5):
    """
    preds: pd.DataFrame
      one row for each observation, one column for each prediction.
      Columns are sorted from left to right descending in order of likelihood.
    truth: pd.Series
      one row for each obeservation.
    """
    truth = truth['country']
    assert(len(preds)==len(truth))
    r = pd.DataFrame(0, index=preds.index, columns=preds.columns, dtype=np.float64)
    for col in preds.columns:
        r[col] = (preds[col] == truth) * 1.0

    score = pd.Series(r.apply(ndcg_at_k, axis=1, reduce=True), name='score')
    return score.mean()


def idcg(n=5):
    """ Ideal Normalized Discounted cumulative gain. """
    idcg = 0
    idcg += float((pow(2, 1) - 1)) / np.log2((0+1)+1)
    for k in range(1, n+1):
        idcg += float((pow(2, 0) - 1)) / np.log2((k+1)+1)

    return idcg


