import logging
import numpy as np
from joblib import Parallel, delayed
from typing import Tuple

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from mne.stats import bootstrap_confidence_interval

from src.utils.logger import get_logger

logger = get_logger(file_name="classification")
logger.setLevel(logging.INFO)

name_to_func = {"balanced": balanced_accuracy_score, "roc-auc": roc_auc_score}

# -------------------------------------------------------------------------------------------------------------------- #


def _get_indices(times: np.array, start: float, end: float, sfreq: float) -> Tuple[int, int]:
    """
    Determine indices at which the time window begins and ends.
    :param times: array of times in seconds
    :param start: start index for the slice
    :param end: end index for the slice
    :param sfreq: sampling frequency
    :return:
        start index, end index
    """

    start_idx = int((start - times[0]) / (1 / sfreq))
    end_idx = int((end - times[0]) / (1 / sfreq))
    return start_idx, end_idx


def _get_t_steps(window_size: int, sfreq: float) -> int:
    """
    Calculate the number of steps in indices that correspond to the window size given in ms
    :param window_size: size of the analysis window in ms
    :param sfreq: sampling frequency
    :return:
        number of time steps to be analysed
    """

    return int(window_size / (1e3 / sfreq))


def get_slice(x: np.array, t_idx, window_size=-1., sfreq=-1):
    """
    todo comment
    :param x:
    :param t_idx:
    :param window_size:
    :param sfreq:
    :return:
    """

    if window_size < 0:
        return x[..., t_idx]
    else:
        t_steps = _get_t_steps(int(window_size), sfreq)
        return x[..., t_idx - t_steps + 1: t_idx + 1].reshape(x.shape[0], -1)


def classify(x: np.array, y: np.array, cv: int, clf: Pipeline, scoring):
    # todo comment

    kf = StratifiedKFold(cv, shuffle=True)
    scores = np.zeros((cv,))
    dummy_scores = np.zeros((cv,))

    y = y[0].reshape(-1,)  # todo tmp
    dummy_clf = make_pipeline(StandardScaler(), DummyClassifier(strategy="stratified"))

    for i, (train_idx, test_idx) in enumerate(kf.split(x, y)):

        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train
        clf = clone(clf)
        clf.fit(x_train, y_train)

        # Test
        y_pred = clf.predict(x_test)
        scores[i] = scoring(y_test, y_pred)

        # Dummy test
        dummy_clf.fit(x_train, y_train)
        y_pred = dummy_clf.predict(x_test)
        dummy_scores[i] = scoring(y_test, y_pred)

    # Estimate confidence intervals
    lower, upper = bootstrap_confidence_interval(scores, ci=.95, n_bootstraps=2000, stat_fun="mean")
    dummy_lower, dummy_upper = bootstrap_confidence_interval(dummy_scores, ci=.95, n_bootstraps=2000, stat_fun="mean")
    return scores, lower, upper, dummy_scores, dummy_lower, dummy_upper


def classify_temporal(x: np.array, y: np.array, params: dict, n_jobs=1):
    # todo comment

    name_to_obj = {"LinearSVC": LinearSVC(max_iter=params["max-iter"])}

    # Time array
    times = np.arange(params["epochs-tmin"], params["epochs-tmax"], 1 / params["sfreq"])
    start_idx, end_idx = _get_indices(times,
                                      params["classification-tmin"],
                                      params["classification-tmax"],
                                      params["sfreq"])
    clf = make_pipeline(StandardScaler(), name_to_obj[params["clf"]])

    # Create parallel functions per time point
    parallel_funcs = []
    for t_idx in range(start_idx, end_idx):  # noqa
        x_slice = get_slice(x=x, t_idx=t_idx, window_size=params["window-size"], sfreq=params["sfreq"])

        func = delayed(classify)(x=x_slice, y=y, cv=params["cv"],
                                 clf=clf, scoring=roc_auc_score)
        parallel_funcs.append(func)

    logger.debug(f"Total of {len(parallel_funcs)} parallel functions added")
    logger.debug(f"Executing {n_jobs} jobs in parallel")

    parallel_pool = Parallel(n_jobs=n_jobs)
    results = parallel_pool(parallel_funcs)
    results = format_results(data=results, params=params)

    logger.debug(f"{len(parallel_funcs)} time steps processed")
    return results


def format_results(data, params):
    """
    todo comment
    :param data:
    :param params:
    :return:
    """
    # todo tidy

    scores, lowers, uppers = [], [], []
    d_scores, d_lowers, d_uppers = [], [], []
    for s, l, u, ds, dl, du in data:  # per time step
        scores.append(s)
        lowers.append(l)
        uppers.append(u)
        d_scores.append(ds)
        d_lowers.append(dl)
        d_uppers.append(du)

    scores = np.stack(scores, axis=1)  # cv x time steps
    lowers = np.array(lowers)
    uppers = np.array(uppers)
    d_scores = np.stack(d_scores, axis=1)
    d_lowers = np.array(d_lowers)
    d_uppers = np.array(d_uppers)

    results = {"meta": params,
               "data":
                   {
                       "scores": scores,
                       "lowers": lowers,
                       "uppers": uppers,
                       "dummy-scores": d_scores,
                       "dummy-lowers": d_lowers,
                       "dummy-uppers": d_uppers
                   }}
    return results
