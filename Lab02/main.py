import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from timed_decorator.simple_timed import timed
from typing import Tuple


predicted = np.array([
    1,1,1,0,1,0,1,1,0,0
])
actual = np.array([
    1,1,1,1,0,0,1,0,0,0
])

big_size = 500000
big_actual = np.repeat(actual, big_size)
big_predicted = np.repeat(predicted, big_size)

@timed(use_seconds=True, show_args=True)
def tp_fp_fn_tn_sklearn(gt: np.ndarray, pred: np.ndarray) -> Tuple[int, ...]:
    tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
    return tp, fp, fn, tn


@timed(use_seconds=True, show_args=True)
def tp_fp_fn_tn_numpy(gt: np.ndarray, pred: np.ndarray) -> Tuple[int, ...]:
    # tn, fp, fn, tp = 0, 0, 0, 0
    # for f, s in zip(gt, pred):
    #     match f, s:
    #         case 1, 1:
    #             tp += 1
    #         case 0, 1:
    #             fp += 1
    #         case 0, 0:
    #             tn += 1
    #         case 1, 0:
    #             fn += 1
    tp = np.sum((gt == 1) & (pred == 1))
    fp = np.sum((gt == 0) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))
    tn = np.sum((gt == 0) & (pred == 0))
    return tp, fp, fn, tn


# DO BETTER!!! din 2 calcule


assert tp_fp_fn_tn_sklearn(actual, predicted) == tp_fp_fn_tn_numpy(actual, predicted)

rez_1 = tp_fp_fn_tn_sklearn(big_actual, big_predicted)
rez_2 = tp_fp_fn_tn_numpy(big_actual, big_predicted)

assert rez_1 == rez_2


# Exercise 2
# Implement a method to retrieve the calculate the accuracy using numpy operations.
@timed(use_seconds=True, show_args=True)
def accuracy_sklearn(gt: np.ndarray, pred: np.ndarray) -> float:
    return accuracy_score(gt, pred)


@timed(use_seconds=True, show_args=True)
def accuracy_numpy(gt: np.ndarray, pred: np.ndarray) -> float:
    return np.sum(np == pred).__truediv__(len(gt))


assert accuracy_sklearn(actual, predicted) == accuracy_numpy(actual, predicted)

rez_1 = accuracy_sklearn(big_actual, big_predicted)
rez_2 = accuracy_numpy(big_actual, big_predicted)

assert np.isclose(rez_1, rez_2)


# Excercise 3
# Implement a method to calculate the F1-Score using numpy operations. Be careful at corner cases (divide by 0).

@timed(use_seconds=True, show_args=True)
def f1_score_sklearn(gt: np.ndarray, pred: np.ndarray) -> float:
    return f1_score(gt, pred)


@timed(use_seconds=True, show_args=True)
def f1_score_numpy(gt: np.ndarray, pred: np.ndarray) -> float:
    precision = np.sum((gt == 1) & (pred == 1)) / (np.sum((gt == 1) & (pred == 1)) + np.sum((gt == 0) & (pred == 1)))
    recall = np.sum((gt == 1) & (pred == 1)) / (np.sum((gt == 1) & (pred == 1)) + np.sum((gt == 1) & (pred == 0)))
    return 2*(precision.__mul__(recall) / (precision + recall))


assert f1_score_sklearn(actual, predicted) == f1_score_numpy(actual, predicted)
rez_1 = f1_score_sklearn(big_actual, big_predicted)
rez_2 = f1_score_numpy(big_actual, big_predicted)

assert np.isclose(rez_1, rez_2)
