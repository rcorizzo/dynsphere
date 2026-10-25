"""Window-ridge control with fitting-only preprocessing and normal calibration."""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsqr
from sklearn.linear_model import Ridge

from .data import history_batches, prepare_training, validate_sequences


class WindowRidgeControl:
    def __init__(self, window=100, max_targets=20000, batch_size=128, random_state=37):
        self.window, self.max_targets = window, max_targets
        self.batch_size, self.random_state = batch_size, random_state

    def fit(self, x, entity_ids, times=None):
        p = prepare_training(x, entity_ids, times, self.window, self.max_targets, self.random_state)
        self.scaler_ = p['scaler']
        xs, groups, targets = p['scaled'], p['groups'], p['selected_targets']
        previous = np.arange(len(x))
        for idx in groups:
            previous[idx[1:]] = idx[:-1]
        y = xs[targets] - xs[previous[targets]]
        n, width = len(targets), self.window * xs.shape[1]
        if n * width * 8 <= 512 * 1024 ** 2:
            features = np.concatenate([w.reshape(len(rows), -1) for rows, w in history_batches(
                xs, groups, targets, self.window, self.batch_size)])
            model = Ridge(alpha=float(n)).fit(features, y)
            self.coef_ = np.asarray(model.coef_).reshape(xs.shape[1], width).T
            self.intercept_ = np.atleast_1d(model.intercept_)
            solver = 'dense-ridge'
        else:
            feature_mean = np.zeros(width)
            for rows, w in history_batches(xs, groups, targets, self.window, self.batch_size):
                feature_mean += w.reshape(len(rows), -1).sum(0) / n
            target_mean = y.mean(0)

            def multiply(v):
                values = [(w.reshape(len(rows), -1) - feature_mean) @ v / np.sqrt(n)
                          for rows, w in history_batches(xs, groups, targets, self.window, self.batch_size)]
                return np.r_[np.concatenate(values), v]

            def transpose(v):
                result = v[n:].copy()
                start = 0
                for rows, w in history_batches(xs, groups, targets, self.window, self.batch_size):
                    result += (w.reshape(len(rows), -1) - feature_mean).T @ v[start:start+len(rows)] / np.sqrt(n)
                    start += len(rows)
                return result

            operator = LinearOperator((n + width, width), matvec=multiply, rmatvec=transpose, dtype=float)
            coefficients = []
            for j in range(y.shape[1]):
                solution = lsqr(operator, np.r_[(y[:, j] - target_mean[j]) / np.sqrt(n), np.zeros(width)],
                                atol=1e-6, btol=1e-6, iter_lim=500)
                if solution[1] not in (0, 1, 2, 4, 5):
                    raise ValueError('Window-ridge matrix-free solver did not converge')
                coefficients.append(solution[0])
            self.coef_ = np.column_stack(coefficients)
            self.intercept_ = target_mean - feature_mean @ self.coef_
            solver = 'matrix-free-lsqr-atol-btol-1e-6-iter-limit-500'
        residuals = []
        for rows, w in history_batches(xs, groups, p['calibration_targets'], self.window, self.batch_size):
            means = xs[previous[rows]] + w.reshape(len(rows), -1) @ self.coef_ + self.intercept_
            residuals.append(xs[rows] - means)
        residual = np.concatenate(residuals)
        self.bias_ = residual.mean(0)
        self.variance_ = np.maximum(np.mean((residual - self.bias_) ** 2, axis=0), 1e-6)
        self.metadata_ = {'solver': solver, 'ridge_alpha_on_mean_loss': 1.,
                          'fit_targets': len(targets), 'calibration_targets': len(residual)}
        return self

    def decision_function(self, x, entity_ids, times=None):
        x, groups, _ = validate_sequences(x, entity_ids, times)
        xs = self.scaler_.transform(x)
        previous = np.arange(len(x))
        for idx in groups:
            previous[idx[1:]] = idx[:-1]
        targets = np.concatenate([idx[self.window:] for idx in groups])
        scores = np.full(len(x), np.nan)
        for rows, w in history_batches(xs, groups, targets, self.window, self.batch_size):
            means = xs[previous[rows]] + w.reshape(len(rows), -1) @ self.coef_ + self.intercept_ + self.bias_
            residual = xs[rows] - means
            scores[rows] = .5 * np.sum(residual ** 2 / self.variance_ + np.log(2*np.pi*self.variance_), axis=1)
        return scores
