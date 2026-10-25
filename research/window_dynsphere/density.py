"""Normal regime-local ridge dynamics and a calibrated Gaussian density."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class LocalDynamicsDensity:
    def __init__(self, n_regimes=1, ridge_alpha=1., variance_floor=1e-6,
                 pseudo_count=32, random_state=37):
        self.n_regimes, self.ridge_alpha = n_regimes, ridge_alpha
        self.variance_floor, self.pseudo_count = variance_floor, pseudo_count
        self.random_state = random_state

    def fit(self, context, descriptor, increments):
        context, descriptor, increments = map(lambda a: np.asarray(a, dtype=float),
                                              (context, descriptor, increments))
        if (self.n_regimes < 1 or self.ridge_alpha < 0 or self.variance_floor <= 0
                or self.pseudo_count <= 0 or len(context) < 2):
            raise ValueError('Invalid local density configuration')
        if any(a.ndim != 2 or len(a) != len(context) or not np.isfinite(a).all()
               for a in (context, descriptor, increments)):
            raise ValueError('Invalid local density fitting arrays')
        self.context_scaler_ = StandardScaler().fit(context)
        self.n_outputs_ = increments.shape[1]
        self.descriptor_scaler_ = StandardScaler().fit(descriptor)
        q, z = self.context_scaler_.transform(context), self.descriptor_scaler_.transform(descriptor)
        k = min(self.n_regimes, len(np.unique(z, axis=0)))
        labels = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit_predict(z)
        minimum = max(32, 2 * (context.shape[1] + 1))
        while len(np.unique(labels)) > 1:
            active, counts = np.unique(labels, return_counts=True)
            small = [int(i) for i, count in zip(active, counts) if count < minimum]
            if not small:
                break
            a = min(small, key=lambda i: (np.sum(labels == i), i))
            eligible = [int(i) for i, count in zip(active, counts) if count >= minimum and i != a]
            if not eligible:
                eligible = [int(i) for i in active if i != a]
            center = z[labels == a].mean(0)
            b = min(eligible, key=lambda i: (np.sum((center - z[labels == i].mean(0)) ** 2), i))
            labels[labels == a] = b
        mapping = {old: new for new, old in enumerate(np.unique(labels))}
        self.labels_ = np.array([mapping[value] for value in labels])
        self.effective_regimes_ = len(mapping)
        self.centers_ = np.stack([z[self.labels_ == i].mean(0) for i in range(len(mapping))])
        self.temperature_ = max(float(np.median(np.sum((z - self.centers_[self.labels_]) ** 2, 1))), 1e-6)
        self.models_ = []
        for i in range(len(mapping)):
            mask = self.labels_ == i
            self.models_.append(Ridge(alpha=self.ridge_alpha * np.sum(mask), fit_intercept=True).fit(
                q[mask], increments[mask]))
        return self

    def _raw(self, context, descriptor, previous_x):
        q = self.context_scaler_.transform(context)
        z = self.descriptor_scaler_.transform(descriptor)
        distances = np.sum((z[:, None, :] - self.centers_[None, :, :]) ** 2, axis=2)
        logits = -distances / (2 * self.temperature_)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        log_weights = logits - logsumexp(logits, axis=1, keepdims=True)
        means = np.stack([model.predict(q).reshape(len(q), self.n_outputs_)
                          for model in self.models_], axis=1)
        return means + np.asarray(previous_x)[:, None, :], log_weights

    def calibrate(self, context, descriptor, previous_x, targets):
        n, d = np.asarray(targets).shape
        if n < 2:
            raise ValueError('Insufficient normal calibration targets')
        mass = np.zeros(self.effective_regimes_)
        sums = np.zeros((self.effective_regimes_, d))
        for start in range(0, n, 512):
            sl = slice(start, start + 512)
            means, logw = self._raw(context[sl], descriptor[sl], previous_x[sl])
            w = np.exp(logw)
            residual = targets[sl, None, :] - means
            mass += w.sum(0)
            sums += (w[:, :, None] * residual).sum(0)
        pooled_mean = sums.sum(0) / n
        biases = np.divide(sums, mass[:, None], out=np.broadcast_to(pooled_mean, sums.shape).copy(),
                           where=mass[:, None] > 1e-12)
        local_squares = np.zeros_like(sums)
        pooled_squares = np.zeros(d)
        for start in range(0, n, 512):
            sl = slice(start, start + 512)
            means, logw = self._raw(context[sl], descriptor[sl], previous_x[sl])
            w = np.exp(logw)[:, :, None]
            residual = targets[sl, None, :] - means
            local_squares += (w * (residual - biases[None, :, :]) ** 2).sum(0)
            pooled_squares += (w * (residual - pooled_mean) ** 2).sum((0, 1))
        pooled_variance = pooled_squares / n
        variance = (local_squares + self.pseudo_count * pooled_variance) / (mass[:, None] + self.pseudo_count)
        self.biases_, self.calibration_mass_ = biases, mass
        self.variances_ = np.maximum(variance, self.variance_floor)
        if not np.isfinite(self.variances_).all() or not np.isfinite(biases).all():
            raise ValueError('Nonfinite normal residual calibration')
        return self

    def predict(self, context, descriptor, previous_x, targets=None):
        means, logw = self._raw(context, descriptor, previous_x)
        means = means + self.biases_[None, :, :]
        result = {'means': means, 'weights': np.exp(logw), 'log_weights': logw}
        if targets is not None:
            residual = np.asarray(targets)[:, None, :] - means
            density = -.5 * np.sum(residual ** 2 / self.variances_[None, :, :]
                                    + np.log(2 * np.pi * self.variances_[None, :, :]), axis=2)
            result.update(log_density=density, scores=-logsumexp(logw + density, axis=1))
        return result
