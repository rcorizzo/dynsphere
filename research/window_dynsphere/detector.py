"""Experimental history-conditioned normal dynamics with one density score."""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import torch
from sklearn.base import BaseEstimator

from .data import history_batches, prepare_training, validate_sequences
from .density import LocalDynamicsDensity
from .network import single_thread, train_encoder


class WindowConditionedDynamicsDetector(BaseEstimator):
    def __init__(self, window=100, hidden_dim=32, latent_dim=8, n_regimes=1,
                 reconstruction_weight=1., epochs=10, batch_size=128,
                 max_targets=20000, random_state=37, channel_independent=False):
        self.window, self.hidden_dim, self.latent_dim = window, hidden_dim, latent_dim
        self.n_regimes, self.reconstruction_weight = n_regimes, float(reconstruction_weight)
        self.epochs, self.batch_size, self.max_targets = epochs, batch_size, max_targets
        self.random_state, self.channel_independent = random_state, channel_independent

    @staticmethod
    def _flatten(xs, groups):
        n, d = xs.shape
        return xs.T.reshape(-1, 1), [idx + j * n for j in range(d) for idx in groups]

    def fit(self, x, entity_ids, times=None, representation_from=None):
        if self.channel_independent and self.n_regimes != 1:
            raise ValueError('The channel-independent control uses one regime')
        p = prepare_training(x, entity_ids, times, self.window, self.max_targets, self.random_state)
        self.scaler_ = p['scaler']
        self.n_features_in_ = p['scaled'].shape[1]
        self.selected_targets_ = p['selected_targets']
        self.fit_rows_, self.calibration_targets_ = p['fit_rows'], p['calibration_targets']
        options = {key: value for key, value in self.get_params().items() if key != 'n_regimes'}
        digest = hashlib.sha256(json.dumps(options, sort_keys=True).encode())
        digest.update(np.ascontiguousarray(p['scaled'][p['fit_rows']]).tobytes())
        digest.update(p['selected_targets'].tobytes())
        for idx in p['groups']:
            digest.update(idx.tobytes())
        self.training_identity_ = digest.hexdigest()
        if representation_from is not None:
            if self.training_identity_ != representation_from.training_identity_:
                raise ValueError('Cannot reuse a representation fitted on different inputs/configuration')
            self.model_ = copy.deepcopy(representation_from.model_)
            self.training_loss_ = list(representation_from.training_loss_)
        else:
            network_x, network_groups = p['scaled'], p['groups']
            network_targets = p['selected_targets']
            if self.channel_independent:
                network_x, network_groups = self._flatten(network_x, network_groups)
                network_targets = np.concatenate([p['selected_targets'] + j * len(x)
                                                  for j in range(self.n_features_in_)])
                if len(network_targets) > self.max_targets:
                    network_targets = network_targets[np.linspace(
                        0, len(network_targets) - 1, self.max_targets, dtype=int)]
            self.model_, self.training_loss_ = train_encoder(
                network_x, network_groups, network_targets, window=self.window,
                hidden_dim=self.hidden_dim, latent_dim=self.latent_dim,
                reconstruction_weight=self.reconstruction_weight, epochs=self.epochs,
                batch_size=self.batch_size, random_state=self.random_state)
        self.model_.eval().requires_grad_(False)
        q, descriptor, previous, _ = self._contexts(p['scaled'], p['groups'], p['selected_targets'])
        cq, cd, cp, cz = self._contexts(p['scaled'], p['groups'], p['calibration_targets'])
        target = p['scaled'][p['selected_targets']]
        calibration_target = p['scaled'][p['calibration_targets']]
        self.densities_ = []
        if self.channel_independent:
            for j in range(self.n_features_in_):
                model = LocalDynamicsDensity(random_state=self.random_state).fit(
                    q[:, j], descriptor[:, j], (target - previous)[:, j:j + 1])
                model.calibrate(cq[:, j], cd[:, j], cp[:, j:j + 1], calibration_target[:, j:j + 1])
                self.densities_.append(model)
        else:
            model = LocalDynamicsDensity(n_regimes=self.n_regimes, random_state=self.random_state).fit(
                q, descriptor, target - previous)
            model.calibrate(cq, cd, cp, calibration_target)
            self.densities_.append(model)
        residual = calibration_target - self._head_predictions(cz)
        self.head_bias_ = residual.mean(0)
        self.head_variance_ = np.maximum(np.mean((residual - self.head_bias_) ** 2, axis=0), 1e-6)
        self.effective_regimes_ = self.densities_[0].effective_regimes_
        self.metadata_ = {
            'requested_regimes': self.n_regimes, 'effective_regimes': self.effective_regimes_,
            'training_loss': self.training_loss_, 'fit_rows': len(p['fit_rows']),
            'calibration_targets': len(p['calibration_targets']),
            'selected_fitting_targets': len(p['selected_targets']),
            'training_identity': self.training_identity_,
            'parameter_count': sum(v.numel() for v in self.model_.parameters()),
            'regime_occupancy': [np.bincount(v.labels_).tolist() for v in self.densities_],
            'variance_min': min(float(v.variances_.min()) for v in self.densities_),
            'variance_max': max(float(v.variances_.max()) for v in self.densities_),
            'score': 'calibrated-predictive-gaussian-mixture-nll',
            'sampling': 'regular-only', 'internal_scaling': 'first-80-percent-per-entity',
            'prediction_units': 'internally-standardized-observations',
        }
        return self

    def _prepare_contexts(self, xs, groups):
        if self.channel_independent:
            xs, groups = self._flatten(xs, groups)
        previous = np.arange(len(xs))
        positions = np.zeros(len(xs), dtype=int)
        group_for = np.zeros(len(xs), dtype=int)
        for group, idx in enumerate(groups):
            previous[idx[1:]] = idx[:-1]
            positions[idx] = np.arange(len(idx))
            group_for[idx] = group
        return xs, groups, previous, positions, group_for

    def _joint_contexts(self, xs, groups, targets, prepared=None):
        targets = np.asarray(targets, dtype=int)
        if prepared is None:
            # This helper receives already flattened values for independent channels.
            previous = np.arange(len(xs))
            positions, group_for = np.zeros(len(xs),dtype=int), np.zeros(len(xs),dtype=int)
            for group, idx in enumerate(groups):
                previous[idx[1:]] = idx[:-1]
                positions[idx] = np.arange(len(idx))
                group_for[idx] = group
        else:
            xs, groups, previous, positions, group_for = prepared
        has_previous_latent = positions[targets] > self.window
        prior_targets = previous[targets[has_previous_latent]]
        needed = np.unique(np.concatenate([targets, prior_targets]))
        values = []
        with single_thread(), torch.no_grad():
            for _, windows in history_batches(xs, groups, needed, self.window, self.batch_size,
                                              lookup=(group_for,positions)):
                values.append(self.model_.encode(torch.as_tensor(windows, dtype=torch.float32)).numpy())
        encoded = np.concatenate(values).astype(float)
        z = encoded[np.searchsorted(needed, targets)]
        velocity = np.zeros_like(z)
        velocity[has_previous_latent] = z[has_previous_latent] - encoded[np.searchsorted(needed, prior_targets)]
        past = xs[previous[targets]]
        descriptor = np.column_stack([z, velocity])
        return np.column_stack([descriptor, past]), descriptor, past, z

    def _contexts(self, xs, groups, targets, prepared=None):
        if not self.channel_independent:
            return self._joint_contexts(xs, groups, targets, prepared)
        if prepared is None:
            prepared = self._prepare_contexts(xs,groups)
        flat, flat_groups = prepared[:2]
        n, d, m = len(xs), xs.shape[1], len(targets)
        requested = np.concatenate([np.asarray(targets) + j * n for j in range(d)])
        q, z, past, latent = self._joint_contexts(flat, flat_groups, requested, prepared)
        return (q.reshape(d, m, -1).transpose(1, 0, 2),
                z.reshape(d, m, -1).transpose(1, 0, 2),
                past.reshape(d, m).T, latent.reshape(d, m, -1).transpose(1, 0, 2))

    def _head_predictions(self, z):
        shape = z.shape
        values = []
        flat = z.reshape(-1, self.latent_dim)
        with single_thread(), torch.no_grad():
            for start in range(0, len(flat), self.batch_size):
                values.append(self.model_.forecast(torch.as_tensor(flat[start:start + self.batch_size],
                                                                   dtype=torch.float32)).numpy())
        predictions = np.concatenate(values).astype(float)
        return predictions.reshape(shape[0], -1) if self.channel_independent else predictions

    def _score(self, x, entity_ids, times, diagnostics, neural_head=False):
        if not hasattr(self, 'densities_'):
            raise ValueError('Fit the estimator before scoring')
        x, groups, _ = validate_sequences(x, entity_ids, times)
        if x.shape[1] != self.n_features_in_:
            raise ValueError('Observation feature count differs from fitting')
        xs = self.scaler_.transform(x)
        prepared = self._prepare_contexts(xs,groups)
        targets = np.concatenate([idx[self.window:] for idx in groups])
        scores = np.full(len(x), np.nan)
        validity = np.zeros(len(x), dtype=bool)
        validity[targets] = True
        result = {'scores': scores, 'validity_mask': validity,
                  'effective_regime_count': self.effective_regimes_}
        if diagnostics:
            k = 1 if self.channel_independent else self.effective_regimes_
            result.update(predictive_means=np.full((len(x), k, x.shape[1]), np.nan),
                          regime_weights=np.full((len(x), k), np.nan),
                          per_regime_log_density=np.full((len(x), k), np.nan))
        for start in range(0, len(targets), self.batch_size):
            rows = targets[start:start + self.batch_size]
            q, descriptor, past, latent = self._contexts(xs, groups, rows, prepared)
            if neural_head:
                means = self._head_predictions(latent) + self.head_bias_
                residual = xs[rows] - means
                scores[rows] = .5 * np.sum(residual ** 2 / self.head_variance_
                                           + np.log(2 * np.pi * self.head_variance_), axis=1)
                continue
            if self.channel_independent:
                parts = [density.predict(q[:, j], descriptor[:, j], past[:, j:j + 1], xs[rows, j:j + 1])
                         for j, density in enumerate(self.densities_)]
                scores[rows] = sum(v['scores'] for v in parts)
                predicted = {'means': np.concatenate([v['means'] for v in parts], axis=2),
                             'weights': np.ones((len(rows), 1)),
                             'log_density': sum(v['log_density'] for v in parts)}
            else:
                predicted = self.densities_[0].predict(q, descriptor, past, xs[rows])
                scores[rows] = predicted['scores']
            if diagnostics:
                result['predictive_means'][rows] = predicted['means']
                result['regime_weights'][rows] = predicted['weights']
                result['per_regime_log_density'][rows] = predicted['log_density']
        return result

    def decision_function(self, x, entity_ids, times=None):
        return self._score(x, entity_ids, times, False)['scores']

    def decision_components(self, x, entity_ids, times=None, diagnostics=False):
        return self._score(x, entity_ids, times, diagnostics)

    def forecast_head_scores(self, x, entity_ids, times=None):
        return self._score(x, entity_ids, times, False, neural_head=True)['scores']
