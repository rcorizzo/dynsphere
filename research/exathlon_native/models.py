"""Label-free, segment-aware interfaces to unchanged model architectures."""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import random
import sys

import numpy as np
import pandas as pd


def segment_slices(entities):
    entities = np.asarray(entities)
    if entities.ndim != 1:
        raise ValueError('Entity IDs must be one dimensional')
    if not len(entities):
        return []
    starts = np.r_[0, np.flatnonzero(entities[1:] != entities[:-1])+1]
    if len(np.unique(entities[starts])) != len(starts):
        raise ValueError('Each entity must occupy one contiguous segment')
    return list(zip(starts, np.r_[starts[1:], len(entities)]))


def window_starts(entities, window, forecast=False):
    if window < 1:
        raise ValueError('Window must be positive')
    arrays = [np.arange(a, b-window+1-int(forecast), dtype=int)
              for a, b in segment_slices(entities) if b-a >= window+int(forecast)]
    return np.concatenate(arrays) if arrays else np.array([], dtype=int)


@contextmanager
def reserved_partition(module, calibration):
    original = module.train_val_split
    def split(train, *args, **kwargs):
        return train.copy(), calibration.copy()
    module.train_val_split = split
    try:
        yield
    finally:
        module.train_val_split = original


def align_observed(length, raw, warmup=0):
    raw = np.asarray(raw, dtype=float).reshape(-1)
    if warmup < 0 or warmup+len(raw) > length or not np.isfinite(raw).all():
        raise ValueError('Invalid genuinely observed model scores')
    values = np.full(length, np.nan)
    values[warmup:warmup+len(raw)] = raw
    return values


def require_support(scores, mask):
    scores, mask = np.asarray(scores), np.asarray(mask, dtype=bool)
    if scores.shape != mask.shape or not np.isfinite(scores[mask]).all():
        raise ValueError('Common support contains unobserved or invalid scores')


def tensor_dataset(values, entities, window, forecast):
    import torch
    from torch.utils.data import Dataset
    starts = window_starts(entities, window, forecast)
    tensor = torch.as_tensor(values, dtype=torch.float32)
    class Windows(Dataset):
        def __len__(self):
            return len(starts)
        def __getitem__(self, index):
            a = starts[index]
            x = tensor[a:a+window]
            y = tensor[a+window:a+window+1] if forecast else x
            return x, y
    dataset = Windows()
    dataset.starts = starts
    return dataset


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class DeepModel:
    def __init__(self, method, configuration, seed):
        self.method, self.config, self.seed = method, dict(configuration), seed
        self.window = int(self.config['window_size'])

    def fit(self, train, train_entities, calibration, calibration_entities):
        if self.method == 'catch':
            self._catch_fit(train, train_entities, calibration, calibration_entities)
        elif self.method == 'mtad_gat':
            self._mtad_fit(train, train_entities)
        else:
            raise ValueError('Unknown native deep comparison')
        return self

    def _catch_fit(self, train, train_entities, calibration, calibration_entities):
        from torch.utils.data import DataLoader
        from benchmarks.official_baselines.adapters.catch import _load_corrected_catch
        cls = _load_corrected_catch()
        module = importlib.import_module(cls.__module__)
        original = module.anomaly_detection_data_provider
        columns = [f'feature_{i}' for i in range(train.shape[1])]
        frame = pd.DataFrame(train, columns=columns)
        validation = pd.DataFrame(calibration, columns=columns)
        def provider(data, batch_size, win_size, step=1, mode='train'):
            if mode not in ('train', 'val'):
                raise ValueError('Unexpected fitting provider mode')
            entities = train_entities if mode == 'train' else calibration_entities
            if len(data) != len(entities):
                raise ValueError('Independent validation partition length differs')
            dataset = tensor_dataset(data.values, entities, win_size, False)
            if not len(dataset):
                raise ValueError('No complete normal within-segment windows')
            return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        seed_all(self.seed)
        self.model = cls(num_epochs=int(self.config['epochs']), seq_len=self.window,
                         batch_size=int(self.config['batch_size']))
        module.anomaly_detection_data_provider = provider
        try:
            with reserved_partition(module, validation):
                self.model.detect_fit(frame, validation)
        finally:
            module.anomaly_detection_data_provider = original
        if not np.allclose(self.model.scaler.mean_, np.mean(train, axis=0)):
            raise ValueError('CATCH scaler did not use the exact fitting execution')
        self.metadata = {'backend': 'author CATCH; coverage-count-v1',
                         'device': str(self.model.device), 'fit_windows': len(self.model.train_data_loader.dataset),
                         'validation_windows': len(self.model.valid_data_loader.dataset)}

    def _mtad_fit(self, train, entities):
        import torch
        from torch.utils.data import DataLoader
        from benchmarks.official_baselines.adapters.common import resolve_torch_device, source_directory
        source = source_directory('mtad_gat')
        sys.path.insert(0, str(source))
        try:
            from mtad_gat import MTAD_GAT
        finally:
            sys.path.pop(0)
        seed_all(self.seed)
        self.device = resolve_torch_device(torch)
        self.model = MTAD_GAT(n_features=train.shape[1], window_size=self.window,
                              out_dim=train.shape[1], use_gatv2=True).to(self.device)
        dataset = tensor_dataset(train, entities, self.window, True)
        if not len(dataset):
            raise ValueError('No complete normal within-segment forecast windows')
        loader = DataLoader(dataset, batch_size=int(self.config['batch_size']), shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config.get('lr', 1e-3)))
        criterion = torch.nn.MSELoss()
        self.model.train()
        for _ in range(int(self.config['epochs'])):
            for windows, targets in loader:
                windows, targets = windows.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                forecast, reconstruction = self.model(windows)
                loss = criterion(forecast, targets.squeeze(1))+criterion(reconstruction, windows)
                loss.backward()
                optimizer.step()
        self.model.eval()
        self.metadata = {'backend': 'ML4ITS MTAD-GAT', 'device': str(self.device), 'fit_windows': len(dataset)}

    def score(self, values, entities):
        scores = np.full(len(values), np.nan)
        for a, b in segment_slices(entities):
            if b-a <= self.window:
                continue
            if self.method == 'catch':
                columns = [f'feature_{i}' for i in range(values.shape[1])]
                raw = self.model.detect_score(pd.DataFrame(values[a:b], columns=columns))[0]
                scores[a:b] = align_observed(b-a, raw)
            else:
                scores[a:b] = self._mtad_score(values[a:b])
        return scores

    def _mtad_score(self, values):
        import torch
        from torch.utils.data import DataLoader
        dataset = tensor_dataset(values, np.zeros(len(values)), self.window, True)
        loader = DataLoader(dataset, batch_size=int(self.config['batch_size']), shuffle=False)
        result = []
        with torch.no_grad():
            for windows, targets in loader:
                windows, targets = windows.to(self.device), targets.to(self.device)
                forecast, reconstruction = self.model(windows)
                fore = torch.mean((forecast-targets.squeeze(1))**2, dim=1)
                recon = torch.mean((reconstruction-windows)**2, dim=(1, 2))
                result.append((fore+recon).cpu().numpy())
        return align_observed(len(values), np.concatenate(result), self.window)
