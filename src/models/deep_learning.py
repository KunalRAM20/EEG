"""
Deep-learning seizure models (CNN, LSTM, CNN-LSTM) implemented in PyTorch.

Each network is wrapped in ``TorchClassifier`` which exposes a scikit-learn style
``fit`` / ``predict`` / ``predict_proba`` interface so the training script can
treat classical and deep models uniformly. The wrapper is picklable (it stores
its architecture + weights) so a selected deep model can be persisted with
joblib and reloaded by the web app.

If PyTorch is not installed the module degrades gracefully: ``TORCH_AVAILABLE``
is False and ``build_all_deep`` returns an empty dict, so the rest of the
pipeline (classical models, XAI, reporting, web app) still runs.
"""
from __future__ import annotations

import numpy as np

import config

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when torch is missing
    TORCH_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Network definitions
# --------------------------------------------------------------------------- #
if TORCH_AVAILABLE:

    class CNN1D(nn.Module):
        """1-D convolutional network over the raw EEG window."""

        def __init__(self, window: int, n_classes: int = 2):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(64, n_classes))

        def forward(self, x):                       # x: (B, 1, window)
            return self.head(self.features(x))

    class LSTMNet(nn.Module):
        """LSTM over the EEG window treated as a time series of small chunks."""

        def __init__(self, window: int, hidden: int, n_classes: int = 2,
                     chunk: int = 16):
            super().__init__()
            self.chunk = chunk
            self.lstm = nn.LSTM(chunk, hidden, num_layers=1, batch_first=True)
            self.head = nn.Linear(hidden, n_classes)

        def forward(self, x):                       # x: (B, 1, window)
            b = x.size(0)
            seq = x.view(b, -1, self.chunk)         # (B, window/chunk, chunk)
            out, _ = self.lstm(seq)
            return self.head(out[:, -1, :])

    class CNNLSTM(nn.Module):
        """CNN front-end for local morphology + LSTM for temporal dynamics."""

        def __init__(self, window: int, hidden: int, n_classes: int = 2):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4),
            )
            self.lstm = nn.LSTM(32, hidden, batch_first=True)
            self.head = nn.Linear(hidden, n_classes)

        def forward(self, x):                       # x: (B, 1, window)
            f = self.cnn(x)                         # (B, 32, window/16)
            seq = f.permute(0, 2, 1)                # (B, T, 32)
            out, _ = self.lstm(seq)
            return self.head(out[:, -1, :])

    _ARCHITECTURES = {"CNN": CNN1D, "LSTM": LSTMNet, "CNN_LSTM": CNNLSTM}


# --------------------------------------------------------------------------- #
# scikit-learn style wrapper
# --------------------------------------------------------------------------- #
class TorchClassifier:
    """Uniform wrapper around a PyTorch network for the training pipeline."""

    def __init__(self, arch: str, window: int = config.DL_WINDOW):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available.")
        self.arch = arch
        self.window = window
        self.hidden = config.DL_PARAMS["hidden_size"]
        torch.manual_seed(config.RANDOM_STATE)
        self._build()

    def _build(self):
        cls = _ARCHITECTURES[self.arch]
        if self.arch == "CNN":
            self.model = cls(self.window)
        else:
            self.model = cls(self.window, self.hidden)
        self.device = torch.device("cpu")
        self.model.to(self.device)

    # -- training ---------------------------------------------------------- #
    def fit(self, X, y, groups=None, verbose: bool = False):
        """
        Train with class-weighted loss and early-stopping on a *validation* split.

        Stopping on validation loss (not training loss) is what keeps the network
        from overfitting the training recordings. When ``groups`` is supplied the
        validation split is recording-wise (no recording spans train and val),
        matching the leakage-free protocol used for the classical models.
        """
        p = config.DL_PARAMS
        Xn = np.asarray(X, dtype=np.float32)
        yn = np.asarray(y)

        tr_idx, va_idx = self._val_split(len(yn), yn, groups)
        Xt = torch.tensor(Xn[tr_idx], dtype=torch.float32).unsqueeze(1)
        yt = torch.tensor(yn[tr_idx], dtype=torch.long)
        Xv = torch.tensor(Xn[va_idx], dtype=torch.float32).unsqueeze(1)
        yv = torch.tensor(yn[va_idx], dtype=torch.long)

        generator = torch.Generator().manual_seed(config.RANDOM_STATE)
        loader = DataLoader(TensorDataset(Xt, yt),
                            batch_size=p["batch_size"], shuffle=True,
                            generator=generator)
        opt = torch.optim.Adam(self.model.parameters(), lr=p["learning_rate"])

        # Inverse-frequency class weights counter the seizure/non-seizure imbalance.
        counts = np.bincount(yn[tr_idx], minlength=2).astype(np.float64)
        w = counts.sum() / (2.0 * np.clip(counts, 1.0, None))
        weight = torch.tensor(w, dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=weight)

        best_val, best_state, patience = float("inf"), None, 0
        for epoch in range(p["epochs"]):
            self.model.train()
            total = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * xb.size(0)
            total /= max(1, len(loader.dataset))

            self.model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model(Xv), yv).item())
            if verbose:
                print(f"    [{self.arch}] epoch {epoch + 1:02d} "
                      f"train={total:.4f} val={val_loss:.4f}")

            if val_loss < best_val - 1e-4:
                best_val, patience = val_loss, 0
                best_state = {k: v.detach().clone()
                              for k, v in self.model.state_dict().items()}
            else:
                patience += 1
                if patience >= p["patience"]:
                    break
        if best_state is not None:                 # restore best-validation weights
            self.model.load_state_dict(best_state)
        return self

    def _val_split(self, n, y, groups):
        """Return (train_idx, val_idx); recording-wise when groups are given."""
        from sklearn.model_selection import StratifiedShuffleSplit
        val_frac = config.DL_PARAMS.get("val_fraction", 0.2)
        if groups is not None:
            groups = np.asarray(groups)
            unique_groups, first = np.unique(groups, return_index=True)
            group_y = np.asarray(y)[first]
            for group, label in zip(unique_groups, group_y):
                if np.any(np.asarray(y)[groups == group] != label):
                    raise ValueError(
                        f"recording {group!r} contains mixed labels")
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=val_frac,
                random_state=config.RANDOM_STATE)
            tr_groups, va_groups = next(
                splitter.split(unique_groups, group_y))
            tr = np.flatnonzero(np.isin(groups, unique_groups[tr_groups]))
            va = np.flatnonzero(np.isin(groups, unique_groups[va_groups]))
        else:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_frac,
                                              random_state=config.RANDOM_STATE)
            tr, va = next(splitter.split(np.zeros(n), y))
        return tr, va

    # -- inference --------------------------------------------------------- #
    def predict_proba(self, X):
        X = torch.tensor(np.asarray(X), dtype=torch.float32).unsqueeze(1)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X.to(self.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    # -- pickling (store architecture + weights) --------------------------- #
    def __getstate__(self):
        return {
            "arch": self.arch,
            "window": self.window,
            "hidden": self.hidden,
            "state_dict": {k: v.cpu().numpy()
                           for k, v in self.model.state_dict().items()},
        }

    def __setstate__(self, state):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to load this model.")
        self.arch = state["arch"]
        self.window = state["window"]
        self.hidden = state["hidden"]
        self._build()
        sd = {k: torch.tensor(v) for k, v in state["state_dict"].items()}
        self.model.load_state_dict(sd)


def build_all_deep() -> dict:
    """Return {name: TorchClassifier} for every deep model, or {} if no torch."""
    if not TORCH_AVAILABLE:
        return {}
    return {name: TorchClassifier(name) for name in _ARCHITECTURES}
